"""
sac.py — Agent Soft Actor-Critic (SAC).

Implémentation de référence :
    Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018
    https://arxiv.org/abs/1812.05905

Caractéristiques :
    - Double Q-network (réduction overestimation)
    - Entropie adaptative (température α apprise automatiquement)
    - Soft update des réseaux cibles (polyak averaging)
    - Action rescalée de (-1,1) vers [0,1] pour BatteryThermalEnv
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from agent.networks import Actor, Critic
from agent.replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    # Réseaux
    hidden_dim: int   = 256    # largeur des couches cachées
    n_layers: int     = 2      # nombre de couches cachées

    # Entraînement
    lr_actor: float   = 3e-4   # learning rate actor
    lr_critic: float  = 3e-4   # learning rate critic
    lr_alpha: float   = 3e-4   # learning rate température

    gamma: float      = 0.99   # facteur de discount
    tau: float        = 0.005  # coefficient de soft update (polyak)
    batch_size: int   = 256    # taille des batches

    # Buffer
    buffer_capacity: int  = 100_000
    learning_starts: int  = 1_000  # steps avant le premier update

    # Entropie
    init_alpha: float     = 0.2    # valeur initiale de α
    target_entropy: float = None   # None → -dim(action) (recommandé)

    # Misc
    device: str = "cpu"            # "cuda" si GPU disponible


class SACAgent:
    """
    Agent SAC avec entropie adaptative.

    Usage :
        agent = SACAgent(obs_dim=5, action_dim=1)
        agent.push(obs, action, reward, next_obs, done)
        if agent.ready:
            metrics = agent.update()
        action = agent.select_action(obs)
    """

    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig = None):
        self.cfg        = config or SACConfig()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = torch.device(self.cfg.device)

        # --- Réseaux ---
        self.actor  = Actor(obs_dim, action_dim, self.cfg.hidden_dim, self.cfg.n_layers).to(self.device)
        self.critic = Critic(obs_dim, action_dim, self.cfg.hidden_dim, self.cfg.n_layers).to(self.device)

        # Réseau cible critic (pas de gradient, mis à jour par soft update)
        self.critic_target = Critic(obs_dim, action_dim, self.cfg.hidden_dim, self.cfg.n_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # --- Optimiseurs ---
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=self.cfg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # --- Entropie adaptative ---
        target_entropy = self.cfg.target_entropy
        if target_entropy is None:
            target_entropy = -float(action_dim)  # recommandé : -dim(A)
        self.target_entropy = target_entropy

        # log_alpha est le paramètre appris (plus stable numériquement que α direct)
        self.log_alpha = torch.tensor(
            np.log(self.cfg.init_alpha), dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(obs_dim, action_dim, self.cfg.buffer_capacity)

        # --- Compteurs ---
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Ajoute une transition dans le replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    @property
    def ready(self) -> bool:
        """True si on peut commencer l'entraînement."""
        return len(self.buffer) >= self.cfg.learning_starts

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Sélectionne une action à partir d'une observation numpy.

        Returns :
            action numpy ∈ [0, 1]  (rescalée depuis (-1,1))
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_t = self.actor.select_action(obs_t, deterministic=deterministic)
        action_scaled = self._scale_action(action_t)
        return action_scaled.cpu().numpy().flatten()

    def update(self) -> dict[str, float]:
        """
        Effectue une étape de mise à jour SAC.

        Returns :
            dict avec les métriques de la mise à jour
        """
        batch = self.buffer.sample(self.cfg.batch_size, device=self.cfg.device)

        critic_loss = self._update_critic(batch)
        actor_loss, entropy = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)
        self._soft_update_target()

        self.total_updates += 1

        return {
            "critic_loss": critic_loss,
            "actor_loss":  actor_loss,
            "alpha_loss":  alpha_loss,
            "alpha":       self.log_alpha.exp().item(),
            "entropy":     entropy,
        }

    # ------------------------------------------------------------------
    # Étapes d'entraînement
    # ------------------------------------------------------------------

    def _update_critic(self, batch: dict) -> float:
        obs, actions, rewards, next_obs, dones = (
            batch["obs"], batch["actions"], batch["rewards"],
            batch["next_obs"], batch["dones"],
        )

        with torch.no_grad():
            # Action suivante + log_prob depuis la politique courante
            next_actions, next_log_prob = self.actor.sample(next_obs)
            next_actions_scaled = self._scale_action(next_actions)

            # Q cible = min(Q1_target, Q2_target) - α * log_prob
            q1_next, q2_next = self.critic_target(next_obs, next_actions_scaled)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_prob
            q_target = rewards + (1.0 - dones) * self.cfg.gamma * q_next

        # Q courants
        actions_scaled = self._scale_action(actions)  # actions déjà en [0,1]
        q1, q2 = self.critic(obs, actions_scaled)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch: dict) -> tuple[float, float]:
        obs = batch["obs"]

        new_actions, log_prob = self.actor.sample(obs)
        new_actions_scaled = self._scale_action(new_actions)

        q1 = self.critic.q1_forward(obs, new_actions_scaled)

        # Maximise Q - α * log_prob  (minimise le négatif)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), -log_prob.mean().item()

    def _update_alpha(self, batch: dict) -> float:
        obs = batch["obs"]

        with torch.no_grad():
            _, log_prob = self.actor.sample(obs)

        # Minimise α * (-log_prob - target_entropy)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def _soft_update_target(self) -> None:
        """Polyak averaging : θ_target ← τ·θ + (1-τ)·θ_target"""
        tau = self.cfg.tau
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)

    # ------------------------------------------------------------------
    # Utilitaire
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_action(action: torch.Tensor) -> torch.Tensor:
        """Rescale action de (-1, 1) → [0, 1] pour BatteryThermalEnv."""
        return (action + 1.0) / 2.0

    def save(self, path: str) -> None:
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "log_alpha":      self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)

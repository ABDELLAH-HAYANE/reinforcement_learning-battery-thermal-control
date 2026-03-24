"""
networks.py — Réseaux de neurones pour SAC.

Actor  : politique stochastique gaussienne avec squashing tanh.
Critic : double Q-network pour réduire le biais d'overestimation.

Architecture MLP configurable (hidden_dim, n_layers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Bornes du log_std pour la stabilité numérique
LOG_STD_MIN = -5
LOG_STD_MAX = 2


def _mlp(in_dim: int, hidden_dim: int, n_layers: int, out_dim: int) -> nn.Sequential:
    """Construit un MLP avec n_layers couches cachées et activation ReLU."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor — politique gaussienne squashée (tanh)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Politique stochastique : π(a|s) = tanh(μ(s) + σ(s) · ε),  ε ~ N(0,I)

    L'action finale est dans (-1, 1) puis rescalée vers [0, 1] pour
    correspondre à l'espace d'action de BatteryThermalEnv.

    Paramètres :
        obs_dim    : dimension de l'observation
        action_dim : dimension de l'action
        hidden_dim : largeur des couches cachées
        n_layers   : nombre de couches cachées
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.trunk   = _mlp(obs_dim, hidden_dim, n_layers, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (mu, log_std) pour l'observation donnée."""
        h = F.relu(self.trunk(obs))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonne une action et calcule sa log-probabilité.

        Returns :
            action   : action squashée ∈ (-1, 1)  [rescale vers [0,1] dans l'agent]
            log_prob : log π(a|s) corrigé pour le squashing tanh
        """
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)

        # Reparameterization trick
        x = dist.rsample()
        action = torch.tanh(x)

        # Correction de log_prob pour tanh (formule Appendix C de SAC paper)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sélectionne une action pour l'inférence (pas de gradient).

        Args:
            deterministic : si True, utilise μ directement (mode évaluation)
        """
        with torch.no_grad():
            mu, log_std = self(obs)
            if deterministic:
                action = torch.tanh(mu)
            else:
                std = log_std.exp()
                action = torch.tanh(Normal(mu, std).rsample())
        return action


# ---------------------------------------------------------------------------
# Critic — Double Q-network
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Double Q-network : Q1(s,a) et Q2(s,a) pour réduire l'overestimation.

    Les deux têtes partagent la même architecture mais ont des poids indépendants.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = _mlp(in_dim, hidden_dim, n_layers, 1)
        self.q2 = _mlp(in_dim, hidden_dim, n_layers, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (Q1, Q2) pour le calcul de la loss critic."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Retourne uniquement Q1 — utilisé pour la mise à jour de l'actor."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)

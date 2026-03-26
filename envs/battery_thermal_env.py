"""
battery_thermal_env.py — Environnement Gymnasium pour le contrôle thermique batterie.

Observation (5 dimensions, normalisées dans [-1, 1]) :
    [T_cell_norm, SoC_norm, I_charge_norm, T_amb_norm, P_cool_prev_norm]

Action (continue, 1 dimension) :
    a ∈ [0, 1]  →  fraction de puissance de refroidissement maximale

Récompense :
    r = - w1 * max(0, T - T_max)²
      - w2 * max(0, T_min - T)²
      - w3 * a
      + w4 * delta_SoC

Fin d'épisode :
    - T > T_cutoff  (sécurité thermique)
    - SoC >= 1.0    (batterie pleine)
    - step >= max_steps
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.thermal_model import ThermalModel
from config import EnvConfig, DEFAULT_CONFIG


class BatteryThermalEnv(gym.Env):
    """
    Environnement de contrôle thermique batterie compatible Gymnasium.

    Usage rapide :
        env = BatteryThermalEnv()
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig = None, render_mode: str = None):
        super().__init__()
        self.cfg = config or DEFAULT_CONFIG
        self.tc = self.cfg.thermal
        self.rc = self.cfg.reward
        self.render_mode = render_mode

        self.model = ThermalModel(self.tc)

        # --- Espaces ---
        # Action : fraction de refroidissement ∈ [0, 1]
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation : [T_norm, SoC_norm, I_norm, T_amb_norm, P_cool_prev_norm]
        # toutes dans [-1, 1] après normalisation
        self.observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(5,),
            dtype=np.float32,
        )

        # État interne
        self._T: float = 0.0
        self._T_prev: float = 0.0   # température au step précédent (pour dT/dt)
        self._SoC: float = 0.0
        self._T_amb: float = 0.0
        self._I: float = 0.0
        self._action_prev: float = 0.0
        self._step_count: int = 0

        # Historique pour render / analyse
        self.history: dict = {}

    # ------------------------------------------------------------------
    # Interface Gymnasium
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._T = self.np_random.uniform(self.tc.T_init_min, self.tc.T_init_max)
        self._T_prev = self._T
        self._SoC = self.np_random.uniform(self.tc.SoC_init_min, self.tc.SoC_init_max)
        self._T_amb = self.np_random.uniform(self.tc.T_amb_min, self.tc.T_amb_max)
        self._I = self._sample_current()
        self._action_prev = 0.0
        self._step_count = 0

        self.history = {"T": [], "SoC": [], "I": [], "T_amb": [], "action": [], "reward": []}

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action_scalar = float(np.clip(np.asarray(action).flat[0], 0.0, 1.0))

        SoC_prev = self._SoC
        T_prev   = self._T   # pour calcul dT

        # --- Mise à jour thermique ---
        T_next, P_gen, P_cool = self.model.step(
            self._T, self._I, action_scalar, self._T_amb
        )

        # --- Mise à jour SoC (intégration courant) ---
        # dSoC = I * dt / (3600 * C_batt)  — I positif = charge
        dSoC = self._I * self.tc.dt / (3600.0 * self.tc.C_batt)
        SoC_next = float(np.clip(self._SoC + dSoC, 0.0, 1.0))

        # --- Variation courant (profil stochastique léger) ---
        self._I = self._update_current()
        self._T_amb = self._update_T_amb()

        # --- Sauvegarde état ---
        self._T_prev = T_prev
        self._T = T_next
        self._SoC = SoC_next
        self._action_prev = action_scalar
        self._step_count += 1

        # --- Récompense ---
        dT = T_next - T_prev
        reward = self._compute_reward(T_next, SoC_next, action_scalar, SoC_prev, dT=dT)

        # --- Conditions de fin ---
        terminated = bool(T_next > self.tc.T_cutoff or SoC_next >= 1.0)
        truncated = bool(self._step_count >= self.tc.max_steps)

        # --- Logging historique ---
        self.history["T"].append(T_next)
        self.history["SoC"].append(SoC_next)
        self.history["I"].append(self._I)
        self.history["T_amb"].append(self._T_amb)
        self.history["action"].append(action_scalar)
        self.history["reward"].append(reward)

        obs = self._get_obs()
        info = self._get_info(P_gen=P_gen, P_cool=P_cool)

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Normalise l'état dans [-1, 1]."""
        T_norm = self._normalize(
            self._T,
            lo=self.tc.T_amb_min - 5.0,
            hi=self.tc.T_cutoff,
        )
        SoC_norm = self._normalize(self._SoC, lo=0.0, hi=1.0)
        I_norm = self._normalize(self._I, lo=self.tc.I_min, hi=self.tc.I_max)
        T_amb_norm = self._normalize(self._T_amb, lo=self.tc.T_amb_min, hi=self.tc.T_amb_max)
        P_cool_prev_norm = self._normalize(self._action_prev, lo=0.0, hi=1.0)

        return np.array(
            [T_norm, SoC_norm, I_norm, T_amb_norm, P_cool_prev_norm],
            dtype=np.float32,
        )

    def _get_info(self, P_gen: float = 0.0, P_cool: float = 0.0) -> dict:
        return {
            "T": self._T,
            "SoC": self._SoC,
            "I": self._I,
            "T_amb": self._T_amb,
            "step": self._step_count,
            "P_gen": P_gen,
            "P_cool": P_cool,
        }

    def _compute_reward(
        self, T: float, SoC: float, action: float, SoC_prev: float, dT: float = 0.0
    ) -> float:
        rc = self.rc
        tc = self.tc

        # Pénalités thermiques (quadratiques)
        penalty_high = rc.w_temp_high * max(0.0, T - tc.T_safe_max) ** 2
        penalty_low  = rc.w_temp_low  * max(0.0, tc.T_safe_min - T) ** 2

        # Coût énergétique du refroidissement
        cost_cooling = rc.w_cooling * action

        # Bonus progression de charge
        bonus_soc = rc.w_soc * max(0.0, SoC - SoC_prev)

        # Pénalité vitesse de montée thermique — anticipe les pics
        # Pénalise uniquement si T dépasse T_warning ET monte (dT > 0)
        penalty_rise = rc.w_delta_T * max(0.0, dT) * max(0.0, T - tc.T_warning)

        return float(-penalty_high - penalty_low - cost_cooling + bonus_soc - penalty_rise)

    def _sample_current(self) -> float:
        """Courant initial aléatoire dans la plage configurée."""
        return float(self.np_random.uniform(self.tc.I_min, self.tc.I_max))

    def _update_current(self) -> float:
        """Profil de courant stochastique : marche aléatoire bornée."""
        noise = self.np_random.normal(0.0, 2.0)
        I_new = float(np.clip(self._I + noise, self.tc.I_min, self.tc.I_max))
        return I_new

    def _update_T_amb(self) -> float:
        """Température ambiante : variation lente."""
        noise = self.np_random.normal(0.0, 0.05)
        return float(np.clip(self._T_amb + noise, self.tc.T_amb_min, self.tc.T_amb_max))

    @staticmethod
    def _normalize(value: float, lo: float, hi: float) -> float:
        """Mappe [lo, hi] → [-1, 1]."""
        if hi == lo:
            return 0.0
        return float(2.0 * (value - lo) / (hi - lo) - 1.0)

    def _render_human(self):
        print(
            f"step={self._step_count:4d} | "
            f"T={self._T:5.1f}°C | "
            f"SoC={self._SoC:.3f} | "
            f"I={self._I:6.1f}A | "
            f"T_amb={self._T_amb:.1f}°C | "
            f"cool={self._action_prev:.2f}"
        )

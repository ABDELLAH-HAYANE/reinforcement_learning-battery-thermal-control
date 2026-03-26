"""
rule_based.py — Contrôleurs déterministes de référence.

Ces contrôleurs ne sont pas entraînés : ils codent en dur une
stratégie thermique simple. Ils servent de baseline pour évaluer
l'apport du SAC.

Observation layout (depuis BatteryThermalEnv._get_obs) :
    obs[0] = T_norm        ∈ [-1, 1]   (T_lo=T_amb_min-5, T_hi=T_cutoff)
    obs[1] = SoC_norm      ∈ [-1, 1]
    obs[2] = I_norm        ∈ [-1, 1]
    obs[3] = T_amb_norm    ∈ [-1, 1]
    obs[4] = P_cool_prev   ∈ [-1, 1]

Tous les contrôleurs exposent la même interface que SACAgent :
    action = controller.select_action(obs, deterministic=True) -> np.ndarray shape (1,)
"""

import numpy as np
from config import ThermalConfig


def _denorm_T(obs0: float, tc: ThermalConfig) -> float:
    """Dénormalise obs[0] → température réelle [°C]."""
    T_lo = tc.T_amb_min - 5.0
    T_hi = tc.T_cutoff
    return (obs0 + 1.0) / 2.0 * (T_hi - T_lo) + T_lo


class RandomController:
    """
    Politique aléatoire uniforme — borne inférieure de performance.
    Compatible avec l'interface select_action.
    """

    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return np.array([self._rng.random()], dtype=np.float32)


class BangBangController:
    """
    Contrôleur tout-ou-rien (bang-bang).

    Règle :
        si T > T_safe_max  → refroidissement à fond (action = 1.0)
        sinon              → pas de refroidissement  (action = 0.0)

    Simple et robuste, mais oscillant et énergivore.
    """

    def __init__(self, tc: ThermalConfig = None):
        self.tc = tc or ThermalConfig()

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        T = _denorm_T(float(obs[0]), self.tc)
        action = 1.0 if T > self.tc.T_safe_max else 0.0
        return np.array([action], dtype=np.float32)


class ProportionalController:
    """
    Contrôleur proportionnel (P-controller).

    Règle :
        T_opt = (T_safe_min + T_safe_max) / 2
        error = T - T_opt
        action = clip(gain * error / (T_cutoff - T_opt), 0, 1)

    Lisse, sans oscillation, mais sans mémoire de l'état passé.
    Le gain contrôle l'agressivité du refroidissement.
    """

    def __init__(self, tc: ThermalConfig = None, gain: float = 1.5):
        self.tc   = tc or ThermalConfig()
        self.gain = gain
        self._T_opt  = (tc.T_safe_min + tc.T_safe_max) / 2.0 if tc else 30.0
        self._T_range = self.tc.T_cutoff - self._T_opt

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        T = _denorm_T(float(obs[0]), self.tc)
        error  = T - self._T_opt
        action = float(np.clip(self.gain * error / self._T_range, 0.0, 1.0))
        return np.array([action], dtype=np.float32)


class HysteresisController:
    """
    Contrôleur bang-bang avec hystérésis.

    Evite les oscillations rapides en ajoutant une zone morte :
        - Active refroidissement max si T > T_safe_max
        - Coupe refroidissement si T < T_safe_max - hysteresis
        - Maintient l'état précédent entre les deux seuils

    Paramètre :
        hysteresis : largeur de la zone morte [°C], ex. 3.0
    """

    def __init__(self, tc: ThermalConfig = None, hysteresis: float = 3.0):
        self.tc          = tc or ThermalConfig()
        self.hysteresis  = hysteresis
        self._cooling_on = False

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        T = _denorm_T(float(obs[0]), self.tc)
        T_on  = self.tc.T_safe_max
        T_off = self.tc.T_safe_max - self.hysteresis

        if T > T_on:
            self._cooling_on = True
        elif T < T_off:
            self._cooling_on = False
        # entre T_off et T_on : maintient état courant

        action = 1.0 if self._cooling_on else 0.0
        return np.array([action], dtype=np.float32)

    def reset(self) -> None:
        """À appeler entre chaque épisode pour réinitialiser l'état interne."""
        self._cooling_on = False

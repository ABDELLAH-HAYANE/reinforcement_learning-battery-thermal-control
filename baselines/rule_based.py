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

    def reset(self) -> None:
        pass


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

    def reset(self) -> None:
        pass


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


class PIDController:
    """
    Contrôleur PID pour le refroidissement thermique batterie.

    Setpoint = T_opt = (T_safe_min + T_safe_max) / 2

    Loi de commande :
        error     = T - T_opt
        u(t)      = Kp*e + Ki*∫e dt + Kd*de/dt
        action    = clip(u / u_max, 0, 1)

    Anti-windup : l'intégrale est bornée à [-windup_limit, +windup_limit]
    pour éviter la saturation lors de grands dépassements prolongés.

    Paramètres par défaut calibrés pour BatteryThermalEnv (T_safe=[15,45]°C) :
        Kp = 0.05   — proportionnel : ~0.5 d'action pour 10°C d'erreur
        Ki = 0.002  — intégral      : correction lente des erreurs persistantes
        Kd = 0.10   — dérivé        : amortit les montées rapides
    """

    def __init__(
        self,
        tc: ThermalConfig = None,
        Kp: float = 0.05,
        Ki: float = 0.002,
        Kd: float = 0.10,
        windup_limit: float = 20.0,
    ):
        self.tc           = tc or ThermalConfig()
        self.Kp           = Kp
        self.Ki           = Ki
        self.Kd           = Kd
        self.windup_limit = windup_limit
        self._T_opt       = (self.tc.T_safe_min + self.tc.T_safe_max) / 2.0
        self._integral    = 0.0
        self._prev_error  = 0.0

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        T     = _denorm_T(float(obs[0]), self.tc)
        error = T - self._T_opt
        dt    = self.tc.dt

        # Terme intégral avec anti-windup
        self._integral = float(np.clip(
            self._integral + error * dt,
            -self.windup_limit,
            self.windup_limit,
        ))

        # Terme dérivé
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        action = float(np.clip(u, 0.0, 1.0))
        return np.array([action], dtype=np.float32)

    def reset(self) -> None:
        """À appeler entre chaque épisode pour réinitialiser l'intégrateur."""
        self._integral   = 0.0
        self._prev_error = 0.0

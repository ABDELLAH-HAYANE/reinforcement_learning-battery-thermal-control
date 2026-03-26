"""
curriculum_env.py — Wrapper Gymnasium pour entraînement par curriculum.

Augmente progressivement la difficulté en élargissant les plages de
température ambiante et de courant à mesure que l'entraînement avance.

Stages (par défaut) :
    0  [    0 – 50k steps]  Facile  : T_amb étroit, courant modéré
    1  [ 50k – 120k steps]  Moyen   : T_amb élargi, courant plus fort
    2  [120k+      steps]   Plein   : config complète (identique à ThermalConfig)

Usage :
    env = CurriculumEnv(BatteryThermalEnv())
    env = CurriculumEnv(BatteryThermalEnv(), stages=CUSTOM_STAGES)
"""

import numpy as np
import gymnasium as gym

from envs.battery_thermal_env import BatteryThermalEnv


# ---------------------------------------------------------------------------
# Définition des stages
# ---------------------------------------------------------------------------

# Chaque stage : (step_min, dict des attributs ThermalConfig à modifier)
DEFAULT_STAGES = [
    (0, dict(
        T_amb_min=18.0, T_amb_max=30.0,   # ambiant stable
        I_max=25.0,     I_min=-5.0,        # courant modéré
    )),
    (50_000, dict(
        T_amb_min=16.0, T_amb_max=33.0,   # ambiant plus variable
        I_max=38.0,     I_min=-8.0,
    )),
    (120_000, dict(
        T_amb_min=15.0, T_amb_max=35.0,   # config complète
        I_max=50.0,     I_min=-10.0,
    )),
]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class CurriculumEnv(gym.Wrapper):
    """
    Wraps BatteryThermalEnv et fait progresser la difficulté automatiquement.

    La transition entre stages se fait au prochain reset() après avoir
    atteint le seuil de steps, pour ne pas interrompre un épisode en cours.

    Attributs utiles :
        .total_steps     : steps cumulés depuis la création
        .current_stage   : index du stage actif (0, 1, 2...)
        .stage_log       : liste des (step, stage) de chaque transition

    Info dict enrichi avec :
        "curriculum_stage"  : stage actif
        "curriculum_steps"  : steps cumulés
    """

    def __init__(self, env: BatteryThermalEnv, stages: list = None):
        super().__init__(env)
        self._stages        = stages or DEFAULT_STAGES
        self.total_steps    = 0
        self.current_stage  = -1          # forcera l'application du stage 0
        self.stage_log: list[tuple] = []
        self._pending_stage = None        # stage à appliquer au prochain reset
        self._apply_stage(0, force=True)

    # ------------------------------------------------------------------
    # Interface Gymnasium
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        # Applique une transition en attente (transition propre à la fin d'épisode)
        if self._pending_stage is not None:
            self._apply_stage(self._pending_stage)
            self._pending_stage = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        # Vérifie si un nouveau stage doit être activé
        next_stage = self._next_stage()
        if next_stage is not None and next_stage != self._pending_stage:
            self._pending_stage = next_stage

        info["curriculum_stage"] = self.current_stage
        info["curriculum_steps"] = self.total_steps
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Gestion des stages
    # ------------------------------------------------------------------

    def _next_stage(self) -> int | None:
        """Retourne le stage cible selon total_steps, ou None si déjà atteint."""
        target = self.current_stage
        for i, (min_steps, _) in enumerate(self._stages):
            if self.total_steps >= min_steps:
                target = i
        return target if target > self.current_stage else None

    def _apply_stage(self, stage_idx: int, force: bool = False) -> None:
        """Applique les overrides du stage sur self.env.tc."""
        if stage_idx == self.current_stage and not force:
            return
        _, overrides = self._stages[stage_idx]
        for k, v in overrides.items():
            setattr(self.env.tc, k, v)
        self.current_stage = stage_idx
        self.stage_log.append((self.total_steps, stage_idx))

    @property
    def stage_names(self) -> list[str]:
        return [f"Stage {i}" for i in range(len(self._stages))]

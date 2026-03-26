"""
test_day7.py — Tests pour le reward tuning (Jour 7).

Vérifie :
    - la pénalité dT/dt (w_delta_T)
    - les nouveaux poids de récompense
    - la rétrocompatibilité des tests existants
"""

import numpy as np
import pytest

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    cfg = EnvConfig(thermal=ThermalConfig(max_steps=200), reward=RewardConfig())
    e = BatteryThermalEnv(config=cfg)
    e.reset(seed=0)
    return e


# ---------------------------------------------------------------------------
# Nouveaux poids — valeurs par défaut
# ---------------------------------------------------------------------------

class TestNewRewardWeights:

    def test_w_temp_high_increased(self):
        """w_temp_high doit être >= 5.0 pour pénaliser plus fort les pics."""
        rc = RewardConfig()
        assert rc.w_temp_high >= 5.0, f"w_temp_high trop bas : {rc.w_temp_high}"

    def test_w_cooling_reduced(self):
        """w_cooling doit être <= 0.02 pour autoriser plus de refroidissement."""
        rc = RewardConfig()
        assert rc.w_cooling <= 0.02, f"w_cooling trop élevé : {rc.w_cooling}"

    def test_w_delta_T_exists(self):
        """w_delta_T doit exister dans RewardConfig."""
        rc = RewardConfig()
        assert hasattr(rc, "w_delta_T"), "w_delta_T manquant dans RewardConfig"
        assert rc.w_delta_T > 0.0

    def test_T_warning_exists(self):
        """T_warning doit exister dans ThermalConfig."""
        tc = ThermalConfig()
        assert hasattr(tc, "T_warning"), "T_warning manquant dans ThermalConfig"
        assert tc.T_safe_min < tc.T_warning < tc.T_safe_max


# ---------------------------------------------------------------------------
# Pénalité dT/dt
# ---------------------------------------------------------------------------

class TestDeltaTReward:

    def _reward_with_dT(self, env, T, dT):
        """Helper : calcule la récompense avec un dT donné."""
        return env._compute_reward(
            T=T,
            SoC=0.5,
            action=0.0,
            SoC_prev=0.5,
            dT=dT,
        )

    def test_rising_T_above_warning_penalized(self, env):
        """Montée rapide au-dessus de T_warning doit réduire la récompense."""
        T = env.tc.T_warning + 2.0   # au-dessus du seuil
        r_rising  = self._reward_with_dT(env, T, dT=+1.0)
        r_stable  = self._reward_with_dT(env, T, dT=0.0)
        assert r_rising < r_stable, "Montée thermique doit réduire la récompense"

    def test_cooling_dT_not_penalized(self, env):
        """Descente de T (dT < 0) ne doit pas ajouter de pénalité dT."""
        T = env.tc.T_warning + 2.0
        r_cooling = self._reward_with_dT(env, T, dT=-1.0)
        r_stable  = self._reward_with_dT(env, T, dT=0.0)
        assert r_cooling >= r_stable, "Refroidissement ne doit pas être pénalisé par w_delta_T"

    def test_rising_T_below_warning_not_penalized(self, env):
        """Montée sous T_warning ne doit pas déclencher la pénalité dT."""
        T = env.tc.T_warning - 2.0   # en dessous du seuil
        r_rising = self._reward_with_dT(env, T, dT=+1.0)
        r_stable = self._reward_with_dT(env, T, dT=0.0)
        assert r_rising == r_stable, "Pénalité dT ne doit pas s'appliquer sous T_warning"

    def test_penalty_scales_with_dT(self, env):
        """Pénalité proportionnelle à la vitesse de montée."""
        T = env.tc.T_warning + 5.0
        r_slow = self._reward_with_dT(env, T, dT=+0.5)
        r_fast = self._reward_with_dT(env, T, dT=+2.0)
        assert r_fast < r_slow, "Montée plus rapide doit donner une pénalité plus forte"

    def test_penalty_scales_with_T(self, env):
        """Pénalité plus forte quand T est plus loin au-dessus de T_warning."""
        r_low  = self._reward_with_dT(env, T=env.tc.T_warning + 1.0, dT=+1.0)
        r_high = self._reward_with_dT(env, T=env.tc.T_warning + 5.0, dT=+1.0)
        assert r_high < r_low, "Pénalité dT doit croître avec T - T_warning"


# ---------------------------------------------------------------------------
# Rétrocompatibilité — les tests Jour 1 doivent toujours passer
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_reward_penalizes_overtemp(self, env):
        """Dépassement thermique → récompense plus basse qu'en zone sûre."""
        env.reset(seed=0)
        env._T = env.tc.T_safe_max + 10.0
        _, r_hot, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

        env.reset(seed=0)
        env._T = (env.tc.T_safe_min + env.tc.T_safe_max) / 2.0
        _, r_ok, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

        assert r_hot < r_ok, "Dépassement thermique doit réduire la récompense"

    def test_step_output_shape(self, env):
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.5], dtype=np.float32)
        )
        assert obs.shape == (5,)
        assert isinstance(reward, float)

    def test_dT_tracked_in_state(self, env):
        """T_prev doit être mis à jour après chaque step."""
        env.reset(seed=0)
        T_before = env._T
        env.step(np.array([0.0], dtype=np.float32))
        assert env._T_prev == T_before, "T_prev doit conserver la T du step précédent"

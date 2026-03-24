"""
test_env.py — Sanity checks pour BatteryThermalEnv et ThermalModel.

Lance avec : pytest tests/test_env.py -v
"""

import numpy as np
import pytest

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.thermal_model import ThermalModel
from envs.battery_thermal_env import BatteryThermalEnv


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def env():
    return BatteryThermalEnv()


@pytest.fixture
def model():
    cfg = ThermalConfig()
    return ThermalModel(cfg)


# -----------------------------------------------------------------------
# ThermalModel
# -----------------------------------------------------------------------

class TestThermalModel:

    def test_no_current_no_cooling_warms_to_ambient(self, model):
        """Sans courant ni refroidissement, T doit converger vers T_amb."""
        T = 50.0
        T_amb = 25.0
        for _ in range(500):
            T, _, _ = model.step(T=T, I=0.0, action=0.0, T_amb=T_amb)
        assert abs(T - T_amb) < 1.0, f"T={T:.2f} devrait converger vers T_amb={T_amb}"

    def test_cooling_lowers_temperature(self, model):
        """Refroidissement max doit faire baisser T si P_gen < P_cool_max."""
        T_start = 50.0
        T, _, _ = model.step(T=T_start, I=0.0, action=1.0, T_amb=25.0)
        assert T < T_start, "Le refroidissement doit réduire T"

    def test_current_raises_temperature(self, model):
        """Un fort courant sans refroidissement doit faire monter T."""
        T_start = 25.0
        T, _, _ = model.step(T=T_start, I=50.0, action=0.0, T_amb=25.0)
        assert T > T_start, "Le courant doit augmenter T"

    def test_action_clipped(self, model):
        """Les actions hors [0,1] doivent être clippées sans erreur."""
        T, _, _ = model.step(T=25.0, I=10.0, action=5.0, T_amb=25.0)
        assert np.isfinite(T)
        T, _, _ = model.step(T=25.0, I=10.0, action=-2.0, T_amb=25.0)
        assert np.isfinite(T)

    def test_steady_state_formula(self, model):
        """Vérification analytique de la température d'équilibre."""
        I, action, T_amb = 10.0, 0.5, 20.0
        T_ss = model.steady_state_temp(I, action, T_amb)
        # Simuler jusqu'à l'équilibre
        T = T_amb
        for _ in range(5000):
            T, _, _ = model.step(T=T, I=I, action=action, T_amb=T_amb)
        assert abs(T - T_ss) < 0.5, f"T_sim={T:.2f} vs T_ss={T_ss:.2f}"


# -----------------------------------------------------------------------
# BatteryThermalEnv — API Gymnasium
# -----------------------------------------------------------------------

class TestEnvAPI:

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,), "L'observation doit avoir 5 dimensions"
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs), "obs doit être dans observation_space"

    def test_obs_in_bounds(self, env):
        """L'observation normalisée doit rester dans [-1, 1] après reset."""
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0), \
                f"obs hors bornes: {obs}"

    def test_step_output_shape(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self, env):
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0

    def test_zero_cooling_warms_up(self, env):
        """Sans refroidissement avec courant fort, T doit monter."""
        env.reset(seed=1)
        env._T = 25.0
        env._I = env.tc.I_max
        T_before = env._T
        for _ in range(10):
            env.step(np.array([0.0], dtype=np.float32))
        assert env._T > T_before

    def test_full_cooling_limits_rise(self, env):
        """Refroidissement max doit limiter la montée en température."""
        env.reset(seed=2)
        env._T = 25.0
        env._I = env.tc.I_max

        T_no_cool = 25.0
        env_no = BatteryThermalEnv()
        env_no.reset(seed=2)
        env_no._T = 25.0
        env_no._I = env.tc.I_max

        for _ in range(50):
            env.step(np.array([1.0], dtype=np.float32))
            env_no.step(np.array([0.0], dtype=np.float32))

        assert env._T < env_no._T, "Refroidissement max doit donner T < sans refroidissement"

    def test_episode_terminates_on_overcooling(self, env):
        """T_cutoff doit déclencher terminated=True."""
        env.reset(seed=0)
        env._T = env.tc.T_cutoff + 1.0  # déjà au-dessus de la coupure
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert terminated, "L'épisode doit se terminer si T > T_cutoff"

    def test_episode_truncates_at_max_steps(self):
        """max_steps doit déclencher truncated=True."""
        cfg = EnvConfig(thermal=ThermalConfig(max_steps=5))
        env = BatteryThermalEnv(config=cfg)
        env.reset(seed=0)
        truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.5], dtype=np.float32))
            if terminated or truncated:
                break
        assert truncated or env._step_count <= 5

    def test_reward_penalizes_overtemp(self, env):
        """Récompense doit être plus basse quand T > T_safe_max."""
        env.reset(seed=0)
        env._T = env.tc.T_safe_max + 10.0
        _, r_hot, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

        env.reset(seed=0)
        env._T = (env.tc.T_safe_min + env.tc.T_safe_max) / 2
        _, r_ok, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

        assert r_hot < r_ok, "Dépassement thermique doit réduire la récompense"

    def test_history_populated(self, env):
        """L'historique doit être rempli après N steps."""
        env.reset(seed=0)
        N = 10
        for _ in range(N):
            env.step(env.action_space.sample())
        for key in ["T", "SoC", "I", "action", "reward"]:
            assert len(env.history[key]) == N, f"history['{key}'] manquant"

    def test_reproducibility(self, env):
        """Même seed → mêmes observations."""
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_custom_config(self):
        """L'environnement doit accepter une config personnalisée."""
        cfg = EnvConfig(
            thermal=ThermalConfig(T_safe_max=40.0, max_steps=100),
            reward=RewardConfig(w_temp_high=5.0),
        )
        env = BatteryThermalEnv(config=cfg)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (5,)

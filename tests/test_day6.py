"""
test_day6.py — Tests pour CurriculumEnv et robustness.py.
"""

import copy
import numpy as np
import pytest

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from envs.curriculum_env import CurriculumEnv, DEFAULT_STAGES
from robustness import _make_scenarios, _eval_scenario
from agent.sac import SACAgent, SACConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_env():
    cfg = EnvConfig(thermal=ThermalConfig(max_steps=50), reward=RewardConfig())
    return BatteryThermalEnv(config=cfg)


@pytest.fixture
def curriculum_env(base_env):
    stages = [
        (0,   dict(T_amb_min=20.0, T_amb_max=25.0, I_max=20.0, I_min=-5.0)),
        (100, dict(T_amb_min=15.0, T_amb_max=35.0, I_max=50.0, I_min=-10.0)),
    ]
    return CurriculumEnv(base_env, stages=stages)


@pytest.fixture
def small_agent(base_env):
    obs_dim    = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    return SACAgent(obs_dim, action_dim, config=SACConfig(hidden_dim=64, n_layers=1))


# ---------------------------------------------------------------------------
# CurriculumEnv — structure
# ---------------------------------------------------------------------------

class TestCurriculumEnvStructure:

    def test_wraps_env(self, curriculum_env):
        assert isinstance(curriculum_env.env, BatteryThermalEnv)

    def test_starts_at_stage_0(self, curriculum_env):
        assert curriculum_env.current_stage == 0

    def test_observation_space_preserved(self, curriculum_env, base_env):
        assert curriculum_env.observation_space.shape == base_env.observation_space.shape

    def test_action_space_preserved(self, curriculum_env, base_env):
        assert curriculum_env.action_space.shape == base_env.action_space.shape

    def test_reset_returns_valid_obs(self, curriculum_env):
        obs, info = curriculum_env.reset(seed=0)
        assert obs.shape == (5,)
        assert curriculum_env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# CurriculumEnv — stage 0 overrides appliqués
# ---------------------------------------------------------------------------

class TestCurriculumStage0:

    def test_stage0_I_max_applied(self, curriculum_env):
        """Stage 0 doit limiter I_max à 20."""
        assert curriculum_env.env.tc.I_max == 20.0

    def test_stage0_T_amb_narrow(self, curriculum_env):
        assert curriculum_env.env.tc.T_amb_max == 25.0


# ---------------------------------------------------------------------------
# CurriculumEnv — transition de stage
# ---------------------------------------------------------------------------

class TestCurriculumTransition:

    def _run_n_steps(self, env, n: int) -> None:
        env.reset(seed=0)
        for _ in range(n):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

    def test_stage_advances_after_threshold(self, curriculum_env):
        """Après 100 steps, le stage doit passer à 1."""
        self._run_n_steps(curriculum_env, 100)
        curriculum_env.reset()  # applique la transition
        assert curriculum_env.current_stage == 1

    def test_stage1_overrides_applied(self, curriculum_env):
        self._run_n_steps(curriculum_env, 100)
        curriculum_env.reset()
        assert curriculum_env.env.tc.I_max == 50.0
        assert curriculum_env.env.tc.T_amb_max == 35.0

    def test_stage_log_recorded(self, curriculum_env):
        self._run_n_steps(curriculum_env, 100)
        curriculum_env.reset()
        assert len(curriculum_env.stage_log) >= 1

    def test_total_steps_incremented(self, curriculum_env):
        self._run_n_steps(curriculum_env, 30)
        assert curriculum_env.total_steps == 30

    def test_info_contains_stage(self, curriculum_env):
        curriculum_env.reset(seed=0)
        action = curriculum_env.action_space.sample()
        _, _, _, _, info = curriculum_env.step(action)
        assert "curriculum_stage" in info
        assert "curriculum_steps" in info

    def test_no_backward_transition(self, curriculum_env):
        """Le stage ne peut pas reculer."""
        self._run_n_steps(curriculum_env, 110)
        curriculum_env.reset()
        stage_after = curriculum_env.current_stage
        self._run_n_steps(curriculum_env, 10)
        curriculum_env.reset()
        assert curriculum_env.current_stage >= stage_after


# ---------------------------------------------------------------------------
# CurriculumEnv — compatibilité avec SACAgent
# ---------------------------------------------------------------------------

class TestCurriculumWithAgent:

    def test_agent_can_step(self, curriculum_env, small_agent):
        obs, _ = curriculum_env.reset(seed=0)
        action = small_agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = curriculum_env.step(action)
        assert next_obs.shape == (5,)
        assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# robustness._make_scenarios
# ---------------------------------------------------------------------------

class TestMakeScenarios:

    def test_returns_all_scenarios(self):
        scenarios = _make_scenarios(ThermalConfig())
        expected = {"Baseline", "Hot ambient", "Cold ambient", "High load", "Weak cooling"}
        assert set(scenarios.keys()) == expected

    def test_baseline_unchanged(self):
        base = ThermalConfig()
        scenarios = _make_scenarios(base)
        assert scenarios["Baseline"].T_amb_min == base.T_amb_min
        assert scenarios["Baseline"].I_max == base.I_max

    def test_hot_ambient_T_higher(self):
        base = ThermalConfig()
        scenarios = _make_scenarios(base)
        assert scenarios["Hot ambient"].T_amb_min > base.T_amb_min

    def test_weak_cooling_reduced(self):
        base = ThermalConfig()
        scenarios = _make_scenarios(base)
        assert scenarios["Weak cooling"].P_cool_max < base.P_cool_max

    def test_scenarios_independent(self):
        """Modifier un scénario ne doit pas affecter les autres."""
        scenarios = _make_scenarios(ThermalConfig())
        scenarios["Hot ambient"].I_max = 999.0
        assert scenarios["Baseline"].I_max != 999.0


# ---------------------------------------------------------------------------
# robustness._eval_scenario
# ---------------------------------------------------------------------------

class TestEvalScenario:

    def test_returns_expected_keys(self, small_agent):
        tc = ThermalConfig(max_steps=50)
        m  = _eval_scenario(small_agent, "Baseline", tc, n_episodes=2)
        for k in ["scenario", "return_mean", "return_std", "pct_safe_mean", "T_max_mean"]:
            assert k in m

    def test_metrics_finite(self, small_agent):
        tc = ThermalConfig(max_steps=50)
        m  = _eval_scenario(small_agent, "Test", tc, n_episodes=2)
        assert np.isfinite(m["return_mean"])
        assert np.isfinite(m["pct_safe_mean"])
        assert 0.0 <= m["pct_safe_mean"] <= 100.0

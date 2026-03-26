"""
test_day4.py — Tests pour evaluate.py et sweep.py.
"""

import os
import json
import numpy as np
import pytest

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from agent.sac import SACAgent, SACConfig
from evaluate import load_agent, run_episode, plot_episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    cfg = EnvConfig(thermal=ThermalConfig(max_steps=100), reward=RewardConfig())
    return BatteryThermalEnv(config=cfg)


@pytest.fixture
def small_agent(small_env):
    obs_dim    = small_env.observation_space.shape[0]
    action_dim = small_env.action_space.shape[0]
    cfg = SACConfig(hidden_dim=64, n_layers=1)
    return SACAgent(obs_dim, action_dim, config=cfg)


@pytest.fixture
def saved_checkpoint(tmp_path, small_agent):
    path = str(tmp_path / "test_ckpt.pt")
    small_agent.save(path)
    return path, small_agent.obs_dim, small_agent.action_dim


# ---------------------------------------------------------------------------
# load_agent
# ---------------------------------------------------------------------------

class TestLoadAgent:

    def test_load_returns_sac_agent(self, saved_checkpoint):
        path, obs_dim, action_dim = saved_checkpoint
        agent = load_agent(path, obs_dim, action_dim, hidden_dim=64, n_layers=1)
        assert isinstance(agent, SACAgent)

    def test_loaded_agent_selects_action(self, saved_checkpoint, small_env):
        path, obs_dim, action_dim = saved_checkpoint
        agent = load_agent(path, obs_dim, action_dim, hidden_dim=64, n_layers=1)
        obs, _ = small_env.reset(seed=0)
        action = agent.select_action(obs, deterministic=True)
        assert action.shape == (action_dim,)
        assert 0.0 <= float(action[0]) <= 1.0


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------

class TestRunEpisode:

    EXPECTED_KEYS = {
        "return", "length", "T_max", "T_min", "T_mean",
        "pct_safe", "SoC_final", "T_hist", "SoC_hist", "action_hist",
    }

    def test_returns_expected_keys(self, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=0)
        assert self.EXPECTED_KEYS.issubset(ep.keys())

    def test_scalar_metrics_finite(self, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=1)
        for k in ["return", "length", "T_max", "T_min", "T_mean", "pct_safe", "SoC_final"]:
            assert np.isfinite(ep[k]), f"Métrique '{k}' non finie"

    def test_pct_safe_in_range(self, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=2)
        assert 0.0 <= ep["pct_safe"] <= 100.0

    def test_histories_non_empty(self, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=3)
        assert len(ep["T_hist"]) > 0
        assert len(ep["T_hist"]) == len(ep["SoC_hist"]) == len(ep["action_hist"])

    def test_actions_in_bounds(self, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=4)
        for a in ep["action_hist"]:
            assert 0.0 <= a <= 1.0, f"Action hors bornes : {a}"

    def test_t_max_consistent_with_history(self, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=5)
        assert abs(ep["T_max"] - max(ep["T_hist"])) < 1e-6

    def test_reproducible_with_same_seed(self, small_agent, small_env):
        ep1 = run_episode(small_agent, small_env, seed=7)
        ep2 = run_episode(small_agent, small_env, seed=7)
        assert ep1["return"] == ep2["return"]


# ---------------------------------------------------------------------------
# plot_episode
# ---------------------------------------------------------------------------

class TestPlotEpisode:

    def test_creates_png(self, tmp_path, small_agent, small_env):
        ep = run_episode(small_agent, small_env, seed=0)
        out = str(tmp_path / "test_plot.png")
        plot_episode(ep, out, small_env)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

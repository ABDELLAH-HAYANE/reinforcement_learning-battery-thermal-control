"""
test_train.py — Smoke tests pour train.py.

Vérifie que la boucle complète (env + agent + logger) tourne sans erreur
sur un petit nombre de steps.
"""

import os
import pytest
import numpy as np

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from agent.sac import SACAgent, SACConfig
from utils.logger import TrainingLogger
from train import evaluate, shaped_reward


# -----------------------------------------------------------------------
# shaped_reward
# -----------------------------------------------------------------------

class TestShapedReward:

    def test_bonus_when_in_safe_range(self):
        cfg = ThermalConfig()
        T_opt = (cfg.T_safe_min + cfg.T_safe_max) / 2.0
        info = {"T": T_opt}
        r = shaped_reward(0.0, info, cfg)
        assert r > 0.0, "Bonus attendu quand T == T_opt"

    def test_no_negative_bonus(self):
        """Le reward shapé ne doit pas diminuer le reward original."""
        cfg = ThermalConfig()
        for T in [cfg.T_safe_min, cfg.T_safe_max, cfg.T_cutoff]:
            info = {"T": T}
            r = shaped_reward(0.0, info, cfg)
            assert r >= 0.0


# -----------------------------------------------------------------------
# Boucle complète — smoke test
# -----------------------------------------------------------------------

class TestTrainLoop:

    def _make_components(self, learning_starts=50):
        env_cfg = EnvConfig(
            thermal=ThermalConfig(max_steps=200),
            reward=RewardConfig(),
        )
        sac_cfg = SACConfig(
            hidden_dim=64, n_layers=1,
            batch_size=32, learning_starts=learning_starts,
            buffer_capacity=1000,
        )
        env   = BatteryThermalEnv(config=env_cfg)
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=sac_cfg,
        )
        return env, agent, env_cfg

    def test_loop_runs_100_steps(self):
        """La boucle doit tourner 100 steps sans erreur."""
        env, agent, env_cfg = self._make_components()
        obs, _ = env.reset(seed=0)
        total_steps = 0

        while total_steps < 100:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_s = shaped_reward(reward, info, env_cfg.thermal)
            agent.push(obs, action, reward_s, next_obs, float(terminated))
            if agent.ready:
                agent.update()
            obs = next_obs
            total_steps += 1
            if terminated or truncated:
                obs, _ = env.reset()

        assert total_steps == 100

    def test_agent_updates_after_learning_starts(self):
        """update() doit retourner des métriques finies après learning_starts."""
        env, agent, env_cfg = self._make_components(learning_starts=32)
        obs, _ = env.reset(seed=1)

        for _ in range(50):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, next_obs, float(terminated))
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        assert agent.ready
        metrics = agent.update()
        for k, v in metrics.items():
            assert np.isfinite(v), f"Métrique '{k}' non finie : {v}"

    def test_evaluate_returns_dict(self):
        env, agent, _ = self._make_components()
        result = evaluate(agent, env, n_episodes=2)
        expected_keys = [
            "eval_return_mean", "eval_return_std",
            "eval_length_mean", "eval_T_max_mean", "eval_pct_safe",
        ]
        for k in expected_keys:
            assert k in result
            assert np.isfinite(result[k])

    def test_logger_creates_csv(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), print_every=5)
        logger.log_episode(ep=1, total_steps=100, ep_return=-5.0, ep_length=100)
        logger.log_update(step=50, critic_loss=0.1, actor_loss=0.2, alpha=0.2,
                          alpha_loss=0.01, entropy=1.0)
        logger.close()

        assert os.path.exists(str(tmp_path / "episodes.csv"))
        assert os.path.exists(str(tmp_path / "updates.csv"))

    def test_multiple_episodes_logged(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), print_every=2)
        for i in range(5):
            logger.log_episode(ep=i+1, total_steps=(i+1)*100,
                               ep_return=float(-i), ep_length=100)
        logger.close()

        import csv
        with open(str(tmp_path / "episodes.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

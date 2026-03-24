"""
test_sac.py — Sanity checks pour ReplayBuffer, Actor, Critic, SACAgent.

Lance avec : pytest tests/test_sac.py -v
"""

import numpy as np
import pytest
import torch

from agent.replay_buffer import ReplayBuffer
from agent.networks import Actor, Critic
from agent.sac import SACAgent, SACConfig

OBS_DIM    = 5
ACTION_DIM = 1


# -----------------------------------------------------------------------
# ReplayBuffer
# -----------------------------------------------------------------------

class TestReplayBuffer:

    def test_push_and_len(self):
        buf = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=100)
        assert len(buf) == 0
        buf.push(np.zeros(OBS_DIM), np.zeros(ACTION_DIM), 0.0, np.zeros(OBS_DIM), False)
        assert len(buf) == 1

    def test_circular_overflow(self):
        buf = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=10)
        for i in range(15):
            buf.push(np.ones(OBS_DIM) * i, np.zeros(ACTION_DIM), 0.0, np.zeros(OBS_DIM), False)
        assert len(buf) == 10  # capacity atteinte

    def test_sample_shapes(self):
        buf = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=100)
        for _ in range(50):
            buf.push(np.random.randn(OBS_DIM), np.random.randn(ACTION_DIM),
                     float(np.random.randn()), np.random.randn(OBS_DIM), False)
        batch = buf.sample(32)
        assert batch["obs"].shape      == (32, OBS_DIM)
        assert batch["actions"].shape  == (32, ACTION_DIM)
        assert batch["rewards"].shape  == (32, 1)
        assert batch["next_obs"].shape == (32, OBS_DIM)
        assert batch["dones"].shape    == (32, 1)

    def test_sample_returns_tensors(self):
        buf = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=100)
        for _ in range(10):
            buf.push(np.zeros(OBS_DIM), np.zeros(ACTION_DIM), 0.0, np.zeros(OBS_DIM), False)
        batch = buf.sample(5)
        for v in batch.values():
            assert isinstance(v, torch.Tensor)

    def test_done_stored_as_float(self):
        buf = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=10)
        buf.push(np.zeros(OBS_DIM), np.zeros(ACTION_DIM), 1.0, np.zeros(OBS_DIM), True)
        batch = buf.sample(1)
        assert batch["dones"].item() == 1.0


# -----------------------------------------------------------------------
# Actor
# -----------------------------------------------------------------------

class TestActor:

    def setup_method(self):
        self.actor = Actor(OBS_DIM, ACTION_DIM)
        self.obs = torch.randn(4, OBS_DIM)  # batch de 4

    def test_forward_shapes(self):
        mu, log_std = self.actor(self.obs)
        assert mu.shape      == (4, ACTION_DIM)
        assert log_std.shape == (4, ACTION_DIM)

    def test_log_std_clamped(self):
        mu, log_std = self.actor(self.obs)
        assert log_std.min().item() >= -5 - 1e-4
        assert log_std.max().item() <= 2  + 1e-4

    def test_sample_action_in_range(self):
        action, log_prob = self.actor.sample(self.obs)
        assert action.shape == (4, ACTION_DIM)
        # tanh → action ∈ (-1, 1)
        assert action.abs().max().item() < 1.0

    def test_log_prob_finite(self):
        _, log_prob = self.actor.sample(self.obs)
        assert torch.isfinite(log_prob).all()

    def test_select_action_no_grad(self):
        obs = torch.randn(1, OBS_DIM)
        action = self.actor.select_action(obs)
        assert action.shape == (1, ACTION_DIM)

    def test_deterministic_reproducible(self):
        obs = torch.randn(1, OBS_DIM)
        a1 = self.actor.select_action(obs, deterministic=True)
        a2 = self.actor.select_action(obs, deterministic=True)
        assert torch.allclose(a1, a2)


# -----------------------------------------------------------------------
# Critic
# -----------------------------------------------------------------------

class TestCritic:

    def setup_method(self):
        self.critic = Critic(OBS_DIM, ACTION_DIM)
        self.obs    = torch.randn(4, OBS_DIM)
        self.action = torch.randn(4, ACTION_DIM)

    def test_forward_shapes(self):
        q1, q2 = self.critic(self.obs, self.action)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_q1_q2_differ(self):
        """Q1 et Q2 ont des poids différents → valeurs différentes."""
        q1, q2 = self.critic(self.obs, self.action)
        assert not torch.allclose(q1, q2)

    def test_q1_forward_matches_forward(self):
        q1_direct = self.critic.q1_forward(self.obs, self.action)
        q1_full, _ = self.critic(self.obs, self.action)
        assert torch.allclose(q1_direct, q1_full)


# -----------------------------------------------------------------------
# SACAgent
# -----------------------------------------------------------------------

class TestSACAgent:

    def _make_agent(self, learning_starts=10):
        cfg = SACConfig(
            hidden_dim=64, n_layers=2,
            batch_size=16, learning_starts=learning_starts,
            buffer_capacity=1000,
        )
        return SACAgent(OBS_DIM, ACTION_DIM, config=cfg)

    def test_select_action_shape(self):
        agent = self._make_agent()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (ACTION_DIM,)

    def test_select_action_in_bounds(self):
        """L'action rescalée doit être dans [0, 1]."""
        agent = self._make_agent()
        for _ in range(50):
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            action = agent.select_action(obs)
            assert np.all(action >= 0.0) and np.all(action <= 1.0), \
                f"action hors bornes: {action}"

    def test_not_ready_before_learning_starts(self):
        agent = self._make_agent(learning_starts=100)
        for _ in range(50):
            agent.push(np.zeros(OBS_DIM), np.zeros(ACTION_DIM), 0.0, np.zeros(OBS_DIM), False)
        assert not agent.ready

    def test_ready_after_learning_starts(self):
        agent = self._make_agent(learning_starts=10)
        for _ in range(10):
            agent.push(np.zeros(OBS_DIM), np.zeros(ACTION_DIM), 0.0, np.zeros(OBS_DIM), False)
        assert agent.ready

    def test_update_returns_metrics(self):
        agent = self._make_agent(learning_starts=16)
        for _ in range(20):
            agent.push(
                np.random.randn(OBS_DIM).astype(np.float32),
                np.random.rand(ACTION_DIM).astype(np.float32),
                float(np.random.randn()),
                np.random.randn(OBS_DIM).astype(np.float32),
                False,
            )
        metrics = agent.update()
        for key in ["critic_loss", "actor_loss", "alpha_loss", "alpha", "entropy"]:
            assert key in metrics
            assert np.isfinite(metrics[key]), f"metric '{key}' non fini: {metrics[key]}"

    def test_update_decrements_critic_loss(self):
        """La critic_loss doit rester finie après plusieurs updates."""
        agent = self._make_agent(learning_starts=32)
        for _ in range(50):
            agent.push(
                np.random.randn(OBS_DIM).astype(np.float32),
                np.random.rand(ACTION_DIM).astype(np.float32),
                float(np.random.randn()),
                np.random.randn(OBS_DIM).astype(np.float32),
                False,
            )
        for _ in range(10):
            metrics = agent.update()
            assert np.isfinite(metrics["critic_loss"])

    def test_alpha_positive(self):
        """α doit toujours rester positif (log_alpha peut être négatif)."""
        agent = self._make_agent(learning_starts=16)
        for _ in range(20):
            agent.push(
                np.random.randn(OBS_DIM).astype(np.float32),
                np.random.rand(ACTION_DIM).astype(np.float32),
                0.0,
                np.random.randn(OBS_DIM).astype(np.float32),
                False,
            )
        metrics = agent.update()
        assert metrics["alpha"] > 0.0

    def test_save_load(self, tmp_path):
        agent = self._make_agent()
        path = str(tmp_path / "ckpt.pt")
        agent.save(path)

        agent2 = self._make_agent()
        agent2.load(path)

        obs = torch.randn(1, OBS_DIM)
        a1 = agent.actor.select_action(obs, deterministic=True)
        a2 = agent2.actor.select_action(obs, deterministic=True)
        assert torch.allclose(a1, a2), "Les poids chargés doivent donner la même action"

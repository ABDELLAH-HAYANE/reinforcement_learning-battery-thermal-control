"""
replay_buffer.py — Buffer circulaire pour SAC.

Stocke les transitions (s, a, r, s', done) sous forme numpy,
retourne des batches sous forme de tenseurs PyTorch.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Buffer circulaire de taille fixe.

    Usage :
        buf = ReplayBuffer(obs_dim=5, action_dim=1, capacity=100_000)
        buf.push(obs, action, reward, next_obs, done)
        batch = buf.sample(256)
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0       # pointeur d'écriture
        self.size = 0      # nombre de transitions stockées

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.FloatTensor(self.obs[idx]).to(device),
            "actions":  torch.FloatTensor(self.actions[idx]).to(device),
            "rewards":  torch.FloatTensor(self.rewards[idx]).to(device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(device),
            "dones":    torch.FloatTensor(self.dones[idx]).to(device),
        }

    def __len__(self) -> int:
        return self.size

    @property
    def ready(self) -> bool:
        """True si le buffer a au moins un batch disponible."""
        return self.size > 0

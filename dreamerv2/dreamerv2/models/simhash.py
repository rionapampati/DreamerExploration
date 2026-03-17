import torch
import torch.nn as nn
from collections import defaultdict


class SimHashModel(nn.Module):
    """
    SimHash count-based intrinsic reward.
    Operates on RSSM latent state (deter + stoch concatenated).
    No trainable parameters — fixed random projection + persistent count table.
    Intrinsic reward = 1 / sqrt(visit_count(hash(state)))
    """

    def __init__(self, input_size: int, k: int = 128):
        super().__init__()
        A = torch.randn(k, input_size)
        self.register_buffer("A", A)          # fixed projection, never updated
        self.register_buffer("reward_running_mean", torch.zeros(1))
        self.register_buffer("reward_running_var", torch.ones(1))
        self.counts: dict = defaultdict(int)  # NOT a buffer — lives in Python

        # store for assertions
        self.input_size = input_size
        self.k = k

    def _hash(self, flat_states: torch.Tensor) -> list:
        """flat_states: [N, D] → list of N tuple keys"""
        assert flat_states.shape[1] == self.input_size, (
            f"SimHash input dim mismatch: got {flat_states.shape[1]}, "
            f"expected {self.input_size}"
        )
        proj = flat_states @ self.A.T        # [N, k]
        bits = (proj > 0).int()              # [N, k]  binary
        return [tuple(row.tolist()) for row in bits]

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: [T, B, D] detached RSSM model states
        Returns:
            intrinsic_reward: [T, B]
        """
        T, B, D = latent_state.shape
        flat = latent_state.detach().reshape(T * B, D)
        keys = self._hash(flat)

        rewards = []
        for key in keys:
            self.counts[key] += 1
            rewards.append(1.0 / (self.counts[key] ** 0.5))

        intrinsic = torch.tensor(
            rewards, dtype=latent_state.dtype, device=latent_state.device
        )
        return intrinsic.reshape(T, B)

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """EMA normalization to keep intrinsic reward scale stable."""
        batch_mean = reward.mean().detach()
        batch_var  = reward.var().detach().clamp(min=1e-8)
        self.reward_running_mean = (
            0.99 * self.reward_running_mean + 0.01 * batch_mean
        )
        self.reward_running_var = (
            0.99 * self.reward_running_var + 0.01 * batch_var
        )
        return (reward - self.reward_running_mean) / (
            self.reward_running_var.sqrt() + 1e-8
        )

    def get_counts(self) -> dict:
        """Returns count table for checkpointing."""
        return dict(self.counts)

    def set_counts(self, counts: dict):
        """Restores count table from checkpoint."""
        self.counts = defaultdict(int, counts)
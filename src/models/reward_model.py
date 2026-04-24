"""Task 4: Reward model for RLHF-style preference learning."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from runtime import configure_torch_runtime
except ModuleNotFoundError:
    from src.runtime import configure_torch_runtime

configure_torch_runtime()


class RewardModel(nn.Module):
    """Score sequences in [0,1] (maps to human1–5 after scaling)."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 256, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.embed(tokens)
        mask = (tokens != self.pad_id).float().unsqueeze(-1)
        summed = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.net(summed).squeeze(-1)

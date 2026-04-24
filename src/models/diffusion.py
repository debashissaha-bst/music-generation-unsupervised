from __future__ import annotations

import torch.nn as nn


class DiffusionPlaceholder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Diffusion training is not implemented in this project baseline.")

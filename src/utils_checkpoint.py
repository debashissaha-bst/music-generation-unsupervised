"""PyTorch version-tolerant checkpoint loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load(path: Path | str, map_location: Any = None):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

"""Optional piano-roll representation for analysis or alternate models."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import pretty_midi
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install pretty_midi") from e


def midi_to_piano_roll(
    path: Path | str,
    fs: int = 8,
    max_time_steps: int = 512,
) -> np.ndarray:

    pm = pretty_midi.PrettyMIDI(str(path))
    roll = pm.get_piano_roll(fs=fs)
    if roll.shape[1] > max_time_steps:
        roll = roll[:, :max_time_steps]
    elif roll.shape[1] < max_time_steps:
        pad = max_time_steps - roll.shape[1]
        roll = np.pad(roll, ((0, 0), (0, pad)), mode="constant")
    roll = np.clip(roll / 127.0, 0.0, 1.0)
    return roll.astype(np.float32)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install pretty_midi") from e


def pitch_histogram(path: Path | str) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(str(path))
    h = np.zeros(12, dtype=np.float64)
    for ins in pm.instruments:
        for n in ins.notes:
            h[n.pitch % 12] += 1.0
    s = h.sum()
    if s > 0:
        h /= s
    return h


def histogram_similarity(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(p - q).sum() / 2.0)


def pairwise_against_reference(
    midi_paths: list[Path], ref: Path
) -> Dict[str, float]:
    r = pitch_histogram(ref)
    out: Dict[str, float] = {}
    for p in midi_paths:
        out[str(p)] = histogram_similarity(r, pitch_histogram(p))
    return out

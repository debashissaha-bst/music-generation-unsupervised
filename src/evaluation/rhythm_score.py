from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pretty_midi
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install pretty_midi") from e


def rhythm_diversity(path: Path | str) -> float:
    pm = pretty_midi.PrettyMIDI(str(path))
    durs = []
    for ins in pm.instruments:
        for n in ins.notes:
            durs.append(round(float(n.end - n.start), 4))
    if not durs:
        return 0.0
    return float(len(set(durs)) / len(durs))


def repetition_ratio(path: Path | str, window: int = 4) -> float:
    pm = pretty_midi.PrettyMIDI(str(path))
    pitches = []
    for ins in pm.instruments:
        for n in sorted(ins.notes, key=lambda x: x.start):
            pitches.append(n.pitch)
    if len(pitches) < window:
        return 0.0
    patterns = []
    for i in range(len(pitches) - window + 1):
        patterns.append(tuple(pitches[i : i + window]))
    if not patterns:
        return 0.0
    uniq = len(set(patterns))
    return float(1.0 - uniq / len(patterns))

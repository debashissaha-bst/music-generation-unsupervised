"""Export token sequences to MIDI files."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pretty_midi
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install pretty_midi") from e

try:
    import config
    from preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID, token_to_note
except ModuleNotFoundError:
    from src import config
    from src.preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID, token_to_note


def tokens_to_midi(
    tokens: Sequence[int] | np.ndarray,
    out_path: Path | str,
    tempo: float = 120.0,
    program: int = 0,
) -> None:
    """Convert note tokens (excluding SOS/PAD) to a single-track MIDI."""
    seconds_per_quarter = 60.0 / tempo
    seconds_per_step = seconds_per_quarter / float(config.STEPS_PER_QUARTER)

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program, is_drum=False)
    t_sec = 0.0
    for tid in tokens:
        t = int(tid)
        if t in (PAD_ID, SOS_ID, EOS_ID):
            if t == EOS_ID:
                break
            continue
        try:
            pitch, dur_steps = token_to_note(t)
        except ValueError:
            continue
        dur_sec = max(seconds_per_step * dur_steps, seconds_per_step * 0.25)
        note = pretty_midi.Note(
            velocity=90,
            pitch=int(np.clip(pitch, 0, 127)),
            start=float(t_sec),
            end=float(t_sec + dur_sec),
        )
        inst.notes.append(note)
        t_sec += dur_sec
    pm.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))

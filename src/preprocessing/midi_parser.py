"""Load MIDI files and extract quantized note events."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError as e:  # pragma: no cover
    raise ImportError("Install pretty_midi: pip install pretty_midi") from e


@dataclass
class NoteEvent:
    start_step: int
    duration_steps: int
    pitch: int
    velocity: int


def load_midi_notes(
    path: Path | str,
    steps_per_quarter: int = 4,
) -> List[NoteEvent]:
    """
    Quantize note starts/durations onto a grid with `steps_per_quarter` per quarter note.
    With 4/4 time, 16 steps per bar => steps_per_quarter = 4.
    """
    path = Path(path)
    pm = pretty_midi.PrettyMIDI(str(path))
    if not pm.instruments:
        return []

    inst = None
    for ins in pm.instruments:
        if not ins.is_drum:
            inst = ins
            break
    if inst is None:
        inst = pm.instruments[0]

    tempo = float(pm.estimate_tempo()) or 120.0
    seconds_per_quarter = 60.0 / tempo
    seconds_per_step = seconds_per_quarter / float(steps_per_quarter)

    notes: List[NoteEvent] = []
    for n in inst.notes:
        start_step = int(round(n.start / seconds_per_step))
        end_step = int(round(n.end / seconds_per_step))
        duration = max(1, end_step - start_step)
        pitch = int(np.clip(n.pitch, 0, 127))
        vel = int(np.clip(n.velocity, 1, 127))
        notes.append(
            NoteEvent(
                start_step=start_step,
                duration_steps=duration,
                pitch=pitch,
                velocity=vel,
            )
        )
    notes.sort(key=lambda x: (x.start_step, x.pitch))
    return notes


def midi_to_note_events(
    path: Path | str,
    steps_per_quarter: int,
    max_duration_steps: int,
) -> List[NoteEvent]:
    raw = load_midi_notes(path, steps_per_quarter=steps_per_quarter)
    clipped: List[NoteEvent] = []
    for e in raw:
        d = min(e.duration_steps, max_duration_steps)
        clipped.append(
            NoteEvent(
                start_step=e.start_step,
                duration_steps=d,
                pitch=e.pitch,
                velocity=e.velocity,
            )
        )
    return clipped


def collect_midi_paths(raw_root: Path) -> List[Tuple[Path, int]]:
    """Return (path, genre_id) for each MIDI.

    Preferred layout is `raw_root/<genre>/*.mid`, but this also supports a flat
    layout with MIDI files directly under `raw_root` (assigned default genre 0).
    """
    try:
        import config
    except ModuleNotFoundError:
        from src import config

    pairs: List[Tuple[Path, int]] = []
    for gid, name in enumerate(config.GENRE_NAMES):
        gdir = raw_root / name
        if not gdir.is_dir():
            continue
        for p in gdir.rglob("*.mid"):
            pairs.append((p, gid))
        for p in gdir.rglob("*.midi"):
            pairs.append((p, gid))

    # Fallback: allow flat folder layout under data/raw_midi.
    if not pairs:
        for p in raw_root.rglob("*.mid"):
            pairs.append((p, 0))
        for p in raw_root.rglob("*.midi"):
            pairs.append((p, 0))
    return pairs

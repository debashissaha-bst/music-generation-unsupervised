from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:
    import config
except ModuleNotFoundError:  # package import: python -m src....
    from src import config

from .midi_parser import NoteEvent

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def build_vocab_size() -> int:
    """pitch in [0,127], duration in [1, max_dur] -> token = 3 + pitch * max_dur + (dur-1)."""
    return 3 + 128 * config.MAX_DURATION_STEPS


def note_to_token(pitch: int, duration_steps: int) -> int:
    d = int(np.clip(duration_steps, 1, config.MAX_DURATION_STEPS))
    p = int(np.clip(pitch, 0, 127))
    return 3 + p * config.MAX_DURATION_STEPS + (d - 1)


def token_to_note(token: int) -> Tuple[int, int]:
    if token < 3:
        raise ValueError("Special token")
    t = token - 3
    dur = (t % config.MAX_DURATION_STEPS) + 1
    pitch = t // config.MAX_DURATION_STEPS
    return int(pitch), int(dur)


class NoteTokenizer:
    def __init__(self, max_seq_len: int | None = None):
        self.max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        self.vocab_size = build_vocab_size()

    def events_to_sequence(self, events: Sequence[NoteEvent]) -> np.ndarray:
        """Chronological note tokens without special tokens."""
        toks: List[int] = []
        for e in sorted(events, key=lambda x: (x.start_step, x.pitch)):
            toks.append(note_to_token(e.pitch, e.duration_steps))
        return np.array(toks, dtype=np.int64)

    def encode_with_special(self, events: Sequence[NoteEvent]) -> np.ndarray:
        notes = self.events_to_sequence(events)
        out = np.full(self.max_seq_len, PAD_ID, dtype=np.int64)
        max_notes = self.max_seq_len - 2
        L = min(len(notes), max_notes)
        out[0] = SOS_ID
        if L > 0:
            out[1 : 1 + L] = notes[:L]
        out[1 + L] = EOS_ID
        return out

    def encode_windows_with_special(
        self,
        events: Sequence[NoteEvent],
        stride: int | None = None,
    ) -> List[np.ndarray]:

        notes = self.events_to_sequence(events)
        max_notes = self.max_seq_len - 2
        if max_notes <= 0:
            raise ValueError("max_seq_len must be at least 3")
        if notes.size == 0:
            return []
        stride = int(stride or max_notes)
        stride = max(1, min(stride, max_notes))

        windows: List[np.ndarray] = []
        for start in range(0, len(notes), stride):
            chunk = notes[start : start + max_notes]
            if chunk.size == 0:
                continue
            out = np.full(self.max_seq_len, PAD_ID, dtype=np.int64)
            out[0] = SOS_ID
            out[1 : 1 + len(chunk)] = chunk
            out[1 + len(chunk)] = EOS_ID
            windows.append(out)
            if start + max_notes >= len(notes):
                break
        return windows

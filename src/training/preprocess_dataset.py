
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import config
    from preprocessing.midi_parser import collect_midi_paths, midi_to_note_events
    from preprocessing.tokenizer import NoteTokenizer
except ModuleNotFoundError:
    from src import config
    from src.preprocessing.midi_parser import collect_midi_paths, midi_to_note_events
    from src.preprocessing.tokenizer import NoteTokenizer


def write_synthetic_midis(out_dir: Path, n_per_genre: int = 3) -> None:
    """Create tiny placeholder MIDIs so the pipeline runs without downloads."""
    try:
        import pretty_midi
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, gname in enumerate(config.GENRE_NAMES):
        gpath = out_dir / gname
        gpath.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_genre):
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0, is_drum=False)
            base = 60 + gid * 2 + i
            t = 0.0
            for k in range(16):
                pitch = int(np.clip(base + (k % 5) - 2, 21, 108))
                n = pretty_midi.Note(
                    velocity=80 + k,
                    pitch=pitch,
                    start=t,
                    end=t + 0.25,
                )
                inst.notes.append(n)
                t += 0.25
            pm.instruments.append(inst)
            pm.write(str(gpath / f"synthetic_{i}.mid"))


def _stable_train_test_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.permutation(n)
    if n <= 1:
        return idx, idx.copy()
    split = int(round(n * config.TRAIN_RATIO))
    split = min(max(split, 1), n - 1)
    return idx[:split], idx[split:]


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    pairs = collect_midi_paths(config.DATA_RAW)
    if not pairs:
        print("No MIDI found; writing synthetic MIDIs under data/raw_midi/")
        write_synthetic_midis(config.DATA_RAW)
        pairs = collect_midi_paths(config.DATA_RAW)

    tok = NoteTokenizer()
    sequences: list[np.ndarray] = []
    genres: list[int] = []
    source_files = 0
    skipped_files = 0

    for path, gid in pairs:
        try:
            events = midi_to_note_events(
                path,
                steps_per_quarter=config.STEPS_PER_QUARTER,
                max_duration_steps=config.MAX_DURATION_STEPS,
            )
        except Exception as e:
            print(f"Skip {path}: {e}")
            skipped_files += 1
            continue
        if len(events) < 2:
            skipped_files += 1
            continue
        windows = tok.encode_windows_with_special(events)
        if not windows:
            skipped_files += 1
            continue
        sequences.extend(windows)
        genres.extend([gid] * len(windows))
        source_files += 1

    if not sequences:
        raise RuntimeError("No valid sequences extracted. Add MIDI files to data/raw_midi/<genre>/.")

    X = np.stack(sequences, axis=0)
    G = np.array(genres, dtype=np.int64)
    n = X.shape[0]
    train_idx, test_idx = _stable_train_test_split(n)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.DATA_SPLIT.mkdir(parents=True, exist_ok=True)

    payload = {
        "tokens": torch.from_numpy(X),
        "genre": torch.from_numpy(G),
        "vocab_size": tok.vocab_size,
        "max_seq_len": tok.max_seq_len,
        "num_source_files": source_files,
        "num_windows": n,
        "skipped_files": skipped_files,
    }
    torch.save(payload, config.DATA_PROCESSED / "dataset.pt")
    with open(config.DATA_SPLIT / "indices.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_idx.tolist(), "test": test_idx.tolist()}, f)

    print(
        f"Saved {n} windows from {source_files} MIDI files "
        f"(skipped={skipped_files}), vocab_size={tok.vocab_size} -> "
        f"{config.DATA_PROCESSED / 'dataset.pt'}"
    )


if __name__ == "__main__":
    main()

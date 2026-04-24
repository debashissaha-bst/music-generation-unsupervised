from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import config
    from generation.midi_export import tokens_to_midi
    from preprocessing.tokenizer import EOS_ID, SOS_ID, note_to_token
except ModuleNotFoundError:
    from src import config
    from src.generation.midi_export import tokens_to_midi
    from src.preprocessing.tokenizer import EOS_ID, SOS_ID, note_to_token


def random_sequence(length: int, rng: random.Random) -> list[int]:
    out = [SOS_ID]
    for _ in range(length - 2):
        p = rng.randint(40, 80)
        d = rng.randint(1, min(8, config.MAX_DURATION_STEPS))
        out.append(note_to_token(p, d))
    out.append(EOS_ID)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out_dir", type=str, default=str(config.OUTPUTS_MIDI / "baseline_random"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        tokens_to_midi(random_sequence(args.len, rng), out / f"random_{i}.mid")
    print(f"Wrote {args.n} random MIDIs to {out}")


if __name__ == "__main__":
    main()

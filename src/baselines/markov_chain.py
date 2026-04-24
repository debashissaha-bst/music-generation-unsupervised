from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

try:
    from utils_checkpoint import torch_load
    import config
    from generation.midi_export import tokens_to_midi
    from preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID
except ModuleNotFoundError:
    from src.utils_checkpoint import torch_load
    from src import config
    from src.generation.midi_export import tokens_to_midi
    from src.preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID


def build_chain(tokens: torch.Tensor) -> dict[int, Counter]:
    trans: dict[int, Counter] = defaultdict(Counter)
    for row in tokens:
        prev = None
        for t in row.tolist():
            if t == PAD_ID:
                continue
            if prev is not None:
                trans[prev][t] += 1
            prev = t
            if t == EOS_ID:
                break
    return trans


def sample_seq(chain: dict[int, Counter], rng: random.Random, max_len: int) -> list[int]:
    cur = SOS_ID
    out = [cur]
    for _ in range(max_len - 1):
        c = chain.get(cur)
        if not c:
            break
        items, weights = zip(*c.items())
        cur = rng.choices(items, weights=weights, k=1)[0]
        out.append(cur)
        if cur == EOS_ID:
            break
    if out[-1] != EOS_ID:
        out.append(EOS_ID)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default=str(config.DATA_PROCESSED / "dataset.pt"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out_dir", type=str, default=str(config.OUTPUTS_MIDI / "baseline_markov"))
    args = ap.parse_args()
    data = torch_load(args.processed, map_location="cpu")
    chain = build_chain(data["tokens"])
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        tokens_to_midi(sample_seq(chain, rng, args.max_len), out / f"markov_{i}.mid")
    print(f"Wrote {args.n} Markov MIDIs to {out}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import config
    from utils_checkpoint import torch_load
except ModuleNotFoundError:
    from src import config
    from src.utils_checkpoint import torch_load


def load_processed():
    path = config.DATA_PROCESSED / "dataset.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Run preprocess first: missing {path}")
    return torch_load(path, map_location="cpu")


def load_split_indices():
    p = config.DATA_SPLIT / "indices.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}; run preprocess_dataset.py")
    with open(p, encoding="utf-8") as f:
        idx = json.load(f)
    return idx["train"], idx["test"]


def _ensure_nonempty_split(indices: list[int]) -> tuple[list[int], list[int]]:
    if not indices:
        return [], []
    if len(indices) == 1:
        return indices[:], indices[:]
    split = int(round(len(indices) * config.TRAIN_RATIO))
    split = min(max(split, 1), len(indices) - 1)
    return indices[:split], indices[split:]


def make_loaders(
    batch_size: int | None = None,
    single_genre_id: int | None = None,
):
    batch_size = batch_size or config.BATCH_SIZE
    data = load_processed()
    tokens = data["tokens"]
    genre = data["genre"]
    train_idx, test_idx = load_split_indices()

    if single_genre_id is not None:
        gmask = genre == single_genre_id
        all_idx = torch.where(gmask)[0].tolist()
        if not all_idx:
            raise RuntimeError(f"No samples found for genre id {single_genre_id}")
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(all_idx)
        train_idx, test_idx = _ensure_nonempty_split(all_idx)

    ds = TensorDataset(tokens, genre)
    tr = Subset(ds, train_idx)
    te = Subset(ds, test_idx)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(te, batch_size=batch_size, shuffle=False),
        data,
    )

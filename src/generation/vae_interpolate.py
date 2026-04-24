"""Latent interpolation between two random codes (Task 2 deliverable helper)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

try:
    from utils_checkpoint import torch_load
    import config
    from generation.midi_export import tokens_to_midi
    from models.vae import MusicVAE
    from preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID
    from training.data_utils import load_processed
except ModuleNotFoundError:
    from src.utils_checkpoint import torch_load
    from src import config
    from src.generation.midi_export import tokens_to_midi
    from src.models.vae import MusicVAE
    from src.preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID
    from src.training.data_utils import load_processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINTS / "vae_best.pt"))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--genre", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(config.OUTPUTS_MIDI / "vae_interpolation"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed()
    vocab_size = int(data["vocab_size"])
    max_len = int(data["max_seq_len"])
    ckpt = torch_load(args.checkpoint, map_location=device)
    model = MusicVAE(
        vocab_size=vocab_size,
        num_genres=config.NUM_GENRES,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_id=PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    z0 = torch.randn(1, config.LATENT_DIM, device=device)
    z1 = torch.randn(1, config.LATENT_DIM, device=device)
    gid = torch.tensor([args.genre], device=device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, alpha in enumerate(torch.linspace(0, 1, args.steps)):
        z = (1 - alpha) * z0 + alpha * z1
        seq = model.generate(z, gid, max_len=max_len, sos_id=SOS_ID, eos_id=EOS_ID, device=device)
        tokens_to_midi(seq[0].cpu().tolist(), out / f"interp_{i:02d}.mid")
    print(f"Wrote {args.steps} interpolation MIDIs to {out}")


if __name__ == "__main__":
    main()

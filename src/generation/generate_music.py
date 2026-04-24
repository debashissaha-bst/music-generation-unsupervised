"""Generate MIDI from trained checkpoints."""
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
    from models.autoencoder import LSTMMusicAutoencoder
    from models.transformer import GenreTransformerLM
    from models.vae import MusicVAE
    from preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID
    from training.data_utils import load_processed
except ModuleNotFoundError:
    from src.utils_checkpoint import torch_load
    from src import config
    from src.generation.midi_export import tokens_to_midi
    from src.models.autoencoder import LSTMMusicAutoencoder
    from src.models.transformer import GenreTransformerLM
    from src.models.vae import MusicVAE
    from src.preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID
    from src.training.data_utils import load_processed


def load_vocab_meta():
    data = load_processed()
    return int(data["vocab_size"]), int(data["max_seq_len"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ae", "vae", "transformer", "rlhf"], required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=str(config.OUTPUTS_MIDI))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    vocab_size, max_seq = load_vocab_meta()
    max_len = args.max_len or max_seq
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "ae":
        ckpt = torch_load(args.checkpoint, map_location=device)
        model = LSTMMusicAutoencoder(
            vocab_size=vocab_size,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_id=PAD_ID,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for i in range(args.n):
            z = torch.randn(1, config.LATENT_DIM, device=device)
            seq = model.generate(z, max_len=max_len, sos_id=SOS_ID, eos_id=EOS_ID, device=device)
            tokens_to_midi(seq[0].cpu().tolist(), out_dir / f"ae_gen_{i}.mid")

    elif args.task == "vae":
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
        for i in range(args.n):
            z = torch.randn(1, config.LATENT_DIM, device=device)
            gid = torch.tensor([i % config.NUM_GENRES], device=device)
            seq = model.generate(
                z, gid, max_len=max_len, sos_id=SOS_ID, eos_id=EOS_ID, device=device
            )
            tokens_to_midi(seq[0].cpu().tolist(), out_dir / f"vae_gen_{i}_g{gid.item()}.mid")

    elif args.task in ("transformer", "rlhf"):
        ckpt = torch_load(args.checkpoint, map_location=device)
        mlen = min(int(ckpt.get("max_len", max_len)), config.MAX_SEQ_TRANSFORMER)
        model = GenreTransformerLM(
            vocab_size=vocab_size,
            num_genres=config.NUM_GENRES,
            d_model=config.EMBED_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            dim_feedforward=config.TRANSFORMER_FF,
            dropout=config.DROPOUT,
            max_len=mlen,
            pad_id=PAD_ID,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for i in range(args.n):
            gid = torch.tensor([i % config.NUM_GENRES], device=device)
            seq = model.generate(
                gid,
                max_new_tokens=min(max_len, mlen) - 1,
                sos_id=SOS_ID,
                eos_id=EOS_ID,
                device=device,
                temperature=0.95,
            )
            tokens_to_midi(seq[0].cpu().tolist(), out_dir / f"{args.task}_gen_{i}.mid")

    print(f"Wrote {args.n} files to {out_dir}")


if __name__ == "__main__":
    main()

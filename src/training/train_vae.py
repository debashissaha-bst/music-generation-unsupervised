"""Task 2: train genre-conditioned VAE."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import config
    from models.vae import MusicVAE
    from preprocessing.tokenizer import PAD_ID
    from training.data_utils import make_loaders
except ModuleNotFoundError:
    from src import config
    from src.models.vae import MusicVAE
    from src.preprocessing.tokenizer import PAD_ID
    from src.training.data_utils import make_loaders


def train(
    epochs: int | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.RANDOM_SEED)
    train_loader, test_loader, data = make_loaders(batch_size=batch_size, single_genre_id=None)

    vocab_size = int(data["vocab_size"])
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
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    history = []
    best = float("inf")
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINTS / "vae_best.pt"

    total_epochs = int(epochs or config.EPOCHS_VAE)
    for epoch in range(total_epochs):
        beta = min(
            config.KL_BETA,
            config.KL_BETA * (epoch + 1) / max(config.KL_ANNEAL_EPOCHS, 1),
        )
        model.train()
        running_loss = running_kl_loss = 0.0
        n = 0
        for xb, gb in tqdm(train_loader, desc=f"VAE epoch {epoch+1}"):
            xb, gb = xb.to(device), gb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, mu, logvar, _ = model(xb, gb)
            loss = criterion(logits.reshape(-1, vocab_size), xb.reshape(-1))
            kl_loss = model.kl_loss(mu, logvar)
            loss = loss + beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * xb.size(0)
            running_kl_loss += kl_loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / max(n, 1)
        train_kl_loss = running_kl_loss / max(n, 1)

        model.eval()
        val_running = val_kl_running = 0.0
        vn = 0
        with torch.no_grad():
            for xb, gb in test_loader:
                xb, gb = xb.to(device), gb.to(device)
                logits, mu, logvar, _ = model(xb, gb)
                loss = criterion(logits.reshape(-1, vocab_size), xb.reshape(-1))
                kl_loss = model.kl_loss(mu, logvar)
                val_running += loss.item() * xb.size(0)
                val_kl_running += kl_loss.item() * xb.size(0)
                vn += xb.size(0)
        val_loss = val_running / max(vn, 1)
        val_kl_loss = val_kl_running / max(vn, 1)
        history.append((train_loss, train_kl_loss, val_loss, val_kl_loss))
        total_val = val_loss + beta * val_kl_loss
        print(
            f"Epoch {epoch+1} beta={beta:.3f}: train={train_loss:.4f} val={val_loss:.4f} val_kl={val_kl_loss:.4f}"
        )
        if total_val < best:
            best = total_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab_size": vocab_size,
                    "num_genres": config.NUM_GENRES,
                },
                best_path,
            )

    config.OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot([h[0] for h in history], label="train")
    plt.plot([h[2] for h in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("CE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_PLOTS / "vae_loss.png", dpi=150)
    plt.close()
    plt.figure()
    plt.plot([h[3] for h in history], label="val_kl")
    plt.xlabel("Epoch")
    plt.ylabel("KL (approx batch mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_PLOTS / "vae_kl_loss.png", dpi=150)
    plt.close()
    print(f"Saved {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Task 2 genre-conditioned VAE")
    ap.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS_VAE")
    ap.add_argument("--batch_size", type=int, default=None, help="Override config.BATCH_SIZE")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_name=args.device)

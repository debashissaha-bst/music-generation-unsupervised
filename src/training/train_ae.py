"""Task 1: train LSTM autoencoder"""
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
    from models.autoencoder import LSTMMusicAutoencoder
    from preprocessing.tokenizer import PAD_ID
    from training.data_utils import make_loaders
except ModuleNotFoundError:
    from src import config
    from src.models.autoencoder import LSTMMusicAutoencoder
    from src.preprocessing.tokenizer import PAD_ID
    from src.training.data_utils import make_loaders


def train(epochs: int | None = None, batch_size: int | None = None, device_name: str | None = None):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.RANDOM_SEED)
    train_loader, test_loader, data = make_loaders(batch_size=batch_size, single_genre_id=0)

    vocab_size = int(data["vocab_size"])
    model = LSTMMusicAutoencoder(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_id=PAD_ID,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    history = []
    best = float("inf")
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINTS / "ae_best.pt"

    total_epochs = int(epochs or config.EPOCHS_AE)
    for epoch in range(total_epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, _ in tqdm(train_loader, desc=f"AE epoch {epoch+1}"):
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = crit(logits.reshape(-1, vocab_size), xb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        model.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                loss = crit(logits.reshape(-1, vocab_size), xb.reshape(-1))
                vloss += loss.item() * xb.size(0)
                vn += xb.size(0)
        val_loss = vloss / max(vn, 1)
        history.append((train_loss, val_loss))
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab_size": vocab_size,
                    "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
                },
                best_path,
            )

    config.OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot([h[0] for h in history], label="train")
    plt.plot([h[1] for h in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_PLOTS / "ae_loss.png", dpi=150)
    plt.close()
    print(f"Saved {best_path} and loss curve.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Task 1 LSTM Autoencoder")
    ap.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS_AE")
    ap.add_argument("--batch_size", type=int, default=None, help="Override config.BATCH_SIZE")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_name=args.device)

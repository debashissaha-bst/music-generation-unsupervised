"""Task 3: train Transformer LM."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import config
    from models.transformer import GenreTransformerLM
    from preprocessing.tokenizer import PAD_ID
    from training.data_utils import make_loaders
except ModuleNotFoundError:
    from src import config
    from src.models.transformer import GenreTransformerLM
    from src.preprocessing.tokenizer import PAD_ID
    from src.training.data_utils import make_loaders


def train(epochs: int | None = None, batch_size: int | None = None, device_name: str | None = None):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.RANDOM_SEED)
    train_loader, test_loader, data = make_loaders(batch_size=batch_size)

    vocab_size = int(data["vocab_size"])
    max_len = min(int(data["max_seq_len"]), config.MAX_SEQ_TRANSFORMER)
    model = GenreTransformerLM(
        vocab_size=vocab_size,
        num_genres=config.NUM_GENRES,
        d_model=config.EMBED_DIM,
        nhead=config.TRANSFORMER_HEADS,
        num_layers=config.TRANSFORMER_LAYERS,
        dim_feedforward=config.TRANSFORMER_FF,
        dropout=config.DROPOUT,
        max_len=max_len,
        pad_id=PAD_ID,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)

    best = float("inf")
    best_ppl = float("inf")
    history = []
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINTS / "transformer_best.pt"

    total_epochs = int(epochs or config.EPOCHS_TRANSFORMER)
    for epoch in range(total_epochs):
        model.train()
        run = 0.0
        nb = 0
        for xb, gb in tqdm(train_loader, desc=f"Transformer {epoch+1}"):
            xb, gb = xb.to(device), gb.to(device)
            if xb.size(1) > max_len:
                xb = xb[:, :max_len]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, gb)
            loss = model.loss(logits, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            nb += 1

        train_loss = run / max(nb, 1)
        model.eval()
        vrun = 0.0
        vb = 0
        with torch.no_grad():
            for xb, gb in test_loader:
                xb, gb = xb.to(device), gb.to(device)
                if xb.size(1) > max_len:
                    xb = xb[:, :max_len]
                logits = model(xb, gb)
                loss = model.loss(logits, xb)
                vrun += loss.item()
                vb += 1
        val_loss = vrun / max(vb, 1)
        ppl = math.exp(val_loss)
        history.append({"train_nll": train_loss, "val_nll": val_loss, "val_ppl": ppl})
        print(f"Epoch {epoch+1}: train_nll={train_loss:.4f} val_nll={val_loss:.4f} ppl={ppl:.2f}")
        if val_loss < best:
            best = val_loss
            best_ppl = ppl
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab_size": vocab_size,
                    "num_genres": config.NUM_GENRES,
                    "max_len": max_len,
                },
                best_path,
            )

    report = {
        "best_val_nll": best,
        "perplexity": best_ppl,
        "epochs": history,
    }
    config.OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUTS_PLOTS / "transformer_perplexity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {best_path} and perplexity report.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Task 3 Transformer LM")
    ap.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS_TRANSFORMER")
    ap.add_argument("--batch_size", type=int, default=None, help="Override config.BATCH_SIZE")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_name=args.device)

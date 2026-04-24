"""
Task 4: RL fine-tuning.

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from utils_checkpoint import torch_load
    import config
    from models.reward_model import RewardModel
    from models.transformer import GenreTransformerLM
    from preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID, token_to_note
    from training.data_utils import load_processed
except ModuleNotFoundError:
    from src.utils_checkpoint import torch_load
    from src import config
    from src.models.reward_model import RewardModel
    from src.models.transformer import GenreTransformerLM
    from src.preprocessing.tokenizer import EOS_ID, PAD_ID, SOS_ID, token_to_note
    from src.training.data_utils import load_processed


def heuristic_reward(tokens: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    b, L = tokens.shape
    out = []
    for i in range(b):
        row = tokens[i]
        valid = row[row != pad_id]
        if valid.numel() < 3:
            out.append(0.2)
            continue
        pitches = []
        for t in valid.tolist():
            if t <= EOS_ID:
                continue
            try:
                p, _ = token_to_note(int(t))
                pitches.append(p)
            except Exception:
                continue
        if not pitches:
            out.append(0.25)
            continue
        uniq = len(set(pitches)) / max(len(pitches), 1)
        length_term = min(1.0, len(pitches) / 32.0)
        out.append(float(0.5 * uniq + 0.5 * length_term))
    return torch.tensor(out, device=tokens.device, dtype=torch.float32)


def train_reward_model(
    reward: RewardModel,
    tokens: torch.Tensor,
    genre: torch.Tensor,
    device: torch.device,
    steps: int = 300,
) -> None:
    opt = torch.optim.Adam(reward.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    reward.train()
    n = tokens.size(0)
    for s in range(steps):
        idx = torch.randint(0, n, (min(config.BATCH_SIZE, n),))
        xb = tokens[idx].to(device)
        gb = genre[idx].to(device)
        with torch.no_grad():
            target = heuristic_reward(xb)
        opt.zero_grad(set_to_none=True)
        pred = reward(xb)
        loss = crit(pred, target)
        loss.backward()
        opt.step()


def train_rlhf(epochs: int | None = None, device_name: str | None = None):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.RANDOM_SEED)
    data = load_processed()
    tokens = data["tokens"]
    genre = data["genre"]
    vocab_size = int(data["vocab_size"])
    max_len = min(int(data["max_seq_len"]), config.MAX_SEQ_TRANSFORMER)

    ckpt_path = config.CHECKPOINTS / "transformer_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Train transformer first: missing {ckpt_path}")

    ckpt = torch_load(ckpt_path, map_location=device)
    policy = GenreTransformerLM(
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
    policy.load_state_dict(ckpt["model"])

    reward_net = RewardModel(vocab_size=vocab_size, pad_id=PAD_ID).to(device)
    train_reward_model(reward_net, tokens, genre, device)

    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(reward_net.state_dict(), config.CHECKPOINTS / "reward_model.pt")

    opt = torch.optim.Adam(policy.parameters(), lr=config.RL_LR)
    policy.train()
    reward_net.eval()

    metrics = []
    total_epochs = int(epochs or config.EPOCHS_RLHF)
    for it in range(total_epochs):
        batch_r = []
        batch_pg = []
        for _ in range(config.RL_SAMPLES_PER_STEP):
            g = torch.randint(0, config.NUM_GENRES, (config.RL_BATCH,), device=device)
            samp = policy.generate(
                g,
                max_new_tokens=max_len - 1,
                sos_id=SOS_ID,
                eos_id=EOS_ID,
                device=device,
                temperature=1.0,
            )
            with torch.no_grad():
                r_model = reward_net(samp)
                r_heur = heuristic_reward(samp)
                r = 0.7 * r_model + 0.3 * r_heur
            logp = policy.sequence_log_prob(samp, g)
            baseline = r.mean()
            adv = r - baseline
            loss_pg = -(adv.detach() * logp).mean()
            opt.zero_grad(set_to_none=True)
            loss_pg.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            batch_r.append(r.mean().item())
            batch_pg.append(loss_pg.item())
        metrics.append(
            {"iter": it + 1, "mean_reward": sum(batch_r) / len(batch_r), "pg_loss": sum(batch_pg) / len(batch_pg)}
        )
        print(
            f"RLHF iter {it+1}: mean_reward={metrics[-1]['mean_reward']:.4f} pg_loss={metrics[-1]['pg_loss']:.4f}"
        )

    torch.save(
        {"model": policy.state_dict(), "vocab_size": vocab_size, "num_genres": config.NUM_GENRES, "max_len": max_len},
        config.CHECKPOINTS / "rlhf_policy.pt",
    )
    config.OUTPUTS_SURVEY.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUTS_SURVEY / "rlhf_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {config.CHECKPOINTS / 'rlhf_policy.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Task 4 RLHF fine-tuning")
    ap.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS_RLHF")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = ap.parse_args()
    train_rlhf(epochs=args.epochs, device_name=args.device)

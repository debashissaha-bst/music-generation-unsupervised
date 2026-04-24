"""Task 3: Causal Transformer language model with genre conditioning."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from runtime import configure_torch_runtime
except ModuleNotFoundError:
    from src.runtime import configure_torch_runtime

configure_torch_runtime()
import torch.nn.functional as F


class GenreTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_genres: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.genre_embed = nn.Embedding(num_genres, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed_scale = math.sqrt(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        genre_id: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, L = x.shape
        tok = self.embed(x) * self.embed_scale
        g = self.genre_embed(genre_id).unsqueeze(1).expand(-1, L, -1)
        h = tok + g + self.pos_embed[:, :L, :]
        if attn_mask is None:
            attn_mask = self._causal_mask(L, x.device)
        key_padding = x == self.pad_id
        out = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding)
        return self.lm_head(out)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=self.pad_id,
        )

    def sequence_log_prob(self, tokens: torch.Tensor, genre_id: torch.Tensor) -> torch.Tensor:
        """Sum of log π(y_t | y_<t) for each sequence (B,)."""
        if tokens.size(1) < 2:
            return torch.zeros(tokens.size(0), device=tokens.device)
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self.forward(inp, genre_id)
        logp = F.log_softmax(logits, dim=-1)
        idx = tgt.unsqueeze(-1)
        gathered = logp.gather(-1, idx).squeeze(-1)
        mask = tgt != self.pad_id
        summed = (gathered * mask.float()).sum(dim=1)
        return summed

    def perplexity_from_loss(self, loss: float) -> float:
        return math.exp(loss)

    @torch.no_grad()
    def generate(
        self,
        genre_id: torch.Tensor,
        max_new_tokens: int,
        sos_id: int,
        eos_id: int,
        device: torch.device,
        temperature: float = 1.0,
    ):
        self.eval()
        b = genre_id.size(0)
        cur = torch.full((b, 1), sos_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            L = cur.size(1)
            if L >= self.max_len:
                break
            logits = self.forward(cur, genre_id)[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, next_t], dim=1)
            if (next_t == eos_id).all():
                break
        return cur

"""Task 2: Genre-conditioned VAE for multi-genre symbolic music."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from runtime import configure_torch_runtime
except ModuleNotFoundError:
    from src.runtime import configure_torch_runtime

configure_torch_runtime()
class MusicVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_genres: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_genres = num_genres

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.genre_embed = nn.Embedding(num_genres, embed_dim)

        self.enc_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        enc_out = hidden_dim * 2
        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

        self.dec_init_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.dec_init_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.dec_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(
        self, x: torch.Tensor, genre_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.genre_embed(genre_id).unsqueeze(1).expand(-1, x.size(1), -1)
        emb = self.embed(x) + g
        lengths = (x != self.pad_id).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.enc_lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self, z: torch.Tensor, tgt: torch.Tensor, genre_id: torch.Tensor
    ) -> torch.Tensor:
        g = self.genre_embed(genre_id).unsqueeze(1).expand(-1, tgt.size(1), -1)
        emb = self.embed(tgt) + g
        b = z.size(0)
        h0 = self.dec_init_h(z).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.dec_init_c(z).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        out, _ = self.dec_lstm(emb, (h0, c0))
        return self.fc_out(out)

    def forward(
        self, x: torch.Tensor, genre_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, genre_id)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x, genre_id)
        return logits, mu, logvar, z

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        genre_id: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
        device: torch.device,
    ):
        self.eval()
        b = z.size(0)
        cur = torch.full((b, 1), sos_id, dtype=torch.long, device=device)
        h0 = self.dec_init_h(z).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.dec_init_c(z).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        h, c = h0, c0
        generated = [cur]
        for _ in range(max_len - 1):
            g = self.genre_embed(genre_id).unsqueeze(1)
            emb = self.embed(cur) + g
            out, (h, c) = self.dec_lstm(emb, (h, c))
            logits = self.fc_out(out[:, -1, :])
            next_t = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_t)
            cur = next_t
            if (next_t == eos_id).all():
                break
        return torch.cat(generated, dim=1)

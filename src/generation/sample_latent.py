"""Sample latent vectors for AE / VAE generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config


def main():
    p = argparse.ArgumentParser(description="Sample z ~ N(0,I) for VAE or uniform for AE")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--latent", type=int, default=config.LATENT_DIM)
    p.add_argument("--out", type=str, default=str(config.OUTPUTS_PLOTS / "latent_samples.pt"))
    args = p.parse_args()
    z = torch.randn(args.n, args.latent)
    torch.save(z, args.out)
    print(f"Saved {args.out} shape={tuple(z.shape)}")


if __name__ == "__main__":
    main()

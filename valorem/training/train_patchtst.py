#!/usr/bin/env python
"""
Train PatchTST on Valorem's daily feature matrix.

Run
----
python -m valorem.training.train_patchtst \
       --batch 32 --past 60 --horizon 5 --epochs 50
"""
from __future__ import annotations
import argparse, time, pathlib, torch
from torch import nn, optim

from valorem.data.patchtst_loader import make_dataloader
from valorem.models.patchtst_model import make_model

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CHK_DIR = pathlib.Path("checkpoints"); CHK_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
def train(batch: int, past: int, horizon: int, epochs: int, lr: float) -> None:
    # ------------------------------------------------------------------ #
    # DataLoader
    # ------------------------------------------------------------------ #
    loader = make_dataloader(
        batch_size=batch,
        past_window=past,
        horizon=horizon,
        shuffle=True,
    )
    D = loader.dataset.df.shape[1]
    H = horizon
    target_idx = loader.dataset.target_idx        # << index of the label

    # ------------------------------------------------------------------ #
    # Model + optimiser
    # ------------------------------------------------------------------ #
    model = make_model(D, H, seq_len=past).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    for epoch in range(1, epochs + 1):
        model.train()
        total_mse, total_mae, n_seen = 0.0, 0.0, 0
        t0 = time.time()

        for past_seq, future_seq in loader:
            past_seq, future_seq = past_seq.to(DEVICE), future_seq.to(DEVICE)

            pred_all = model(past_seq)                  # [B, H, D]
            pred     = pred_all[:, :, target_idx]       # [B, H]  <- slice

            loss = loss_fn(pred, future_seq)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bsz = past_seq.size(0)
            total_mse += loss.item() * bsz
            total_mae += (pred - future_seq).abs().mean().item() * bsz
            n_seen += bsz

        mse = total_mse / n_seen
        mae = total_mae / n_seen
        print(f"E{epoch:03d}  mse={mse:.4e}  mae={mae:.4e}  "
              f"time={time.time() - t0:.1f}s")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "mse": mse,
                "mae": mae,
            },
            CHK_DIR / f"patchtst_epoch{epoch:03d}.pt",
        )

    print("Training complete!")

# ─────────────────────────────────────────────────────────────
def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--past",    type=int,   default=60)
    ap.add_argument("--horizon", type=int,   default=5)
    ap.add_argument("--epochs",  type=int,   default=20)
    ap.add_argument("--lr",      type=float, default=1e-4)
    train(**vars(ap.parse_args()))

if __name__ == "__main__":
    cli()

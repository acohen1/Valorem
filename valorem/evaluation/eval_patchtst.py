# valorem/evaluation/eval_patchtst.py
"""
One-shot hold-out evaluation for PatchTST.

Usage
-----
python -m valorem.evaluation.eval_patchtst \
       --ckpt checkpoints/patchtst_epoch030.pt \
       --batch 64 --past 60 --horizon 5 --test_span 0.2 \
       --out preds_vs_actuals.csv

* `ckpt`       - path to a `.pt` checkpoint produced by train_patchtst.py
* `test_span`  - fraction of the dataset (or days) reserved for testing
                 (default: last 20 % of rows)
The script prints MAE / RMSE / directional-accuracy and writes a CSV
with columns   date, y_true, y_pred   for downstream plots.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from valorem.data.patchtst_loader import (
    daily_dataframe,
    PatchTSTDataset,
)
from valorem.models.patchtst_model import make_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
def build_test_loader(past: int, horizon: int, batch: int, span: float) -> tuple[DataLoader, int, int, int]:
    """Return a DataLoader on the hold-out tail portion of the dataset."""
    df_all = daily_dataframe()

    # last N rows → test
    test_len = int(len(df_all) * span)
    df_test  = df_all.iloc[-test_len:]

    ds = PatchTSTDataset(df_test, past, horizon)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False)

    D = ds.df.shape[1]
    H = ds.hrz
    tgt_idx = ds.target_idx
    return loader, D, H, tgt_idx


# ----------------------------------------------------------------------
def evaluate(
    ckpt_path: str | pathlib.Path,
    batch: int,
    past: int,
    horizon: int,
    test_span: float,
    out_csv: str | None,
) -> None:
    loader, D, H, tgt_idx = build_test_loader(past, horizon, batch, test_span)

    model = make_model(D, H, seq_len=past).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mae_list, mse_list, rows = [], [], []

    with torch.no_grad():
        for batch_no, (past_seq, future_seq) in enumerate(loader):
            past_seq, future_seq = past_seq.to(DEVICE), future_seq.to(DEVICE)
            preds_all = model(past_seq)
            preds     = preds_all[:, :, tgt_idx]

            mae_list.append(torch.abs(preds - future_seq).cpu().numpy())
            mse_list.append(((preds - future_seq) ** 2).cpu().numpy())

            if out_csv:
                start_row = batch_no * batch
                end_row   = start_row + len(preds)
                dates = loader.dataset.df.index[start_row:end_row]  # aligns rows

                last_true = future_seq[:, -1].cpu().numpy()
                last_pred = preds[:, -1].cpu().numpy()
                rows.extend(zip(dates, last_true, last_pred))

    mae  = np.mean(np.concatenate(mae_list))
    rmse = np.sqrt(np.mean(np.concatenate(mse_list)))
    print(f"TEST  MAE={mae:.4f}  RMSE={rmse:.4f}")

    if out_csv:
        pd.DataFrame(rows, columns=["date", "y_true", "y_pred"]).to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")


# ----------------------------------------------------------------------
def cli() -> None:
    p = argparse.ArgumentParser(description="Hold-out evaluation for PatchTST")
    p.add_argument("--ckpt", required=True, help="checkpoint .pt file")
    p.add_argument("--batch",   type=int,   default=64, help="mini-batch size")
    p.add_argument("--past",    type=int,   default=60, help="context window")
    p.add_argument("--horizon", type=int,   default=5,  help="forecast horizon")
    p.add_argument("--test_span", type=float, default=0.2,
                   help="fraction of data reserved as test set (default 0.2)")
    p.add_argument("--out", help="CSV path to dump (date, y_true, y_pred)")
    args = p.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        batch=args.batch,
        past=args.past,
        horizon=args.horizon,
        test_span=args.test_span,
        out_csv=args.out,
    )


if __name__ == "__main__":
    cli()

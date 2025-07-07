# valorem/models/patchtst_model.py
"""
Wrapper that instantiates the **official PatchTST `Model` class** with a
minimal synthetic `configs` namespace, so Valorem code can build the
network with a single function call.

Directory layout
----------------
Valorem/
├─ external/
│   └─ PatchTST/
│       └─ PatchTST_supervised/       <- cloned repo
│           └─ models/PatchTST.py     <- upstream Model class
└─ valorem/
    └─ models/patchtst_model.py       <- this file
"""
from __future__ import annotations

import sys
import pathlib
from types import SimpleNamespace
import torch.nn as nn

# Make upstream repo importable (no __init__.py needed)
ROOT = pathlib.Path(__file__).resolve().parents[2]            # Valorem/
PATCHTST_ROOT = ROOT / "external" / "PatchTST"
PATCHTST_SUP  = PATCHTST_ROOT / "PatchTST_supervised"

sys.path.append(str(PATCHTST_ROOT))   # exposes `PatchTST_supervised`
sys.path.append(str(PATCHTST_SUP))    # exposes its sibling `layers.*`

# Import the official backbone
from PatchTST_supervised.models.PatchTST import Model as PatchTST  # noqa: E402

# Valorem wrapper
class ValoremPatchTST(nn.Module):
    """
    PatchTST with a clean constructor::

        model = ValoremPatchTST(
            num_features=D,        # feature dim D
            horizon=H,             # forecast steps H
            seq_len=T,             # context window T
            patch_len=16, stride=16,
            d_model=128, n_heads=8, n_layers=3,
        )
    """

    def __init__(
        self,
        *,
        num_features: int,
        horizon: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Build the minimal `configs` object the upstream Model expects
        cfg = SimpleNamespace(
            # core dims
            enc_in=num_features,
            seq_len=seq_len,
            pred_len=horizon,
            # architecture
            e_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff or 4 * d_model,
            # regularisation / misc
            dropout=dropout,
            fc_dropout=dropout,
            head_dropout=0.0,
            individual=False,
            # patch params
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            # RevIN & decomposition defaults
            revin=True,
            affine=True,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )

        self.core = PatchTST(cfg)

    def forward(self, x):                       # x: [B, T, D]
        out = self.core(x)                      # out: [B, H, 1]
        return out.squeeze(-1)                  # -> [B, H]


# Factory for training scripts
def make_model(
    D: int,
    H: int,
    *,
    seq_len: int,
    **kw,
) -> nn.Module:
    """
    Convenience factory matching the Valorem DataLoader shapes.

    Parameters
    ----------
    D : int
        Number of feature columns (``past.shape[-1]``).
    H : int
        Forecast horizon (matches DataLoader ``horizon``).
    seq_len : int
        Context window length (matches DataLoader ``past_window``).
    **kw
        Extra hyper-parameters forwarded to ``ValoremPatchTST``.
    """
    return ValoremPatchTST(num_features=D, horizon=H, seq_len=seq_len, **kw)

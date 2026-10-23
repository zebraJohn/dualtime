"""Phase-Times2D: a single Times2D branch over a day-aligned ("phase") view.

Experiment 1 of the Dual-Times2D plan -- does the phase view have predictive
power on its own?

Structurally identical to the official Times2D baseline (same backbone, same
FSDH, same head); the ONLY difference is `period_list`/`patch_len`, which
control how the series is folded into a 2D grid:

  period   = fold width, i.e. what counts as one cycle
  patch_len = width of a patch within a cycle, i.e. the phase-bin size

The official list is [720, 360, 110, 96, 48] -- every entry is a *multi-day*
scale (30d / 15d / ~4.6d / 4d / 2d), none of them a single day. The phase view
instead uses periods that are whole numbers of days, so a row of the folded
grid is a day and a column is an hour-of-day.
"""
__all__ = ['PhaseTimes2D']

import torch
from torch import nn

from models.Times2D import Times2DBackbone


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = Times2DBackbone(
            configs,
            period_list=configs.phase_period_list,
            patch_len=configs.phase_patch_len,
            use_fsdh=not configs.no_fsdh,
            build_head=True,
            # Legacy Times2D indexes FSDH weights by position within the batch
            # (shape (batch_size, 2)). Kept here so this model differs from the
            # baseline ONLY in period_list -- otherwise any metric change could
            # be blamed on the FSDH parameterisation instead of the phase view.
            per_sample_weights=True,
        )

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        x = batch_x.permute(0, 2, 1)   # [B, N, T]
        x = self.model(x)              # [B, N, pred_len]
        x = x.permute(0, 2, 1)         # [B, pred_len, N]
        return x

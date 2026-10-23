__all__ = ['Times2D']

# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.fft import rfft
import numpy as np
import math
from typing import Callable, Optional
from einops import rearrange
from layers.RevIN import RevIN
from layers.Conv_Blocks import Inception_Block_V1
from layers.encoders import TSTiEncoder, TSTEncoder, TSTEncoderLayer
from layers.PeriodAttention import PeriodAttentionConv, FSDHPeriodAttention

# Official Times2D period list (paper / ETTh1 script). Branch A of Dual-Times2D
# relies on this being the default -- do not edit in place, pass
# `period_list=` to Times2DBackbone instead.
DEFAULT_PERIOD_LIST = [720, 360, 110, 96, 48]


def remap_legacy_state_dict(state_dict):
    """Remap a pre-refactor Times2D checkpoint onto the FSDH-split layout.

    FSDH's `conv`, `weights` and `heatmap_to_pred` used to live directly on the
    backbone; they now live under `model.fsdh.*`. Everything else is untouched,
    so a checkpoint already in the new layout passes straight through.
    """
    legacy = ('model.conv.', 'model.heatmap_to_pred.')
    if not any(k == 'model.weights' or k.startswith(legacy) for k in state_dict):
        return state_dict  # already refactored

    out = {}
    for k, v in state_dict.items():
        if k == 'model.weights':
            out['model.fsdh.weights'] = v
        # NB: the trailing dot in 'model.conv.' is load-bearing.
        # 'model.conv2D.0.weight'.startswith('model.conv') is True, so matching
        # on 'model.conv' without the dot would silently rename all ten conv2D
        # keys to 'model.fsdh.conv2D.*' and the load would fail.
        elif k.startswith('model.conv.'):
            out['model.fsdh.conv.' + k[len('model.conv.'):]] = v
        elif k.startswith('model.heatmap_to_pred.'):
            out['model.fsdh.heatmap_to_pred.' + k[len('model.heatmap_to_pred.'):]] = v
        else:
            out[k] = v
    return out


def compute_derivative_heatmaps(x):
    # Calculate first derivative manually
    first_derivative = x[:, 1:] - x[:, :-1]
    # Pad the first_derivative to maintain original length
    first_derivative = torch.cat([torch.zeros_like(first_derivative[:, :1, :]), first_derivative], dim=1)

    # Calculate second derivative manually
    second_derivative = first_derivative[:, 1:] - first_derivative[:, :-1]
    # Pad the second_derivative to maintain original length
    second_derivative = torch.cat([torch.zeros_like(second_derivative[:, :1, :]), second_derivative], dim=1)

    # Stack the derivatives along a new dimension
    heatmap = torch.stack([first_derivative, second_derivative], dim=-1)
    heatmap = heatmap.permute(0,2,1,3)  # Adjust shape to [Batch, Channels, Time, Derivative]
    return heatmap


class FSDH(nn.Module):
    """Times2D's First/Second Derivative Heatmap branch.

    Layout contract: input is the RevIN-normalised series as **[B, T, N]**
    (time on dim 1 -- RevIN reduces over range(1, ndim-1), see layers/RevIN.py).
    Returns [B, N, pred_len].

    Extracted verbatim from Times2DBackbone so the dual model can share a single
    instance instead of computing the identical heatmap twice.
    """

    def __init__(self, configs, per_sample_weights=True):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.data = configs.data

        # Optional PHAT-style X-shaped attention replacing the Inception conv.
        # Build exactly one of `conv` / `x_conv` so the baseline path consumes
        # the same RNG as before and keeps the same state_dict keys.
        self.fsdh_x_attn = int(getattr(configs, 'fsdh_x_attn', 0))
        if self.fsdh_x_attn:
            fsdh_dropout = getattr(configs, 'fsdh_x_dropout', None)
            if fsdh_dropout is None:
                fsdh_dropout = float(getattr(configs, 'x_attn_dropout', 0.1))
            self.x_conv = FSDHPeriodAttention(
                n_channels=configs.enc_in,
                seq_len=self.seq_len,
                period=int(getattr(configs, 'fsdh_period', 24)),
                patch_len=int(getattr(configs, 'fsdh_patch_len', 2)),
                d_model=int(getattr(configs, 'fsdh_x_dim', 32)),
                n_heads=int(getattr(configs, 'fsdh_x_heads', 4)),
                negative=bool(getattr(configs, 'x_attn_negative', 1)),
                modulation=bool(getattr(configs, 'x_attn_modulation', 1)),
                dropout=float(fsdh_dropout),
                norm=str(getattr(configs, 'x_attn_norm', 'dyt')),
                lambda_init=float(getattr(configs, 'x_attn_lambda', 0.2)),
                mu_init=float(getattr(configs, 'x_attn_mu', 1.0)),
                distance=str(getattr(configs, 'x_attn_distance', 'periodic')),
            )
            self.conv = None
        else:
            self.x_conv = None
            self.conv = nn.Sequential(
                Inception_Block_V1(configs.enc_in, configs.d_ff,
                                   num_kernels=configs.num_kernels),
                nn.GELU(),
                Inception_Block_V1(configs.d_ff, configs.enc_in,
                                   num_kernels=configs.num_kernels))

        # Legacy Times2D indexes these by *position within the batch*: weights[b]
        # is applied to sample b. That couples the parameter shape to batch_size
        # and is kept as the default so existing checkpoints (and the reported
        # baseline numbers) reproduce exactly. Pass per_sample_weights=False for
        # a single shared (1, 2) that broadcasts -- what Dual-Times2D uses.
        n_w = configs.batch_size if per_sample_weights else 1
        self.weights = nn.Parameter(torch.randn(n_w, 2))

        self.heatmap_to_pred = nn.Conv1d(in_channels=self.seq_len, out_channels=self.pred_len, kernel_size=1)

    def forward(self, x):  # x: [B, T, N], RevIN-normalised
        B, T, N = x.size()

        heatmap = compute_derivative_heatmaps(x)                      # [B, N, T, 2]
        if self.x_conv is not None:
            heatmap_features = self.x_conv(heatmap).permute(0, 2, 1, 3)  # [B, T, N, 2]
        else:
            heatmap_features = self.conv(heatmap).permute(0, 2, 1, 3)    # [B, T, N, 2]

        w = self.weights
        if w.shape[0] == 1:
            w = w.expand(B, -1)
        elif w.shape[0] != B:
            raise RuntimeError(
                "FSDH weights are batch-coupled: model has "
                f"{w.shape[0]} but this batch has {B}. Either run with "
                f"batch_size={w.shape[0]}, or build with per_sample_weights=False."
            )

        # [B, T, N]. The original had two consecutive permutes here which cancel
        # out, so heatmap_to_pred is fed [B, T, N] (Conv1d channels = seq_len).
        final = torch.sum(heatmap_features * w.view(B, 1, 1, 2), -1)
        final = self.heatmap_to_pred(final)                           # [B, pred_len, N]
        final = final.permute(0, 2, 1)                                # [B, N, pred_len]
        return final


class Times2DBackbone(nn.Module):
    def __init__(self, configs, period_list=None, patch_len=None,
                 use_fsdh=True, build_head=True, per_sample_weights=True, **kwargs):
        super(Times2DBackbone, self).__init__()

        # --- resolve period/patch FIRST: everything below derives from them ---
        # (applying these after the normal setup would leave top_k, dim_list,
        # tokens_list, Head and conv2D stale and mis-shaped)
        # `period_list` mirrors `patch_len`'s three-way fallback below: the
        # kwarg wins, then configs, then the ETTh1 default. The configs step was
        # MISSING -- period_list took the kwarg or DEFAULT_PERIOD_LIST and never
        # looked at configs, so `--period_list` was a dead CLI argument that
        # silently did nothing (verified: no caller passes the kwarg for the
        # plain Times2D path, and configs.period_list was read nowhere).
        #
        # It matters for any dataset whose period structure differs from
        # ETTh1's: upstream's own weather value is [720, 360, 140, 70, 48],
        # which upstream can only express by editing the hardcoded line.
        #
        # run.py's --period_list default is None for this reason. It used to
        # default to [24, 12] (a TSL-template leftover); making the configs
        # fallback live without also changing that default would have silently
        # re-pointed every existing Times2D run at a different period list.
        if period_list is not None:
            self.period_list = list(period_list)
        elif getattr(configs, 'period_list', None) is not None:
            self.period_list = list(configs.period_list)
        else:
            self.period_list = list(DEFAULT_PERIOD_LIST)
        resolved_patch = list(patch_len) if patch_len is not None else list(configs.patch_len)
        if len(resolved_patch) < len(self.period_list):
            raise ValueError(
                f"need one patch length per period: {len(self.period_list)} periods "
                f"{self.period_list} but only {len(resolved_patch)} patch lengths {resolved_patch}"
            )
        self.patch_len = resolved_patch[:len(self.period_list)]

        # Load parameters from configs
        self.data = configs.data
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.add = configs.add
        self.affine = configs.affine
        self.head_dropout = configs.head_dropout
        self.subtract_last = configs.subtract_last
        self.revin_layer = RevIN(self.enc_in, affine=self.affine, subtract_last=self.subtract_last)
        self.conv_blocks = nn.ModuleList()
        self.backbone = nn.ModuleList()
        self.n_layers = configs.e_layers
        self.wo_conv = configs.wo_conv
        self.serial_conv = configs.serial_conv
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.attn_dropout = configs.attn_dropout
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = nn.Flatten(start_dim=2)
        # Dead in the forward pass, but present in every existing checkpoint --
        # dropping it would make strict load_state_dict fail on unexpected keys.
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        self.dropout = configs.dropout
        self.act = 'gelu'
        self.norm = 'BatchNorm'
        self.key_padding_mask = 'auto'
        self.padding_var = None
        self.attn_mask = None
        self.res_attention = True
        self.pre_norm = False
        self.store_attn = False
        self.pe = 'zeros'
        self.learn_pe = True
        self.verbose = False
        self.batch = configs.batch_size

        # Period decomposition. `period` is the fold width (what counts as one
        # cycle) and `patch_len` is the patch width within a cycle, so
        # tokens = ceil(period / patch_len), while period_len = ceil(seq_len /
        # period) is the number of cycles folded into the 2D grid.
        self.top_k = len(self.period_list)
        self.period_len = [math.ceil(self.seq_len / i) for i in self.period_list]

        # Define kernel_list and stride_list
        self.kernel_list = [(n, self.patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = self.kernel_list

        # Define dim_list and tokens_list
        self.dim_list = [k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]

        # TSTiEncoder silently rebuilds its positional encoding if the token
        # count is wrong, which would swap in a freshly-random PE instead of
        # failing. Assert the derived count matches what the 2D conv emits.
        for i, period in enumerate(self.period_list):
            width = math.ceil(period / self.patch_len[i]) * self.patch_len[i]
            emitted = (width - self.patch_len[i]) // self.patch_len[i] + 1
            if self.tokens_list[i] != emitted:
                raise ValueError(
                    f"period {period} + patch_len {self.patch_len[i]} yields {emitted} tokens "
                    f"but tokens_list says {self.tokens_list[i]}"
                )

        # Optional PHAT-style X-shaped attention replacing each period's conv2D.
        # The output shape is identical, so the TST encoder / Head / dual
        # interfaces below are untouched.  The parameters are NOT: a PNA
        # checkpoint cannot be loaded into the conv path (or vice versa).
        self.x_attn = int(getattr(configs, 'x_attn', 0))
        if self.x_attn:
            x_dim = int(getattr(configs, 'x_attn_dim', 32))
            x_heads = int(getattr(configs, 'x_attn_heads', 4))
            if x_dim % x_heads != 0:
                raise ValueError(
                    f'x_attn_dim ({x_dim}) must be divisible by x_attn_heads ({x_heads})')
            self.conv2D = nn.ModuleList([
                PeriodAttentionConv(
                    period=period,
                    patch_len=self.patch_len[i],
                    seq_len=self.seq_len,
                    d_model=x_dim,
                    n_heads=x_heads,
                    out_dim=self.dim_list[i],
                    negative=bool(getattr(configs, 'x_attn_negative', 1)),
                    modulation=bool(getattr(configs, 'x_attn_modulation', 1)),
                    dropout=float(getattr(configs, 'x_attn_dropout', 0.1)),
                    norm=str(getattr(configs, 'x_attn_norm', 'dyt')),
                    lambda_init=float(getattr(configs, 'x_attn_lambda', 0.2)),
                    mu_init=float(getattr(configs, 'x_attn_mu', 1.0)),
                    distance=str(getattr(configs, 'x_attn_distance', 'periodic')),
                )
                for i, period in enumerate(self.period_list)
            ])
        else:
            self.conv2D = nn.ModuleList([
                nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s)
                for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
            ])

        self.fsdh = FSDH(configs, per_sample_weights=per_sample_weights) if use_fsdh else None
        self.build_head = build_head
        self.head = Head(self.seq_len, self.top_k, self.pred_len, head_dropout=self.head_dropout,
                         Concat=not self.add) if build_head else None

        self.backbone = nn.ModuleList([nn.Sequential(
            TSTiEncoder(self.enc_in, patch_num=token, patch_len=self.dim_list[i], max_seq_len=self.seq_len,
                        n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, d_k=None, d_v=None,
                        d_ff=self.d_ff, norm=self.norm, attn_dropout=self.attn_dropout, head_dropout=self.head_dropout, dropout=self.dropout,
                        act=self.act,
                        key_padding_mask=self.key_padding_mask, padding_var=self.padding_var, attn_mask=self.attn_mask,
                        res_attention=self.res_attention, pre_norm=self.pre_norm, store_attn=self.store_attn, pe=self.pe,
                        learn_pe=self.learn_pe, verbose=self.verbose, **self.kwargs),
            nn.Flatten(start_dim=-2),
            nn.Linear(self.tokens_list[i] * self.d_model, self.seq_len)
            if self.tokens_list[i] * self.d_model != self.seq_len else nn.Identity()
        ) for i, token in enumerate(self.tokens_list)])

        # Device placement is owned by the caller (Exp_Basic does
        # `self._build_model().to(self.device)`); doing it here as well used to
        # leave self.weights stranded on CPU.

    def _period_encoder_inputs(self, x_norm):
        """Shared front-end: the per-period 2D-conv output fed to each encoder.

        Yields (i, out) where out is [B, N, dim_list[i], tokens_list[i]]. Split
        out from period_features so the token-level accessor can reuse exactly
        the same fold/conv path rather than duplicating it.
        """
        B, N, T = x_norm.size()
        # Guard the [B, N, T] contract here too: a [B, T, N] tensor reaching this
        # point otherwise fails deep inside the fold reshape with an opaque
        # "shape ... is invalid for input of size ..." error.
        if N != self.enc_in:
            raise ValueError(
                f"expected {self.enc_in} channels on dim 1 of [B, N, T], got {N}. "
                f"Got shape {tuple(x_norm.shape)}; RevIN returns [B, T, N], so permute "
                "to [B, N, T] before passing it here."
            )
        x = x_norm
        for i, (period, (kernel_height, kernel_width)) in enumerate(zip(self.period_list, self.kernel_list)):
            if self.seq_len % period != 0:
                pad1 = nn.ConstantPad1d((0, period - self.seq_len % period), 0)
                padded_X = pad1(x)
                padded_X = padded_X.reshape(padded_X.shape[0], padded_X.shape[1], padded_X.shape[2] // period, period)
            else:
                padded_X = x.reshape(B, N, T // period, period)

            if period % kernel_width != 0:
                pad2 = nn.ConstantPad1d((0, kernel_width - period % kernel_width), 0)
                out = pad2(padded_X)
            else:
                out = padded_X  # [B, N, patch, periods]

            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
            out = out.unsqueeze(-3)
            out = self.conv2D[i](out)
            out = self.flatten(out)
            out = rearrange(out, '(b n) d p -> b n d p', b=x.size(0))  # [B, N, dim_list[i], tokens_list[i]]
            yield i, out

    def period_features(self, x_norm):
        """Fold -> 2D conv -> TST encoder -> back-projection, once per period.

        x_norm: [B, N, T], already RevIN-normalised. Returns a list of top_k
        tensors of shape [B, N, seq_len]. Normalisation is deliberately NOT done
        here, so that normalise and denormalise use the same RevIN instance.
        """
        return [self.backbone[i](out) for i, out in self._period_encoder_inputs(x_norm)]

    def period_token_features(self, x_norm):
        """Per-period TST encoder outputs, i.e. before the back-projection.

        Returns a list of top_k tensors of shape [B, N, d_model, tokens_list[i]]
        (the encoder's native layout). These are proper token sequences, which is
        what cross-attention needs -- the concatenated [B, N, top_k*seq_len] from
        `feature_map` has already collapsed the token axis.

        No new parameters and no renamed weights: this calls the same submodules
        as `period_features`, just stopping before the per-period Linear.
        """
        return [self.backbone[i][0](out) for i, out in self._period_encoder_inputs(x_norm)]

    def project_period_tokens(self, token_feats):
        """Inverse of `period_token_features`: [B,N,d_model,tokens_i] per period
        -> the concatenated [B, N, top_k*seq_len] that `feature_map` returns.

        Uses the backbone's own Flatten + Linear submodules so the element order
        matches exactly (Flatten(start_dim=-2) is d_model-major, which a naive
        reshape from a [B,N,tokens,d_model] tensor would get wrong).
        """
        outs = []
        for i, t in enumerate(token_feats):
            outs.append(self.backbone[i][2](self.backbone[i][1](t)))
        return torch.cat(outs, dim=-1)

    def feature_map(self, x_norm):
        """Concatenated per-period features: [B, N, top_k * seq_len]."""
        return torch.cat(self.period_features(x_norm), dim=-1)

    def _normalise(self, x):
        """[B, N, T] raw -> [B, T, N] RevIN-normalised, with the layout asserted."""
        B, N, T = x.shape
        # Layout contract: this module takes [B, N, T], but RevIN and FSDH both
        # need [B, T, N]. Getting this wrong is the dangerous case -- with
        # affine=False RevIN does not raise, it normalises the wrong axis and
        # silently produces a different model.
        if N != self.enc_in:
            raise ValueError(
                f"expected {self.enc_in} channels on dim 1 of [B, N, T], got {N}. "
                "Note the model takes [B, N, T]; RevIN/FSDH internally use [B, T, N]."
            )
        return self.revin_layer(x.permute(0, 2, 1), 'norm')

    def forward_features(self, x):
        """x: [B, N, T] raw series.

        Returns (feat [B, N, top_k*seq_len], heatmap [B, N, pred_len] or None).
        """
        x_btn = self._normalise(x)                                      # [B, T, N]
        heatmap = None if self.fsdh is None else self.fsdh(x_btn)       # [B, N, pred_len]
        feat = self.feature_map(x_btn.permute(0, 2, 1))                 # [B, N, top_k*T]
        return feat, heatmap

    def forward(self, x):  # x: [B, N, T]
        if not self.build_head:
            raise RuntimeError("this backbone was built without a head; call forward_features() instead")

        x_btn = self._normalise(x)                                      # [B, T, N]
        heatmap = None if self.fsdh is None else self.fsdh(x_btn)
        res = self.period_features(x_btn.permute(0, 2, 1))

        z = self.head(res)                                              # [B, N, pred_len]
        combined = z if heatmap is None else z + heatmap
        combined = combined.permute(0, 2, 1)                            # [B, pred_len, N]
        combined = self.revin_layer(combined, 'denorm')
        combined = combined.permute(0, 2, 1)                            # [B, N, pred_len]
        return combined


class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0,
                 Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(context_window * (num_period if Concat else 1), target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: list of [bs x nvars x context_window]
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = Times2DBackbone(configs, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Checkpoints written before the FSDH split carry model.conv.* /
        # model.weights / model.heatmap_to_pred.* at the backbone level.
        state_dict = remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        x = batch_x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x

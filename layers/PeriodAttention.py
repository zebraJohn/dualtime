"""PHAT-style X-shaped attention for Times2D's folded period grids.

The folded grid is ``[H, W]`` per channel, where

  H = ceil(seq_len / period)       number of periods ("cycles", aligned axis)
  W = ceil(period / patch_len)     patches within one period ("offsets")

For a token at ``(h, w)`` the two axes are:

  * period-offset  -- vary w at fixed h (within one cycle);
  * period-aligned -- vary h at fixed w (same phase across cycles).

PHAT's PNA applies a softmax over each axis and composes the two.  The offset
axis additionally uses positive/negative logits with a periodic-distance
modulation; the aligned axis is a plain softmax.

Two conventions are pinned here.  PHAT's main text writes the positive set as
``{s : d(i,s) < d(i,j)} U {i}`` and omits the union for the negative set, but
its own appendix proof (Eqs. 22-23) subtracts Softplus at the **target** j:

  * positive set: ``{s : d(i,s) < d(i,j)} U {j}``;
  * negative set: ``{s : d(i,s) > d(i,j)} U {j}``.

Target-inclusive is the convention that makes the stick-breaking identity
``exp(a_tilde[i,j]) = sigmoid(a[i,j]) * prod_{closer} (1 - sigmoid(a[i,s]))``
hold.  Ties are excluded from both sets.  ``d`` is circular by default (PHAT),
with a ``linear`` option for fold widths that are not true periods.

Everything is computed in the module's dtype.  When autocast is active it is
disabled for the attention so the ``finfo.min/2`` mask fill cannot overflow to
-inf in fp16 and produce NaNs.
"""

__all__ = [
    'DynamicTanh',
    'PeriodAttn',
    'PeriodAttentionConv',
    'FSDHPeriodAttention',
    'distance_index',
    'apply_modulation',
    'separable_combine',
    'fold_series',
    'unfold_series',
]

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# distance / modulation
# --------------------------------------------------------------------------- #

def distance_index(n_offsets: int, mode: str = 'periodic',
                   device: Optional[torch.device] = None
                   ) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (order, n_lt, n_gt) for an offset axis of length ``n_offsets``.

    order[i] sorts candidates by distance from i.  n_lt[i, j] / n_gt[i, j] are
    the numbers of candidates strictly closer / farther than j.  The first
    ``n_lt[i, j]`` entries of ``order[i]`` are therefore exactly the closer
    set, and the last ``n_gt[i, j]`` entries are the farther set.
    """
    W = int(n_offsets)
    if W < 1:
        raise ValueError(f'n_offsets must be >= 1, got {W}')
    idx = torch.arange(W, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    if mode == 'periodic':
        dist = torch.minimum(dist, W - dist)
    elif mode != 'linear':
        raise ValueError(f"distance mode must be 'periodic' or 'linear', got {mode!r}")

    order = torch.argsort(dist, dim=-1, stable=True)
    # n_lt[i, j] = sum_s 1[dist[i, s] < dist[i, j]]
    n_lt = (dist[:, None, :] < dist[:, :, None]).sum(dim=-1)
    n_gt = (dist[:, None, :] > dist[:, :, None]).sum(dim=-1)
    return order, n_lt, n_gt


def _expand_to(t: Tensor, like: Tensor) -> Tensor:
    return t.to(device=like.device).expand_as(like)


def apply_modulation(scores: Tensor, order: Tensor, n_lt: Tensor,
                     n_gt: Tensor, kind: str,
                     key_valid: Optional[Tensor] = None) -> Tensor:
    """Sum of Softplus(scores) over the closer/farther set.

    ``scores`` has shape ``[..., W, W]`` (query i, candidate s).  The returned
    tensor has the same shape.  This is an exact O(W^2) evaluation of PHAT's
    set sums -- no W^3 mask is materialised.

    ``key_valid`` is an optional bool tensor broadcastable to ``scores``
    (True = candidate is a real patch).  Invalid candidates contribute zero to
    the sums, so changing padded inputs cannot change a valid token's
    modulation through the invalid logits.
    """
    if scores.ndim < 2:
        raise ValueError('scores must have at least two dimensions')
    W = scores.shape[-1]
    if scores.shape[-2] != W:
        raise ValueError(f'expected a square offset axis, got {tuple(scores.shape)}')
    if W == 0:
        return torch.zeros_like(scores)
    if kind not in ('close', 'far'):
        raise ValueError(f"kind must be 'close' or 'far', got {kind!r}")

    target_sp = F.softplus(scores)
    if key_valid is not None:
        key_valid = key_valid.to(device=scores.device, dtype=scores.dtype)
        target_sp = target_sp * key_valid

    order_e = _expand_to(order, scores)
    # Gather the masked Softplus values directly: invalid entries are zero, so
    # the cumulative sums only count real candidates.
    a_sorted = torch.gather(target_sp, -1, order_e)
    cum = torch.cumsum(a_sorted, dim=-1)
    total = cum[..., -1]

    if kind == 'close':
        # Strictly-closer set = the first n_lt entries (for j != i this already
        # contains self, because d(i,i)=0 < d(i,j)).  PHAT's proof multiplies
        # the target j's own sigmoid in, which corresponds to subtracting
        # Softplus at j as well -- see the module docstring.
        idx = (_expand_to(n_lt, scores).clamp(min=1) - 1)
        close = torch.gather(cum, -1, idx)
        close = torch.where(_expand_to(n_lt, scores) > 0, close,
                            torch.zeros_like(close))
        return close + target_sp

    # Farther set = total minus everything up to the last non-farther entry
    # (which includes the target j itself), plus the target j's own Softplus.
    n_gt_e = _expand_to(n_gt, scores)
    before_idx = ((W - n_gt_e).clamp(min=1) - 1)
    before = torch.gather(cum, -1, before_idx)
    far = total[..., :, None] - before
    far = torch.where(n_gt_e > 0, far, torch.zeros_like(far))
    return far + target_sp


def masked_softmax(logits: Tensor, mask: Optional[Tensor] = None,
                   dim: int = -1) -> Tensor:
    """Softmax with a finite negative fill (never -inf)."""
    if mask is not None:
        fill = torch.finfo(logits.dtype).min / 2
        logits = logits.masked_fill(~mask.to(device=logits.device), fill)
    return F.softmax(logits, dim=dim)


# --------------------------------------------------------------------------- #
# separable composition
# --------------------------------------------------------------------------- #

def separable_combine(a_off: Tensor, a_aln: Tensor, v: Tensor,
                      chunk_elems: int = 8_000_000) -> Tensor:
    """Compose offset and aligned attention over a [H, W] grid.

    a_off: [B, h, H, Wq, Wp]   offset weights (same cycle, across patches)
    a_aln: [B, h, Hq, W, Hk]   aligned weights (same patch, across cycles)
    v:     [B, h, Hk, Wp, d]   values (key cycle, key patch, feature)

    out[b, h, i, w, c] = sum_{k,p} a_aln[b,h,i,w,k] * a_off[b,h,i,w,p]
                                 * v[b,h,k,p,c]
    """
    B, heads, H, Wq, Wp = a_off.shape
    if a_aln.shape[0] != B or a_aln.shape[1] != heads:
        raise ValueError('a_aln batch/head dimensions do not match a_off')
    if a_aln.shape[2] != H or a_aln.shape[3] != Wq or a_aln.shape[4] != v.shape[2]:
        raise ValueError('a_aln shape is incompatible with a_off/v')
    if v.shape[0] != B or v.shape[1] != heads or v.shape[3] != Wp:
        raise ValueError('v shape is incompatible with a_off')

    d = v.shape[-1]
    denom = max(1, heads * H * Wq * Wp * d)
    chunk = max(1, int(chunk_elems // denom))
    outs = []
    for start in range(0, B, chunk):
        end = min(B, start + chunk)
        v_c = v[start:end]
        tmp = torch.einsum('bhiwk,bhkpc->bhiwpc',
                           a_aln[start:end], v_c)
        out = torch.einsum('bhiwp,bhiwpc->bhiwc', a_off[start:end], tmp)
        outs.append(out)
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

class DynamicTanh(nn.Module):
    """DyT: gamma * tanh(alpha * x) + beta, per channel."""

    def __init__(self, dim: int, alpha_init: float = 1.0,
                 gamma_init: float = 1.0, beta_init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.gamma = nn.Parameter(torch.full((dim,), float(gamma_init)))
        self.beta = nn.Parameter(torch.full((dim,), float(beta_init)))

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


def _make_norm(norm: str, dim: int) -> nn.Module:
    if norm == 'dyt':
        return DynamicTanh(dim)
    if norm == 'layernorm':
        return nn.LayerNorm(dim)
    if norm == 'none':
        return nn.Identity()
    raise ValueError(f"x_attn_norm must be 'dyt', 'layernorm' or 'none', got {norm!r}")


# --------------------------------------------------------------------------- #
# PNA
# --------------------------------------------------------------------------- #

class PeriodAttn(nn.Module):
    """Positive-negative period attention on a ``[H, W]`` patch grid."""

    def __init__(self, d_model: int, n_heads: int, n_cycles: int, n_offsets: int,
                 negative: bool = True, modulation: bool = True,
                 dropout: float = 0.1, norm: str = 'dyt',
                 lambda_init: float = 0.2, mu_init: float = 1.0,
                 distance: str = 'periodic'):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by n_heads ({n_heads})')
        if n_cycles < 1 or n_offsets < 1:
            raise ValueError(f'n_cycles/n_offsets must be >= 1, got {n_cycles}/{n_offsets}')
        if not 0.0 < lambda_init < 1.0:
            raise ValueError(f'lambda_init must be in (0, 1), got {lambda_init}')
        if mu_init <= 0:
            raise ValueError(f'mu_init must be > 0, got {mu_init}')

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_cycles = int(n_cycles)
        self.n_offsets = int(n_offsets)
        self.d_head = self.d_model // self.n_heads
        self.negative = bool(negative)
        self.modulation = bool(modulation)
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if self.negative:
            self.q2 = nn.Linear(d_model, d_model)
            self.k2 = nn.Linear(d_model, d_model)
            lam = torch.full((n_heads,), math.log(lambda_init / (1.0 - lambda_init)))
            self.lambda_raw = nn.Parameter(lam)
        else:
            self.register_parameter('lambda_raw', None)

        if self.n_cycles > 1:
            # softplus(mu_raw) == mu_init
            mu_raw = math.log(math.expm1(mu_init))
            self.mu_raw = nn.Parameter(torch.full((n_heads,), mu_raw))
        else:
            self.register_parameter('mu_raw', None)

        self.norm = _make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        order, n_lt, n_gt = distance_index(n_offsets, distance)
        self.register_buffer('order', order, persistent=False)
        self.register_buffer('n_lt', n_lt, persistent=False)
        self.register_buffer('n_gt', n_gt, persistent=False)

    def _project(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, H, W, _ = x.shape
        shape = (B, H, W, self.n_heads, self.d_head)

        def heads(y: Tensor) -> Tensor:
            return y.reshape(shape).permute(0, 3, 1, 2, 4)  # [B,h,H,W,dh]

        return heads(self.q1(x)), heads(self.k1(x)), heads(self.v(x))

    def forward(self, z: Tensor, valid: Optional[Tensor] = None) -> Tensor:
        """z: [B, H, W, d_model]; valid: optional bool [H, W] (True = real)."""
        B, H, W, D = z.shape
        if H != self.n_cycles or W != self.n_offsets:
            raise ValueError(
                f'expected grid [{self.n_cycles}, {self.n_offsets}], got [{H}, {W}]')
        if D != self.d_model:
            raise ValueError(f'expected d_model={self.d_model}, got {D}')

        if valid is not None:
            valid = valid.to(device=z.device, dtype=torch.bool)
            if valid.shape != (H, W):
                raise ValueError(f'valid must be [{H}, {W}], got {tuple(valid.shape)}')
            mask_off = valid[None, None, :, None, :]            # [1,1,H,1,W]
            mask_aln = valid.transpose(0, 1)[None, None, None, :, :]  # [1,1,1,W,H]
            key_valid = mask_off
        else:
            mask_off = None
            mask_aln = None
            key_valid = None

        # Disable autocast only for this block: fp16 + a large negative mask can
        # overflow to -inf and produce NaNs.  Compute in the parameter dtype so
        # an outer AMP autocast cannot hand a fp16 activation to fp32 weights.
        orig_dtype = z.dtype
        with torch.autocast(device_type=z.device.type, enabled=False):
            x = z.to(self.q1.weight.dtype)
            q1, k1, v = self._project(x)

            if valid is not None:
                # The separable product A_aln * A_off can select V[k, p] where
                # k is a valid aligned key at offset w and p is a valid offset
                # key at cycle i, even when (k, p) itself is invalid.  Zeroing
                # invalid values removes those cross terms without breaking the
                # separable (X-shaped) structure.
                v = v * valid[None, None, :, :, None]

            alpha = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
            if self.modulation:
                alpha = alpha - apply_modulation(
                    alpha, self.order, self.n_lt, self.n_gt, 'close',
                    key_valid=key_valid)
            a_off = masked_softmax(alpha, mask_off, dim=-1)

            if self.negative:
                q2 = self.q2(x).reshape(B, H, W, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)
                k2 = self.k2(x).reshape(B, H, W, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)
                beta = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
                if self.modulation:
                    beta = beta - apply_modulation(
                        beta, self.order, self.n_lt, self.n_gt, 'far',
                        key_valid=key_valid)
                lam = torch.sigmoid(self.lambda_raw).view(1, self.n_heads, 1, 1, 1)
                a_off = a_off - lam * masked_softmax(beta, mask_off, dim=-1)

            a_off = self.attn_dropout(a_off)

            if H > 1:
                gamma = torch.einsum('bhiwc,bhkwc->bhiwk', q1, k1)
                mu = F.softplus(self.mu_raw).view(1, self.n_heads, 1, 1, 1)
                a_aln = masked_softmax(gamma * self.scale * mu, mask_aln, dim=-1)
            else:
                # The aligned axis has a single cycle.  An all-ones matrix is the
                # algebraic identity; skipping the mu parameter avoids a dead one.
                a_aln = torch.ones(B, self.n_heads, H, W, H,
                                   device=z.device, dtype=v.dtype)
            a_aln = self.attn_dropout(a_aln)

            out = separable_combine(a_off, a_aln, v)
            out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, D)
            out = self.dropout(self.out_proj(out))
            x = self.norm(x + out)
            if valid is not None:
                x = x * valid[None, :, :, None]
            x = x.to(orig_dtype)
        return x


# --------------------------------------------------------------------------- #
# drop-in replacement for Times2D's per-period conv2D
# --------------------------------------------------------------------------- #

class PeriodAttentionConv(nn.Module):
    """Drop-in replacement for ``Conv2d(1, dim_list, (H, patch_len), stride=s)``.

    Input  [B*N, 1, H, W_pad] -> output [B*N, H*patch_len, 1, W].
    The patch embedding is 1 x patch_len: it never mixes cycles.  Cross-cycle
    mixing is the aligned attention's job.
    """

    def __init__(self, period: int, patch_len: int, seq_len: int,
                 d_model: int, n_heads: int, out_dim: Optional[int] = None,
                 negative: bool = True, modulation: bool = True,
                 dropout: float = 0.1, norm: str = 'dyt',
                 lambda_init: float = 0.2, mu_init: float = 1.0,
                 distance: str = 'periodic'):
        super().__init__()
        if period <= 0 or patch_len <= 0 or seq_len <= 0:
            raise ValueError('period, patch_len and seq_len must all be > 0')
        self.period = int(period)
        self.patch_len = int(patch_len)
        self.seq_len = int(seq_len)
        self.n_cycles = math.ceil(seq_len / period)
        self.n_offsets = math.ceil(period / patch_len)
        self.dim_list = self.n_cycles * self.patch_len
        if out_dim is not None and int(out_dim) != self.dim_list:
            raise ValueError(
                f'out_dim={out_dim} does not match H*patch_len={self.dim_list}')

        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=(1, self.patch_len),
            stride=(1, self.patch_len))
        self.attn = PeriodAttn(
            d_model, n_heads, self.n_cycles, self.n_offsets,
            negative=negative, modulation=modulation, dropout=dropout,
            norm=norm, lambda_init=lambda_init, mu_init=mu_init,
            distance=distance)
        self.out_proj = nn.Linear(d_model, self.patch_len)

        h = torch.arange(self.n_cycles)[:, None]
        w = torch.arange(self.n_offsets)[None, :]
        valid = (h * self.period + w * self.patch_len) < self.seq_len
        self.register_buffer('valid', valid, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, Wp = x.shape
        if C != 1 or H != self.n_cycles or Wp != self.n_offsets * self.patch_len:
            raise ValueError(
                f'expected [B,1,{self.n_cycles},{self.n_offsets * self.patch_len}], '
                f'got {tuple(x.shape)}')
        z = self.patch_embed(x)                       # [B, d, H, W]
        z = z.permute(0, 2, 3, 1)                     # [B, H, W, d]
        z = self.attn(z, self.valid)                  # [B, H, W, d]
        z = self.out_proj(z)                          # [B, H, W, patch_len]
        # out_proj has a bias; zero the invalid cycle/offset blocks again so a
        # padded query cannot leak its bias into the output token.
        z = z * self.valid[None, :, :, None]
        z = z.permute(0, 1, 3, 2)                     # [B, H, patch_len, W]
        z = z.reshape(B, self.dim_list, self.n_offsets)
        return z.unsqueeze(2)                         # [B, dim_list, 1, W]


# --------------------------------------------------------------------------- #
# FSDH's derivative-heatmap path
# --------------------------------------------------------------------------- #

def fold_series(x: Tensor, n_cycles: int, period: int,
                n_offsets: int, patch_len: int) -> Tensor:
    """[..., T, C] -> [..., n_cycles, n_offsets, patch_len, C] (zero padded)."""
    C = x.shape[-1]
    T = x.shape[-2]
    pad_t = n_cycles * period - T
    if pad_t:
        x = F.pad(x, (0, 0, 0, pad_t))
    x = x.reshape(*x.shape[:-2], n_cycles, period, C)
    pad_p = n_offsets * patch_len - period
    if pad_p:
        x = F.pad(x, (0, 0, 0, pad_p))
    return x.reshape(*x.shape[:-3], n_cycles, n_offsets, patch_len, C)


def unfold_series(y: Tensor, seq_len: int, n_cycles: int, period: int,
                  n_offsets: int, patch_len: int) -> Tensor:
    """Inverse of fold_series: [..., H, W, patch_len, C] -> [..., T, C]."""
    C = y.shape[-1]
    y = y.reshape(*y.shape[:-4], n_cycles, n_offsets * patch_len, C)
    y = y[..., :period, :]
    y = y.reshape(*y.shape[:-3], n_cycles * period, C)
    return y[..., :seq_len, :]


class FSDHPeriodAttention(nn.Module):
    """Replacement for FSDH's Inception conv on [B, N, T, 2] -> [B, N, T, 2].

    A pointwise channel mix replaces the old conv's cross-variate mixing; the
    temporal derivative maps are folded by ``period`` and processed by PNA.
    """

    def __init__(self, n_channels: int, seq_len: int, period: int,
                 patch_len: int, d_model: int, n_heads: int,
                 negative: bool = True, modulation: bool = True,
                 dropout: float = 0.1, norm: str = 'dyt',
                 lambda_init: float = 0.2, mu_init: float = 1.0,
                 distance: str = 'periodic'):
        super().__init__()
        if period <= 0 or patch_len <= 0 or seq_len <= 0:
            raise ValueError('period, patch_len and seq_len must all be > 0')
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.period = int(period)
        self.patch_len = int(patch_len)
        self.n_cycles = math.ceil(seq_len / period)
        self.n_offsets = math.ceil(period / patch_len)

        self.channel_mix = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1)
        self.patch_embed = nn.Conv2d(
            2, d_model, kernel_size=(1, self.patch_len),
            stride=(1, self.patch_len))
        self.attn = PeriodAttn(
            d_model, n_heads, self.n_cycles, self.n_offsets,
            negative=negative, modulation=modulation, dropout=dropout,
            norm=norm, lambda_init=lambda_init, mu_init=mu_init,
            distance=distance)
        self.out_proj = nn.Linear(d_model, 2 * self.patch_len)

        h = torch.arange(self.n_cycles)[:, None]
        w = torch.arange(self.n_offsets)[None, :]
        valid = (h * self.period + w * self.patch_len) < self.seq_len
        self.register_buffer('valid', valid, persistent=False)

    def forward(self, heatmap: Tensor) -> Tensor:
        B, N, T, C = heatmap.shape
        if N != self.n_channels or T != self.seq_len or C != 2:
            raise ValueError(
                f'expected [B,{self.n_channels},{self.seq_len},2], '
                f'got {tuple(heatmap.shape)}')

        x = self.channel_mix(heatmap)                 # [B, N, T, 2]
        x = fold_series(x, self.n_cycles, self.period,
                        self.n_offsets, self.patch_len)   # [B,N,H,W,k,2]
        x = x.permute(0, 1, 5, 2, 3, 4)               # [B,N,2,H,W,k]
        x = x.reshape(B * N, 2, self.n_cycles,
                      self.n_offsets * self.patch_len)

        z = self.patch_embed(x)                       # [B*N,d,H,W]
        z = z.permute(0, 2, 3, 1)                     # [B*N,H,W,d]
        z = self.attn(z, self.valid)
        z = self.out_proj(z)                          # [B*N,H,W,2k]
        z = z * self.valid[None, :, :, None]
        z = z.reshape(B * N, self.n_cycles, self.n_offsets,
                      self.patch_len, 2)
        z = unfold_series(z, T, self.n_cycles, self.period,
                          self.n_offsets, self.patch_len)    # [B*N,T,2]
        return z.reshape(B, N, T, 2)

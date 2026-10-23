"""Dual-branch Patch + FreEformer model whose cross-attention is CROSS-CHANNEL.

Every earlier dual -- DualTimes2D, DualPhaseFormer, DualFreDN -- packed its CA
tokens with batch dimension B*N, so channel c could only ever attend to channel
c. PROGRESS 11 measured the consequence: rho(Times2D, FreDN) = 0.9593, and
tests/test_cross_channel.py's INV3-control reads exactly 0.000e+00 for FreDN.

This model puts the CHANNEL on the token axis instead:

    patch side  one token per (channel, period)   ->  N*top_k tokens
    freq side   one token per channel             ->  N tokens

The two lengths differ and that is fine: BidirectionalCrossAttention uses
nn.MultiheadAttention, which does not require L == S.

Consequences worth stating, because they are easy to misread:

  * There is NO FeatureAdapter. Each side is natively its own width wide --
    `d_patch` for the patch branch, `d_freq` for the frequency branch -- so the
    rank bottleneck that shaped DualFreDN (see its "KNOWN LIMITATION") does not
    arise, and `2*fre_embed_size == d_model` is NOT a constraint here. Do not
    add an adapter. In v1 both widths were `configs.d_model`; v2 lets each
    branch keep its native architecture (Times2D 64, FreEformer 512), which is
    the whole point of the unequal-width CA below.

  * `F_chan` is a READ-ONLY copy of the frequency branch. CA must not write
    back: that branch's fc and fre_trans_*[2] receive gradient only through
    y_freq, and disturbing that path either kills them or makes y_freq stop
    being that branch's own output. Same rule as DualFreDN -- see its INV5
    note -- and tests/test_dual_freeformer_invariants.py INV4/INV6 pin it.

  * The fused path reads patch_feat AND the CA-updated tokens. DualFreDN's
    fused path read only the CA-updated tensors, but here the patch side had to
    pool each period's token axis down to one token to get the channel onto the
    token axis, so P can no longer be unpacked back into patch_feat. Reading
    both is what keeps "CA did not help" from being confounded with "we threw
    the patch detail away" (spec 2.3).
"""
__all__ = ['Model']

import torch
from torch import nn

from layers.RevIN import RevIN
from models.Times2D import Times2DBackbone, FSDH
from models.FreEformer import Model as FreEformerBackbone
from models.FreEformer import _cfg


class UnequalBiCrossAttention(nn.Module):
    """Bidirectional cross-attention between two DIFFERENT widths.

    The sibling of models/DualPhaseFormer.py's BidirectionalCrossAttention,
    which hardcodes one `d_model` for both its LayerNorms and both its MHAs and
    is shared by DualTimes2D / DualPhaseFormer / DualFreDN. That class is left
    alone: three other duals depend on it and none of them is unequal-width, so
    a branch added there would be permanently unreachable for them.

    nn.MultiheadAttention takes kdim/vdim, so neither direction needs a
    projection, and each returns at its own query width (PROGRESS 12.11
    verified both directions at 64 <-> 512 before this was written).

    ⚠️ V1 CHECKPOINTS DO NOT LOAD HERE. Ten state_dict keys differ from the
    class above (norm_phase -> norm_freq, patch_queries_phase ->
    patch_queries_freq, phase_queries_patch -> freq_queries_patch, each with
    .weight and .bias, plus the three in_proj/out_proj sets). The v1 fallback
    keeps the run PATH and the setting NAME valid -- a config that sets no
    fe_* field still reproduces v1 bitwise at construction, which is what
    tests/test_fre_branch.py F2 pins -- but
    `exp/exp_long_term_forecasting.py:469` loads with strict=True, so
    re-evaluating one of the v1 DualFreEformer runs now raises a missing-keys
    RuntimeError. That is a real regression: PROGRESS 12.10.2's workflow
    (evaluate a checkpoint you already have) no longer works for the nine v1
    ca0/ca1 directories.

    Re-evaluate v1 with v1's code (`git checkout 20dc970`) rather than
    retraining, or pass strict=False and accept that the CA blocks are then
    freshly initialised. Renaming these attributes to v1's was the alternative
    and was rejected: "phase" is the wrong word for the frequency branch here,
    and the confusion would outlive the inconvenience.

    WHAT THIS DOES NOT BUY: the exchange capacity is still capped by the
    narrower side. Direction 1's queries come from a d_patch-wide input X, so
    Q = X @ W_q has rank <= d_patch no matter how the attention is arranged.
    v2 may claim "each branch runs at its native width"; it may NOT claim "the
    CA got more capacity" (spec 2.5).
    """

    def __init__(self, d_patch, d_freq, n_heads, dropout=0.0):
        super().__init__()
        if d_patch % n_heads or d_freq % n_heads:
            raise ValueError(
                f'n_heads={n_heads} must divide BOTH widths: d_patch={d_patch}, '
                f'd_freq={d_freq}. nn.MultiheadAttention requires embed_dim % '
                f'n_heads == 0 for each direction, and the two directions have '
                f'different embed_dims.')
        self.d_patch = d_patch
        self.d_freq = d_freq
        self.norm_patch = nn.LayerNorm(d_patch)
        self.norm_freq = nn.LayerNorm(d_freq)
        self.patch_queries_freq = nn.MultiheadAttention(
            embed_dim=d_patch, num_heads=n_heads, kdim=d_freq, vdim=d_freq,
            dropout=dropout)
        self.freq_queries_patch = nn.MultiheadAttention(
            embed_dim=d_freq, num_heads=n_heads, kdim=d_patch, vdim=d_patch,
            dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, f_patch, f_freq):
        """f_patch: [Lp, B, d_patch], f_freq: [Lq, B, d_freq] (seq-first)."""
        p_norm = self.norm_patch(f_patch)
        f_norm = self.norm_freq(f_freq)

        upd_patch, _ = self.patch_queries_freq(p_norm, f_norm, f_norm)
        f_patch = f_patch + self.drop(upd_patch)

        upd_freq, _ = self.freq_queries_patch(f_norm, p_norm, p_norm)
        f_freq = f_freq + self.drop(upd_freq)
        return f_patch, f_freq


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        # v2: two widths. `d_patch` is the Times2D branch (configs.d_model, 64);
        # `d_freq` is the FreEformer branch and may be its native 512. None
        # falls back, so a config that sets nothing behaves exactly as v1.
        d_patch = configs.d_model
        d_freq = _cfg(configs, 'fe_d_model', configs.d_model)
        self.d_patch, self.d_freq = d_patch, d_freq

        # --- branches ---------------------------------------------------
        # Same flags as DualFreDN: the patch branch carries no head and no
        # FSDH of its own (the shared head replaces the former, and a single
        # FSDH instance is shared by both paths).
        self.branch_patch = Times2DBackbone(
            configs, use_fsdh=False, build_head=False, per_sample_weights=True)
        self.branch_freq = FreEformerBackbone(configs)

        self.revin_layer = RevIN(configs.enc_in, affine=configs.affine,
                                 subtract_last=configs.subtract_last)
        self.fsdh = None if configs.no_fsdh else FSDH(configs)

        # --- cross-attention --------------------------------------------
        # Token axis is the CHANNEL. There is no width adapter, so neither
        # branch is projected: the capacity limit is the narrower side's own
        # width, which is a property of the branches, not of this code (2.5).
        self.top_k = self.branch_patch.top_k
        self.enc_in = configs.enc_in
        self.use_cross_attn = bool(int(getattr(configs, 'cross_attn', 0) or 0))
        ca_dropout = float(getattr(configs, 'ca_dropout', 0.0) or 0.0)
        ca_heads = max(1, int(getattr(configs, 'ca_heads', 4) or 4))
        ca_layers = max(1, int(getattr(configs, 'ca_layers', 1) or 1))
        if d_patch % ca_heads or d_freq % ca_heads:
            raise ValueError(
                f'ca_heads={ca_heads} must divide both branch widths '
                f'(d_patch={d_patch}, d_freq={d_freq})')

        # real/imag -> one channel token, at the FREQ branch's width.
        self.reim_proj = nn.Linear(2 * d_freq, d_freq)

        # Built for BOTH arms, so the parameter sets are identical and the
        # ablation isolates cross-attention alone (INV3).
        self.cross_blocks = nn.ModuleList([
            UnequalBiCrossAttention(d_patch, d_freq, ca_heads, ca_dropout)
            for _ in range(ca_layers)])

        # --- fusion -----------------------------------------------------
        h = int(getattr(configs, 'dual_hidden', 128) or 128)
        dropout = float(getattr(configs, 'adapter_dropout', 0.0) or 0.0)
        patch_dim = self.top_k * configs.seq_len

        self.patch_proj = nn.Sequential(nn.Linear(patch_dim, h), nn.GELU(),
                                        nn.Dropout(dropout))
        self.chan_proj = nn.Sequential(
            nn.Linear(self.top_k * d_patch, h), nn.GELU(), nn.Dropout(dropout))
        self.freq_proj = nn.Sequential(nn.Linear(d_freq, h), nn.GELU(),
                                       nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(),
                                    nn.Linear(h, h))
        self.head = nn.Linear(h, configs.pred_len)

        # Per-branch prediction terms, same shape as DualFreDN's:
        #     Y = fused(z) + alpha * Y_patch + beta * Y_freq
        # y_freq is what keeps the FreEformer branch's own prediction head and
        # its fre_trans_*[2] projections alive; drop it and they become
        # registered-but-never-called dead weight (INV6).
        self.patch_head = nn.Linear(patch_dim, configs.pred_len)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # token assembly
    # ------------------------------------------------------------------
    def _patch_tokens(self, toks):
        """List of [B,N,d_patch,tokens_i] -> P [B, N*top_k, d_patch].

        Each period is pooled over its own token axis first, because one token
        per (channel, period) is what puts the CHANNEL on the CA token axis.
        The pooling is the price of that axis: it loses the within-period token
        resolution, which is why the fused path also reads patch_feat.

        The reshape is (N, top_k) row-major: token j belongs to channel
        j // top_k. A (top_k, N) order has the same total length and is wrong;
        INV2 is the guard.
        """
        per_period = [t.mean(dim=3) for t in toks]              # [B,N,d_patch] each
        P = torch.stack(per_period, dim=2)                      # [B,N,top_k,d_patch]
        B, N, K, D = P.shape
        return P.reshape(B, N * K, D)

    def _freq_tokens(self, x_enc):
        """x_enc [B,T,N] raw -> (y_norm [B,pred_len,N], F_chan [B,N,d_freq]).

        The single source of the frequency branch's two outputs, so the y_freq
        that ships and the F_chan the tests inspect cannot drift apart. Called
        exactly once per forward: it runs the branch.
        """
        y_norm, chan_real, chan_imag = self.branch_freq.forward_features(x_enc)
        F_chan = self.reim_proj(torch.cat([chan_real, chan_imag], dim=-1))
        return y_norm, F_chan

    def channel_tokens(self, x_norm, x_enc):
        """(P [B, N*top_k, d_patch], F [B, N, d_freq]) BEFORE cross-attention.

        Introspection hook for INV2; not used by forward(). Two inputs because
        the two branches normalise with their own RevIN instances -- x_norm is
        the dual-normalised [B,N,T] for the patch side, x_enc the raw [B,T,N]
        for the frequency side. One tensor cannot serve both.
        """
        toks = self.branch_patch.period_token_features(x_norm)
        P = self._patch_tokens(toks)
        _y_norm, F_chan = self._freq_tokens(x_enc)
        return P, F_chan

    # ------------------------------------------------------------------
    def forward(self, batch_x, batch_x_mark=None, dec_inp=None,
                batch_y_mark=None, batch_y=None):
        x = batch_x.permute(0, 2, 1)                            # [B,N,T]
        if x.shape[1] != self.branch_patch.enc_in:
            raise ValueError(
                f'expected {self.branch_patch.enc_in} channels on dim 1 of [B,N,T], '
                f'got {x.shape[1]}')
        B, N, _T = x.shape

        x_btn = self.revin_layer(x.permute(0, 2, 1), 'norm')    # [B,T,N]
        heatmap = None if self.fsdh is None else self.fsdh(x_btn)
        x_norm = x_btn.permute(0, 2, 1)                         # [B,N,T]

        # --- patch side -------------------------------------------------
        toks = self.branch_patch.period_token_features(x_norm)  # [B,N,d,L_i] each
        # From the ORIGINAL tokens, never from the CA output: this is the
        # branch's own detail, and the CA-updated summary lives in P_hat.
        patch_feat = self.branch_patch.project_period_tokens(toks)
        P = self._patch_tokens(toks)                            # [B, N*top_k, d]

        # --- frequency side (one call; it runs the branch) --------------
        y_freq_norm, F_chan = self._freq_tokens(batch_x)        # [B,pred_len,N], [B,N,d]

        # --- cross-attention, over the CHANNEL axis ---------------------
        packed_p = P.permute(1, 0, 2)                           # [N*top_k, B, d]
        packed_f = F_chan.permute(1, 0, 2)                      # [N, B, d]
        if self.use_cross_attn:
            for blk in self.cross_blocks:
                packed_p, packed_f = blk(packed_p, packed_f)
        P_hat = packed_p.permute(1, 0, 2)                       # [B, N*top_k, d]
        F_hat = packed_f.permute(1, 0, 2)                       # [B, N, d]

        # --- fuse + predict ---------------------------------------------
        h_p = self.patch_proj(patch_feat)
        # (N, top_k) row-major again, matching _patch_tokens.
        h_c = self.chan_proj(P_hat.reshape(B, N, self.top_k * P_hat.shape[-1]))
        h_f = self.freq_proj(F_hat)
        z = self.fusion(torch.cat([h_p, h_c, h_f], dim=-1))
        y_fused = self.head(z)                                  # [B,N,pred_len]

        y_patch = self.patch_head(patch_feat)                   # [B,N,pred_len]
        y_freq = y_freq_norm.permute(0, 2, 1)                   # [B,N,pred_len]

        pred = y_fused + self.alpha * y_patch + self.beta * y_freq

        combined = pred if heatmap is None else pred + heatmap
        combined = combined.permute(0, 2, 1)                    # [B,pred_len,N]
        return self.revin_layer(combined, 'denorm')

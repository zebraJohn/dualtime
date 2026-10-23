"""Dual-Times2D: two Times2D branches over different temporal organisations.

Experiment 2 of the plan -- are the patch view and the phase view complementary?

    x [B,N,T]
       |
    RevIN norm (once)
       |
       +--------------+--------------+
       v              v              v
    Branch A       Branch B        FSDH (once)
    patch periods  phase periods   -> heatmap [B,N,pred_len]
       |              |
    F_patch        F_phase
    [B,N,topk*T]   [B,N,topk*T]
       +------+-------+
              v
        concat [B,N,2*topk*T]
              v
        Head: Linear -> pred_len     (shared)
              v
          z + heatmap
              v
        RevIN denorm (same instance)
              v
        Y_hat [B,N,pred_len]

Branches differ ONLY in `period_list`/`patch_len`: the fold width determines
what counts as one cycle, and patch_len the phase-bin width within it. Both
branches are capacity-matched (~10% apart).

FSDH is computed once and shared. It derives from the same RevIN-normalised
input for both branches, so a per-branch copy would be numerically identical
and purely redundant. Both branches are built with build_head=False, since the
shared head replaces theirs (keeping them would leave ~3M dead parameters per
branch in the optimiser).
"""
__all__ = ['DualTimes2D']

import torch
from torch import nn

from layers.RevIN import RevIN
from models.Times2D import FSDH, Head, Times2DBackbone


class BidirectionalCrossAttention(nn.Module):
    """One bidirectional cross-attention block between the two branches.

    Follows Dual-Times2D.md section 14:

        F_patch' = Attn(Q=F_patch, K=F_phase, V=F_phase)
        F_phase' = Attn(Q=F_phase, K=F_patch, V=F_patch)

    i.e. the patch branch is enriched with phase information and vice versa.
    Both directions read the *pre-block* features, so they are independent
    rather than sequential. Pre-norm residual connections.

    Works on token sequences, not the collapsed [B,N,top_k*seq_len] features --
    see Times2DBackbone.period_token_features for why.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.norm_patch = nn.LayerNorm(d_model)
        self.norm_phase = nn.LayerNorm(d_model)
        # Q=patch, KV=phase
        self.patch_queries_phase = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # Q=phase, KV=patch
        self.phase_queries_patch = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, f_patch, f_phase):
        """f_patch: [Lp, BN, D], f_phase: [Lq, BN, D] (seq-first, Lp != Lq allowed)."""
        p_norm = self.norm_patch(f_patch)
        q_norm = self.norm_phase(f_phase)

        upd_patch, _ = self.patch_queries_phase(p_norm, q_norm, q_norm)
        f_patch = f_patch + self.drop(upd_patch)

        upd_phase, _ = self.phase_queries_patch(q_norm, p_norm, p_norm)
        f_phase = f_phase + self.drop(upd_phase)
        return f_patch, f_phase


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        # NOTE: keep this identical to PhaseTimes2D / the Times2D baseline so
        # that a metric change can only come from the architecture, not from a
        # different FSDH parameterisation. See PhaseTimes2D for the rationale.
        fsdh_kwargs = dict(per_sample_weights=True)

        self.branch_patch = Times2DBackbone(
            configs,
            use_fsdh=False,
            build_head=False,
            **fsdh_kwargs,
        )
        self.branch_phase = Times2DBackbone(
            configs,
            period_list=configs.phase_period_list,
            patch_len=configs.phase_patch_len,
            use_fsdh=False,
            build_head=False,
            **fsdh_kwargs,
        )

        # One RevIN for both branches, and for both norm and denorm -- RevIN
        # caches mean/stdev on the instance, so denorm must use the module that
        # normalised.
        self.revin_layer = RevIN(configs.enc_in, affine=configs.affine,
                                 subtract_last=configs.subtract_last)
        self.fsdh = FSDH(configs, **fsdh_kwargs) if not configs.no_fsdh else None

        # Fusion head. Passing a 2-element list makes Head concat along the last
        # dim, i.e. Linear((topk_a + topk_b) * seq_len -> pred_len).
        fusion_dim = int(getattr(configs, 'fusion_dim', 0) or 0)
        if fusion_dim > 0:
            # Bottleneck fusion. The raw concat head is the single largest block
            # in the model -- at seq_len=720/pred_len=720 it is
            # Linear(7200 -> 720) = 5.18M params, twice the baseline's head --
            # and is the main suspect for the long-horizon overfitting.
            self.fusion_patch = nn.Sequential(
                nn.Linear(self.branch_patch.top_k * configs.seq_len, fusion_dim), nn.GELU())
            self.fusion_phase = nn.Sequential(
                nn.Linear(self.branch_phase.top_k * configs.seq_len, fusion_dim), nn.GELU())
            self.head = Head(fusion_dim, 2, configs.pred_len,
                             head_dropout=configs.head_dropout, Concat=True)
        else:
            self.fusion_patch = None
            self.fusion_phase = None
            n_periods = self.branch_patch.top_k + self.branch_phase.top_k
            self.head = Head(configs.seq_len, n_periods, configs.pred_len,
                             head_dropout=configs.head_dropout, Concat=True)

        # Stage 2: bidirectional cross-attention over the branches' token
        # sequences. Opt-in -- with cross_attn=0 this model is exactly the
        # stage-1 Dual, so the baseline stays reproducible.
        self.use_cross_attn = bool(int(getattr(configs, 'cross_attn', 0) or 0))
        if self.use_cross_attn:
            ca_layers = max(1, int(getattr(configs, 'ca_layers', 1) or 1))
            ca_heads = max(1, int(getattr(configs, 'ca_heads', 4) or 4))
            ca_dropout = float(getattr(configs, 'ca_dropout', 0.0) or 0.0)
            d_model = self.branch_patch.d_model
            if d_model % ca_heads != 0:
                raise ValueError(f'ca_heads={ca_heads} must divide d_model={d_model}')
            self.cross_blocks = nn.ModuleList([
                BidirectionalCrossAttention(d_model, ca_heads, ca_dropout)
                for _ in range(ca_layers)])
        else:
            self.cross_blocks = None

    def _cross_attend(self, x_norm):
        """Run the bidirectional cross-attention blocks, then back-project.

        Attention happens at the token level (patch: 66 tokens, phase: 60, both
        d_model wide) because the concatenated [B,N,top_k*seq_len] features have
        already collapsed the token axis. Afterwards each branch's tokens go
        through its own existing per-period Flatten+Linear, so this adds no
        parameters beyond the attention blocks themselves.
        """
        toks_patch = self.branch_patch.period_token_features(x_norm)   # [B,N,D,L_i] each
        toks_phase = self.branch_phase.period_token_features(x_norm)

        packed_p, B, N, D = self._pack(toks_patch)
        packed_q, _, _, _ = self._pack(toks_phase)

        for block in self.cross_blocks:
            packed_p, packed_q = block(packed_p, packed_q)

        toks_patch = self._unpack(packed_p, self.branch_patch.tokens_list, B, N, D)
        toks_phase = self._unpack(packed_q, self.branch_phase.tokens_list, B, N, D)

        f_patch = self.branch_patch.project_period_tokens(toks_patch)
        f_phase = self.branch_phase.project_period_tokens(toks_phase)
        return f_patch, f_phase

    @staticmethod
    def _pack(token_list):
        """[B,N,D,L_i] per period -> ([L_total, B*N, D], B, N, D) for nn.MultiheadAttention."""
        cat = torch.cat(token_list, dim=3)              # [B, N, D, L]
        B, N, D, L = cat.shape
        return cat.permute(3, 0, 1, 2).reshape(L, B * N, D), B, N, D

    @staticmethod
    def _unpack(attn_out, tokens_list, B, N, D):
        """[L_total, B*N, D] -> list of [B,N,D,L_i], splitting back per period."""
        L = attn_out.shape[0]
        cat = attn_out.reshape(L, B, N, D).permute(1, 2, 3, 0)   # [B, N, D, L]
        return list(torch.split(cat, tokens_list, dim=3))

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        x = batch_x.permute(0, 2, 1)                                  # [B, N, T]

        if x.shape[1] != self.branch_patch.enc_in:
            raise ValueError(
                f"expected {self.branch_patch.enc_in} channels on dim 1 of [B, N, T], "
                f"got {x.shape[1]}. RevIN/FSDH internally use [B, T, N]."
            )

        x_btn = self.revin_layer(x.permute(0, 2, 1), 'norm')          # [B, T, N]
        heatmap = None if self.fsdh is None else self.fsdh(x_btn)     # [B, N, pred_len]
        x_norm = x_btn.permute(0, 2, 1)                               # [B, N, T]

        if self.cross_blocks is None:
            f_patch = self.branch_patch.feature_map(x_norm)           # [B, N, topk_a*T]
            f_phase = self.branch_phase.feature_map(x_norm)           # [B, N, topk_b*T]
        else:
            f_patch, f_phase = self._cross_attend(x_norm)

        if self.fusion_patch is not None:
            f_patch = self.fusion_patch(f_patch)                      # [B, N, fusion_dim]
            f_phase = self.fusion_phase(f_phase)                      # [B, N, fusion_dim]

        z = self.head([f_patch, f_phase])                             # [B, N, pred_len]
        combined = z if heatmap is None else z + heatmap

        combined = combined.permute(0, 2, 1)                          # [B, pred_len, N]
        combined = self.revin_layer(combined, 'denorm')
        # RevIN denorm needs [B, T, N]; the exp pipeline wants [B, pred_len, N].
        return combined

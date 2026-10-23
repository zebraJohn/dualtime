"""Dual-branch Patch + Phase model with bidirectional cross-attention.

The minimal experiment that answers the one question no probe could:
does letting the two branches attend to each other's FEATURES create
information that neither branch had alone?

    x [B,T,N]
       |
    RevIN norm (patch branch)      PhaseFormer's own RevIN (phase branch)
       |                                       |
    Times2D token features          PhaseFormer latent Z [B,C,L,latent]
    [B*N, 66, d_model]                          |
       |                              FeatureAdapter(latent -> d_model)
       |                                       |
       |                              [B*C, 24, d_model]
       +------------------+--------------------+
                          |
           BidirectionalCrossAttention  (optional, cross_attn=1)
                          |
       +------------------+--------------------+
       |                                       |
    project_period_tokens              flatten tokens
    [B,N,top_k*seq_len]                [B,N,L*d_model]
       |                                       |
    patch_proj(->h)                    phase_proj(->h)
       +------------------+--------------------+
                          |
                  concat -> fusion MLP -> head
                          |
                     + FSDH heatmap
                          |
                    RevIN denorm -> [B,pred_len,N]

WHY THIS SHAPE
--------------
* CA runs at the TOKEN level, not on the collapsed [B,N,top_k*seq_len] feature.
  That concatenated feature has already flattened the token axis, so attending
  over it would be attending over channels -- see Times2DBackbone.period_token_features.
* cross_attn=0 must be a plain feature-concat dual, so that CA is the ONLY
  difference between the two runs. That is what makes the ablation mean
  something; tests/test_dual_phaseformer_invariants.py enforces it.
* No MoE, no Memory Bank, and no per-branch prediction terms. Each of those
  would add a second way for the model to improve, and then a win could not be
  attributed to CA. Add them later, on top of a CA result that stands alone.

KNOWN LIMITATION -- read before interpreting a null result
----------------------------------------------------------
The phase branch enters CA through FeatureAdapter(latent_dim -> d_model). At
latent_dim=6 that is a rank-<=6 map into a 64-wide space, so the attention's
queries and keys live in a 6-dimensional subspace and cannot express much. This
is a structural limit, not a tuning one.

The plan (phase2d-v2.md section 8.2) therefore raises latent_dim to 64 first,
which makes the adapter an identity and removes the bottleneck. A null result
at latent_dim=6 says almost nothing; a null result at latent_dim=64 is evidence.
"""
__all__ = ['DualPhaseFormer']

import torch
import torch.nn.functional as F
from torch import nn

from layers.RevIN import RevIN
from models.Times2D import Times2DBackbone, FSDH
from models.PhaseFormer import Model as PhaseFormerBackbone


class BidirectionalCrossAttention(nn.Module):
    """One bidirectional block between the two token sequences.

    Follows Dual-Times2D.md section 14:

        F_patch' = Attn(Q=F_patch, K=F_phase, V=F_phase)
        F_phase' = Attn(Q=F_phase, K=F_patch, V=F_patch)

    Both directions read the *pre-block* features, so they are independent
    rather than sequential. Pre-norm residual connections. L_patch (66) and
    L_phase (24) need not match -- nn.MultiheadAttention handles that.

    Lifted unchanged from models/DualTimes2D.py, where it is already covered by
    a smoke test.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.norm_patch = nn.LayerNorm(d_model)
        self.norm_phase = nn.LayerNorm(d_model)
        self.patch_queries_phase = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.phase_queries_patch = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, f_patch, f_phase):
        """f_patch: [Lp, BN, D], f_phase: [Lq, BN, D] (seq-first)."""
        p_norm = self.norm_patch(f_patch)
        q_norm = self.norm_phase(f_phase)

        upd_patch, _ = self.patch_queries_phase(p_norm, q_norm, q_norm)
        f_patch = f_patch + self.drop(upd_patch)

        upd_phase, _ = self.phase_queries_patch(q_norm, p_norm, p_norm)
        f_phase = f_phase + self.drop(upd_phase)
        return f_patch, f_phase


class FeatureAdapter(nn.Module):
    """Project one branch's tokens to the shared CA width.

    Identity when the widths already match, so a model configured with
    latent_dim == d_model pays nothing for it.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        if in_dim == out_dim:
            self.net = nn.Identity()
        else:
            mid = max(out_dim, 2 * in_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, mid), nn.GELU(), nn.LayerNorm(mid),
                nn.Linear(mid, out_dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        d_model = configs.d_model

        # --- branches ---------------------------------------------------
        # Patch branch carries no head and no FSDH of its own: the shared head
        # below replaces the former (keeping it would leave dead parameters),
        # and a single FSDH instance is shared by both paths.
        self.branch_patch = Times2DBackbone(
            configs, use_fsdh=False, build_head=False, per_sample_weights=True)
        self.branch_phase = PhaseFormerBackbone(configs)

        self.revin_layer = RevIN(configs.enc_in, affine=configs.affine,
                                 subtract_last=configs.subtract_last)
        self.fsdh = None if configs.no_fsdh else FSDH(configs)

        # --- cross-attention --------------------------------------------
        self.use_cross_attn = bool(int(getattr(configs, 'cross_attn', 0) or 0))
        if self.use_cross_attn:
            ca_heads = max(1, int(getattr(configs, 'ca_heads', 4) or 4))
            ca_layers = max(1, int(getattr(configs, 'ca_layers', 1) or 1))
            ca_dropout = float(getattr(configs, 'ca_dropout', 0.0) or 0.0)
            if d_model % ca_heads != 0:
                raise ValueError(f'ca_heads={ca_heads} must divide d_model={d_model}')
            self.phase_adapter = FeatureAdapter(
                self.branch_phase.latent_dim, d_model,
                float(getattr(configs, 'adapter_dropout', 0.0) or 0.0))
            self.cross_blocks = nn.ModuleList([
                BidirectionalCrossAttention(d_model, ca_heads, ca_dropout)
                for _ in range(ca_layers)])
        else:
            # Built even when CA is off, so the parameter set is identical
            # between the two runs and the ablation isolates CA alone.
            self.phase_adapter = FeatureAdapter(
                self.branch_phase.latent_dim, d_model,
                float(getattr(configs, 'adapter_dropout', 0.0) or 0.0))
            self.cross_blocks = None

        # --- fusion -----------------------------------------------------
        h = int(getattr(configs, 'dual_hidden', 128) or 128)
        dropout = float(getattr(configs, 'adapter_dropout', 0.0) or 0.0)
        patch_dim = self.branch_patch.top_k * configs.seq_len
        phase_dim = self.branch_phase.period_len * d_model
        self.patch_proj = nn.Sequential(nn.Linear(patch_dim, h), nn.GELU(), nn.Dropout(dropout))
        self.phase_proj = nn.Sequential(nn.Linear(phase_dim, h), nn.GELU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(),
                                    nn.Linear(h, h))
        self.head = nn.Linear(h, configs.pred_len)

        # Per-branch prediction terms (phase2d-v2.md section 10.2):
        #     Y = decoder(z) + alpha * Y_patch + beta * Y_phase
        #
        # Two jobs. It gives each branch a direct path to the loss, which the
        # feature-only path does not; and if the fused path learns nothing, the
        # optimiser can push alpha/beta up and fall back to prediction-level
        # fusion rather than collapsing to a single branch.
        #
        # It does NOT muddy the CA ablation: both arms (cross_attn=0 and 1) have
        # these terms, so their difference still isolates cross-attention. What
        # it does mean is that "dual beats a single branch" answers a different
        # question than "cross-attention helps" -- report them separately.
        #
        # It also keeps PhaseFormer's predictor alive: without this, the
        # predictor's parameters are registered but never called, i.e. dead
        # weight that never trains (caught by
        # tests/test_dual_phaseformer_invariants.py INV5).
        self.patch_head = nn.Linear(patch_dim, configs.pred_len)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # packing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pack(token_list):
        """Times2D tokens: [B,N,D,L_i] per period -> ([L_total, B*N, D], B, N, D)."""
        cat = torch.cat(token_list, dim=3)              # [B,N,D,L]
        B, N, D, L = cat.shape
        return cat.permute(3, 0, 1, 2).reshape(L, B * N, D), B, N, D

    @staticmethod
    def _unpack(attn_out, tokens_list, B, N, D):
        """[L_total, B*N, D] -> list of [B,N,D,L_i], split back per period."""
        L = attn_out.shape[0]
        cat = attn_out.reshape(L, B, N, D).permute(1, 2, 3, 0)   # [B,N,D,L]
        return list(torch.split(cat, tokens_list, dim=3))

    @staticmethod
    def _pack_phase(Z):
        """PhaseFormer latent: [B,C,L,D] -> ([L, B*C, D], B, C, D).

        Note the axis order differs from the Times2D side: PhaseFormer's Z is
        [B,C,L,D] while Times2D's tokens are [B,N,D,L]. Packing without the
        permute would silently transpose the token axis.
        """
        B, C, L, D = Z.shape
        return Z.permute(2, 0, 1, 3).reshape(L, B * C, D), B, C, D

    @staticmethod
    def _unpack_phase(packed, B, C, D):
        """[L, B*C, D] -> [B,C,L,D]."""
        L = packed.shape[0]
        return packed.reshape(L, B, C, D).permute(1, 2, 0, 3)

    # ------------------------------------------------------------------
    # phase branch, in normalised space
    # ------------------------------------------------------------------
    def phase_latent(self, x_enc):
        """x_enc: [B,T,N] raw -> PhaseFormer latent Z [B,C,L,latent_dim].

        Deliberately written against the phase backbone's OWN submodules rather
        than reimplementing them, so the arithmetic is identical to the
        standalone model. Neither normalise nor denormalise happens here -- the
        phase branch keeps its own RevIN, and the dual model denormalises once
        at the very end.
        """
        m = self.branch_phase
        if m.use_revin:
            x_in, _ = m.revin.normalize(x_enc)
        else:
            x_in = x_enc.float()
        x = x_in.permute(0, 2, 1)                       # [B,C,T]
        B, C, _ = x.shape
        if m.pad_seq_len > 0:
            x = F.pad(x, (0, m.pad_seq_len), mode='circular')
        x_periods = x.view(B, C, m.num_periods_input, m.period_len)
        phase_series = m._to_phase_series(x_periods)

        Z = m.embedding(phase_series)
        cur = phase_series
        for i, unit in enumerate(m.routing_layers):
            Z, y_steps = unit(cur, Z)
            if i < len(m.routing_layers) - 1:
                cur = y_steps
        return Z

    def phase_predict_normalised(self, x_enc):
        """[B,T,N] raw -> y_hat [B,pred_len,N] in NORMALISED space.

        The dual model needs PhaseFormer's prediction path applied to a latent
        that CA may have modified, so it cannot simply call the backbone's
        forward. This reproduces steps 9-10 of that forward minus the
        denormalisation; tests/test_dual_phaseformer_invariants.py asserts the
        result matches the standalone model bitwise once denormalised.
        """
        return self._phase_head(self.phase_latent(x_enc))

    def _phase_head(self, Z):
        m = self.branch_phase
        y_steps = m.predictor(Z)                                    # [B,C,L,P_out]
        y_periods = m._from_phase_steps_to_periods(y_steps)         # [B,C,P_out,L]
        B, C = y_periods.shape[0], y_periods.shape[1]
        y_full = y_periods.reshape(B, C, -1)[..., :m.pred_len]      # [B,C,pred_len]
        return y_full.permute(0, 2, 1)                              # [B,pred_len,C]

    # ------------------------------------------------------------------
    def forward(self, batch_x, batch_x_mark=None, dec_inp=None,
                batch_y_mark=None, batch_y=None):
        x = batch_x.permute(0, 2, 1)                                # [B,N,T]
        if x.shape[1] != self.branch_patch.enc_in:
            raise ValueError(
                f'expected {self.branch_patch.enc_in} channels on dim 1 of [B,N,T], '
                f'got {x.shape[1]}')

        x_btn = self.revin_layer(x.permute(0, 2, 1), 'norm')        # [B,T,N]
        heatmap = None if self.fsdh is None else self.fsdh(x_btn)   # [B,N,pred_len]
        x_norm = x_btn.permute(0, 2, 1)                             # [B,N,T]

        # --- patch tokens ---
        toks = self.branch_patch.period_token_features(x_norm)      # [B,N,d,L_i] each
        packed_p, Bp, Np, Dp = self._pack(toks)                     # [66, B*N, d]

        # --- phase tokens ---
        Z = self.phase_latent(batch_x)                              # [B,C,L,latent]
        Q = self.phase_adapter(Z)                                   # [B,C,L,d]
        packed_q, Bq, Cq, Dq = self._pack_phase(Q)                  # [24, B*C, d]

        if self.cross_blocks is not None:
            for blk in self.cross_blocks:
                packed_p, packed_q = blk(packed_p, packed_q)

        toks = self._unpack(packed_p, self.branch_patch.tokens_list, Bp, Np, Dp)
        patch_feat = self.branch_patch.project_period_tokens(toks)  # [B,N,topk*T]

        Q = self._unpack_phase(packed_q, Bq, Cq, Dq)                # [B,C,L,d]
        phase_feat = Q.reshape(Q.shape[0], Q.shape[1], -1)          # [B,C,L*d]

        # --- fuse + predict ---
        h_p = self.patch_proj(patch_feat)
        h_q = self.phase_proj(phase_feat)
        z = self.fusion(torch.cat([h_p, h_q], dim=-1))
        y_fused = self.head(z)                                      # [B,N,pred_len]

        # Per-branch predictions, in the same normalised space. y_phase comes
        # from PhaseFormer's own predictor applied to the UNMODIFIED latent, so
        # it is genuinely that branch's output; cross-attention acts on the
        # fused path only, which is exactly what we want to isolate.
        y_patch = self.patch_head(patch_feat)                       # [B,N,pred_len]
        y_phase = self._phase_head(Z).permute(0, 2, 1)              # [B,N,pred_len]

        pred = y_fused + self.alpha * y_patch + self.beta * y_phase

        combined = pred if heatmap is None else pred + heatmap
        combined = combined.permute(0, 2, 1)                        # [B,pred_len,N]
        combined = self.revin_layer(combined, 'denorm')
        return combined

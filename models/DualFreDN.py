"""Dual-branch Patch + Frequency model with bidirectional cross-attention.

The same experiment as models/DualPhaseFormer.py, with the frequency branch
swapped from PhaseFormer to FreDN to answer one question the PhaseFormer run
could not: was cross-attention useless because the MECHANISM is useless, or
because PhaseFormer was not a different enough second branch (rho was 0.9377)?

    x [B,T,N]
       |
    RevIN norm (patch branch)          FreDN's own RevIN (freq branch)
       |                                          |
    Times2D token features              spectrum X [B,N,D,F], F=seq_len//2+1
    [B*N, 66, d_model]                            |
       |                              ReIm concat -> [B,N,2D,F]
       |                                          |
       |                              _pack_freq -> [F, B*N, 2D]
       |                                          |
       |                              FeatureAdapter (Identity when 2D==d_model)
       |                                          |
       +------------------+-----------------------+
                          |
           BidirectionalCrossAttention  (optional, cross_attn=1)
                          |
       +------------------+-----------------------+
       |                                          |
    project_period_tokens                 _unpack_freq -> flatten
    [B,N,top_k*seq_len]                        [B,N,2D*F]
       |                                          |
    patch_proj(->h)                            freq_proj(->h)
       +------------------+-----------------------+
                          |
                  concat -> fusion MLP -> head
                          |
                     + FSDH heatmap
                          |
                    RevIN denorm -> [B,pred_len,N]

WHAT "FULL FreDN" MEANS HERE -- read before interpreting a result
-----------------------------------------------------------------
The alpha/beta path uses ALL of FreDN (season spectrum + trend MLP). The FUSED
path sees only the SEASON SPECTRUM, because the frequency axis is FreDN's only
axis with sequence semantics for attention; `trend` is [B,T,N,D] and its only
sequence axis is the raw input itself. That is a deliberate trade, not an
oversight: flattening trend into the fused feature would add a T*D = 23040-wide
term with no token structure to attend over.

So: freq_feat is spectrum-only; y_freq is full FreDN. Do not conflate them.

WHY y_freq MUST STAY IN THE alpha/beta SUM
------------------------------------------
_from_spectrum's res_proj/proj_layers/pred_layers and _trend_forward receive
gradient ONLY through y_freq -- they do not appear in freq_feat. Drop y_freq and
they become registered-but-never-called dead weight, which trains fine and
reports fine while FreDN's prediction head is never trained. INV5 asserts
against it.

KNOWN LIMITATION -- the rank bottleneck
---------------------------------------
The frequency branch enters CA through FeatureAdapter(2*fre_embed_size, d_model).
If 2*fre_embed_size != d_model that is a low-rank map, and the attention's
queries and keys live in a subspace too small to express much -- the same
structural limit scripts/run_ca_experiment.sh:16-22 warns about for PhaseFormer.
Configs must therefore satisfy 2*fre_embed_size == d_model. INV6 pins this.
"""
__all__ = ['Model']

import torch
import torch.nn.functional as F
from torch import nn

from layers.RevIN import RevIN
from models.Times2D import Times2DBackbone, FSDH
from models.FreDN import Model as FreDNBackbone
from models.DualPhaseFormer import BidirectionalCrossAttention, FeatureAdapter


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        d_model = configs.d_model
        embed_size = int(getattr(configs, 'fre_embed_size', 32))

        # Fail here rather than three hundred lines later. The frequency branch
        # enters CA through FeatureAdapter(2*embed_size, d_model); if the two
        # differ, the CA queries and keys live in a low-rank subspace -- the
        # structural limit scripts/run_ca_experiment.sh:16-22 records for
        # PhaseFormer, and the reason this model is configured at
        # fre_embed_size == d_model/2.
        #
        # This is NOT a cosmetic check. It is also not a case of "the model
        # trains but means less": forward() unpacks CA's output with the
        # PRE-adapter width returned by _pack_freq while Q is POST-adapter, so a
        # mismatch dies in a reshape with an opaque size error, after the data
        # has loaded. See INV6.
        if 2 * embed_size != d_model:
            raise ValueError(
                f'2*fre_embed_size ({2 * embed_size}) must equal d_model '
                f'({d_model}), got fre_embed_size={embed_size}. Anything else '
                f'makes FeatureAdapter a low-rank map into the CA space, which '
                f'is the structural limit that made the PhaseFormer '
                f'cross-attention result uninterpretable.')

        # --- branches ---------------------------------------------------
        # Same flags as DualPhaseFormer: the patch branch carries no head and no
        # FSDH of its own (the shared head replaces the former, and a single
        # FSDH instance is shared by both paths).
        self.branch_patch = Times2DBackbone(
            configs, use_fsdh=False, build_head=False, per_sample_weights=True)
        self.branch_freq = FreDNBackbone(configs)

        self.revin_layer = RevIN(configs.enc_in, affine=configs.affine,
                                 subtract_last=configs.subtract_last)
        self.fsdh = None if configs.no_fsdh else FSDH(configs)

        # --- cross-attention --------------------------------------------
        # A 2*embed_size-wide token is the same rank limit PhaseFormer hit at
        # latent_dim=6; see scripts/run_ca_experiment.sh's header.
        self.ca_in_dim = 2 * embed_size
        self.use_cross_attn = bool(int(getattr(configs, 'cross_attn', 0) or 0))
        ca_dropout = float(getattr(configs, 'ca_dropout', 0.0) or 0.0)
        if self.use_cross_attn:
            ca_heads = max(1, int(getattr(configs, 'ca_heads', 4) or 4))
            ca_layers = max(1, int(getattr(configs, 'ca_layers', 1) or 1))
            if d_model % ca_heads != 0:
                raise ValueError(f'ca_heads={ca_heads} must divide d_model={d_model}')

        # Built for BOTH arms, so the parameter sets are identical and the
        # ablation isolates cross-attention alone (INV4).
        self.freq_adapter = FeatureAdapter(
            self.ca_in_dim, d_model,
            float(getattr(configs, 'adapter_dropout', 0.0) or 0.0))
        self.cross_blocks = None
        if self.use_cross_attn:
            self.cross_blocks = nn.ModuleList([
                BidirectionalCrossAttention(d_model, ca_heads, ca_dropout)
                for _ in range(ca_layers)])

        # --- fusion -----------------------------------------------------
        h = int(getattr(configs, 'dual_hidden', 128) or 128)
        dropout = float(getattr(configs, 'adapter_dropout', 0.0) or 0.0)
        patch_dim = self.branch_patch.top_k * configs.seq_len
        # spectrum-only: [B,N,2*embed_size,F] flattened
        self.freq_dim = self.ca_in_dim * (configs.seq_len // 2 + 1)
        self.patch_proj = nn.Sequential(nn.Linear(patch_dim, h), nn.GELU(), nn.Dropout(dropout))
        self.freq_proj = nn.Sequential(nn.Linear(self.freq_dim, h), nn.GELU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(),
                                    nn.Linear(h, h))
        self.head = nn.Linear(h, configs.pred_len)

        # Per-branch prediction terms, same shape as DualPhaseFormer's:
        #     Y = decoder(z) + alpha * Y_patch + beta * Y_freq
        # Keeps each branch's own prediction path alive (see the module
        # docstring: without y_freq, FreDN's prediction head never trains).
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
    def _pack_freq(S):
        """FreDN spectrum: [B,N,E,F] -> ([F, B*N, E], B, N, E).

        THREE different axis conventions live in this repo, and FreDN's is the
        one that contradicts Times2D's:

            Times2D      [B,N,D,L]  -> [L,   B*N, D]   token axis THIRD
            PhaseFormer  [B,C,L,D]  -> [L,   B*C, D]   token axis THIRD
            FreDN        [B,N,D,F]  -> [F,   B*N, E]   token axis FOURTH

        FreDN's token axis is LAST, i.e. the opposite of Times2D's. Copying
        Times2D's permute here transposes the frequency bins against the
        embedding dim silently: shapes all still line up. INV2 is the guard.
        """
        B, N, E, F = S.shape
        return S.permute(3, 0, 1, 2).reshape(F, B * N, E), B, N, E

    @staticmethod
    def _unpack_freq(packed, B, N, E):
        """[F, B*N, E] -> [B,N,E,F]."""
        F = packed.shape[0]
        return packed.reshape(F, B, N, E).permute(1, 2, 3, 0)

    # ------------------------------------------------------------------
    # frequency branch, in normalised space
    # ------------------------------------------------------------------
    def freq_branch_forward(self, x_enc):
        """x_enc: [B,T,N] raw -> (X [B,N,D,F] complex, trend [B,T,N,D],
        y_norm [B,pred_len,N]).

        Driven through the frequency backbone's OWN submodules rather than
        reimplemented, so the arithmetic is identical to the standalone model.
        The branch keeps its own RevIN; neither normalise nor denormalise is
        duplicated here -- the dual denormalises once at the very end.

        NOTE the return shape: y_norm is [B,pred_len,N], channel LAST, whereas
        patch_head and freq_feat are [B,N,...]. The call site must permute.
        """
        m = self.branch_freq
        x_norm = m.revin_layer(x_enc, 'norm')
        season, trend = m.decomp(m._embed(x_norm))
        X = m._to_spectrum(season)
        y_norm = m._from_spectrum(X) + m._trend_forward(trend)
        return X, trend, y_norm

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

        # --- frequency tokens ---
        X, _trend, y_freq_norm = self.freq_branch_forward(batch_x)  # X [B,N,D,F]
        S = torch.cat([X.real, X.imag], dim=2)                      # [B,N,E,F]
        packed_f, Bf, Nf, Ef = self._pack_freq(S)                   # [F, B*N, E]
        Q = self.freq_adapter(packed_f)                             # Identity at 2D==d_model

        if self.cross_blocks is not None:
            for blk in self.cross_blocks:
                packed_p, Q = blk(packed_p, Q)

        toks = self._unpack(packed_p, self.branch_patch.tokens_list, Bp, Np, Dp)
        patch_feat = self.branch_patch.project_period_tokens(toks)  # [B,N,topk*T]

        Q = self._unpack_freq(Q, Bf, Nf, Ef)                        # [B,N,E,F]
        freq_feat = Q.reshape(Q.shape[0], Q.shape[1], -1)           # [B,N,E*F]

        # --- fuse + predict ---
        h_p = self.patch_proj(patch_feat)
        h_q = self.freq_proj(freq_feat)
        z = self.fusion(torch.cat([h_p, h_q], dim=-1))
        y_fused = self.head(z)                                      # [B,N,pred_len]

        # Per-branch predictions, in the same normalised space. y_freq comes
        # from the UNMODIFIED spectrum, so it is genuinely that branch's output;
        # cross-attention acts on the fused path only. The permute is required:
        # freq_branch_forward returns [B,pred_len,N].
        y_patch = self.patch_head(patch_feat)                       # [B,N,pred_len]
        y_freq = y_freq_norm.permute(0, 2, 1)                       # [B,N,pred_len]

        pred = y_fused + self.alpha * y_patch + self.beta * y_freq

        combined = pred if heatmap is None else pred + heatmap
        combined = combined.permute(0, 2, 1)                        # [B,pred_len,N]
        combined = self.revin_layer(combined, 'denorm')
        return combined

"""FreEformer (AAAI 2026), ported from D:\\pycharm\\FreEformer.

Ported from `model/FrePatchTST3_attn_ablation.py` -- the variant upstream's own
scripts/Weather_freformer_attn1.sh:5 names as `model_name`, i.e. the paper's
main model, not one of the architecture ablations.

    x [B,T,N] -> RevIN -> tokenEmb -> [B,N,T,D]        D = embed_size
              -> rfft over T -> [B,N,D,F]              F = valid_fre_points
              -> flatten(-2) -> [B,N,D*F]
              -> fre_trans_{real,imag}: Linear(D*F -> d_model)
                                        -> Encoder_ori  (attention over N)
                                        -> Linear(d_model -> D*F)
              -> reshape -> [B,N,D,F] -> irfft -> [B,N,T,D]
              -> + residual -> fc(flatten(-2)) -> [B,pred_len,N]

Note the frequency axis does NOT survive the flatten(-2): the attention runs on
a per-channel vector in which (D,F) has already been collapsed. That is why
the tap this module exposes sits BEFORE the Encoder (see `_fre_trans_features`).

CONTROLLED CHANGES -- one, not three
------------------------------------
The spec predicted three (RevIN signature, a split of the frequency path, a
feature tap). Two turned out to be non-changes:

  1. RevIN needs NOTHING. layers/RevIN.py and upstream's RevIN have identical
     __init__ signatures and the same positional `forward(x, mode)`; upstream
     just carries an unused `mask=None`.
  2. The tap needs NO structural change. Upstream's `fre_trans_real` is already
     `nn.Sequential(Linear, Encoder_ori, Linear)`, so the pre-Encoder tensor is
     reachable by indexing [0] / [1] / [2]. Unrolling the Sequential into
     separate members -- which the implementation plan initially proposed --
     would have been WRONG: it moves state_dict keys from
     `fre_trans_real.0.weight` to a new name, breaking the checkpoint contract
     it claimed to preserve. The Sequential stays.

So the only real change is the new forward_features entry point and the tap
method behind it, both of which are reads.

V2 ADDS ONE MORE: PER-BRANCH WIDTH
----------------------------------
v1 put this branch into the Times2D protocol (d_model=64, n_heads=16,
dropout=0.5) because both branches read the same `configs` attribute, and that
made a standalone FreEformer score 0.727 against Times2D's 0.372 -- while the
same weights at d_model=512 reproduce upstream's 0.47068 exactly (PROGRESS
12.10). The 0.727 was the configuration, not the model.

So v2 reads five `fe_*` fields through `_cfg()`, each falling back to the
shared field when unset. Unset means v1, bitwise. See spec 2.1.

`embeddings2` and `embeddings_time` are upstream parameters that upstream never
reads. They are kept so that state_dict keys match upstream exactly; they are
32 scalars each and are documented here rather than silently dropped.
"""
__all__ = ['Model']

import torch
import torch.nn as nn

from layers.FreEncDec import (Encoder_ori, FreEncoderLayer,
                              FreAttentionLayer, FullAttention_ablation)
from layers.RevIN import RevIN


def _cfg(configs, name, fallback):
    """Per-branch override: use configs.<name> when set, else the shared value.

    `None` -- and ONLY None -- means "fall back". An explicit value equal to the
    shared one is a no-op by construction, which is what makes v1 bitwise
    reproducible (tests/test_fre_branch.py F2). Note the sentinel is None and not
    falsiness: `fe_dropout=0.0` is a legitimate "no dropout" and must be used,
    not silently replaced by the shared 0.5.

    v2 gives this branch its own width/heads/layers/dropout so it can run at its
    native architecture (d_model=512, n_heads=8, e_layers=2, dropout=0.2) while
    the patch branch stays at Times2D's 64/16/3/0.5. Spec 2.1.
    """
    v = getattr(configs, name, None)
    return fallback if v is None else v


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        # d_model/d_ff here are the FREQ BRANCH's, not the dual's: v2 lets this
        # branch run at its native 512 while the patch branch stays at 64
        # (spec 2.1). None falls back to the shared field.
        self.d_model = _cfg(configs, 'fe_d_model', configs.d_model)
        self.hidden_size = self.d_model
        self.d_ff = _cfg(configs, 'fe_d_ff', configs.d_ff)
        self.time_branch = configs.time_branch

        # Dead on this path: upstream's forward never reads either one. See the
        # module docstring for why they are kept.
        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride

        self.embed_size = configs.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings2 = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings_time = nn.Parameter(torch.randn(1, self.embed_size))

        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        # token_num=enc_in is upstream's choice and it is load-bearing: the
        # learned weight_mat is [1,1,N,N] and multiplies the attention matrix
        # elementwise, so the attention is over the CHANNEL axis. l, s and
        # token_num must all be enc_in or this raises at the multiply.
        self.encoder_fre_real = self._make_encoder(configs)
        self.fre_trans_real = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, self.d_model),
            self.encoder_fre_real,
            nn.Linear(self.d_model, self.valid_fre_points * self.embed_size)
        )
        self.encoder_fre_imag = self._make_encoder(configs)
        self.fre_trans_imag = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, self.d_model),
            self.encoder_fre_imag,
            nn.Linear(self.d_model, self.valid_fre_points * self.embed_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(_cfg(configs, 'fe_dropout', configs.dropout))

    @staticmethod
    def _make_encoder(configs):
        """Upstream's Encoder_ori stack, minus CKA and minus save_folder.

        `save_folder` is absent on purpose: it existed only to feed the removed
        plot branches, and building its f-string would have forced a
        `plot_mat_label` config field to exist for a path that can no longer
        run. See layers/FreEncDec.py's docstring.

        The five fe_* values are resolved once, here, and reused for the six
        uses below -- resolving them inline would let them drift apart.
        """
        d_model = _cfg(configs, 'fe_d_model', configs.d_model)
        d_ff = _cfg(configs, 'fe_d_ff', configs.d_ff)
        n_heads = _cfg(configs, 'fe_n_heads', configs.n_heads)
        e_layers = _cfg(configs, 'fe_e_layers', configs.e_layers)
        dropout = _cfg(configs, 'fe_dropout', configs.dropout)

        return Encoder_ori(
            [
                FreEncoderLayer(
                    FreAttentionLayer(
                        FullAttention_ablation(
                            False, configs.factor,
                            attention_dropout=dropout,
                            output_attention=configs.output_attention,
                            token_num=configs.enc_in,
                            SF_mode=configs.attn_enhance,
                            softmax_flag=configs.attn_softmax_flag,
                            weight_plus=configs.attn_weight_plus,
                            outside_softmax=configs.attn_outside_softmax,
                            plot_mat_flag=configs.plot_mat_flag and _ == e_layers - 1,
                            plot_grad_flag=configs.plot_grad_flag and _ == e_layers - 1),
                        d_model, n_heads),
                    d_model, d_ff,
                    dropout=dropout, activation=configs.activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            one_output=True)

    def tokenEmb(self, x, embeddings):
        """[B,T,N] -> [B,N,T,D]. Upstream verbatim."""
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def _fre_trans_features(self, x):
        """[B,N,T,D] -> (out [B,N,T,D], chan_real [B,N,d_model], chan_imag [B,N,d_model]).

        Re-derives upstream's `Fre_Trans` by stepping through the SAME
        Sequential it would have called -- [0], [1], [2] in order -- so `out`
        is bitwise what upstream computes, and the tensors the dual needs fall
        out of the middle without any weight being renamed or moved.

        The tap is PRE-Encoder. That makes it each channel's own spectrum
        summary, before this branch's channel attention mixes anything, so the
        dual's cross-attention is the only channel exchange on this path. Read
        only: callers must not write back into the branch, or the branch's own
        prediction stops being its own (see models/DualFreEformer.py).
        """
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')
        assert x_fre.shape[-1] == self.valid_fre_points

        y_real = x_fre.real.flatten(-2)                      # [B,N,D*F]
        y_imag = x_fre.imag.flatten(-2)

        chan_real = self.fre_trans_real[0](y_real)           # [B,N,d_model]
        chan_imag = self.fre_trans_imag[0](y_imag)

        y_real = self.fre_trans_real[2](self.fre_trans_real[1](chan_real))
        y_imag = self.fre_trans_imag[2](self.fre_trans_imag[1](chan_imag))

        y = torch.complex(y_real.reshape(B, N, D, self.valid_fre_points),
                          y_imag.reshape(B, N, D, self.valid_fre_points))

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        out = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')
        out = out.transpose(-1, -2)                          # [B,N,T,D]
        return out, chan_real, chan_imag

    def Fre_Trans(self, x):
        """Upstream's method, unchanged in behaviour: the tap is a read."""
        return self._fre_trans_features(x)[0]

    def forward_features(self, x_enc):
        """x_enc: [B,T,N] raw.

        Returns (y_norm [B,pred_len,N], chan_real [B,N,d_model],
                 chan_imag [B,N,d_model]).

        `y_norm` is in NORMALISED space (no denorm) -- the dual denormalises
        once at the very end, the same contract as
        models/FreDN.py:forward_features.

        The dropout upstream applies before denorm happens HERE, not in
        forward(), so that the y_norm the dual consumes during training is
        exactly what standalone FreEformer would feed into its denorm. Putting
        it in forward() instead would have made forward_features and forward
        diverge in train mode.
        """
        x = self.revin_layer(x_enc, mode='norm')
        x_ori = x

        # [B, T, N] -> [B, N, T, D]
        h = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, T, D]; upstream's `Fre_Trans(x) + x` adds THIS same tensor
        # back, so it is computed once and reused rather than recomputed.
        x, chan_real, chan_imag = self._fre_trans_features(h)
        x = x + h

        # [B, N, T*D] --> [B, N, pred_len] --> [B, pred_len, N]
        y_norm = self.fc(x.flatten(-2)).transpose(-1, -2)

        return self.dropout(y_norm), chan_real, chan_imag

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """x: [Batch, Input length, Channel] -> [B, pred_len, N], denormalised.

        Delegates to forward_features so the two cannot drift: this is
        denorm(forward_features(...)[0]) by construction, which is what
        tests/test_fre_branch.py B1 asserts.
        """
        y_norm, _chan_real, _chan_imag = self.forward_features(x)
        return self.revin_layer(y_norm, mode='denorm')

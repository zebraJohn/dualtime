"""FreEformer's attention stack, ported for models/FreEformer.py.

WHY THIS FILE EXISTS INSTEAD OF ADDING TO layers/SelfAttention_Family.py
-----------------------------------------------------------------------
layers/Transformer_EncDec.py:27 already defines an EncoderLayer, and
layers/SelfAttention_Family.py:134 already defines an AttentionLayer. Neither
is signature-compatible with upstream's: upstream's EncoderLayer takes
MLP_flag and its forward takes tau/delta/token_weight; upstream's
AttentionLayer takes dp_rank/imp_mode and its forward takes the same three.
Reusing the names here would bind the wrong class with no error at all --
the call would just take the other code path.

Class names are not part of state_dict keys (those are attribute paths like
`fre_trans_real.1.attn_layers.0.attention.query_projection.weight`), so
renaming two classes costs nothing at checkpoint time.

Upstream lives at D:\\pycharm\\FreEformer\\layers\\ -- Transformer_EncDec.py
for EncoderLayer/Encoder_ori and SelfAttention_Family.py for
AttentionLayer/FullAttention_ablation.

WHAT WAS DELIBERATELY DROPPED (dropped, not forgotten)
------------------------------------------------------
1. AttentionLayer's dp_rank / imp_mode branch. Upstream's attn_ablation
   builds it with three positional args, so dp_rank is always None on this
   path and that branch cannot execute; keeping it would drag in the
   dynamic_projection dependency for dead code.
2. Encoder_ori's CKA_flag branch. It imports CudaCKA and calls
   random.uniform(0, 1) inside forward. CKA_flag is fixed at 0 here, but the
   draw is inside the forward pass, and seed reproducibility is load-bearing
   for every number this repo reports -- so it goes, rather than merely
   staying switched off.
3. FullAttention_ablation's two plot branches (plot_mat_flag and
   plot_grad_flag). They call plot_mat (a utils.tools dependency) and draw
   from the global RNG via random.random()/random.randint(). Note that
   upstream's plot_mat guard is `random.random() < 1`, i.e. always true, and
   the plot_grad guard is NOT wrapped in `not self.training` -- so with
   either flag on, eval (and training) would consume RNG state and write
   files. Both branches now raise instead, so the flags cannot be turned on
   by accident and no draw is left behind them.
4. FullAttention_ablation's four print() calls in __init__, which fire once
   per layer construction and pollute every run log.

FullAttention_ablation keeps upstream's learnable weight_mat [1,1,N,N] and
tau [1,1,N,1]. weight_mat multiplies the attention matrix A [b,h,l,s]
elementwise, so l == s == token_num is a structural requirement, not a
coincidence -- upstream sets token_num=enc_in, i.e. the attention is over the
CHANNEL axis by design. That is exactly the axis this repo's DualFreEformer
puts its cross-attention on.
"""
__all__ = ['FreEncoderLayer', 'Encoder_ori', 'FreAttentionLayer',
           'FullAttention_ablation']

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask


class FreEncoderLayer(nn.Module):
    """Upstream EncoderLayer, renamed. Verbatim otherwise.

    One translation: upstream tests `isinstance(x, List)` with typing.List,
    which is deprecated for isinstance checks; (tuple, list) is the same
    test without the deprecation.
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", MLP_flag=True):
        super(FreEncoderLayer, self).__init__()
        self.MLP_flag = MLP_flag
        # dff is defaulted at 4*d_model
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.MLP_flag:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            self.norm2 = nn.LayerNorm(d_model)
            self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, token_weight=None):
        list_flag = isinstance(x, (tuple, list))
        if list_flag:
            k_ori = x[1]
            if len(x) not in {2, 3}:
                raise ValueError('Input error in EncoderLayer')
            q, k, v = (x[0], x[1], x[1]) if len(x) == 2 else (x[0], x[1], x[2])
            x = q
        else:
            q, k, v = x, x, x

        new_x, attn = self.attention(
            q, k, v,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            token_weight=token_weight
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        output = y

        if self.MLP_flag:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            output = self.norm2(x + y)

        if list_flag:
            return [output, k_ori], attn
        else:
            return output, attn


class Encoder_ori(nn.Module):
    """Upstream Encoder_ori minus the CKA branch (see the module docstring).

    `conv_layers` is accepted and ignored, as it is upstream.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None,
                 one_output=False):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, nvars, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if isinstance(x, (tuple, list)):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        else:
            return x, attns


class FreAttentionLayer(nn.Module):
    """Upstream AttentionLayer minus the dp_rank/imp_mode path."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(FreAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                token_weight=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
            token_weight=token_weight.to(queries.device) if token_weight is not None else None
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention_ablation(nn.Module):
    """Upstream verbatim except the dropped prints and plot branches.

    The enhanced path is not a wrapper around softmax -- it learns an N x N
    prior (weight_mat) over the token axis and folds it into the attention
    matrix. With token_num=enc_in that prior is over CHANNELS.
    """

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, token_num=None, SF_mode=1, softmax_flag=1,
                 weight_plus=0, outside_softmax=0, plot_mat_flag=False,
                 save_folder='./', plot_grad_flag=False):
        super(FullAttention_ablation, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.SF_mode = SF_mode
        self.softmax_flag = softmax_flag
        self.token_num = token_num
        self.outside_softmax = outside_softmax
        self.weight_plus = weight_plus
        self.plot_mat_flag = plot_mat_flag
        self.plot_grad_flag = plot_grad_flag
        self.save_folder = save_folder
        self.num_heads = 1

        if self.SF_mode and self.token_num is not None:
            # [1,1,N,1]
            if self.softmax_flag:
                self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

            init_weight_mat = (torch.eye(self.token_num) * 1.0 +
                               torch.randn(self.token_num, self.token_num) * 1.0)
            self.weight_mat = nn.Parameter(
                init_weight_mat[None, None, :, :].repeat(1, self.num_heads or 1, 1, 1))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None
        if self.SF_mode and self.token_num is not None:
            # 2d
            if self.softmax_flag:
                weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)
            else:
                weight_mat = F.softplus(self.weight_mat)

        if self.SF_mode and weight_mat is not None:
            if self.outside_softmax:
                if self.weight_plus:
                    A = A.softmax(dim=-1) + weight_mat
                else:
                    A = A.softmax(dim=-1) * weight_mat
                A = F.normalize(A, p=1, dim=-1)
            else:
                if self.weight_plus:
                    A = A + weight_mat
                else:
                    A = A * weight_mat

                # attention matrix [b,h,l,s]
                A = torch.softmax(A, dim=-1)
        else:
            A = torch.softmax(A, dim=-1)

        # Both plot branches are gone. Upstream's plot_mat guard is
        # `random.random() < 1` (always true) and its plot_grad guard is not
        # wrapped in `not self.training`, so leaving them in -- even behind a
        # flag -- would keep two global-RNG draws wired into the forward pass
        # of a repo whose every reported number depends on seed
        # reproducibility.
        if not self.training and self.plot_mat_flag:
            raise NotImplementedError(
                'plot_mat_flag must stay 0: this port removed the plotting '
                'branch because its guard drew from the global RNG on every '
                'eval. See the module docstring.')
        if self.plot_grad_flag:
            raise NotImplementedError(
                'plot_grad_flag must stay 0: this port removed the plotting '
                'branch because its guard drew from the global RNG during '
                'training. See the module docstring.')

        # dropout, reserved
        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

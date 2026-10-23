"""A frequency-domain branch that mixes ACROSS CHANNELS.

WHY THIS EXISTS
---------------
Every branch this project has tested -- Times2D, PhaseFormer, FreDN -- is
per-channel: the channel axis is a batch axis and no information crosses between
channels. Measured consequence: rho(Times2D, FreDN) = 0.9593 on weather, 0.9602
on ETTh1, against a same-model-different-seed ceiling of 0.988. Their residuals
are ~96% shared.

Three attempts to break that failed: swapping the branch made it WORSE (FreDN is
more homogeneous with Times2D than PhaseFormer was, 0.9602 vs 0.9377), swapping
the dataset did nothing (0.9593 vs 0.9602), and feature-level cross-attention
produced nothing at all (5/12, mean -0.0020, PROGRESS.md 4.5).

The one structural bias absent from ALL of them is cross-channel modelling --
and the data is full of it. Weather's 21 variables are physically coupled; ETTh1's
7 channels are loads on the same transformer. Every model so far has thrown that
information away. This is the cheapest thing that does not.

WHAT IT IS
----------
The mechanism of FreEformer's FrePatchTST3_linear_cha, reduced to its essentials:

    x [B,T,N] -> RevIN -> emb -> FFT over T -> X [B,N,D,F]
                                            -> permute -> [B,D,F,N]
                                            -> MLP over the N axis
                                            -> permute back -> iFFT + long skip
                                            -> per-channel head -> [B,tau,N]

The per-channel models do their MLPs over the FREQUENCY axis, per channel. This
one does it over the CHANNEL axis, per (frequency, embedding) pair. That single
axis change is the whole difference, and tests/test_cross_channel.py proves it
is real: perturbing one channel's input moves other channels' outputs, whereas
FreDN provably moves them by exactly 0.

WHAT IT IS NOT
--------------
This is NOT FreEformer. FreEformer also changes the embedding, the depth, the
width, the attention mechanism and the training recipe (23 config fields against
this model's 4). If rho drops here, cross-channel is implicated; if it does not,
only THIS cross-channel mixer is excluded, not the idea in general. The point of
the reduced version is that it isolates ONE variable: everything except the
channel axis is the same kind of thing Times2D and FreDN already are.

Real/imag get SEPARATE mixers, following linear_cha's comment that sharing them
performs worse.

No cross-attention, no dual model, no MoE: this is measured standalone against
Times2D with scripts/analyze_complementarity.py, exactly like the rho guard.
"""
__all__ = ['Model']

import torch
import torch.nn as nn

from layers.RevIN import RevIN


class _ChannelMixer(nn.Module):
    """An MLP over the LAST axis, applied pointwise over every other axis.

    With the input shaped [..., N] this mixes channels at each of the (batch,
    embedding, frequency) positions independently.
    """

    def __init__(self, n_channels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, z):
        return self.net(z)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        c = configs
        self.seq_len = c.seq_len
        self.pred_len = c.pred_len
        self.enc_in = c.enc_in
        self.embed_size = int(getattr(c, 'cc_embed_size', 8))
        self.mix_hidden = int(getattr(c, 'cc_hidden', 32))
        self.head_hidden = int(getattr(c, 'cc_head_hidden', 256))
        self.dropout = float(getattr(c, 'cc_dropout', 0.1))

        self.emb = nn.Parameter(torch.empty(self.seq_len, self.embed_size))
        nn.init.xavier_uniform_(self.emb)

        # affine=True, subtract_last=False -- upstream FreDN's setting, and the
        # same one this repo's FreDN port uses, so the two are comparable.
        self.revin_layer = RevIN(self.enc_in)

        self.cc_real = _ChannelMixer(self.enc_in, self.mix_hidden, self.dropout)
        self.cc_imag = _ChannelMixer(self.enc_in, self.mix_hidden, self.dropout)

        self.head = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden, self.pred_len),
        )

    def forward(self, x_enc, x_enc_mark=None, x_dec=None, x_dec_mark=None):
        """x_enc: [B, seq_len, N] raw -> [B, pred_len, N] raw."""
        B, T, N = x_enc.shape
        if T != self.seq_len or N != self.enc_in:
            raise ValueError(
                f'expected [B, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}')

        x = self.revin_layer(x_enc, 'norm')
        x = x.view(B, T, N, 1) * self.emb.view(1, T, 1, -1)     # [B, T, N, D]
        x = x.permute(0, 2, 3, 1)                               # [B, N, D, T]

        X = torch.fft.rfft(x, dim=-1, norm='ortho')             # [B, N, D, F]
        X = X.permute(0, 2, 3, 1)                               # [B, D, F, N]  <-- channels LAST

        # THE cross-channel step: real and imag each get an MLP over the channel
        # axis, independently at every (embedding, frequency) position.
        Y = torch.complex(self.cc_real(X.real), self.cc_imag(X.imag))
        Y = Y.permute(0, 3, 1, 2)                               # [B, N, D, F]

        y = torch.fft.irfft(Y, n=T, dim=-1, norm='ortho')       # [B, N, D, T]
        y = y + x                                               # long skip
        y = y.permute(0, 1, 3, 2)                               # [B, N, T, D]

        out = self.head(y.flatten(-2)).transpose(-1, -2)        # [B, tau, N]
        return self.revin_layer(out, 'denorm')

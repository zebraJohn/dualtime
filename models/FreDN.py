"""FreDN, ported from D:\\pycharm\\FreDN\\models\\FreDN.py (AAAI 2026).

Three deliberate differences from upstream, and nothing else:

1. RevIN signature. Upstream is `RevIN(args, num_features, ...)`; this repo's
   `layers.RevIN.RevIN` is `RevIN(num_features, eps, affine, subtract_last)`.
   Only the first argument is dropped. Upstream's `test_norm` mode is defined
   but never called anywhere in that repo, so it is not ported; its
   `subtract_short_terms` branch only fires when subtract_last=True, which this
   model never uses. `RevIN(self.feature_size)` therefore reproduces upstream's
   `RevIN(self.configs, self.feature_size)` exactly (affine=True,
   subtract_last=False).

2. `_freq_forward` is split into `_to_spectrum` / `_from_spectrum`, and the
   embedding line is lifted into `_embed`, so DualFreDN can mount
   cross-attention between the FFT and the MLP tail. The arithmetic is
   unchanged -- tests/test_fredn_refactor_parity.py (INV0) asserts the split
   reproduces upstream's monolithic version bitwise.

3. `forward_features()` is added for the dual model's feature tap. It returns
   the latent in NORMALISED space without denormalising, mirroring
   models/PhaseFormer.py's same-named method, so DualFreDN can denormalise once
   at the very end.

`embed_size` is read from `configs.fre_embed_size` (default 32). The `if False:`
branch in `_init_freq_components` is kept verbatim so the two files stay
diffable against each other.
"""
__all__ = ['Model', 'FreqDecomp']

import torch
import torch.nn as nn

from layers.RevIN import RevIN


class FreqDecomp(nn.Module):
    def __init__(self, T, N, D):
        super().__init__()
        self.T = T
        self.N = N
        self.D = D
        self.num_freq = T // 2 + 1

        freq_idx = torch.arange(self.num_freq)
        base_val = 1 / (freq_idx.float() + 1.0) ** 0.5
        base_val = (base_val / base_val.min() * 5).unsqueeze(-1).unsqueeze(-1)
        self.mask = nn.Parameter(base_val.repeat(1, N, D))
        with torch.no_grad():
            self.mask[..., 0] = 0.0

    def forward(self, x):
        """
        Input: x [B, T, N, D]
        Output: season [B, T, N, D], trend [B, T, N, D]
        """
        B, T, N, D = x.shape
        assert T == self.T and N == self.N and D == self.D

        # FFT on T dimension [B, T, N, D]
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freq_mask = torch.sigmoid(self.mask).unsqueeze(0)
        # Trend component spectrum [B, F, N, D] * [1, F, N, D]
        trend_fft = x_fft * freq_mask

        # IFFT to recover trend component [B, T, N, D]
        trend = torch.fft.irfft(trend_fft, n=T, dim=1, norm='ortho')
        season = x - trend
        return season, trend


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self._init_hyperparams()
        self._init_components()

    def _init_hyperparams(self):
        """Hyperparameters"""
        c = self.configs
        self.seq_len = c.seq_len
        self.pred_len = c.pred_len
        self.embed_size = getattr(c, 'fre_embed_size', 32)
        self.hidden_size = getattr(c, 'fre_hidden_size', 64)
        self.hidden_layers = getattr(c, 'fre_hidden_layers', 1)
        self.feature_size = c.enc_in
        self.dropout = getattr(c, 'fre_dropout', 0.1)

    def _init_components(self):
        """Initialize all submodules"""
        # Embedding module
        self.emb = nn.Parameter(torch.Tensor(self.seq_len, self.embed_size))
        nn.init.xavier_uniform_(self.emb)
        # Decomposition module
        self.decomp = FreqDecomp(T=self.seq_len, N=self.feature_size, D=self.embed_size)
        # RevIN
        self.revin_layer = RevIN(self.feature_size)
        # Frequency domain learning components
        self._init_freq_components()
        # Trend prediction components
        self._init_trend_components()
        # Embedding fusion
        self.emb_proj = nn.Linear(self.embed_size, 1)

    def _init_freq_components(self):
        """Initialize frequency domain learning components (real-imaginary separation)"""
        self.freq_learner = nn.ModuleDict()
        if False:
            # Independent parameters
            self.freq_learner['real_part'] = self._create_learner(self.seq_len // 2 + 1, self.pred_len // 2 + 1)
            self.freq_learner['imag_part'] = self._create_learner(self.seq_len // 2 + 1, self.pred_len // 2 + 1)
        else:
            # Shared parameters
            shared_learner = self._create_learner(self.seq_len // 2 + 1, self.pred_len // 2 + 1)
            self.freq_learner['real_part'] = shared_learner
            self.freq_learner['imag_part'] = shared_learner

    def _init_trend_components(self):
        """Initialize trend learning components"""
        self.trend_learner = self._create_learner(self.seq_len, self.pred_len)

    def _create_learner(self, input_size, output_size):
        hidden_dims = self._calc_hidden_dims(self.hidden_size, output_size)
        layers = []
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        for i in range(self.hidden_layers):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i % 2 == 0:  # Apply every other layer
                layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))

        return nn.ModuleDict({
            'res_proj': nn.Linear(input_size, hidden_dims[-1]),
            'proj_layers': nn.Sequential(*layers),
            'pred_layers': nn.Linear(hidden_dims[-1], output_size)
        })

    def _calc_hidden_dims(self, input_dim, output_dim):
        """Calculate hidden dimensions using geometric progression, 0-th is input dimension"""
        ratio = (input_dim / output_dim) ** (1 / (self.hidden_layers + 1))
        return [int(input_dim / (ratio ** i)) for i in range(self.hidden_layers + 1)]

    # ------------------------------------------------------------------
    # split of upstream's `_freq_forward` -- see the module docstring
    # ------------------------------------------------------------------
    def _embed(self, x_norm):
        """[B,T,N] -> [B,T,N,D]. Upstream inlines this line in forward()."""
        B, T, N = x_norm.shape
        return x_norm.view(B, T, N, 1) * self.emb.view(1, self.seq_len, 1, -1)

    def _to_spectrum(self, season):
        """[B,T,N,D] -> [B,N,D,F] complex, F = seq_len//2 + 1.

        Note the axis order: the time axis is MOVED TO THE BACK before rfft, so
        the result is [B,N,D,F], not [B,N,F,D]. DualFreDN packs this with the F
        axis as tokens -- see models/DualFreDN.py::_pack_freq.
        """
        return torch.fft.rfft(season.permute(0, 2, 3, 1), dim=3, norm="ortho")

    def _from_spectrum(self, x_freq):
        """[B,N,D,F] complex -> [B,pred_len,N].

        Note this is [B,pred_len,N] -- channel LAST -- whereas Times2D's
        features are [B,N,...]. DualFreDN permutes at the call site.
        """
        # Frequency domain residual
        res_real = self.freq_learner['real_part']['res_proj'](x_freq.real)
        res_imag = self.freq_learner['imag_part']['res_proj'](x_freq.imag)
        # Frequency domain projection
        proj_real = self.freq_learner['real_part']['proj_layers'](x_freq.real)
        proj_imag = self.freq_learner['imag_part']['proj_layers'](x_freq.imag)
        # Frequency domain prediction
        pred_real = self.freq_learner['real_part']['pred_layers'](proj_real + res_real)
        pred_imag = self.freq_learner['imag_part']['pred_layers'](proj_imag + res_imag)
        pred_freq = torch.complex(pred_real, pred_imag)              # [B,N,D,H//2+1]
        pred = torch.fft.irfft(pred_freq, n=self.pred_len, dim=3, norm="ortho")  # [B,N,D,H]
        return self.emb_proj(pred.permute(0, 3, 1, 2)).squeeze(-1)   # [B,H,N]

    def _trend_forward(self, trend):
        x = trend.permute(0, 2, 3, 1)                                # [B,N,D,T]
        # Trend residual
        res = self.trend_learner['res_proj'](x)
        # Trend projection
        proj = self.trend_learner['proj_layers'](x)
        # Trend prediction
        pred = self.trend_learner['pred_layers'](proj + res)         # [B,N,D,H]
        return self.emb_proj(pred.permute(0, 3, 1, 2)).squeeze(-1)   # [B,H,N]

    # ------------------------------------------------------------------
    def forward(self, x_enc, x_enc_mark=None, x_dec=None, x_dec_mark=None):
        """Input [B,T,N] raw -> Output [B,pred_len,N] raw."""
        x_norm = self.revin_layer(x_enc, 'norm')
        season, trend = self.decomp(self._embed(x_norm))
        season_pred = self._from_spectrum(self._to_spectrum(season))
        trend_pred = self._trend_forward(trend)
        return self.revin_layer(trend_pred + season_pred, 'denorm')

    def forward_features(self, x_enc):
        """x_enc: [B,T,N] raw.

        Returns (X [B,N,D,F] complex, trend [B,T,N,D], y_norm [B,pred_len,N]).

        `y_norm` is in NORMALISED space (season + trend, no denorm), mirroring
        PhaseFormer.forward_features returning a pre-denorm latent. DualFreDN
        denormalises once at the very end.
        """
        x_norm = self.revin_layer(x_enc, 'norm')
        season, trend = self.decomp(self._embed(x_norm))
        X = self._to_spectrum(season)
        y_norm = self._from_spectrum(X) + self._trend_forward(trend)
        return X, trend, y_norm

# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


import torch as t

def divide_no_nan(a, b):
    result = t.zeros_like(a)
    non_zero_mask = b != 0
    result[non_zero_mask] = a[non_zero_mask] / b[non_zero_mask]
    return result

class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        device = forecast.device
        target = target.to(device)
        mask = mask.to(device)
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class HuberLoss(nn.Module):
    """Huber loss matching the PhaseFormer reference implementation exactly.

    Deliberately NOT nn.HuberLoss. That one computes

        |d| <= delta :  0.5 * d^2
        |d| >  delta :  delta * |d| - 0.5 * delta^2

    whereas PhaseFormer-test/src/models/PhaseFormer.py `_compute_loss` computes

        |d| <= delta :  0.5 * d^2 / delta
        |d| >  delta :  |d| - 0.5 * delta

    The two coincide only at delta == 1.0 (the reference's setting) and differ
    by a factor of delta everywhere else. Since --huber_delta is exposed, using
    nn.HuberLoss would silently rescale the loss -- and with it the val_loss
    that checkpoint selection and early stopping are driven by -- as soon as
    anyone changes the delta.
    """

    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        if delta <= 0:
            raise ValueError('huber delta must be > 0, got {}'.format(delta))
        self.delta = float(delta)

    def forward(self, pred, target):
        abs_diff = t.abs(pred - target)
        delta = abs_diff.new_tensor(self.delta)
        quadratic = t.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        return (0.5 * quadratic ** 2 / delta + linear).mean()


class FreqMAELoss(nn.Module):
    """Frequency-domain MAE, matching upstream FreDN's training objective.

    Copied from FreDN/exp/exp_long_term_forecasting.py:148-161, the
    `auxi_mode='rfft', auxi_type='complex', auxi_loss='MAE', module_first=1`
    path. Upstream's ETTh1 script sets lambda=0.0, so `rec_lambda=0` and
    `auxi_lambda=1`: the TIME-DOMAIN LOSS IS WEIGHTED ZERO and this is the
    model's entire training objective. That is deliberate -- the paper's
    "Loss Function Analysis" derives it as L_F = (1/tau_freq) * ||eps_tilde||_1.

    Note `torch.fft.rfft` defaults to norm=None (unnormalised), so this loss
    lands roughly sqrt(pred_len) times larger than the time-domain MAE. That is
    upstream's behaviour and is reproduced on purpose; --loss freqmae runs are
    expected to be watched for scale on their first epoch.
    """

    def forward(self, pred, target):
        diff = t.fft.rfft(pred, dim=1) - t.fft.rfft(target, dim=1)
        return diff.abs().mean()

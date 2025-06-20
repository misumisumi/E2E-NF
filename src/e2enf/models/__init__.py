import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from .discriminator import *  # NOQA
from .feature_mapping import *  # NOQA
from .generator import *  # NOQA


def dilated_factor(batch_f0, fs, dense_factor):
    """Pitch-dependent dilated factor

    Args:
        batch_f0 (ndarray): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle

    Return:
        dilated_factors(np array):
            float array of the pitch-dependent dilated factors (T)

    """
    _batch_f0 = batch_f0.copy(order="C")
    _batch_f0[_batch_f0 == 0] = fs / dense_factor
    dilated_factors = np.ones(_batch_f0.shape) * fs / dense_factor / _batch_f0
    assert np.all(dilated_factors > 0)

    return dilated_factors


def dilated_factor_torch(batch_f0: Tensor, fs: int, dense_factor: int) -> Tensor:
    _batch_f0 = batch_f0.detach().clone()
    _batch_f0[_batch_f0 == 0] = fs / dense_factor
    dilated_factors = torch.ones_like(_batch_f0) * fs / dense_factor / _batch_f0
    assert torch.all(dilated_factors > 0)

    return dilated_factors


def f0_linear_conv(
    batch_f0: ndarray, f0_mean_src: ndarray, f0_mean_tgt: ndarray, f0_std_src: ndarray, f0_std_tgt: ndarray
) -> ndarray:
    conved_batch_f0 = (batch_f0 - f0_mean_src) * f0_std_tgt / f0_std_src + f0_mean_tgt
    conved_batch_f0 = np.where(batch_f0 > 0, conved_batch_f0, 0)

    return conved_batch_f0


def f0_linear_conv_torch(batch_f0: Tensor, mean_x: Tensor, mean_y: Tensor, std_x: Tensor, std_y: Tensor):
    conved_batch_f0 = (batch_f0 - mean_x) * std_y / std_x + mean_y
    conved_batch_f0 = torch.where(batch_f0 > 0, conved_batch_f0, 0)

    return conved_batch_f0

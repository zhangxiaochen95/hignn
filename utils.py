r"""
This script includes functions shared by both 'train_hignn.py' and 'train_dnn.py'.
"""

# Import packages for mathematics.
import numpy as np
# Import packages for ML.
import torch
from torch import Tensor


def cmp2real(x):
    """Converts a complex ndarray to a real one with an additional dimension -1."""
    return np.concatenate([np.expand_dims(x.real, -1), np.expand_dims(x.imag, -1)], axis=-1)


def cmp_matmul(a, b):
    """Matrix multiplication of two (complex) Tensors."""
    assert isinstance(a, Tensor)
    if a.dtype == torch.cfloat:
        a_real, a_imag = a.real, a.imag
    else:
        a_real, a_imag = a[:, :, 0], a[:, :, 1]
    b_real, b_imag = b[:, :, 0], b[:, :, 1]
    return [a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real]


def build_diag_block(blk_list):
    """Build a block diagonal Tensor from a list of Tensors."""
    if blk_list[0].ndim == 2:
        return torch.block_diag(*blk_list)
    elif blk_list[0].ndim == 3:
        blks = []
        for idx_ant in range(blk_list[0].shape[-1]):
            blks_per_ant = []
            for idx_link in range(len(blk_list)):
                blks_per_ant.append(blk_list[idx_link][:, :, idx_ant])
            blks.append(torch.block_diag(*blks_per_ant))
        return torch.dstack(blks)
    else:
        raise Exception("Invalid input dimension")


def weighted_sum_rate(h, x, var_awgn, w):
    """
    Computes the weighted sum rate from the outputs from models.
    Note that this function is different from the one in 'envs.py'.
    """
    rx_power = {}
    for rtype, ttype in h.keys():
        rx_power[(rtype, ttype)] = cmp_matmul(h[(rtype, ttype)], x[ttype])[0].sum(-1).pow(2) + \
                                   cmp_matmul(h[(rtype, ttype)], x[ttype])[1].sum(-1).pow(2)
    # Compute the interference level at each receiver.
    interference = {}
    for rtype in x.keys():
        interference[rtype] = - torch.diag(rx_power[(rtype, rtype)])
        for ttype in x.keys():
            interference[rtype] += rx_power[(rtype, ttype)].sum(-1)
    # Compute the signal-to-interference-plus-noise ratio (SINR) and achievable rate.
    sinr, rate = {}, {}
    for rtype in x.keys():
        sinr[rtype] = torch.diag(rx_power[(rtype, rtype)]) / (interference[rtype] + var_awgn[rtype].squeeze())
        rate[rtype] = torch.log(1 + sinr[rtype])

    # Accumulate the WSR throughout link types.
    wsr = 0.  # Weighted sum rate
    for ltype in w.keys():
        wsr += w[ltype].squeeze().dot(rate[ltype])
    return wsr
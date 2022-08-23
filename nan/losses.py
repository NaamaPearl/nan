from functools import partial

import torch

from nan.utils.general_utils import TINY_NUMBER


def mean_with_mask(x, mask):
    return torch.sum(x * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def loss_with_mask(fn, x, y, mask=None):
    """
    @param x: img 1, [(...), 3]
    @param y: img 2, [(...), 3]
    @param mask: optional, [(...)]
    @param fn: loss function
    """
    loss = fn(x, y)
    if mask is None:
        return torch.mean(loss)
    else:
        return mean_with_mask(loss, mask)


l2_loss  = partial(loss_with_mask, lambda x, y: (x - y) ** 2)
l1_loss  = partial(loss_with_mask, lambda x, y: torch.abs(x - y))
gen_loss = partial(loss_with_mask, lambda x, y: x, 0)
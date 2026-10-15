"""The training loss: cross-entropy against the labels, Kullback-Leibler divergence from the teacher's class
probabilities, optional soft Dice. Pixels without a class (no data, unknown mask values) count in none of them."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice(logits, labels):
    """1 - mean over classes of the soft Dice coefficient, over the pixels with a class (labels >= 0)."""
    valid = (labels >= 0).unsqueeze(1).float()
    p = torch.softmax(logits, 1) * valid
    onehot = F.one_hot(labels.clamp(min=0), logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()


def loss(logits, labels, t_logp, valid, class_weights=None, kd_weight=0.0, temperature=1.0, dice_weight=0.0):
    """(1 - kd_weight) x (CE + dice_weight x Dice) + kd_weight x T^2 x KL(teacher || model), the KL averaged over
    the pixels with data (valid). Computed in FP32. Returns (loss, CE, KL)."""
    logits = logits.float()
    zero = logits.sum() * 0.0
    ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=-1) if (labels >= 0).any() else zero
    sup = ce + (dice_weight * soft_dice(logits, labels) if dice_weight else 0.0)
    kd = zero
    if kd_weight > 0 and t_logp is not None:
        log_q = F.log_softmax(logits / temperature, 1)
        log_p = F.log_softmax(t_logp / temperature, 1)
        kl = (log_p.exp() * (log_p - log_q)).sum(1)
        kd = (kl * valid).sum() / valid.sum().clamp(min=1.0) * temperature ** 2
    return (1 - kd_weight) * sup + kd_weight * kd, ce, kd

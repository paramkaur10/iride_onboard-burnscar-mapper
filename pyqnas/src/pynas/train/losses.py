import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class CategoricalCrossEntropyLoss(nn.Module):
    """Standard cross-entropy. Handles one-hot targets and ignore_index."""
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        if targets.ndim == 4:
            targets = torch.argmax(targets, dim=1)
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """
    Focal loss with class weighting and ignore_index support.

    pt is computed from an UNWEIGHTED cross-entropy so it reflects actual
    model confidence. Class weight is applied as a separate multiplier on
    the final per-pixel loss. Previously pt was derived from the WEIGHTED
    ce, which distorted per-class confidence and made single-class collapse
    a cheap stable minimum.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.reduction    = reduction
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if targets.ndim == 4:
            targets = torch.argmax(targets, dim=1)

        valid_mask   = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        # pt from UNWEIGHTED ce — confidence must not be distorted by class weight
        ce_raw = F.cross_entropy(
            logits, targets_safe, weight=None,
            reduction="none", ignore_index=-100,
        )
        ce_raw = ce_raw * valid_mask.float()
        pt           = torch.exp(-ce_raw)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        if self.weight is not None:
            w = self.weight.to(logits.device)[targets_safe]
        else:
            w = 1.0

        focal_loss = focal_weight * w * ce_raw

        n_valid = valid_mask.sum().clamp(min=1)
        if self.reduction == "mean":
            return focal_loss.sum() / n_valid
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

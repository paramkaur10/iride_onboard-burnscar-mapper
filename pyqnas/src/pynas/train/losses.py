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

    Original had neither — class weights were silently dropped and nodata
    pixels (label=-1) were fed directly into F.cross_entropy, corrupting
    gradients or triggering CUDA asserts.
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
        self.alpha       = alpha
        self.gamma       = gamma
        self.reduction   = reduction
        self.weight      = weight      # per-class weights tensor
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if targets.ndim == 4:
            targets = torch.argmax(targets, dim=1)

        # mask out ignore_index before computing cross-entropy
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0   # temporarily set to valid class

        ce_loss = F.cross_entropy(
            logits,
            targets_safe,
            weight=self.weight,
            reduction="none",
            ignore_index=-100,   # handled manually above
        )
        # zero out ignored pixels
        ce_loss = ce_loss * valid_mask.float()

        pt           = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss   = focal_weight * ce_loss

        n_valid = valid_mask.sum().clamp(min=1)
        if self.reduction == "mean":
            return focal_loss.sum() / n_valid
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
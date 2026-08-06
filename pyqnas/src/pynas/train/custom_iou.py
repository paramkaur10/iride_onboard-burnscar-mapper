import torch
from typing import Optional

@torch.no_grad()
def calculate_iou(
    preds_or_logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    from_logits: bool = True,
    ignore_index: Optional[int] = None,
):
    """
    Computes IoU with the SAME rules we’ll use in Myriad eval:
    - If from_logits=True, uses argmax over channel dim (multiclass) or sigmoid>0.5 (binary).
    - targets can be one-hot (N,C,H,W) or label masks (N,H,W).
    - Per-image: mean over classes present+absent (unless ignored), then mean over batch.
    """
    # ---- get predicted labels ----
    if preds_or_logits.dim() != 4:
        raise ValueError("Expected (N,C,H,W) or (N,1,H,W)")

    if preds_or_logits.shape[1] == 1:
        # Binary: channel=1
        if from_logits:
            probs = torch.sigmoid(preds_or_logits)
        else:
            probs = preds_or_logits
        pred_labels = (probs > 0.5).long().squeeze(1)  # (N,H,W)
        n_classes = 2
    else:
        # Multiclass
        if from_logits:
            pred_labels = torch.argmax(preds_or_logits, dim=1)  # (N,H,W)
        else:
            pred_labels = torch.argmax(preds_or_logits, dim=1)
        n_classes = num_classes

    # ---- prepare target labels ----
    if targets.dim() == 4:     # one-hot -> labels
        target_labels = torch.argmax(targets, dim=1)           # (N,H,W)
    elif targets.dim() == 3:   # already labels
        target_labels = targets.long()
    else:
        raise ValueError("targets must be (N,H,W) or (N,C,H,W)")

    # optional ignore mask
    if ignore_index is not None:
        valid = (target_labels != ignore_index)
    else:
        valid = torch.ones_like(target_labels, dtype=torch.bool)

    # ---- classwise IoU per image ----
    batch_ious = []
    for b in range(pred_labels.size(0)):
        per_class = []
        for cls in range(n_classes):
            if ignore_index is not None and cls == ignore_index:
                continue
            pred_mask   = (pred_labels[b] == cls) & valid[b]
            target_mask = (target_labels[b] == cls) & valid[b]

            inter = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            iou = (inter + 1e-6) / (union + 1e-6)
            per_class.append(iou)
        # mean over classes (including empty ones, as on GPU now)
        batch_ious.append(torch.stack(per_class).mean())
    return torch.stack(batch_ious).mean()  # mean over batch

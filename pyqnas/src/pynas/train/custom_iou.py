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
    Computes mean IoU over the batch.

    Key correctness fix vs original:
    - Classes absent from BOTH pred AND target in an image are EXCLUDED
      from that image's mean, rather than contributing 1.0 via the epsilon
      trick (inter+1e-6)/(union+1e-6). A collapsed model that predicts one
      class everywhere would otherwise score ~0.83 mean IoU on 6 classes.
    - ignore_index pixels are masked out before any intersection/union calc.
    """
    if preds_or_logits.dim() != 4:
        raise ValueError("Expected (N,C,H,W)")

    if preds_or_logits.shape[1] == 1:
        probs = torch.sigmoid(preds_or_logits) if from_logits else preds_or_logits
        pred_labels = (probs > 0.5).long().squeeze(1)
        n_classes = 2
    else:
        pred_labels = torch.argmax(preds_or_logits, dim=1)
        n_classes = num_classes

    if targets.dim() == 4:
        target_labels = torch.argmax(targets, dim=1)
    elif targets.dim() == 3:
        target_labels = targets.long()
    else:
        raise ValueError("targets must be (N,H,W) or (N,C,H,W)")

    if ignore_index is not None:
        valid = target_labels != ignore_index
    else:
        valid = torch.ones_like(target_labels, dtype=torch.bool)

    batch_ious = []
    for b in range(pred_labels.size(0)):
        v = valid[b]
        per_class = []
        for cls in range(n_classes):
            if ignore_index is not None and cls == ignore_index:
                continue
            pred_m   = (pred_labels[b] == cls) & v
            target_m = (target_labels[b] == cls) & v
            inter    = (pred_m & target_m).sum().float()
            union    = (pred_m | target_m).sum().float()
            # Only include class if it is present in pred OR target.
            # Absent classes are excluded — not given 1.0 via epsilon.
            if union > 0:
                per_class.append(inter / union)
        if per_class:
            batch_ious.append(torch.stack(per_class).mean())

    if not batch_ious:
        return torch.tensor(0.0, device=preds_or_logits.device)
    return torch.stack(batch_ious).mean()
"""Training pieces shared by the teacher (notebook 05), the NAS (06) and the final model (07):
the loss, the metrics, one epoch of training, and evaluation over fixed windows.

Classes are the training ids from burnscar.labels: 0 clear, 1 fresh burn, 2 old burn, 3 cloud,
4 cloud shadow, 5 water. -1 (ignore, no data) is left out of the loss and of every metric.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import TRAIN_NAMES

N_CLASSES = len(TRAIN_NAMES)
IGNORE = -1


def build_unet(encoder="resnet50", in_channels=7, classes=N_CLASSES, pretrained=True):
    """U-Net from segmentation_models_pytorch. With ImageNet weights the first layer is adapted to in_channels."""
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_channels, classes=classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class SegLoss(nn.Module):
    """Cross-entropy with class weights, plus soft Dice over the classes present in the batch.
    Pixels with target -1 count in neither."""

    def __init__(self, class_weights=None, dice_weight=1.0, ignore_index=IGNORE):
        super().__init__()
        w = torch.ones(N_CLASSES) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weight", w)
        self.dice_weight, self.ignore_index = dice_weight, ignore_index

    def forward(self, logits, target):
        logits = logits.float()
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index)
        if self.dice_weight <= 0:
            return ce
        v = valid.unsqueeze(1).float()
        prob = logits.softmax(1) * v
        onehot = F.one_hot(target.clamp(min=0), N_CLASSES).permute(0, 3, 1, 2).float() * v
        inter = (prob * onehot).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        present = onehot.sum((0, 2, 3)) > 0
        dice = (1 - (2 * inter + 1) / (denom + 1))[present].mean()
        return ce + self.dice_weight * dice


class Confusion:
    """Confusion matrix over the pixels with a label (target >= 0). Rows: label, columns: prediction."""

    def __init__(self, n=N_CLASSES):
        self.n = n
        self.m = torch.zeros(n, n, dtype=torch.int64)

    @torch.no_grad()
    def add(self, pred, target):
        valid = target >= 0
        idx = (target[valid] * self.n + pred[valid]).flatten()
        self.m += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n).cpu()

    def __iadd__(self, other):
        self.m += other.m
        return self

    def table(self):
        """Per class: labelled pixels, precision, recall, F1 and IoU."""
        m = self.m.double()
        tp = m.diag()
        fn, fp = m.sum(1) - tp, m.sum(0) - tp
        prec = tp / (tp + fp).clamp(min=1)
        rec = tp / (tp + fn).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
        iou = tp / (tp + fp + fn).clamp(min=1)
        return pd.DataFrame({"pixels": m.sum(1).long().numpy(), "precision": prec.numpy(), "recall": rec.numpy(),
                             "f1": f1.numpy(), "iou": iou.numpy()}, index=TRAIN_NAMES[:self.n])

    def fresh(self):
        """Precision, recall, F1 and IoU of fresh burn (class 1)."""
        return self.table().loc["fresh", ["precision", "recall", "f1", "iou"]].to_dict()

    def merged(self, classes):
        """Precision, recall, F1 and IoU of several classes taken as one: a pixel of any of them
        predicted as any of them is right."""
        m = self.m.numpy().astype(np.float64)
        s = np.zeros(self.n, bool)
        s[list(classes)] = True
        tp, fn, fp = m[np.ix_(s, s)].sum(), m[np.ix_(s, ~s)].sum(), m[np.ix_(~s, s)].sum()
        prec, rec = tp / max(tp + fp, 1.0), tp / max(tp + fn, 1.0)
        return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                    iou=tp / max(tp + fp + fn, 1.0))

    def burn(self):
        """Burn of any age: fresh (1) and old (2) as one class."""
        return self.merged((1, 2))


def warmup_cosine(optimizer, warmup_steps, total_steps, floor=0.01):
    """Learning rate: linear warm-up, then a cosine down to floor x the base rate."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device, amp_dtype):
    """Mixed precision when amp_dtype is set, plain float32 otherwise."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=amp_dtype)


def amp_settings(device):
    """bfloat16 autocast where the GPU has it (no scaler needed), else float16 with a gradient scaler."""
    if device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    return torch.float16, (scaler_cls("cuda") if scaler_cls else torch.cuda.amp.GradScaler())


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, amp_dtype=None, scaler=None,
                clip=1.0, log_every=100, extra_loss=None):
    """One pass over loader. extra_loss(x, logits, y) can add a term (notebook 06 adds distillation)."""
    model.train()
    total, n, t0 = 0.0, 0, time.time()
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            logits = model(x)
        loss = loss_fn(logits, y)
        if extra_loss is not None:
            loss = loss + extra_loss(x, logits, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach())
        n += 1
        if log_every and i % log_every == 0:
            print(f"    {i}/{len(loader)}  loss {total / n:.4f}  {(time.time() - t0) / i:.2f} s per batch")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    """Confusion matrix of model over every window in loader."""
    model.eval()
    cm = Confusion()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            logits = model(x)
        cm.add(logits.argmax(1), y)
    return cm


@torch.no_grad()
def predict(model, x, device, amp_dtype=None):
    """Class map (N, H, W) for a batch of images (N, 7, H, W) given as numpy or torch."""
    model.eval()
    x = torch.as_tensor(np.asarray(x)).to(device)
    with autocast(device, amp_dtype):
        return model(x).argmax(1).cpu().numpy()

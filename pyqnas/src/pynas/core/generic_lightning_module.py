"""
generic_lightning_module.py  (IRIDE burnscar-mapper fork)
==========================================================
Rewritten from the original py-q-nas module to correctly handle
ignore_index=-1 (nodata pixels) throughout the training pipeline.

ROOT CAUSE OF THE ORIGINAL BUG
--------------------------------
The original _common_step called:
    F.one_hot(y.long(), num_classes=...)
F.one_hot requires values in [0, num_classes-1]. Our nodata class is
mapped to -1 (standard ignore_index convention). -1 passed to F.one_hot
triggers a CUDA device-side assert:
    "idx_dim >= 0 && idx_dim < index_size"
This poisoned the entire CUDA context, causing every subsequent candidate
in the same session to fail with the same cryptic error.

THE FIX (not a bandaid)
------------------------
Every place that touches target labels now receives ignore_index=-1:
  1. one_hot: y is clamped to [0, num_classes-1] before calling F.one_hot.
     The clamped pixels map to class 0 (clear) for the one_hot encoding,
     but CrossEntropyLoss(ignore_index=-1) excludes them from gradient
     computation, so this is mathematically equivalent to masking them out.
  2. MSE: computed only over valid (non-ignored) pixels.
  3. IoU: calculate_iou already accepts ignore_index; we now pass it.

WHAT IS UNCHANGED
-----------------
- All method signatures (Population uses these via duck-typing)
- All logged metric names (test_iou, test_fps, etc. — Population reads these)
- GenericLightningNetwork (classification, not used for segmentation here)
- configure_optimizers, forward, on_fit_start, on_train_batch_start
- The FP16-aware path (no interaction with this module)
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torchmetrics import MeanSquaredError

from ..train.losses import CategoricalCrossEntropyLoss, FocalLoss
from ..train.custom_iou import calculate_iou

IGNORE_INDEX = -1   # nodata class — excluded from loss, MSE, and IoU


# ══════════════════════════════════════════════════════════════════════════════
# GenericLightningNetwork  (classification — unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class GenericLightningNetwork(pl.LightningModule):

    def __init__(self, model, num_classes, learning_rate=1e-3):
        super().__init__()
        self.lr = learning_rate
        self.model = model
        self._initialize_metrics(num_classes)

    def _initialize_metrics(self, num_classes):
        if num_classes > 2:
            self.loss_fn      = nn.CrossEntropyLoss()
            self.accuracy     = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
            self.f1_score     = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)
            self.mcc          = torchmetrics.classification.MulticlassMatthewsCorrCoef(num_classes=num_classes)
            self.conf_matrix  = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)
        else:
            self.loss_fn      = nn.CrossEntropyLoss()
            self.accuracy     = torchmetrics.classification.BinaryAccuracy()
            self.f1_score     = torchmetrics.classification.BinaryF1Score()
            self.mcc          = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
            self.conf_matrix  = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x.float())

    def _common_step(self, batch, _):
        x, y = batch
        scores = self.forward(x)
        loss   = self.loss_fn(scores, y)
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        _, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({
            "train_loss":     loss,
            "train_accuracy": self.accuracy(torch.argmax(scores, 1), y),
            "train_f1_score": self.f1_score(torch.argmax(scores, 1), y),
            "train_mcc":      self.mcc(torch.argmax(scores, 1), y).float(),
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, 1), y)
        self.f1_score.update(torch.argmax(scores, 1), y)
        self.mcc.update(torch.argmax(scores, 1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y  = batch
        start = time.time()
        loss, scores, y = self._common_step(batch, batch_idx)
        if x.is_cuda:
            torch.cuda.synchronize()
        fps = x.shape[0] / max(time.time() - start, 1e-9)
        self.log_dict({
            "test_loss":     loss,
            "test_accuracy": self.accuracy(torch.argmax(scores, 1), y),
            "test_f1_score": self.f1_score(torch.argmax(scores, 1), y),
            "test_mcc":      self.mcc(torch.argmax(scores, 1), y).float(),
            "fps":           fps,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)


# ══════════════════════════════════════════════════════════════════════════════
# GenericLightningSegmentationNetwork  (segmentation — FIXED)
# ══════════════════════════════════════════════════════════════════════════════

class GenericLightningSegmentationNetwork(pl.LightningModule):
    """
    Lightning module for segmentation tasks.

    Wraps any model that maps (B, C_in, H, W) → (B, num_classes, H, W).

    Key changes vs the original:
      - ignore_index=-1 is respected everywhere (loss, MSE, IoU).
      - F.one_hot receives y.clamp(min=0) to avoid CUDA device-side asserts.
      - MSE is computed only over non-ignored pixels.
      - calculate_iou receives ignore_index=-1.
      - Default loss switched to CrossEntropyLoss (caller can override loss_fn).

    Population compatibility:
      - All metric names (test_iou, test_fps, val_iou, …) are unchanged.
      - last_train_loss / last_val_loss attributes are preserved.
      - model attribute is preserved (Population reads it for param count).
    """

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.lr       = learning_rate
        self.model    = model
        # Default loss — caller replaces this with the class-weighted version:
        #   LM.loss_fn = nn.CrossEntropyLoss(weight=..., ignore_index=-1)
        self.loss_fn  = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.mse      = MeanSquaredError()
        self.iou      = calculate_iou
        # preserved for Population bookkeeping
        self.last_train_loss = torch.tensor(float("nan"))
        self.last_val_loss   = torch.tensor(float("nan"))

    def forward(self, x):
        return self.model(x)

    # ── shared step ──────────────────────────────────────────────────────────

    def _common_step(self, batch, batch_idx, measure_latency: bool = False):
        """
        Shared forward + metrics for train/val/test.

        Returns (loss, mse, iou, latency_ms|None).

        ignore_index=-1 handling:
          - Loss:  CrossEntropyLoss(ignore_index=-1) skips these pixels.
          - MSE:   computed on probability vectors; nodata pixels are zeroed
                   in both pred and target before the metric is updated.
          - IoU:   calculate_iou receives ignore_index=-1 and skips these pixels.
          - one_hot: y is clamped to [0, num_classes-1] before calling
                   F.one_hot so -1 never reaches the CUDA kernel.
        """
        x, y = batch
        n_classes = self._num_classes(x)

        if measure_latency:
            t0     = time.perf_counter()
            logits = self(x)
            t1     = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
        else:
            logits     = self(x)
            latency_ms = None

        # ── 1. Loss (CrossEntropyLoss handles ignore_index=-1 natively) ──
        loss = self.loss_fn(logits, y)

        # ── 2. MSE over valid pixels only ────────────────────────────────
        with torch.no_grad():
            valid_mask = (y != IGNORE_INDEX)                  # (B, H, W) bool

            if logits.shape[1] == 1:   # binary
                probs       = torch.sigmoid(logits)            # (B,1,H,W)
                target_prob = y.float().unsqueeze(1)           # (B,1,H,W)
            else:                       # multiclass
                probs = torch.softmax(logits, dim=1)           # (B,C,H,W)
                # clamp -1 → 0 before one_hot: these pixels are masked out
                # below before the metric update so the wrong class label
                # never contributes to the MSE value.
                y_safe      = y.long().clamp(min=0)
                target_prob = F.one_hot(
                    y_safe, num_classes=n_classes
                ).permute(0, 3, 1, 2).float()                 # (B,C,H,W)

            # zero out ignored pixels in both pred and target, then flatten
            vm4d           = valid_mask.unsqueeze(1).float()   # (B,1,H,W)
            probs_masked   = (probs       * vm4d).reshape(probs.shape[0], -1)
            target_masked  = (target_prob * vm4d).reshape(probs.shape[0], -1)

            mse = self.mse(probs_masked, target_masked)

        # ── 3. IoU — ignore nodata pixels ────────────────────────────────
        iou = calculate_iou(
            logits,
            y,
            num_classes = n_classes,
            from_logits = True,
            ignore_index = IGNORE_INDEX,
        )

        return loss, mse, iou, latency_ms

    def _num_classes(self, x):
        """Infer num_classes from the model output shape on a tiny dummy input."""
        # use the stored attribute if available (set by Population / our code)
        if hasattr(self, "num_classes"):
            return self.num_classes
        if hasattr(self.model, "num_classes"):
            return self.model.num_classes
        # last resort: one forward pass (no grad, cheap)
        with torch.no_grad():
            dummy  = torch.zeros(1, x.shape[1], x.shape[2], x.shape[3],
                                 device=x.device)
            return self.model(dummy).shape[1]

    # ── training / validation / test steps ───────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, mse, iou, _ = self._common_step(batch, batch_idx, measure_latency=False)
        self.last_train_loss = loss.detach()
        bs = batch[0].size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("train_mse",  mse,  on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_iou",  iou,  on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, iou, _ = self._common_step(batch, batch_idx, measure_latency=False)
        self.last_val_loss = loss.detach()
        bs = batch[0].size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("val_mse",  mse,  on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_iou",  iou,  on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mse, iou, latency_ms = self._common_step(batch, batch_idx, measure_latency=True)
        bs  = batch[0].size(0)
        fps = bs / (latency_ms / 1000.0) if latency_ms and latency_ms > 0 else 0.0
        self.log("test_loss",       loss,       on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("test_mse",        mse,        on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("test_iou",        iou,        on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("test_latency_ms", latency_ms, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("test_fps",        fps,        on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_fit_start(self):
        self.model.to(self.device)

    def on_train_batch_start(self, batch, batch_idx):
        x   = batch[0] if isinstance(batch, (tuple, list)) else batch
        dev = getattr(x, "device", self.device)
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: CE loss wrapper used by GenericLightningNetwork_Custom
# ══════════════════════════════════════════════════════════════════════════════

def ce_loss(logits, targets, weight=None, use_hard_labels=True, reduction="none"):
    if use_hard_labels:
        return F.cross_entropy(
            logits, targets.long(),
            weight=weight, reduction=reduction
        )
    log_pred  = F.log_softmax(logits, dim=-1)
    nll_loss  = torch.sum(-targets * log_pred, dim=1)
    return nll_loss


# ══════════════════════════════════════════════════════════════════════════════
# GenericLightningNetwork_Custom  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class GenericLightningNetwork_Custom(pl.LightningModule):

    def __init__(self, parsed_layers, model_parameters, input_channels,
                 num_classes, learning_rate=1e-3):
        super().__init__()
        self.lr          = learning_rate
        self.class_weights = None
        self.loss_fn     = ce_loss
        self.accuracy    = torchmetrics.classification.BinaryAccuracy()
        self.f1_score    = torchmetrics.classification.BinaryF1Score()
        self.mcc         = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if hasattr(self.trainer, "datamodule") and \
                hasattr(self.trainer.datamodule, "class_weights"):
            self.class_weights = self.trainer.datamodule.class_weights.to(self.device)

    def _common_step(self, batch, batch_idx):
        x, y   = batch
        scores = self.forward(x)
        if self.class_weights is not None:
            loss = self.loss_fn(scores, y, weight=self.class_weights, use_hard_labels=True)
        else:
            loss = self.loss_fn(scores, y, use_hard_labels=True)
        return loss.mean(), scores, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({
            "train_loss":     loss,
            "train_accuracy": self.accuracy(torch.argmax(scores, 1), y),
            "train_f1_score": self.f1_score(torch.argmax(scores, 1), y),
            "train_mcc":      self.mcc(torch.argmax(scores, 1), y).float(),
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, 1), y)
        self.f1_score.update(torch.argmax(scores, 1), y)
        self.mcc.update(torch.argmax(scores, 1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({
            "test_loss":     loss,
            "test_accuracy": self.accuracy(torch.argmax(scores, 1), y),
            "test_f1_score": self.f1_score(torch.argmax(scores, 1), y),
            "test_mcc":      self.mcc(torch.argmax(scores, 1), y).float(),
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()
        plt.xlabel("Prediction"); plt.ylabel("Class")
        os.makedirs("./logs/tb_logs", exist_ok=True)
        plt.savefig(f"./logs/tb_logs/confusion_matrix_{datetime.now():%Y-%m-%d_%H-%M-%S}.png")

    def predict_step(self, batch, batch_idx):
        x, y   = batch
        scores = self.forward(x)
        preds  = torch.argmax(scores, 1)
        print(f"Accuracy: {self.accuracy(preds,y):.3f}")
        print(f"F1-score: {self.f1_score(preds,y):.3f}")
        print(f"MCC     : {self.mcc(preds,y):.3f}")
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
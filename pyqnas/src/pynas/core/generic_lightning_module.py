import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError

import os 
from datetime import datetime
import matplotlib.pyplot as plt

from ..train.losses import CategoricalCrossEntropyLoss, FocalLoss
from ..train.custom_iou import calculate_iou


class GenericLightningNetwork(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork, self).__init__()
        self.lr = learning_rate
        self.model = model
        self._initialize_metrics(num_classes)



    def _initialize_metrics(self, num_classes):
        # Metrics
        if num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss()
            self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
            self.f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)
            self.mcc = torchmetrics.classification.MulticlassMatthewsCorrCoef(num_classes=num_classes)
            self.conf_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)
            self.conf_matrix_pred = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.accuracy = torchmetrics.classification.BinaryAccuracy()
            self.f1_score = torchmetrics.classification.BinaryF1Score()
            self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
            self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
            self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        _, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        import time
        x, y = batch

        start_time = time.time()
        loss, scores, y = self._common_step(batch, batch_idx)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        fps = x.shape[0] / elapsed_time if elapsed_time > 0 else 0.0

        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
            'fps': fps,
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss


    def on_test_end(self):
        self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(rf"./logs/tb_logs", exist_ok=True)
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        plt.show()

    def _common_step(self, batch, _):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, _):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")   
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer


class GenericLightningSegmentationNetwork(pl.LightningModule):
    """
    GenericLightningSegmentationNetwork is a PyTorch Lightning module designed for segmentation tasks. 
    It wraps a given model and provides training, validation, testing, and prediction steps, 
    along with logging for loss, mean squared error (MSE), and intersection over union (IoU).
    """
    def __init__(self, model, learning_rate=1e-3):
        super(GenericLightningSegmentationNetwork, self).__init__()
        self.lr = learning_rate
        self.model = model
        
        #self.loss_fn = CategoricalCrossEntropyLoss()
        self.loss_fn = FocalLoss()
        self.mse = MeanSquaredError()
        self.iou = calculate_iou

        # optional: initialize these so they always exist
        self.last_train_loss = torch.tensor(float("nan"))
        self.last_val_loss = torch.tensor(float("nan"))

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, measure_latency: bool = False):
        """
        Shared computation for train/val/test.

        If measure_latency=True, it also measures forward-pass latency in ms
        and returns it as the 4th value; otherwise latency_ms is None.
        """
        x, y = batch

        if measure_latency:
            t0 = time.perf_counter()
            logits = self(x)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
        else:
            logits = self(x)
            latency_ms = None

        # --- loss on logits (unchanged) ---
        loss = self.loss_fn(logits, y)

        # --- probabilities for MSE parity ---
        if logits.shape[1] == 1:  # binary
            probs = torch.sigmoid(logits)
            if y.dtype.is_floating_point and y.shape == probs.shape:
                target_prob = y
            else:
                target_prob = y.float().unsqueeze(1)
        else:                     # multiclass
            probs = torch.softmax(logits, dim=1)
            if y.dtype.is_floating_point and y.shape == probs.shape:
                target_prob = y
            else:
                target_prob = torch.nn.functional.one_hot(
                    y.long(), num_classes=logits.shape[1]
                ).permute(0, 3, 1, 2).float()

        # MSE in the same domain (probs vs one-hot/prob targets)
        mse = self.mse(
            probs.reshape(probs.size(0), -1),
            target_prob.reshape(target_prob.size(0), -1)
        )

        # IoU with canonical function; from_logits=True to match eval
        iou = calculate_iou(
            logits,
            y,
            num_classes=getattr(self, "num_classes", self.model.num_classes),
            from_logits=True,
        )

        return loss, mse, iou, latency_ms

    def training_step(self, batch, batch_idx):
        # we don’t need latency in training
        loss, mse, iou, _ = self._common_step(batch, batch_idx, measure_latency=False)

        # remember last train loss for Population
        self.last_train_loss = loss.detach()

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        bs = x.size(0)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_mse", mse,  on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_iou", iou,  on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, iou, _ = self._common_step(batch, batch_idx, measure_latency=False)

        # remember last val loss for Population
        self.last_val_loss = loss.detach()

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        bs = x.size(0)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("val_mse",  mse,  on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_iou",  iou,  on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        # here we DO want latency
        loss, mse, iou, latency_ms = self._common_step(batch, batch_idx, measure_latency=True)

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        bs = x.size(0)

        if latency_ms is not None and latency_ms > 0:
            fps = bs / (latency_ms / 1000.0)
        else:
            fps = 0.0

        # log metrics that Population expects:
        self.log("test_loss",       loss,       on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("test_mse",        mse,        on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("test_iou",        iou,        on_step=False, on_epoch=True, prog_bar=True,  batch_size=bs)
        self.log("test_latency_ms", latency_ms, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("test_fps",        fps,        on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_fit_start(self):
        # ensure inner model is on the trainer device at the start of fit
        self.model.to(self.device)
    
    def on_train_batch_start(self, batch, batch_idx):
        # batch is typically (x, y)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        dev = getattr(x, "device", self.device)
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)



class GenericLightningNetwork_Custom(pl.LightningModule):
    def __init__(self, parsed_layers, model_parameters, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork_Custom, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            model_parameters=model_parameters,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        self.class_weights = None  # Initialize with a default value

        # Metrics
        self.loss_fn = ce_loss  # Use custom loss function
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1_score = torchmetrics.classification.BinaryF1Score()
        self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
        self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # Ensure the datamodule is attached and has class_weights
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'class_weights'):
            self.class_weights = self.trainer.datamodule.class_weights.to(self.device)
            print(f"GenericLightningNetwork class_weights set: {self.class_weights}")
        else:
            print("GenericLightningNetwork class_weights NOT set")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        # plt.show()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        if self.class_weights is not None:
            loss = self.loss_fn(logits=scores, targets=y, weight=self.class_weights, use_hard_labels=True)
        else:
            loss = self.loss_fn(logits=scores, targets=y, use_hard_labels=True)

        loss = loss.mean()
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer


def ce_loss(logits, targets, weight=None, use_hard_labels=True, reduction="none"):
    """
    Wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        weight: weights for loss if hard labels are used.
        use_hard_labels: If True, targets have [Batch size] shape with int values.
                         If False, the target is vector. Default to True.
    """
    if use_hard_labels:
        if weight is not None:
            return F.cross_entropy(
                logits, targets.long(), weight=weight, reduction=reduction
            )
        else:
            return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


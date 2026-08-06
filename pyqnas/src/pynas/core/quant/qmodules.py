"""
Quantized module wrappers built on the hand-written fake-quant primitives.

Design principle: quantize what the COMPILER quantizes, where it quantizes it.
Edge compilers (Hailo DFC included) fuse conv+BN(+activation) before assigning
quantization parameters, so simulating quantization on the *unfused* conv
weights trains the network against noise the deployed graph never sees.
QuantConv2d therefore folds a tracked BatchNorm into the conv weights each
forward pass BEFORE fake-quantizing them.

Wrappers:
  QuantConv2d  : wraps an existing nn.Conv2d (+ optional nn.BatchNorm2d),
                 weight fake-quant (symmetric int) + input activation
                 fake-quant (asymmetric uint).
  QuantLinear  : same for nn.Linear.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import QuantOpts, fake_quant_weight, ActFakeQuant


class QuantConv2d(nn.Module):
    """Fake-quantized Conv2d with optional BN folding.

    Wraps the ORIGINAL conv (and BN) modules rather than copying parameters,
    so the optimizer keeps updating the original weights and the wrapper stays
    removable (see prepare.remove_quant).
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: Optional[nn.BatchNorm2d],
        opts: QuantOpts,
    ):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.opts = opts
        self.act_quant = ActFakeQuant(opts)

    def _folded_weight_bias(self):
        """Fold BN statistics into conv weight/bias (training-time folding).

        Standard BN folding:
            w_fold = w * gamma / sqrt(var + eps)
            b_fold = (b - mean) * gamma / sqrt(var + eps) + beta

        During training we fold with the BN's RUNNING statistics, which is the
        deployment-time arithmetic (what the compiler bakes in). BN's batch
        statistics still update through the untouched self.bn module ONLY via
        its running buffers; we do not run BN as a separate op in the forward.
        """
        w = self.conv.weight
        b = self.conv.bias if self.conv.bias is not None else torch.zeros(
            w.shape[0], device=w.device, dtype=w.dtype
        )
        if self.bn is None:
            return w, b

        gamma = self.bn.weight
        beta = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var
        eps = self.bn.eps

        std = torch.sqrt(var + eps)
        factor = gamma / std  # (C_out,)
        w_fold = w * factor.reshape(-1, 1, 1, 1)
        b_fold = (b - mean) * factor + beta
        return w_fold, b_fold

    @torch.no_grad()
    def _update_bn_stats(self, x: torch.Tensor) -> None:
        """Keep BN running stats updating during training.

        Since we bypass the BN op in the folded forward, we run the raw conv
        under no_grad and let the BN module observe its output distribution.
        This costs one extra conv per step but keeps folding consistent with
        what a separately-trained conv+BN pair would converge to.
        """
        if self.bn is not None and self.training:
            raw = F.conv2d(
                x, self.conv.weight, self.conv.bias,
                self.conv.stride, self.conv.padding,
                self.conv.dilation, self.conv.groups,
            )
            self.bn(raw)  # updates running_mean / running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) quantize the incoming activation (uint8 asymmetric)
        x = self.act_quant(x)

        # 2) keep BN statistics alive
        self._update_bn_stats(x)

        # 3) fold BN into weights, then quantize folded weights (int8 symmetric,
        #    per-channel) — mirroring compiler fusion-then-quantize order.
        w_fold, b_fold = self._folded_weight_bias()
        w_q = fake_quant_weight(w_fold, self.opts)

        # NOTE: bias is intentionally NOT fake-quantized to int8. Integer
        # hardware keeps biases in wider accumulators (int32 on Hailo-class
        # NPUs), whose rounding error is negligible at this simulation level.
        return F.conv2d(
            x, w_q, b_fold,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


class QuantLinear(nn.Module):
    """Fake-quantized Linear (no BN folding — rare after linear in these nets)."""

    def __init__(self, linear: nn.Linear, opts: QuantOpts):
        super().__init__()
        self.linear = linear
        self.opts = opts
        self.act_quant = ActFakeQuant(opts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_quant(x)
        w_q = fake_quant_weight(self.linear.weight, self.opts)
        return F.linear(x, w_q, self.linear.bias)
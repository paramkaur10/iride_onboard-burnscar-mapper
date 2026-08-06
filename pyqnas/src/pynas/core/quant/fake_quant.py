"""
Hand-written fake quantization primitives (no torch.ao).

Implements the Hailo-8 quantization contract as a differentiable GPU simulation:

  * Weights:     signed int, SYMMETRIC (zero_point = 0), per-channel or per-tensor,
                 bit-width 4/8/16 (Hailo DFC supports all three).
  * Activations: unsigned int, ASYMMETRIC (zero_point in [0, 2^b - 1]),
                 per-tensor, bit-width 8/16.

Every scheme decision is a parameter so alternatives (pow2 scales, per-tensor
weights, symmetric activations) can be ablated against the same training loop.

Gradient flow uses the straight-through estimator (STE): the forward pass
applies real round/clamp; the backward pass treats quantization as identity
inside the representable range and zero outside it (clamped STE).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# STE primitives
# ----------------------------------------------------------------------------

class _RoundSTE(torch.autograd.Function):
    """round() with identity gradient."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


class _ClampSTE(torch.autograd.Function):
    """clamp() with zeroed gradient outside the clamp range (clamped STE).

    Zeroing (rather than passing) the out-of-range gradient is what lets the
    network learn to pull values back inside the representable range instead
    of oscillating: values pinned at the clamp edge receive no push further out.
    """

    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward((x >= lo) & (x <= hi))
        return torch.clamp(x, lo, hi)

    @staticmethod
    def backward(ctx, g):
        (inside,) = ctx.saved_tensors
        return g * inside, None, None


def clamp_ste(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return _ClampSTE.apply(x, lo, hi)


# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class QuantOpts:
    """Scheme configuration. Defaults = Hailo-8 basic scheme (W8A8)."""

    # ---- weights ----
    weight_bits: int = 8              # Hailo supports 4 / 8 / 16
    weight_per_channel: bool = True   # per-output-channel scales (dim 0)
    weight_pow2_scale: bool = False   # constrain scales to powers of two (ablation)
    # symmetric signed range: [-(2^(b-1) - 1), +(2^(b-1) - 1)], e.g. [-127, 127].
    # Narrow range (dropping -128) keeps the grid symmetric around 0, matching
    # symmetric-scheme hardware conventions.

    # ---- activations ----
    act_bits: int = 8                 # Hailo supports 8 / 16
    act_unsigned: bool = True         # Hailo activations are uint => range [0, 2^b - 1]
    act_symmetric: bool = False       # ablation switch (True => signed symmetric acts)
    act_pow2_scale: bool = False
    act_ema_decay: float = 0.99       # EMA for running min/max range observation

    # ---- training dynamics ----
    freeze_ranges_after: Optional[int] = None  # batches after which observers freeze
                                               # (None = never freeze automatically)


# ----------------------------------------------------------------------------
# Weight quantization (stateless: recomputed from live weights each forward)
# ----------------------------------------------------------------------------

def _maybe_pow2(scale: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Optionally snap scales to the nearest power of two (in log2 space)."""
    if not enabled:
        return scale
    # guard against zero scales before log
    scale = torch.clamp(scale, min=1e-12)
    return torch.pow(2.0, torch.round(torch.log2(scale)))


def fake_quant_weight(w: torch.Tensor, opts: QuantOpts) -> torch.Tensor:
    """Symmetric signed fake quantization of a weight tensor.

    Per-channel: one scale per output channel (dim 0), which is Hailo's (and
    most compilers') convention for conv/linear weights.
    """
    qmax = 2 ** (opts.weight_bits - 1) - 1  # e.g. 127 for 8-bit

    if opts.weight_per_channel and w.dim() > 1:
        # max |w| over all dims except 0, keepdim for broadcast
        reduce_dims = tuple(range(1, w.dim()))
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        max_abs = w.abs().amax()

    scale = max_abs / qmax
    scale = torch.clamp(scale, min=1e-12)
    scale = _maybe_pow2(scale, opts.weight_pow2_scale)

    w_int = clamp_ste(round_ste(w / scale), -qmax, qmax)
    return w_int * scale


# ----------------------------------------------------------------------------
# Activation quantization (stateful: EMA range observer)
# ----------------------------------------------------------------------------

class ActFakeQuant(nn.Module):
    """Asymmetric unsigned fake quantization for activations (Hailo scheme).

    Tracks running min/max with EMA during training; uses frozen ranges for
    eval. zero_point is an integer so that real 0.0 is exactly representable
    (required for zero-padding correctness on integer hardware).
    """

    def __init__(self, opts: QuantOpts):
        super().__init__()
        self.opts = opts
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("num_batches", torch.tensor(0))
        self.frozen = False

    @torch.no_grad()
    def _observe(self, x: torch.Tensor) -> None:
        x_min = x.amin()
        x_max = x.amax()
        if not bool(self.initialized):
            self.running_min.copy_(x_min.to(self.running_min.device))
            self.running_max.copy_(x_max.to(self.running_max.device))
            self.initialized.fill_(True)
        else:
            d = self.opts.act_ema_decay
            self.running_min.mul_(d).add_(x_min.to(self.running_min.device) * (1 - d))
            self.running_max.mul_(d).add_(x_max.to(self.running_max.device) * (1 - d))
        self.num_batches += 1
        if (
            self.opts.freeze_ranges_after is not None
            and int(self.num_batches) >= self.opts.freeze_ranges_after
        ):
            self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.frozen:
            self._observe(x)
        if not bool(self.initialized):
            return x  # first eval call before any observation: pass through

        if self.opts.act_symmetric:
            # signed symmetric (ablation mode)
            qmax = 2 ** (self.opts.act_bits - 1) - 1
            max_abs = torch.maximum(self.running_max.abs(), self.running_min.abs())
            scale = torch.clamp(max_abs / qmax, min=1e-12)
            scale = _maybe_pow2(scale, self.opts.act_pow2_scale)
            x_int = clamp_ste(round_ste(x / scale), -qmax, qmax)
            return x_int * scale

        # unsigned asymmetric (Hailo default)
        qmax = 2 ** self.opts.act_bits - 1  # 255 for uint8
        r_min = torch.minimum(self.running_min, torch.tensor(0.0, device=x.device))
        r_max = torch.maximum(self.running_max, torch.tensor(0.0, device=x.device))
        scale = torch.clamp((r_max - r_min) / qmax, min=1e-12)
        scale = _maybe_pow2(scale, self.opts.act_pow2_scale)
        # integer zero point so that fp 0.0 is exactly representable
        zero_point = round_ste(-r_min / scale)
        zero_point = torch.clamp(zero_point, 0, qmax)

        x_int = clamp_ste(round_ste(x / scale) + zero_point, 0, qmax)
        return (x_int - zero_point) * scale
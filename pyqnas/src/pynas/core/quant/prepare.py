"""
Model surgery: install / remove hand-written fake quantization.

Contract mirrors qat_utils.prepare_fp16_aware so population.py integration is
a one-branch addition:

    ctx = prepare_hailo_quant(model, opts)   # swaps modules in-place
    ... train ...
    ctx.close()                              # optional: restore original modules

Conv+BN fusion detection: we walk each nn.Sequential-like container in order
and pair every nn.Conv2d with an IMMEDIATELY FOLLOWING nn.BatchNorm2d. The BN
module is replaced by nn.Identity (its arithmetic now lives inside the folded
QuantConv2d), matching the fused graph the compiler deploys.
"""

from __future__ import annotations

import configparser
import dataclasses
from typing import List, Optional, Tuple

import torch.nn as nn

from .fake_quant import QuantOpts
from .qmodules import QuantConv2d, QuantLinear


# ----------------------------------------------------------------------------
# Config reading
# ----------------------------------------------------------------------------

def read_quant_opts(cfg: configparser.ConfigParser) -> QuantOpts:
    """Build QuantOpts from the [QuantSim] section, with safe fallbacks.

    Example config:

        [QuantSim]
        weight_bits = 8
        weight_per_channel = true
        weight_pow2_scale = false
        act_bits = 8
        act_unsigned = true
        act_symmetric = false
        act_pow2_scale = false
        act_ema_decay = 0.99
        freeze_ranges_after = 200
    """
    s = "QuantSim"
    g = cfg.getint
    gb = cfg.getboolean
    gf = cfg.getfloat

    freeze = cfg.get(s, "freeze_ranges_after", fallback="").strip()
    return QuantOpts(
        weight_bits=g(s, "weight_bits", fallback=8),
        weight_per_channel=gb(s, "weight_per_channel", fallback=True),
        weight_pow2_scale=gb(s, "weight_pow2_scale", fallback=False),
        act_bits=g(s, "act_bits", fallback=8),
        act_unsigned=gb(s, "act_unsigned", fallback=True),
        act_symmetric=gb(s, "act_symmetric", fallback=False),
        act_pow2_scale=gb(s, "act_pow2_scale", fallback=False),
        act_ema_decay=gf(s, "act_ema_decay", fallback=0.99),
        freeze_ranges_after=int(freeze) if freeze else None,
    )


# ----------------------------------------------------------------------------
# Surgery
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class _Swap:
    parent: nn.Module
    conv_name: str
    original_conv: nn.Module
    bn_name: Optional[str]
    original_bn: Optional[nn.Module]


class QuantContext:
    """Handle for removing the quantization wrappers (restores originals)."""

    def __init__(self, model: nn.Module, swaps: List[_Swap]):
        self._model = model
        self._swaps = swaps

    @property
    def num_quantized(self) -> int:
        return len(self._swaps)

    def close(self) -> None:
        for s in self._swaps:
            setattr(s.parent, s.conv_name, s.original_conv)
            if s.bn_name is not None:
                setattr(s.parent, s.bn_name, s.original_bn)
        self._swaps.clear()


def _ordered_children(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    return list(module.named_children())


def prepare_hailo_quant(model: nn.Module, opts: QuantOpts) -> QuantContext:
    """Swap Conv2d(+BN2d) and Linear modules for fake-quantized wrappers.

    Walks the tree recursively. Within each container, a BatchNorm2d directly
    following a Conv2d is treated as fused into it (compiler behavior): the
    conv is wrapped WITH the BN, and the BN slot becomes Identity.
    """
    swaps: List[_Swap] = []
    _prepare_recursive(model, opts, swaps)
    return QuantContext(model, swaps)


def _prepare_recursive(module: nn.Module, opts: QuantOpts, swaps: List[_Swap]) -> None:
    children = _ordered_children(module)
    consumed_bn = set()

    for i, (name, child) in enumerate(children):
        if isinstance(child, QuantConv2d) or isinstance(child, QuantLinear):
            continue  # already prepared

        if isinstance(child, nn.Conv2d):
            # look ahead for an immediately-following BN in the same container
            bn_name, bn_mod = None, None
            if i + 1 < len(children):
                nxt_name, nxt = children[i + 1]
                if isinstance(nxt, nn.BatchNorm2d):
                    bn_name, bn_mod = nxt_name, nxt

            wrapper = QuantConv2d(child, bn_mod, opts)
            setattr(module, name, wrapper)
            if bn_name is not None:
                setattr(module, bn_name, nn.Identity())
                consumed_bn.add(bn_name)
            swaps.append(_Swap(module, name, child, bn_name, bn_mod))

        elif isinstance(child, nn.Linear):
            wrapper = QuantLinear(child, opts)
            setattr(module, name, wrapper)
            swaps.append(_Swap(module, name, child, None, None))

        elif isinstance(child, nn.BatchNorm2d) and name in consumed_bn:
            continue  # already replaced with Identity above

        else:
            _prepare_recursive(child, opts, swaps)


def freeze_all_ranges(model: nn.Module) -> int:
    """Manually freeze every activation observer (e.g. for final finetune phase).

    Returns the number of observers frozen.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "act_quant"):
            m.act_quant.frozen = True
            n += 1
    return n
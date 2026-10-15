"""Pretrained encoders of segmentation_models_pytorch (smp), cut after a number of stages.

smp adapts the first convolution of the pretrained weights to any number of input bands. What an encoder holds
but its returned feature maps do not need is removed after building: the stages past the cut, which smp keeps,
and parts some encoders compute without returning them. Each removal is checked: the feature maps of a random
input must stay exactly the same, or the encoder is kept whole. Parameter counts, sizes and latencies are then
those of what runs.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _weights(value):
    v = (value or "").strip()
    return None if v.lower() in ("", "none", "null", "false", "random") else v


class Encoder(nn.Module):
    """smp encoder `name` cut after `depth` stages; forward gives the feature maps at strides 2 .. 2^depth."""

    def __init__(self, name, in_channels, depth, weights="imagenet", input_hw=(256, 256)):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.name, self.depth, self.in_channels = name, int(depth), int(in_channels)
        self.encoder = smp.encoders.get_encoder(name, in_channels=self.in_channels, depth=self.depth,
                                                weights=_weights(weights))
        self.out_channels = [int(c) for c in self.encoder.out_channels[1:]]
        self.trimmed = self._remove_unused(input_hw)

    def forward(self, x):
        return list(self.encoder(x)[1:])

    def _remove_unused(self, input_hw):
        """Replace by a pass-through (1) submodules a forward pass never runs and (2) submodules whose parameters
        get no gradient from the returned feature maps. Returns True when the result passed the check."""
        enc = self.encoder
        was_training = enc.training
        enc.eval()
        x = torch.randn(1, self.in_channels, *input_hw)
        backup = copy.deepcopy(enc)
        with torch.no_grad():
            before = [f.clone() for f in enc(x)]

        def leaves(m):
            return [c for c in m.modules() if not list(c.children())]

        def prune(parent, keep):
            for name, child in list(parent.named_children()):
                if not keep(child):
                    setattr(parent, name, PassThrough())
                elif list(child.children()):
                    prune(child, keep)

        ok = True
        try:
            ran = set()
            hooks = [m.register_forward_hook(lambda mod, i, o: ran.add(id(mod))) for m in leaves(enc)]
            try:
                with torch.no_grad():
                    enc(x)
            finally:
                for h in hooks:
                    h.remove()
            prune(enc, lambda m: any(id(c) in ran for c in leaves(m)))
            flags = {id(p): p.requires_grad for p in enc.parameters()}
            for p in enc.parameters():
                p.requires_grad_(True)
                p.grad = None
            with torch.enable_grad():
                sum(f.float().sum() for f in enc(x)[1:]).backward()
            fed = lambda m: (not list(m.parameters())) or any(p.grad is not None for p in m.parameters())  # noqa
            prune(enc, fed)
            for p in enc.parameters():
                p.requires_grad_(flags.get(id(p), True))
                p.grad = None
            with torch.no_grad():
                after = enc(x)
            if len(after) != len(before) or any(a.shape != b.shape or not torch.allclose(a, b, rtol=1e-4, atol=1e-5)
                                                for a, b in zip(after, before)):
                raise RuntimeError("feature maps changed")
        except Exception:                                    # noqa: BLE001 - keep the encoder whole
            enc, ok = backup, False
        for p in enc.parameters():
            p.grad = None
        self.encoder = enc
        self.encoder.train(was_training)
        return ok


class PassThrough(nn.Module):
    """Stands in for a part of an encoder that its feature maps do not need."""

    def forward(self, x, *args, **kwargs):
        return x


def prefetch(names, in_channels=3, weights="imagenet"):
    """Build each encoder once with its pretrained weights, so that they are downloaded before parallel workers
    start (several processes downloading the same file can leave it broken). Returns {name: 'ok' or error}."""
    out = {}
    for n in names:
        try:
            import segmentation_models_pytorch as smp
            smp.encoders.get_encoder(n, in_channels=in_channels, depth=5, weights=_weights(weights))
            out[n] = "ok"
        except Exception as e:                               # noqa: BLE001 - reported
            out[n] = f"{type(e).__name__}: {e}"
    return out

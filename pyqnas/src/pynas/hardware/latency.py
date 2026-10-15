"""Predicted Myriad X latency of a network, from nn-Meter's kernel predictors ([Hardware] latency = nnmeter).

nn-Meter (Microsoft Research, MobiSys 2021) splits a model into the kernels the device runs (a convolution
with its batch norm and activation is one kernel) and predicts each kernel's time with a model fitted on
kernels timed on a Neural Compute Stick 2 (Myriad X). Its own ONNX converter stops on U-Nets (resizing), so
kernel_units() lists the kernels of any PyTorch network from one forward pass with hooks, in nn-Meter's format,
and tools/nnmeter_kernels.py (in [Hardware] nnmeter_env) predicts them.

As in nn-Meter: a conv (with its batch norm and ReLU) is one conv-bn-relu kernel, a depthwise conv one
dwconv-bn-relu; GELU, SiLU, Hardswish and the like add an hswish kernel; squeeze-and-excitation is one se
kernel; pooling is maxpool / avgpool; a residual addition is timed as add-relu; concatenations are left out;
bilinear upsampling is timed as an add-relu on the upsampled tensor and nearest as a relu (about as much
memory moved). The numbers are for ranking candidates, not a measurement.
"""
from __future__ import annotations

import json
import os
import tempfile

from .myriad import env_python, nnmeter_env, run, TOOLS, _records

_SE = {"seblock", "squeezeexcite", "squeezeexcitation", "semodule", "effectivesqueezeexcite", "selayer"}
_RESIDUAL = {"mbconv", "resblock", "resnetblock", "basicblock", "bottleneck", "invertedresidual", "mbconvblock",
             "depthwiseseparableconv", "edgeresidual"}
_SWISH_LIKE = ("gelu", "silu", "swish", "hardswish", "mish", "hardsigmoid")


def kernel_units(model, input_shape):
    """nn-Meter kernel units of a model for one input of shape (bands, H, W) (NHWC sizes, as nn-Meter)."""
    import torch
    import torch.nn as nn
    units = []
    se_inside = set()
    for m in model.modules():
        if type(m).__name__.lower() in _SE:
            se_inside.update(id(c) for c in m.modules() if c is not m)
    t = lambda x: [1, int(x.shape[2]), int(x.shape[3]), int(x.shape[1])]  # noqa: E731

    def hook(m, inp, out):
        if id(m) in se_inside or not isinstance(out, torch.Tensor) or out.dim() != 4:
            return
        x = inp[0] if inp and isinstance(inp[0], torch.Tensor) else None
        name = type(m).__name__.lower()
        leaf = not list(m.children())
        if isinstance(m, nn.Conv2d) and x is not None:
            dw = m.groups > 1 and m.groups == m.in_channels
            units.append(dict(op="dwconv-bn-relu" if dw else "conv-bn-relu", inputh=int(x.shape[2]),
                              inputw=int(x.shape[3]), cin=int(m.in_channels), cout=int(m.out_channels),
                              ks=list(m.kernel_size), strides=list(m.stride)))
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)) and x is not None:
            ks = m.kernel_size if isinstance(m.kernel_size, (tuple, list)) else (m.kernel_size, m.kernel_size)
            st = m.stride if isinstance(m.stride, (tuple, list)) else (m.stride, m.stride)
            units.append(dict(op="maxpool" if isinstance(m, nn.MaxPool2d) else "avgpool", inputh=int(x.shape[2]),
                              inputw=int(x.shape[3]), cin=int(x.shape[1]), cout=int(out.shape[1]),
                              ks=list(ks), strides=list(st)))
        elif name in _SE and x is not None:
            units.append(dict(op="se", input_tensors=[t(x)]))
        elif name in _RESIDUAL and x is not None and tuple(x.shape) == tuple(out.shape):
            units.append(dict(op="add-relu", input_tensors=[t(out), t(out)]))
        elif isinstance(m, nn.Upsample) or name == "upsample":
            if (getattr(m, "mode", "bilinear") or "nearest") == "nearest":
                units.append(dict(op="relu", input_tensors=[t(out)]))
            else:
                units.append(dict(op="add-relu", input_tensors=[t(out), t(out)]))
        elif leaf and any(k in name for k in _SWISH_LIKE):
            units.append(dict(op="hswish", inputh=int(out.shape[2]), cin=int(out.shape[1])))

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    was_training = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
        with torch.no_grad():
            model(torch.zeros(1, *input_shape, device=dev))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return units


def kernel_macs(units):
    """Multiply-adds of the conv kernels in a list of kernel units."""
    m = 0
    for u in units:
        if u["op"] in ("conv-bn-relu", "dwconv-bn-relu"):
            ho = u["inputh"] // u["strides"][0]
            wo = u.get("inputw", u["inputh"]) // u["strides"][1]
            k = u["ks"][0] * u["ks"][1]
            m += ho * wo * k * u["cout"] * (1 if u["op"].startswith("dw") else u["cin"])
    return m


def predict(models, cfg, breakdown=False, hw_max=224, timeout=3600):
    """{name: kernel units} -> {name: record with latency_ms} (tools/nnmeter_kernels.py in [Hardware]
    nnmeter_env; predictors fitted up to 224 x 224, larger maps scaled by their area)."""
    py = env_python(nnmeter_env(cfg))
    if not py.exists():
        raise RuntimeError(f"the nn-Meter environment is missing ({py}); run pynas.hardware.myriad.setup(cfg)")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(models, f)
    try:
        cmd = [py, TOOLS / "nnmeter_kernels.py", f.name, "--hw-max", str(hw_max or 0)]
        p = run(cmd + (["--breakdown"] if breakdown else []), timeout=timeout, show=0)
    finally:
        os.unlink(f.name)
    recs = _records(p.stdout)
    if not recs:
        print((p.stdout + p.stderr)[-3000:])
    return {r["model"]: r for r in recs}


def latency_ms(models, input_shape, cfg):
    """{name: model} -> {name: predicted Myriad X ms per forward pass, or None}."""
    units = {n: kernel_units(m, input_shape) for n, m in models.items()}
    recs = predict(units, cfg)
    return {n: (recs.get(n, {}).get("latency_ms") if recs.get(n, {}).get("ok") else None) for n in models}

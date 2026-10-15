"""The network as the Myriad X runs it: BatchNorm folded into the convolution before it, in FP16.

The OpenVINO Model Optimizer folds every BatchNorm whose input is the output of a convolution into that
convolution (weights x gamma / sqrt(var + eps), bias (b - mean) x gamma / sqrt(var + eps) + beta). fold() does
the same in PyTorch, on a copy, and checks that the outputs of the network are unchanged (eval mode, FP32); a
pair that would change them is left unfolded. The folded network is the one trained in the last phase of an
onboard model's training and the one exported, so what is trained, scored and flown is the same network.

    folded, pairs = fold(model, example)           # pairs: [(conv name, BatchNorm name), ...]
    fold_structure(fresh_model, pairs)             # the same layout without values, to load folded weights
    m16 = deploy_copy(model, example, half=True)   # folded (if not yet), eval mode, half precision
"""
from __future__ import annotations

import copy
from collections import Counter

import torch
import torch.nn as nn

BN = nn.modules.batchnorm._BatchNorm
PLAIN_BN = (nn.BatchNorm2d,)
FOLDED = "_pynas_folded"            # attribute of a folded model: its list of folded pairs


def find_pairs(model, example):
    """(conv name, BatchNorm name) for every BatchNorm whose input is exactly the output of a convolution, found by
    one forward pass in eval mode. Modules called more than once are left out."""
    names = {m: n for n, m in model.named_modules()}
    outs, pairs = {}, []

    def conv_hook(m, inp, out):
        if torch.is_tensor(out):
            outs[id(out)] = (names[m], out)

    def bn_hook(m, inp):
        hit = outs.get(id(inp[0]))
        if hit is not None and hit[1] is inp[0]:
            pairs.append((hit[0], names[m]))

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    # plain BatchNorm only: subclasses that do more (e.g. BatchNorm + activation in one module) are left alone
    hooks += [m.register_forward_pre_hook(bn_hook) for m in model.modules() if type(m) in PLAIN_BN]
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    nc, nb = Counter(c for c, _ in pairs), Counter(b for _, b in pairs)
    return [p for p in dict.fromkeys(pairs) if nc[p[0]] == 1 and nb[p[1]] == 1]


def _set(model, name, module):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


@torch.no_grad()
def fold_values(conv, bn):
    """The weight and bias of conv with bn folded in (FP32)."""
    std = torch.sqrt(bn.running_var.float() + bn.eps)
    gamma = bn.weight.float() if bn.weight is not None else torch.ones_like(std)
    beta = bn.bias.float() if bn.bias is not None else torch.zeros_like(std)
    scale = gamma / std
    w = conv.weight.float() * scale.reshape(-1, 1, 1, 1)
    b0 = conv.bias.float() if conv.bias is not None else torch.zeros_like(std)
    return w, (b0 - bn.running_mean.float()) * scale + beta


def _fold_into(model, pairs):
    for c, b in pairs:
        conv, bn = model.get_submodule(c), model.get_submodule(b)
        w, bias = fold_values(conv, bn)
        conv.weight = nn.Parameter(w.to(conv.weight.dtype))
        conv.bias = nn.Parameter(bias.to(conv.weight.dtype))
        _set(model, b, nn.Identity())
    setattr(model, FOLDED, list(getattr(model, FOLDED, [])) + [list(p) for p in pairs])
    return model


@torch.no_grad()
def _out(model, example):
    was = model.training
    model.eval()
    try:
        return model(example).float()
    finally:
        model.train(was)


def _same(a, b, rtol):
    return bool(torch.isfinite(a).all()) and float((a - b).abs().max()) <= rtol * max(float(b.abs().max()), 1e-6)


def fold(model, example, rtol=1e-3):
    """A copy of model with its BatchNorm layers folded into the convolutions before them, checked to give the same
    outputs for example (eval mode, FP32; largest difference within rtol of the largest output: float rounding
    stays far below it, a wrong pair far above). Pairs whose folding would change the outputs stay unfolded (their
    BatchNorm keeps its statistics). Returns (folded copy, folded pairs)."""
    example = example.float()
    pairs = find_pairs(model, example)
    ref = _out(model, example)
    folded = _fold_into(copy.deepcopy(model), pairs)
    if _same(_out(folded, example), ref, rtol):
        return folded, pairs
    keep = []
    for p in pairs:                                   # one by one: keep the pairs that change nothing
        trial = _fold_into(copy.deepcopy(model), keep + [p])
        if _same(_out(trial, example), ref, rtol):
            keep.append(p)
    return _fold_into(copy.deepcopy(model), keep), keep


def fold_structure(model, pairs):
    """Give model the layout of a folded one (BatchNorm replaced by identity, conv with bias) without computing
    values, so that saved folded weights can be loaded into it. Returns model."""
    for c, b in pairs:
        conv = model.get_submodule(c)
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels, dtype=conv.weight.dtype))
        _set(model, b, nn.Identity())
    setattr(model, FOLDED, [list(p) for p in pairs])
    return model


def is_folded(model):
    return hasattr(model, FOLDED)


def freeze_bn(model):
    """BatchNorm layers in eval mode: fixed statistics while the rest trains."""
    for m in model.modules():
        if isinstance(m, BN):
            m.eval()
    return model


def deploy_copy(model, example=None, half=False):
    """The network as it runs on the device: a copy, folded (unless it already is; example needed then), eval
    mode, and in half precision with half=True."""
    if is_folded(model):
        m = copy.deepcopy(model)
    else:
        if example is None:
            raise ValueError("deploy_copy of an unfolded model needs an example input")
        m, _ = fold(model, example)
    m = m.eval()
    return m.half() if half else m.float()

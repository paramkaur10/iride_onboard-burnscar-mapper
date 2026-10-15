"""Building blocks of the searched layers and of the decoder. They use only operations that the MYRIAD
plugin of OpenVINO 2022.3 runs: convolution (also depthwise), BatchNorm (folded into the convolution before
it), ReLU, Swish, sigmoid, element-wise add and multiply, pooling, global average pooling, concatenation and
bilinear interpolation. Every convolution followed by BatchNorm has no bias (folding gives it one)."""
from __future__ import annotations

import torch
import torch.nn as nn


def activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name}")


class ConvBNAct(nn.Sequential):
    """k x k convolution (stride, groups), BatchNorm, activation (none when act is None)."""

    def __init__(self, cin, cout, k=3, stride=1, groups=1, act="relu"):
        layers = [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(activation(act))
        super().__init__(*layers)


class DWSep(nn.Sequential):
    """Depthwise separable: k x k depthwise conv-BN-act, then 1 x 1 conv-BN-act."""

    def __init__(self, cin, cout, k=3, act="relu"):
        super().__init__(ConvBNAct(cin, cin, k, groups=cin, act=act), ConvBNAct(cin, cout, 1, act=act))


class SqueezeExcite(nn.Module):
    """Channel attention: global average, 1 x 1 conv down, activation, 1 x 1 conv up, sigmoid, times the input."""

    def __init__(self, ch, squeeze, act="relu"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Conv2d(ch, squeeze, 1)
        self.act = activation(act)
        self.up = nn.Conv2d(squeeze, ch, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        return x * self.gate(self.up(self.act(self.down(self.pool(x)))))


class MBConv(nn.Module):
    """Inverted residual (MobileNetV2 / EfficientNet): 1 x 1 expansion, k x k depthwise, optional
    squeeze-excitation, 1 x 1 projection without activation; the input is added when the shapes match."""

    def __init__(self, cin, cout, k=3, expansion=4, act="relu", se=False):
        super().__init__()
        hidden = cin * expansion
        layers = [ConvBNAct(cin, hidden, 1, act=act)] if expansion != 1 else []
        layers.append(ConvBNAct(hidden, hidden, k, groups=hidden, act=act))
        if se:
            layers.append(SqueezeExcite(hidden, max(1, cin // 4), act))
        layers.append(ConvBNAct(hidden, cout, 1, act=None))
        self.body = nn.Sequential(*layers)
        self.residual = cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.residual else y


class ResBlock(nn.Module):
    """Residual basic block: two k x k conv-BN (activation after the first), the input added (through a 1 x 1
    conv-BN when the width changes), activation."""

    def __init__(self, cin, cout, k=3, act="relu"):
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(cin, cout, k, act=act), ConvBNAct(cout, cout, k, act=None))
        self.short = nn.Identity() if cin == cout else ConvBNAct(cin, cout, 1, act=None)
        self.act = activation(act)

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


def block(kind, cin, cout, k=3, expansion=1, act="relu"):
    """The block of a layer gene."""
    if kind == "conv":
        return ConvBNAct(cin, cout, k, act=act)
    if kind == "dws":
        return DWSep(cin, cout, k, act=act)
    if kind in ("mb", "mbse"):
        return MBConv(cin, cout, k, expansion, act, se=kind == "mbse")
    if kind == "res":
        return ResBlock(cin, cout, k, act=act)
    raise ValueError(f"unknown block {kind}")


def pool(kind):
    if kind == "max":
        return nn.MaxPool2d(2)
    if kind == "avg":
        return nn.AvgPool2d(2)
    raise ValueError(f"unknown pooling {kind}")


class Upsample(nn.Module):
    """Bilinear interpolation to a fixed size (known when the network is built, so that the exported graph
    has constant sizes)."""

    def __init__(self, size):
        super().__init__()
        self.size = tuple(int(s) for s in size)

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, mode="bilinear", align_corners=False)

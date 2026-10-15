"""Networks from specifications: a chromosome of the search space (genes.py) or a fixed model of
segmentation_models_pytorch, `smp:<Architecture>:<encoder>[:depth=<d>][:decoder=<c1>-<c2>-...]` (the
teacher and the baselines).

    net = build("E:eb0:4/L:mb:64:3:4:silu:max/D:conv:48:2", in_channels=7, num_classes=6)
    net = build("smp:Unet:resnet50", 7, 6)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import genes
from .blocks import ConvBNAct, DWSep, Upsample, block, pool
from .encoders import Encoder


class Stem(nn.Module):
    """Without a pretrained encoder: one 3 x 3 stride-2 conv-BN-ReLU; its output is the first feature map."""

    def __init__(self, cin, width):
        super().__init__()
        self.conv = ConvBNAct(cin, width, 3, stride=2, act="relu")
        self.out_channels = [width]

    def forward(self, x):
        return [self.conv(x)]


def _decoder_block(kind, cin, cout):
    return ConvBNAct(cin, cout, 3, act="relu") if kind == "conv" else DWSep(cin, cout, 3, act="relu")


class Network(nn.Module):
    """Encoder (pretrained, cut, or a stem), the searched layers (each: pooling, block), a U-Net decoder from the
    deepest feature map up to stride 2 (every stage: bilinear upsampling to the next feature map, concatenation
    with it, the decoder blocks), a 1 x 1 convolution to the classes and bilinear upsampling to the input size."""

    def __init__(self, chromosome, in_channels, num_classes, input_hw=(256, 256), encoder_weights="imagenet"):
        super().__init__()
        ch = genes.read(chromosome) if isinstance(chromosome, str) else chromosome
        self.chromosome = ch.write()
        H, W = (int(v) for v in input_hw)
        if H % (2 ** ch.depth) or W % (2 ** ch.depth):
            raise ValueError(f"input {H}x{W} is not divisible by the stride {2 ** ch.depth} of {self.chromosome}")
        e = ch.encoder
        if e.pretrained:
            self.encoder = Encoder(genes.ENCODERS[e.code], in_channels, e.depth, encoder_weights, (H, W))
        else:
            self.encoder = Stem(in_channels, e.width)
        with torch.no_grad():
            was = self.encoder.training
            self.encoder.eval()
            shapes = [tuple(f.shape[1:]) for f in self.encoder(torch.zeros(1, in_channels, H, W))]
            self.encoder.train(was)
        chans, sizes = [s[0] for s in shapes], [s[1:] for s in shapes]
        self.layers = nn.ModuleList()
        for g in ch.layers:
            self.layers.append(nn.Sequential(pool(g.pool), block(g.block, chans[-1], g.width, g.kernel, g.expansion, g.act)))
            chans.append(g.width)
            sizes.append((sizes[-1][0] // 2, sizes[-1][1] // 2))
        d = ch.decoder
        self.ups, self.stages = nn.ModuleList(), nn.ModuleList()
        cin = chans[-1]
        for i in range(len(chans) - 2, -1, -1):
            self.ups.append(Upsample(sizes[i]))
            convs = [_decoder_block(d.block, cin + chans[i], d.width)]
            convs += [_decoder_block(d.block, d.width, d.width) for _ in range(d.convs - 1)]
            self.stages.append(nn.Sequential(*convs))
            cin = d.width
        self.head = nn.Conv2d(cin, num_classes, 1)
        self.out = Upsample((H, W))
        self.feature_channels, self.feature_sizes = chans, sizes

    def forward(self, x):
        feats = list(self.encoder(x))
        for layer in self.layers:
            feats.append(layer(feats[-1]))
        y = feats[-1]
        for k, (up, stage) in enumerate(zip(self.ups, self.stages)):
            y = stage(torch.cat([up(y), feats[len(feats) - 2 - k]], 1))
        return self.out(self.head(y))

    def encoder_parameters(self):
        """Parameters of the pretrained encoder (trained at [Train] encoder_lr_scale x the rate); none for a stem."""
        return list(self.encoder.parameters()) if isinstance(self.encoder, Encoder) else []


def _smp(spec, in_channels, num_classes, encoder_weights):
    import segmentation_models_pytorch as smp
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(f"{spec}: expected smp:<Architecture>:<encoder>[:depth=<d>][:decoder=<c1>-<c2>-...]")
    kw = {}
    for opt in parts[3:]:
        k, _, v = opt.partition("=")
        if k == "depth":
            kw["encoder_depth"] = int(v)
        elif k == "decoder":
            kw["decoder_channels"] = [int(c) for c in v.split("-")]
        else:
            raise ValueError(f"{spec}: unknown option {k}")
    weights = None if (encoder_weights or "").lower() in ("", "none", "random") else encoder_weights
    model = getattr(smp, parts[1])(encoder_name=parts[2], encoder_weights=weights, in_channels=in_channels,
                                   classes=num_classes, **kw)
    model.spec = spec
    return model


def build(spec, in_channels, num_classes, input_hw=(256, 256), encoder_weights="imagenet"):
    """The network of a specification (chromosome or smp:...). encoder_weights='none' builds without pretrained
    weights (enough for sizes, checks and loading saved weights)."""
    spec = str(spec)
    if spec.startswith("smp:"):
        return _smp(spec, in_channels, num_classes, encoder_weights)
    return Network(spec, in_channels, num_classes, input_hw, encoder_weights)


def encoder_parameters(model):
    """The parameters trained at [Train] encoder_lr_scale x the rate: a Network's pretrained encoder, an smp
    model's encoder."""
    if hasattr(model, "encoder_parameters"):
        return list(model.encoder_parameters())
    if hasattr(model, "encoder") and isinstance(model.encoder, nn.Module):
        return list(model.encoder.parameters())
    return []


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def n_params_used(model, input_shape):
    """Parameters the forward pass reaches on an input of input_shape (C, H, W): what the exported network holds.
    smp keeps encoder stages that a shallower U-Net never runs; n_params counts them, this does not."""
    import torch
    model = model.float().eval()
    model.zero_grad(set_to_none=True)
    model(torch.zeros((1,) + tuple(input_shape))).sum().backward()
    n = int(sum(p.numel() for p in model.parameters() if p.grad is not None))
    model.zero_grad(set_to_none=True)
    return n


def pretrained_encoder(spec):
    """The smp encoder name whose pretrained weights a specification needs, or None."""
    spec = str(spec)
    if spec.startswith("smp:"):
        return spec.split(":")[2]
    enc = genes.read(spec).encoder
    return genes.ENCODERS[enc.code] if enc.pretrained else None


def prefetch_weights(specs, weights="imagenet"):
    """Download the pretrained weights of the specifications once, before parallel workers build them."""
    from .encoders import prefetch
    names = sorted({n for n in (pretrained_encoder(s) for s in specs) if n})
    if not names or (weights or "").lower() in ("", "none", "random"):
        return {}
    return prefetch(names, weights=weights)

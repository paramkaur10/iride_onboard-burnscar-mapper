"""Chromosomes of the search space, and the operations of the search on them.

A chromosome is an encoder gene, zero or more layer genes and a decoder gene, written '/'-separated:

    E:<encoder>:<depth>[:<stem width>]     E:eb0:4 (EfficientNet-B0 cut after 4 stages), E:none:1:24 (a 3x3
                                           stride-2 stem of 24 channels instead of a pretrained encoder)
    L:<block>:<width>:<kernel>:<expansion>:<activation>:<pool>
                                           pooling (halves the resolution), then the block
    D:<block>:<width>:<convs>              every decoder stage

e.g. E:eb0:4/L:mb:64:3:4:silu:max/D:conv:48:2. write() and read() are exact inverses. The expansion of a block
that does not expand is always 1, so that a chromosome has one written form.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..config import get_list

ENCODERS = {"eb0": "efficientnet-b0", "eb1": "efficientnet-b1", "mv2": "mobilenet_v2", "r18": "resnet18",
            "r34": "resnet34"}
BLOCKS = ("conv", "dws", "mb", "mbse", "res")
EXPANDING = ("mb", "mbse")
ACTIVATIONS = ("relu", "silu")
POOLS = ("max", "avg")
DECODER_BLOCKS = ("conv", "dws")


@dataclass(frozen=True)
class EncoderGene:
    code: str
    depth: int
    width: int = 0                       # stem width, for code 'none' only

    @property
    def pretrained(self):
        return self.code != "none"

    def write(self):
        return f"E:{self.code}:{self.depth}" + ("" if self.pretrained else f":{self.width}")


@dataclass(frozen=True)
class LayerGene:
    block: str
    width: int
    kernel: int
    expansion: int
    act: str
    pool: str

    def write(self):
        return f"L:{self.block}:{self.width}:{self.kernel}:{self.expansion}:{self.act}:{self.pool}"


@dataclass(frozen=True)
class DecoderGene:
    block: str
    width: int
    convs: int

    def write(self):
        return f"D:{self.block}:{self.width}:{self.convs}"


@dataclass(frozen=True)
class Chromosome:
    encoder: EncoderGene
    layers: tuple
    decoder: DecoderGene

    def write(self):
        return "/".join([self.encoder.write()] + [g.write() for g in self.layers] + [self.decoder.write()])

    def __str__(self):
        return self.write()

    @property
    def depth(self):
        """Number of halvings of the resolution down to the deepest feature map."""
        return self.encoder.depth + len(self.layers)


# ---- reading ---------------------------------------------------------------------------------------
def _int(s, what, code):
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"{code}: {what} '{s}' is not a whole number") from None


def read(code):
    """The Chromosome written as code (vocabulary checked; limits of a Space are checked by Space.check)."""
    parts = [p for p in str(code).strip().split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"{code}: a chromosome needs an encoder gene and a decoder gene")
    e = parts[0].split(":")
    if e[0] != "E" or len(e) not in (3, 4):
        raise ValueError(f"{code}: the first gene must be E:<encoder>:<depth>[:<stem width>]")
    if e[1] == "none":
        if len(e) != 4:
            raise ValueError(f"{code}: E:none needs a stem width (E:none:1:<width>)")
        enc = EncoderGene("none", _int(e[2], "depth", code), _int(e[3], "stem width", code))
        if enc.depth != 1:
            raise ValueError(f"{code}: E:none has depth 1 (the stem)")
    else:
        if e[1] not in ENCODERS or len(e) != 3:
            raise ValueError(f"{code}: unknown encoder '{e[1]}' (known: none, {', '.join(ENCODERS)})")
        enc = EncoderGene(e[1], _int(e[2], "depth", code))
    layers = []
    for g in parts[1:-1]:
        f = g.split(":")
        if f[0] != "L" or len(f) != 7:
            raise ValueError(f"{code}: '{g}' is not L:<block>:<width>:<kernel>:<expansion>:<activation>:<pool>")
        layer = LayerGene(f[1], _int(f[2], "width", code), _int(f[3], "kernel", code), _int(f[4], "expansion", code),
                          f[5], f[6])
        if layer.block not in BLOCKS or layer.act not in ACTIVATIONS or layer.pool not in POOLS:
            raise ValueError(f"{code}: unknown block, activation or pooling in '{g}'")
        if layer.block not in EXPANDING and layer.expansion != 1:
            raise ValueError(f"{code}: block {layer.block} does not expand; its expansion is written 1")
        layers.append(layer)
    d = parts[-1].split(":")
    if d[0] != "D" or len(d) != 4 or d[1] not in DECODER_BLOCKS:
        raise ValueError(f"{code}: the last gene must be D:<{'|'.join(DECODER_BLOCKS)}>:<width>:<convs>")
    return Chromosome(enc, tuple(layers), DecoderGene(d[1], _int(d[2], "width", code), _int(d[3], "convs", code)))


# ---- the space ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Space:
    encoders: tuple
    encoder_depths: tuple
    stem_widths: tuple
    blocks: tuple
    widths: tuple
    kernels: tuple
    expansions: tuple
    activations: tuple
    pools: tuple
    min_layers_plain: int
    min_layers_encoder: int
    max_layers: int
    max_stride: int
    decoder_blocks: tuple
    decoder_widths: tuple
    decoder_convs: tuple

    @classmethod
    def from_config(cls, cfg):
        L = lambda opt, cast=str: tuple(get_list(cfg, "Space", opt, cast))   # noqa: E731
        sp = cls(encoders=L("encoders"), encoder_depths=L("encoder_depths", int), stem_widths=L("stem_widths", int),
                 blocks=L("blocks"), widths=L("widths", int), kernels=L("kernels", int),
                 expansions=L("expansions", int), activations=L("activations"), pools=L("pools"),
                 min_layers_plain=cfg.getint("Space", "min_layers_plain"),
                 min_layers_encoder=cfg.getint("Space", "min_layers_encoder"),
                 max_layers=cfg.getint("Space", "max_layers"), max_stride=cfg.getint("Space", "max_stride"),
                 decoder_blocks=L("decoder_blocks"), decoder_widths=L("decoder_widths", int),
                 decoder_convs=L("decoder_convs", int))
        unknown = ([e for e in sp.encoders if e != "none" and e not in ENCODERS]
                   + [b for b in sp.blocks if b not in BLOCKS] + [a for a in sp.activations if a not in ACTIVATIONS]
                   + [p for p in sp.pools if p not in POOLS] + [b for b in sp.decoder_blocks if b not in DECODER_BLOCKS])
        if unknown:
            raise ValueError(f"[Space]: unknown names {unknown}")
        if 2 ** int(round(math.log2(sp.max_stride))) != sp.max_stride:
            raise ValueError("[Space] max_stride must be a power of 2")
        return sp

    @property
    def max_depth(self):
        return int(round(math.log2(self.max_stride)))

    def layer_range(self, enc):
        """(fewest, most) searched layers after this encoder gene."""
        lo = self.min_layers_encoder if enc.pretrained else self.min_layers_plain
        return lo, min(self.max_layers, self.max_depth - enc.depth)

    def check(self, ch):
        """None if the chromosome is in the space, else the reason it is not."""
        e = ch.encoder
        if e.code not in self.encoders:
            return f"encoder {e.code} not in [Space] encoders"
        if e.pretrained and e.depth not in self.encoder_depths:
            return f"encoder depth {e.depth} not in [Space] encoder_depths"
        if not e.pretrained and e.width not in self.stem_widths:
            return f"stem width {e.width} not in [Space] stem_widths"
        lo, hi = self.layer_range(e)
        if not lo <= len(ch.layers) <= hi:
            return f"{len(ch.layers)} searched layers, allowed {lo}..{hi}"
        for g in ch.layers:
            if (g.block not in self.blocks or g.width not in self.widths or g.kernel not in self.kernels
                    or g.act not in self.activations or g.pool not in self.pools
                    or (g.block in EXPANDING and g.expansion not in self.expansions)):
                return f"layer {g.write()} outside [Space]"
        d = ch.decoder
        if d.block not in self.decoder_blocks or d.width not in self.decoder_widths or d.convs not in self.decoder_convs:
            return f"decoder {d.write()} outside [Space]"
        return None


# ---- random genes ----------------------------------------------------------------------------------------
def random_encoder(space, rng):
    code = rng.choice(space.encoders)
    if code == "none":
        return EncoderGene("none", 1, rng.choice(space.stem_widths))
    depths = [d for d in space.encoder_depths if d <= space.max_depth]
    return EncoderGene(code, rng.choice(depths))


def random_layer(space, rng):
    block = rng.choice(space.blocks)
    return LayerGene(block, rng.choice(space.widths), rng.choice(space.kernels),
                     rng.choice(space.expansions) if block in EXPANDING else 1, rng.choice(space.activations),
                     rng.choice(space.pools))


def random_decoder(space, rng):
    return DecoderGene(rng.choice(space.decoder_blocks), rng.choice(space.decoder_widths), rng.choice(space.decoder_convs))


def random_chromosome(space, rng):
    for _ in range(200):
        enc = random_encoder(space, rng)
        lo, hi = space.layer_range(enc)
        if hi >= lo:
            n = rng.randint(lo, hi)
            return Chromosome(enc, tuple(random_layer(space, rng) for _ in range(n)), random_decoder(space, rng))
    raise ValueError("[Space] allows no chromosome: check its limits")


# ---- variation ----------------------------------------------------------------------------------------------
def _other(values, current, rng):
    choices = [v for v in values if v != current]
    return rng.choice(choices) if choices else current


def _mutate_encoder(e, space, rng):
    what = rng.choice(["code", "size"])
    if what == "code" and len(space.encoders) > 1:
        code = _other(space.encoders, e.code, rng)
        if code == "none":
            return EncoderGene("none", 1, rng.choice(space.stem_widths))
        return EncoderGene(code, e.depth if (e.pretrained and e.depth in space.encoder_depths)
                           else rng.choice([d for d in space.encoder_depths if d <= space.max_depth]))
    if e.pretrained:
        return replace(e, depth=_other([d for d in space.encoder_depths if d <= space.max_depth], e.depth, rng))
    return replace(e, width=_other(space.stem_widths, e.width, rng))


def _mutate_layer(g, space, rng):
    fields = ["block", "width", "kernel", "act", "pool"] + (["expansion"] if g.block in EXPANDING else [])
    f = rng.choice(fields)
    if f == "block":
        block = _other(space.blocks, g.block, rng)
        exp = (g.expansion if g.block in EXPANDING else rng.choice(space.expansions)) if block in EXPANDING else 1
        return replace(g, block=block, expansion=exp)
    pool = {"width": space.widths, "kernel": space.kernels, "act": space.activations, "pool": space.pools,
            "expansion": space.expansions}[f]
    return replace(g, **{f: _other(pool, getattr(g, f), rng)})


def _mutate_decoder(d, space, rng):
    f = rng.choice(["block", "width", "convs"])
    pool = {"block": space.decoder_blocks, "width": space.decoder_widths, "convs": space.decoder_convs}[f]
    return replace(d, **{f: _other(pool, getattr(d, f), rng)})


def repair(ch, space, rng):
    """The chromosome within the layer limits of its encoder: layers past the most allowed are dropped, random
    layers are added up to the fewest allowed."""
    lo, hi = space.layer_range(ch.encoder)
    layers = list(ch.layers)[:max(hi, 0)]
    while len(layers) < lo:
        layers.append(random_layer(space, rng))
    return Chromosome(ch.encoder, tuple(layers), ch.decoder)


def mutate(ch, space, rng, p):
    """Each gene changes with probability p (one of its fields gets another value); with probability p a
    layer is also added or removed. At least one change is made."""
    for attempt in range(50):
        force = attempt > 0
        enc = _mutate_encoder(ch.encoder, space, rng) if rng.random() < p else ch.encoder
        layers = [(_mutate_layer(g, space, rng) if rng.random() < p else g) for g in ch.layers]
        if rng.random() < p:
            lo, hi = space.layer_range(enc)
            if layers and (len(layers) > lo) and (len(layers) >= hi or rng.random() < 0.5):
                del layers[rng.randrange(len(layers))]
            elif len(layers) < hi:
                layers.insert(rng.randint(0, len(layers)), random_layer(space, rng))
        dec = _mutate_decoder(ch.decoder, space, rng) if rng.random() < p else ch.decoder
        child = repair(Chromosome(enc, tuple(layers), dec), space, rng)
        if child != ch:
            return child
        if force:                                    # nothing changed by chance: change one gene for sure
            k = rng.randrange(len(ch.layers) + 2)
            if k == 0:
                child = repair(replace(ch, encoder=_mutate_encoder(ch.encoder, space, rng)), space, rng)
            elif k == len(ch.layers) + 1:
                child = replace(ch, decoder=_mutate_decoder(ch.decoder, space, rng))
            else:
                layers = list(ch.layers)
                layers[k - 1] = _mutate_layer(layers[k - 1], space, rng)
                child = replace(ch, layers=tuple(layers))
            if child != ch:
                return child
    return child


def crossover(a, b, rng, space):
    """One-point crossover of the layer genes; the first child has a's encoder and b's decoder, the second
    the other way round. Both are repaired to the layer limits."""
    i, j = rng.randint(0, len(a.layers)), rng.randint(0, len(b.layers))
    c1 = Chromosome(a.encoder, a.layers[:i] + b.layers[j:], b.decoder)
    c2 = Chromosome(b.encoder, b.layers[:j] + a.layers[i:], a.decoder)
    return repair(c1, space, rng), repair(c2, space, rng)

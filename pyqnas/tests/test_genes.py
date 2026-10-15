"""Chromosomes: writing and reading are inverses, the space check, mutation and crossover stay in the space."""
import random

from pynas import config
from pynas.space import genes


def _space(**over):
    cfg = config.load(overrides={f"Space.{k}": v for k, v in over.items()})
    return genes.Space.from_config(cfg)


def test_write_read_inverse():
    sp, rng = _space(), random.Random(0)
    for _ in range(2000):
        c = genes.random_chromosome(sp, rng)
        assert genes.read(c.write()) == c
        assert sp.check(c) is None, (c.write(), sp.check(c))


def test_known_codes():
    c = genes.read("E:eb0:4/L:mbse:64:3:4:silu:max/D:conv:48:2")
    assert c.encoder.code == "eb0" and c.encoder.depth == 4 and len(c.layers) == 1 and c.decoder.convs == 2
    assert c.depth == 5
    p = genes.read("E:none:1:24/L:conv:32:3:1:relu:max/L:res:64:5:1:relu:avg/L:dws:64:3:1:silu:max/D:dws:32:1")
    assert not p.encoder.pretrained and p.encoder.width == 24 and p.depth == 4


def test_bad_codes_rejected():
    bad = ["E:eb0:4", "E:xx:4/D:conv:48:2", "E:eb0:4/L:conv:32:3:2:relu:max/D:conv:48:2",
           "E:none:1/D:conv:48:2", "E:eb0:4/L:mb:64:3:4:tanh:max/D:conv:48:2", "E:eb0:4/D:sep:48:2"]
    for code in bad:
        try:
            genes.read(code)
        except ValueError:
            continue
        raise AssertionError(f"accepted {code}")


def test_space_limits():
    sp = _space(max_stride="32")
    c = genes.read("E:eb0:5/L:conv:32:3:1:relu:max/D:conv:48:2")      # stride 64 > 32
    assert sp.check(c) is not None
    c = genes.read("E:none:1:24/L:conv:32:3:1:relu:max/D:conv:48:2")    # 1 layer < min_layers_plain 3
    assert sp.check(c) is not None


def test_mutation_changes_and_stays_in_space():
    sp, rng = _space(), random.Random(1)
    for _ in range(1000):
        c = genes.random_chromosome(sp, rng)
        m = genes.mutate(c, sp, rng, 0.15)
        assert m != c
        assert sp.check(m) is None, (c.write(), m.write(), sp.check(m))


def test_crossover_stays_in_space():
    sp, rng = _space(), random.Random(2)
    for _ in range(1000):
        a, b = genes.random_chromosome(sp, rng), genes.random_chromosome(sp, rng)
        for c in genes.crossover(a, b, rng, sp):
            assert sp.check(c) is None, (a.write(), b.write(), c.write(), sp.check(c))
            assert c.encoder in (a.encoder, b.encoder) and c.decoder in (a.decoder, b.decoder)


def test_sampling_is_reproducible():
    sp = _space()
    a = [genes.random_chromosome(sp, random.Random(5)).write() for _ in range(3)]
    b = [genes.random_chromosome(sp, random.Random(5)).write() for _ in range(3)]
    assert a == b


if __name__ == "__main__":
    from _run import run
    run(dict(globals()))

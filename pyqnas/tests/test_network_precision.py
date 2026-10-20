"""Networks from chromosomes (shapes, skip wiring), BatchNorm folding (outputs unchanged, structure reloadable),
the FP16 deploy copy. With a real PyTorch the numbers are checked; elsewhere only the plumbing."""
import random

import torch

from pynas import config
from pynas.space import genes
from pynas.space.network import build, n_params
from pynas.train import precision

REAL = not getattr(torch, "__fake__", False)


def _plain(seed=0, n=20, **over):
    cfg = config.load(overrides={"Space.encoders": "none", **{f"Space.{k}": v for k, v in over.items()}})
    sp, rng = genes.Space.from_config(cfg), random.Random(seed)
    return [genes.random_chromosome(sp, rng) for _ in range(n)]


def test_shapes_of_random_networks():
    for c in _plain(n=30):
        net = build(c.write(), 7, 6, (64, 64), encoder_weights="none").eval()
        with torch.no_grad():
            y = net(torch.randn(2, 7, 64, 64))
        assert tuple(y.shape) == (2, 6, 64, 64), (c.write(), tuple(y.shape))
        assert net.feature_channels[0] == c.encoder.width
        assert len(net.stages) == c.depth - 1


def test_fold_unchanged_outputs_and_reload():
    torch.manual_seed(0)
    for c in _plain(seed=3, n=8):
        net = build(c.write(), 7, 6, (64, 64), encoder_weights="none")
        for m in net.modules():                    # BatchNorm statistics that are not the identity
            if isinstance(m, precision.BN):
                m.running_mean.data = torch.randn(m.num_features) * 0.5 if REAL else m.running_mean.data
                m.running_var.data = torch.rand(m.num_features) + 0.5 if REAL else m.running_var.data
        x = torch.randn(3, 7, 64, 64)
        net.eval()
        with torch.no_grad():
            ref = net(x)
        folded, pairs = precision.fold(net, x)
        n_bn = sum(isinstance(m, precision.BN) for m in net.modules())
        assert len(pairs) == n_bn, (c.write(), len(pairs), n_bn)          # every BN of our blocks follows a conv
        assert not any(isinstance(m, precision.BN) for m in folded.modules())
        with torch.no_grad():
            out = folded.eval()(x)
        if REAL:
            assert float((out - ref).abs().max()) <= 1e-4 * float(ref.abs().max()) + 1e-6
        fresh = precision.fold_structure(build(c.write(), 7, 6, (64, 64), encoder_weights="none"), pairs)
        fresh.load_state_dict(folded.state_dict())                          # same layout, loads strictly
        assert n_params(fresh) == n_params(folded)


def test_deploy_copy_half():
    c = _plain(seed=7, n=1)[0]
    net = build(c.write(), 7, 6, (64, 64), encoder_weights="none")
    x = torch.randn(2, 7, 64, 64)
    m16 = precision.deploy_copy(net, x, half=True)
    assert precision.is_folded(m16)
    assert all(p.dtype == torch.float16 for p in m16.parameters()) if REAL else True


def test_input_mask_image_keeps_unlabelled_image_pixels():
    import torch
    if getattr(torch, "__fake__", False):
        return
    from pynas.train.data import DataSpec, batch_input
    spec = DataSpec(mask_values=[0, 2], class_names=["clear", "fresh"], target=1, nodata=255, scale=10000.0)
    refl = torch.full((1, 3, 4, 4), 1000, dtype=torch.int16)
    refl[0, :, 0, 0] = 0                                   # the image has no data here
    mask = torch.zeros((1, 4, 4), dtype=torch.uint8)
    mask[0, 1, :] = 255                                     # a row without a label, inside the image
    by_labels, by_image = batch_input(refl, mask, spec, "labels"), batch_input(refl, mask, spec, "image")
    assert float(by_labels[0, :, 1, 2].abs().sum()) == 0.0 and float(by_image[0, :, 1, 2].abs().sum()) > 0.0
    assert float(by_image[0, :, 0, 0].abs().sum()) == 0.0


if __name__ == "__main__":
    from _run import run
    run(dict(globals()))

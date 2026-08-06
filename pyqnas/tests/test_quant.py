"""
Self-tests for the hand-written quantization module.

Run from the repo root:  python tests/test_quant.py
(Requires the package importable, i.e. `pip install -e .` / `uv sync` done.)

Verifies, in order:
  1. Weight quantizer produces at most 2^b - 1 distinct levels, symmetric grid.
  2. Per-channel scales isolate channel ranges.
  3. Activation quantizer produces uint grid with exact-zero representability.
  4. STE gradients flow.
  5. BN folding matches conv->bn reference output in eval mode.
  6. prepare/close round-trip restores the original model exactly.
  7. A small quantized model actually trains (loss decreases, observers freeze).
  8. pow2 scale mode yields power-of-two scales.
"""

import torch
import torch.nn as nn

from pynas.core.quant import (
    QuantOpts, fake_quant_weight, ActFakeQuant,
    prepare_hailo_quant, freeze_all_ranges,
)
from pynas.core.quant.qmodules import QuantConv2d

torch.manual_seed(0)
PASS = "  [PASS]"


def test_weight_grid():
    opts = QuantOpts(weight_bits=8, weight_per_channel=False)
    w = torch.randn(64, 32, 3, 3) * 0.1
    wq = fake_quant_weight(w, opts)
    scale = w.abs().amax() / 127
    ints = torch.round(wq / scale)
    n_levels = ints.unique().numel()
    assert n_levels <= 255, f"too many levels: {n_levels}"
    assert ints.min() >= -127 and ints.max() <= 127, "grid not in [-127,127]"
    err = (wq - w).abs().max() / scale
    assert err <= 0.5 + 1e-4, f"rounding error exceeds half a step: {err}"
    print(PASS, f"weight grid: {n_levels} levels within [-127,127], max err {err:.3f} steps")


def test_weight_per_channel():
    opts = QuantOpts(weight_bits=8, weight_per_channel=True)
    w = torch.randn(8, 4, 3, 3)
    w[0] *= 100.0
    wq = fake_quant_weight(w, opts)
    err_others = (wq[1:] - w[1:]).abs().max()
    per_tensor = fake_quant_weight(w, QuantOpts(weight_bits=8, weight_per_channel=False))
    err_others_pt = (per_tensor[1:] - w[1:]).abs().max()
    assert err_others < err_others_pt, "per-channel not better than per-tensor on mixed ranges"
    print(PASS, f"per-channel isolates ranges (err {err_others:.4f} vs per-tensor {err_others_pt:.4f})")


def test_act_grid_and_zero():
    opts = QuantOpts(act_bits=8, act_unsigned=True)
    aq = ActFakeQuant(opts)
    aq.train()
    x = torch.randn(16, 8, 10, 10) * 2 + 1.0
    for _ in range(5):
        _ = aq(x)
    aq.eval()
    y = aq(torch.zeros(1, 8, 2, 2))
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), \
        f"fp 0.0 not exactly representable, got {y.abs().max()}"
    y2 = aq(x)
    scale = (torch.maximum(aq.running_max, torch.tensor(0.0))
             - torch.minimum(aq.running_min, torch.tensor(0.0))) / 255
    n_levels = torch.round(y2 / scale).unique().numel()
    assert n_levels <= 256, f"too many activation levels: {n_levels}"
    print(PASS, f"activation grid: {n_levels} levels, exact zero preserved")


def test_ste_gradients():
    opts = QuantOpts()
    w = torch.randn(16, 8, 3, 3, requires_grad=True)
    wq = fake_quant_weight(w, opts)
    loss = (wq ** 2).sum()
    loss.backward()
    assert w.grad is not None and w.grad.abs().sum() > 0, "no gradient through STE"
    print(PASS, f"STE gradients flow (grad norm {w.grad.norm():.3f})")


def test_bn_folding_matches_reference():
    conv = nn.Conv2d(4, 8, 3, padding=1, bias=True)
    bn = nn.BatchNorm2d(8)
    bn.train()
    for _ in range(10):
        bn(conv(torch.randn(4, 4, 16, 16)))
    conv.eval(); bn.eval()

    x = torch.randn(2, 4, 16, 16)
    ref = bn(conv(x))

    opts = QuantOpts(weight_bits=16, act_bits=16)
    qc = QuantConv2d(conv, bn, opts)
    qc.eval()
    qc.act_quant.train(); qc.act_quant(x); qc.act_quant.eval()
    out = qc(x)

    err = (out - ref).abs().max()
    assert err < 1e-2, f"BN folding mismatch vs reference: {err}"
    print(PASS, f"BN folding matches conv->bn reference (max err {err:.2e} at 16-bit)")


def test_prepare_and_restore():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Flatten(), nn.Linear(16 * 8 * 8, 10),
    )
    orig_ids = {name: id(m) for name, m in model.named_modules()}
    ctx = prepare_hailo_quant(model, QuantOpts())
    assert ctx.num_quantized == 3, f"expected 3 quantized modules, got {ctx.num_quantized}"
    n_wrap = sum(isinstance(m, QuantConv2d) for m in model.modules())
    assert n_wrap == 2, "conv wrappers not installed"
    n_id = sum(isinstance(m, nn.Identity) for m in model.children())
    assert n_id == 2, "BN slots not replaced with Identity"
    ctx.close()
    for name, m in model.named_modules():
        assert id(m) == orig_ids.get(name, id(m)), f"module {name} not restored"
    print(PASS, "prepare/close round-trip restores original modules")


def test_quantized_training_converges():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 4, 3, padding=1),
    ).to(device)
    ctx = prepare_hailo_quant(model, QuantOpts(freeze_ranges_after=20))

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 3, 32, 32, device=device)
    target = torch.randint(0, 4, (8, 32, 32), device=device)
    lossfn = nn.CrossEntropyLoss()

    model.train()
    first, last = None, None
    for step in range(120):
        opt.zero_grad()
        out = model(x)
        loss = lossfn(out, target)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 0.95, f"quantized training not converging: {first:.3f} -> {last:.3f}"
    n_frozen = sum(getattr(m.act_quant, "frozen", False)
                   for m in model.modules() if hasattr(m, "act_quant"))
    assert n_frozen == 2, "observers did not auto-freeze"
    ctx.close()
    print(PASS, f"quantized training converges ({first:.3f} -> {last:.3f}), observers froze")


def test_pow2_scales():
    opts = QuantOpts(weight_bits=8, weight_per_channel=True, weight_pow2_scale=True)
    w = torch.randn(8, 4, 3, 3)
    wq = fake_quant_weight(w, opts)
    # recover scale per channel: scale = max(|w|) / 127
    # with pow2 snapping: scale = 2^round(log2(max(|w|)/127))
    for c in range(8):
        max_abs = w[c].abs().amax()
        raw_scale = max_abs / 127
        if raw_scale < 1e-12:
            continue
        implied_log2 = torch.log2(raw_scale)
        pow2_scale = 2.0 ** torch.round(implied_log2)
        # verify the quantized values are multiples of that pow2 scale
        residuals = wq[c] / pow2_scale
        residuals_rounded = torch.round(residuals)
        err = (residuals - residuals_rounded).abs().max()
        assert err < 1e-4, \
            f"channel {c}: quantized values not multiples of pow2 scale {pow2_scale:.6f}, err={err}"
    print(PASS, "pow2 mode: all channel scales are powers of two")


if __name__ == "__main__":
    print("Running hand-written quantization self-tests...\n")
    test_weight_grid()
    test_weight_per_channel()
    test_act_grid_and_zero()
    test_ste_gradients()
    test_bn_folding_matches_reference()
    test_prepare_and_restore()
    test_quantized_training_converges()
    test_pow2_scales()
    print("\nAll tests passed.")
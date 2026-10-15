"""Whether candidates can fly, cheapest test first, before any of them is trained.

1. size estimate: the weights in FP16 (built without pretrained weights); over [Hardware] max_ir_mb -> infeasible,
   nothing compiled;
2. ([Hardware] check = compile) ONNX -> FP16 IR -> compiled for MYRIAD: not compiled, IR over max_ir_mb, or IR
   layer types missing from the ArgOS list ([Hardware] argos_layers) -> infeasible;
3. ([Hardware] latency = nnmeter) predicted latency; over [Hardware] max_latency_ms (when set) -> infeasible.

    recs = feasibility({"c0": "E:eb0:4/.../D:conv:48:2"}, cfg, "search/checks/gen0", (7, 256, 256), 6)
    recs["c0"]["feasible"], recs["c0"]["reason"]
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..space.network import build, n_params
from . import latency, myriad


def feasibility(specs, cfg, out_dir, input_shape, num_classes, log=None):
    """{name: specification} -> {name: record}. A record has feasible, reason (None when feasible), params,
    est_mb and, from the later tests, ir_mb, compiled, layers, argos_unlisted, latency_ms, error."""
    mode = cfg.get("Hardware", "check").strip().lower()
    lim = myriad.limits(cfg)
    recs, models = {}, {}
    for name, spec in specs.items():
        rec = recs[name] = dict(spec=spec, feasible=True, reason=None)
        try:
            net = build(spec, input_shape[0], num_classes, tuple(input_shape[1:]), encoder_weights="none").eval()
        except Exception as e:                                   # noqa: BLE001 - recorded
            rec.update(feasible=False, reason=f"does not build: {type(e).__name__}: {e}")
            continue
        rec["params"] = n_params(net)
        rec["est_mb"] = round(myriad.fp16_mb(net), 3)
        if mode != "off" and rec["est_mb"] > lim["max_ir_mb"]:
            rec.update(feasible=False, reason=f"weights {rec['est_mb']:.2f} MB in FP16 > {lim['max_ir_mb']} MB")
            continue
        models[name] = net
    if mode == "compile" and models:
        if log:
            log(f"compiling {len(models)} candidates for MYRIAD")
        checked = myriad.check(models, Path(out_dir), input_shape, cfg, compile=True)
        _tidy(Path(out_dir))
        for name, c in checked.items():
            rec = recs[name]
            rec.update({k: c.get(k) for k in ("ir_mb", "compiled", "layers", "argos_unlisted", "error", "mapping")})
            if not c.get("compiled"):
                rec.update(feasible=False, reason=f"not compiled for MYRIAD: {(c.get('error') or '')[:300]}")
            elif c.get("ir_mb") is not None and c["ir_mb"] > lim["max_ir_mb"]:
                rec.update(feasible=False, reason=f"IR {c['ir_mb']:.2f} MB > {lim['max_ir_mb']} MB")
            elif c.get("argos_unlisted"):
                rec.update(feasible=False, reason=f"IR layers not in the ArgOS list: {c['argos_unlisted']}")
    if cfg.get("Hardware", "latency").strip().lower() == "nnmeter":
        alive = {n: m for n, m in models.items() if recs[n]["feasible"]}
        if alive:
            lat = latency.latency_ms(alive, tuple(input_shape), cfg)
            for name in alive:
                recs[name]["latency_ms"] = lat.get(name)
                if lim["max_latency_ms"] is not None and (lat.get(name) is None or lat[name] > lim["max_latency_ms"]):
                    recs[name].update(feasible=False, reason=f"latency {lat.get(name)} ms > {lim['max_latency_ms']} ms")
    return recs


def _tidy(out_dir):
    """Keep the IR .xml and .mapping of every check (small: layer types, names); drop the ONNX files, the IR weights
    and the compiled blobs (the final export makes them again)."""
    shutil.rmtree(out_dir / "onnx", ignore_errors=True)
    for f in (out_dir / "ir").rglob("*"):
        if f.suffix in (".bin", ".blob"):
            f.unlink()

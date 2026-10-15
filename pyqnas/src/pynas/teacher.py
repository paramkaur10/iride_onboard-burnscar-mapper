"""The teacher: a fixed large network ([Teacher] spec) trained by the same trainer as every other model, with the
options of [Teacher] on top of [Train]; it stays on the ground, so it trains in [Teacher] precision without
folding. Its class probabilities (mean of its seeds, averaged over the 8 rotations and flips) are written into the
window pool as <set>_<[Data] teacher>.npy for the training set and the scoring sets: the soft labels distillation
reads.

    q = queue(cfg, "models/pynas/teacher")      # then runner.launch(q, gpus), runner.wait(q)
    write_soft_labels(q, cfg)                   # once every seed is trained
"""
from __future__ import annotations

from pathlib import Path

from . import config as C
from . import runner


def plan(cfg):
    spec = cfg.get("Teacher", "spec").strip()
    return [runner.make_run(spec, "teacher", cfg, {"Train.seed": s}, label="teacher", seed=s)
            for s in C.get_list(cfg, "Teacher", "seeds", int)]


def queue(cfg, out):
    """Write the teacher's queue (its pretrained weights downloaded first). Returns the queue path."""
    from .space.network import prefetch_weights
    runs = plan(cfg)
    prefetch_weights([r["model"] for r in runs], cfg.get("Space", "encoder_weights").strip())
    return runner.write_queue(Path(out) / "runs", runs, cfg)


def models(q, device="cpu"):
    """The trained teacher seeds of the queue."""
    Q = runner.read_queue(q)
    missing = [r["run"] for r in Q["runs"] if runner.done_scores(Q, r) is None]
    if missing:
        raise RuntimeError(f"teacher runs not done yet: {missing}")
    return [runner.load_result(Path(Q["out"]) / r["run"], device) for r in Q["runs"]]


def write_soft_labels(q, cfg, sets=None, batch=16):
    """Write the teacher's stored class log-probabilities for the training set and the search and validation sets
    (or the given sets). Returns {set: path}."""
    import torch
    from .train.data import DataSpec, write_predictions
    spec = DataSpec.from_config(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nets = models(q, device)
    names = sets or [spec.train] + spec.search + spec.val
    tta = cfg.getboolean("Teacher", "tta")
    return {n: write_predictions(nets, spec, n, device, batch=batch, tta=tta) for n in names}

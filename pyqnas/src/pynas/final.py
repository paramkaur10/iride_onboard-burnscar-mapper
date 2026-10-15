"""The final stage, after the search.

1. the winner of the search (winner.json) trained again by the same trainer, [Final] steps, one run per seed;
   the baselines ([Final] baselines, fixed smp models) and the ablations of the winner ([Final] ablations: no_kd,
   without distillation; fp32, trained in FP32 with BatchNorm frozen instead of folded, converted afterwards) with
   the same seeds; all scored on the search and validation sets;
2. deploy(): the winner seed with the best validation score in FP16 is exported as the files ArgOS takes (FP16 IR,
   compiled for MYRIAD), its IR run on the OpenVINO CPU plugin against PyTorch, and the FP16 / FP32 agreement;
3. test(): every final model, the teacher and the deployed one scored once on [Data] test; the result is kept and
   read back afterwards, never computed again unless forced.

With [Final] finalists = k > 0 the winner is not the search's best candidate but comes from a finalist stage
before step 1: the k candidates with the best search fitness are trained like final models ([Final] steps, every
seed) and compared on [Data] val, which the search never saw. A single search score is the best of many noisy
measurements and so overstates its candidate (the winner's curse); the finalists' validation scores are fresh
measurements. Finalists within one standard error of the best mean validation score count as equal and the one
with the lowest predicted latency is taken.

    fq = finalist_queue(cfg, "models/pynas/search", "models/pynas/final_r2")  # then launch, wait
    pick, table = pick_finalist(fq, cfg, "models/pynas/search", "models/pynas/final_r2")
    q = queue(cfg, "models/pynas/search", "models/pynas/final_r2")            # the final stage of that winner

    q = queue(cfg, "models/pynas/search", "models/pynas/final")   # then runner.launch(q, gpus), runner.wait(q)
    dep = deploy(q, cfg, "models/onboard", "burnscar_pynas")
    t = test(q, cfg, teacher_queue="models/pynas/teacher/runs/queue.json")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import runner

ABLATIONS = {"no_kd": {"Train.kd_weight": 0}, "fp32": {"Train.precision": "fp32", "Train.fold": "false"}}


def winner(search_out):
    w = json.load(open(Path(search_out) / "winner.json"))
    if not w.get("spec"):
        raise RuntimeError(f"no winner in {search_out}/winner.json: {w}")
    return w


def _common(cfg):
    return {"Train.steps": cfg.getint("Final", "steps"), "Train.eval_every": cfg.getint("Final", "eval_every")}


def chosen(cfg, search_out, out=None):
    """The model the final stage trains: the finalist stage's pick ([Final] finalists > 0) or the search's winner."""
    if cfg.getint("Final", "finalists") > 0:
        f = Path(out or ".") / "finalists.json"
        if not f.exists():
            raise RuntimeError(f"[Final] finalists > 0 and no {f}: run finalist_queue, then pick_finalist")
        return json.load(open(f))
    return winner(search_out)


def plan(cfg, search_out, out=None):
    w = chosen(cfg, search_out, out)
    common = _common(cfg)
    runs = []
    for s in C.get_list(cfg, "Final", "seeds", int):
        runs.append(runner.make_run(w["spec"], "final", cfg, {**common, "Train.seed": s}, label="winner", seed=s))
        for b in C.get_list(cfg, "Final", "baselines"):
            runs.append(runner.make_run(b, "baseline", cfg, {**common, "Train.seed": s}, label=b, seed=s))
        for a in C.get_list(cfg, "Final", "ablations"):
            if a not in ABLATIONS:
                raise ValueError(f"[Final] ablations: unknown {a} (known: {', '.join(ABLATIONS)})")
            runs.append(runner.make_run(w["spec"], "ablation", cfg, {**common, **ABLATIONS[a], "Train.seed": s},
                                        label=f"winner, {a}", seed=s))
    return runs


def queue(cfg, search_out, out, reuse=()):
    """Write the final stage's queue (pretrained weights downloaded first). Runs already done with the same
    settings in the finalist stage or in the folders of reuse are copied, not trained again. Returns the queue path."""
    from .space.network import prefetch_weights
    runs = plan(cfg, search_out, out)
    prefetch_weights([r["model"] for r in runs], cfg.get("Space", "encoder_weights").strip())
    _reuse(runs, Path(out) / "runs", [Path(out) / "finalists" / "runs", *reuse], cfg)
    return runner.write_queue(Path(out) / "runs", runs, cfg)


def _reuse(runs, dst, sources, cfg):
    """Copy into dst every run of runs that is done, with the same deciding settings (so the same name), in one of
    the source folders; each copy gets reused.json saying where it came from. Returns the names copied."""
    import shutil
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for r in runs:
        if (dst / r["run"]).exists():
            continue
        settings = runner.deciding_settings(r, C.with_overrides(cfg, r["overrides"]))
        for src in sources:
            d = Path(src) / r["run"]
            if d.resolve() != (dst / r["run"]).resolve() and runner._done(d, settings):
                shutil.copytree(d, dst / r["run"])
                with open(dst / r["run"] / "reused.json", "w") as f:
                    json.dump(dict(source=str(d.resolve()), time=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
                copied.append(r["run"])
                break
    return copied


# ---- finalist stage ----------------------------------------------------------------------------------------
def finalists(cfg, search_out):
    """The [Final] finalists trained candidates of the search with the best search fitness, best first."""
    t = pd.read_csv(Path(search_out) / "trained.csv")
    t = t[np.isfinite(pd.to_numeric(t.score, errors="coerce"))].copy()
    t = t.assign(_lat=pd.to_numeric(t.latency_ms, errors="coerce").fillna(np.inf),
                 _par=pd.to_numeric(t.params, errors="coerce").fillna(np.inf))
    # the order of the search's winner rule (best fitness, then lowest latency, then fewest parameters), so that
    # finalist 1 is the search's winner
    t = t.sort_values(["score", "_lat", "_par"], ascending=[False, True, True]).drop_duplicates("spec")
    return t.drop(columns=["_lat", "_par"]).head(cfg.getint("Final", "finalists")).reset_index(drop=True)


def finalist_queue(cfg, search_out, out, reuse=()):
    """Write the finalist stage's queue, <out>/finalists/runs: every finalist trained like a final model, one run
    per seed. Runs already done with the same settings in the folders of reuse are copied. Returns the queue path."""
    from .space.network import prefetch_weights
    f = finalists(cfg, search_out)
    if not len(f):
        raise RuntimeError("no finalists: [Final] finalists = 0 or no trained candidate in the search")
    runs = [runner.make_run(r.spec, "final", cfg, {**_common(cfg), "Train.seed": s}, label=f"finalist {i + 1}", seed=s)
            for i, r in enumerate(f.itertuples()) for s in C.get_list(cfg, "Final", "seeds", int)]
    prefetch_weights([r["model"] for r in runs], cfg.get("Space", "encoder_weights").strip())
    d = Path(out) / "finalists" / "runs"
    _reuse(runs, d, reuse, cfg)
    return runner.write_queue(d, runs, cfg)


def one_se_pick(t):
    """t: one row per model with val_mean, val_se, latency_ms. The models within one standard error of the best
    val_mean count as equal; of those, the lowest latency (then the higher val_mean). Returns (index, bar)."""
    best = t.val_mean.idxmax()
    bar = t.val_mean[best] - t.val_se[best]
    tied = t[t.val_mean >= bar]
    lat = pd.to_numeric(tied.latency_ms, errors="coerce").fillna(np.inf)
    order = pd.DataFrame({"lat": lat, "neg": -tied.val_mean}).sort_values(["lat", "neg"])
    return order.index[0], float(bar)


def pick_finalist(fq, cfg, search_out, out):
    """Pick the winner of the finalist stage (one_se_pick on the mean over seeds of the validation fitness in FP16)
    and write <out>/finalists.json. Returns (record, table); the table also has each finalist's search score and
    its re-measured score on the search split, mean over seeds (the size of the winner's curse)."""
    Q = runner.read_queue(fq)
    missing = [r["run"] for r in Q["runs"] if runner.done_scores(Q, r) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} finalist runs not done yet")
    res = results(fq)
    res["val"] = [_val_score(r, cfg) for _, r in res.iterrows()]
    res["search_remeasured"] = [np.nan if (v := runner.fitness(r, cfg)) is None else v for _, r in res.iterrows()]
    g = res.groupby("model", sort=False)
    t = pd.DataFrame({"label": g.label.first(), "seeds": g.size(), "val_mean": g.val.mean(), "val_std": g.val.std(),
                      "search_remeasured": g.search_remeasured.mean()})
    t["val_se"] = (t.val_std / np.sqrt(t.seeds)).fillna(0.0)
    f = finalists(cfg, search_out).set_index("spec")
    t = t.join(f[["score", "latency_ms", "ir_mb", "params", "gen"]].rename(columns={"score": "search_score"}))
    pick, bar = one_se_pick(t)
    t["within_one_se"] = t.val_mean >= bar
    t.index.name = "spec"
    w = t.loc[pick]
    rec = dict(spec=pick, label=w.label, val_mean=float(w.val_mean), val_se=float(w.val_se),
               latency_ms=float(w.latency_ms), search_score=float(w.search_score),
               search_remeasured=float(w.search_remeasured), bar=bar, finalists=int(len(t)),
               rule=(f"best {len(t)} candidates by search fitness, each trained with seeds {C.get_list(cfg, 'Final', 'seeds')} "
                     f"for {cfg.getint('Final', 'steps')} batches; mean validation fitness in FP16 "
                     f"({cfg.get('Search', 'fitness').strip()}) over seeds; finalists within one standard error of the "
                     "best count as equal; of those, the lowest predicted latency"),
               time=time.strftime("%Y-%m-%d %H:%M:%S"), table=t.reset_index().to_dict("records"))
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "finalists.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec, t


def results(q):
    """One row per done run: label, seed, role, model and its scores."""
    Q = runner.read_queue(q)
    rows = []
    for r in Q["runs"]:
        sc = runner.done_scores(Q, r)
        if sc:
            rows.append(dict(label=r.get("label"), seed=r.get("seed"), **{k: v for k, v in sc.items() if k != "settings"}))
    return pd.DataFrame(rows)


def _val_score(row, cfg, prec="fp16"):
    """Mean over the validation sets of the target IoU, as [Search] fitness defines it (scene-balanced or pooled)."""
    v = runner.fitness(row, cfg, prec, sets=C.get_list(cfg, "Data", "val"))
    return np.nan if v is None else v


def deploy(q, cfg, out_dir, name, windows=8):
    """Export the winner seed with the best mean validation score in FP16. Returns a record (also written as
    <out_dir>/<name>/deploy.json)."""
    import torch
    from .hardware import myriad
    from .train import data as D
    from .train.metrics import evaluate, summary
    Q = runner.read_queue(q)
    res = results(q)
    win = res[res.label == "winner"].copy()
    if not len(win):
        raise RuntimeError("no winner run is done yet")
    win["val_fp16"] = [_val_score(r, cfg) for _, r in win.iterrows()]
    pick = win.sort_values("val_fp16", ascending=False).iloc[0]
    net = runner.load_result(Path(Q["out"]) / pick.run)
    spec = D.DataSpec.from_config(cfg)
    shape = D.input_shape(spec)
    rec = dict(name=name, run=pick.run, seed=int(pick.seed), model=pick.model, val_fp16=float(pick.val_fp16),
               rule=f"winner seed with the best mean validation fresh-burn IoU in FP16 ({cfg.get('Search', 'fitness').strip()})", time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if myriad.status(cfg)["model_optimizer"]:
        chk = myriad.export_ir(net, out_dir, name, shape, cfg)
        rec.update(ir_mb=chk.get("ir_mb"), compiled=chk.get("compiled"), mapping=chk.get("mapping"), files=chk.get("files"),
                   layers=chk.get("layers"), argos_unlisted=chk.get("argos_unlisted"), error=chk.get("error"))
        xml = Path(out_dir) / name / f"{name}.xml"
        if xml.exists():
            rows = []
            for n in spec.val:
                s = D.open_set(spec, n, teacher=False)
                idx = np.unique(np.linspace(0, len(s["refl"]) - 1, windows).astype(int))
                x = D.batch_input(torch.from_numpy(np.ascontiguousarray(s["refl"][idx])),
                                  torch.from_numpy(np.ascontiguousarray(s["mask"][idx])), spec)
                with torch.no_grad():
                    y_pt = net.float().cpu()(x).numpy()
                y_ir = myriad.run_ir(xml, x.numpy(), cfg)
                v = np.asarray(s["mask"][idx]) != spec.nodata
                rows.append(dict(set=n, windows=int(len(idx)), largest_logit=float(np.abs(y_pt).max()),
                                 largest_difference=float(np.abs(y_ir - y_pt).max()),
                                 same_class=float((y_ir.argmax(1) == y_pt.argmax(1))[v].mean())))
            rec["ir_vs_pytorch"] = rows
    else:
        rec["error"] = "no OpenVINO 2022.3.1 environment (pynas.hardware.myriad.setup(cfg))"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rec["fp16_agreement"] = {n: summary(evaluate(net, D.open_set(spec, n, teacher=False), spec, device, "fp32",
                                                 agreement=True), spec).get("fp16_agreement") for n in spec.val}
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), model=pick.model, run=pick.run, seed=int(pick.seed)), d / f"{name}.pt")
    with open(d / "deploy.json", "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return rec


def test(q, cfg, teacher_queue=None, out=None, force=False):
    """Scores on [Data] test: every final run (FP16 and FP32; the teacher in FP32), the teacher seeds and their
    mean. Written once to <out>/test_scores.csv, test_per_gsd.csv and test.json; afterwards read back unless
    force. Returns (scores, per resolution)."""
    import torch
    from .train import data as D
    from .train.metrics import evaluate, summary
    Q = runner.read_queue(q)
    out = Path(out or Path(Q["out"]).parent)
    f_scores, f_gsd, f_lock = out / "test_scores.csv", out / "test_per_gsd.csv", out / "test.json"
    if f_lock.exists() and not force:
        return pd.read_csv(f_scores), pd.read_csv(f_gsd)
    spec = D.DataSpec.from_config(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sets = {n: D.open_set(spec, n, teacher=False) for n in spec.test}
    rows, per = [], []

    def score(label, seed, role, run, net, precs):
        for prec in precs:
            row = dict(label=label, seed=seed, role=role, run=run, precision=prec)
            for n, s in sets.items():
                e = evaluate(net, s, spec, device, prec)
                row.update(summary(e, spec, prefix=f"{n}_"))
                lab = e[e.ref == "labels"]
                per.extend(dict(label=label, seed=seed, precision=prec, set=n, gsd=str(g), iou=v)
                           for g, v in zip(lab.gsd, lab[f"{spec.target_name}_iou"]))
            rows.append(row)

    for r in Q["runs"]:
        if runner.done_scores(Q, r):
            net = runner.load_result(Path(Q["out"]) / r["run"], device)
            score(r.get("label"), r.get("seed"), r["role"], r["run"], net, r["evaluate"])
    if teacher_queue:
        from . import teacher
        nets = teacher.models(teacher_queue, device)
        TQ = runner.read_queue(teacher_queue)
        for r, net in zip(TQ["runs"], nets):
            score("teacher", r.get("seed"), "teacher", r["run"], net, ["fp32"])
        if len(nets) > 1:
            score("teacher, mean of seeds", None, "teacher", None, Mean(nets), ["fp32"])
    scores, per_gsd = pd.DataFrame(rows), pd.DataFrame(per)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(f_scores, index=False)
    per_gsd.to_csv(f_gsd, index=False)
    with open(f_lock, "w") as f:
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), sets=spec.test, runs=len(rows)), f, indent=1)
    return scores, per_gsd


def Mean(models):
    """Class probabilities averaged over several models, as log-probabilities (a module)."""
    import torch

    class _Mean(torch.nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.models = torch.nn.ModuleList(ms)

        def forward(self, x):
            p = torch.stack([torch.softmax(m(x).float(), 1) for m in self.models]).mean(0)
            return torch.log(p.clamp(min=1e-12))
    return _Mean(models)

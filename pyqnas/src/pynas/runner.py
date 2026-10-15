"""Trainings in the background: one worker process per GPU, working through a queue of runs.

A run is a model specification (a chromosome or smp:...), a role (candidate, final, baseline, ablation, teacher),
per-run overrides of the configuration, the stored sets it is scored on and the precisions it is scored in. Each
worker takes the next run nobody has taken, trains it (train/trainer.py), scores it, writes <out>/<run>/ and takes
the next one. The workers do not depend on the process that started them.

    q = write_queue(out, runs, cfg)
    launch(q, gpus=[0, 1])           # returns at once
    wait(q)                          # progress every few minutes until the queue is empty
    stop(q)                          # only to stop early (the runs are given out again at the next launch)

A run is named after everything that decides its result (run_name), and is done when <out>/<run>/scores.json
holds the same settings: a run met again (a duplicate, a restarted search) is read back, never trained twice.

<out>/<run>/: run.json (what it is and the environment), config.ini (the effective configuration),
history.csv, model.pt (the result: weights, folded layers, specification), scores.json (written last).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

CLAIM, FAILED, SCORES, WORKERS = "claim.json", "failed.txt", "scores.json", "workers.json"
MARK = "pynas.runner"
ROLES = ("candidate", "final", "baseline", "ablation", "teacher")
# options that decide what a training gives; [Run] and the loader/logging options do not
DECIDING = ("Data", "Augment", "Train", "Teacher")
_NOT_DECIDING = {("Train", "loader_workers"), ("Train", "log_every"), ("Data", "test")}
_STEP = re.compile(r"^\S+ \S+ (\S+) step (\d+)/(\d+)")
_PROCS = {}


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---- runs and names ---------------------------------------------------------------------------------
def make_run(model, role, cfg, overrides=None, score_sets=None, evaluate=None, **meta):
    """A run: model specification, role, overrides {"Section.option": value}, the sets it is scored on (default:
    [Data] search for candidates, search + val otherwise), the precisions it is scored in (default fp16 and fp32,
    fp32 only for the teacher), and any other key to keep with it (gen, label, seed)."""
    if role not in ROLES:
        raise ValueError(f"role {role}: one of {ROLES}")
    ov = {k: str(v) for k, v in (overrides or {}).items()}
    cfg_run = C.with_overrides(cfg, ov)
    if score_sets is None:
        score_sets = C.get_list(cfg_run, "Data", "search") + ([] if role == "candidate" else C.get_list(cfg_run, "Data", "val"))
    if evaluate is None:
        evaluate = ["fp32"] if role == "teacher" else ["fp16", "fp32"]
    r = dict(model=str(model), role=role, overrides=ov, score_sets=list(score_sets), evaluate=list(evaluate), **meta)
    r["run"] = run_name(r, cfg_run)
    return r


def deciding_settings(run, cfg_run):
    """Everything that decides a run's result: its model, role, score sets and precisions, and the deciding options
    of its effective configuration."""
    out = dict(model=run["model"], role=run["role"], score_sets=run["score_sets"], evaluate=run["evaluate"],
               encoder_weights=cfg_run.get("Space", "encoder_weights").strip())
    for sec in DECIDING:
        if sec == "Teacher" and run["role"] != "teacher":
            continue
        for k, v in cfg_run.items(sec, raw=True):
            if (sec, k) not in _NOT_DECIDING:
                out[f"{sec}.{k}"] = v.strip()
    kd = cfg_run.getfloat("Teacher" if run["role"] == "teacher" else "Train", "kd_weight")
    if kd > 0:                       # soft labels rewritten (a new teacher) make a new run
        f = Path(C.path(cfg_run, "Data", "pool")) / f"{cfg_run.get('Data', 'train').strip()}_{cfg_run.get('Data', 'teacher').strip()}.npy"
        out["teacher_file"] = f"{f.stat().st_size}:{int(f.stat().st_mtime)}" if f.exists() else "missing"
    return out


def run_name(run, cfg_run):
    """r_<12 hex>: the same deciding settings give the same name."""
    h = hashlib.sha1(json.dumps(deciding_settings(run, cfg_run), sort_keys=True).encode()).hexdigest()
    return f"r_{h[:12]}"


# ---- the queue ------------------------------------------------------------------------------------------
def write_queue(out, runs, cfg):
    """<out>/queue.json with the runs and the configuration their overrides apply to. Returns its path."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = [r["run"] for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("two runs of the queue have the same name (the same run twice)")
    q, tmp = out / "queue.json", out / "queue.json.tmp"
    with open(tmp, "w") as f:
        json.dump(dict(out=str(out), config=C.text(cfg), runs=runs), f, indent=1)
    os.replace(tmp, q)
    return q


def read_queue(q):
    with open(q) as f:
        return json.load(f)


def run_config(Q, run):
    return C.loads(Q["config"], run.get("overrides"))


def _done(d, settings):
    try:
        with open(Path(d) / SCORES) as f:
            return json.load(f).get("settings") == settings
    except (OSError, ValueError):
        return False


def done_scores(Q, run):
    """The scores of a run that is done with its settings, or None."""
    d = Path(Q["out"]) / run["run"]
    if _done(d, deciding_settings(run, run_config(Q, run))):
        with open(d / SCORES) as f:
            return json.load(f)
    return None


# ---- workers -----------------------------------------------------------------------------------------------
def _alive(pid):
    """A worker still running: not finished, not a zombie, and not another program that got its pid since."""
    p = _PROCS.get(pid)
    if p is not None:
        return p.poll() is None
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if not os.path.isdir("/proc"):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            if MARK.encode() not in f.read():
                return False
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def workers(q):
    try:
        with open(Path(read_queue(q)["out"]) / WORKERS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def state(q):
    """Every run: done, running (and on which GPU), failed, stopped (its worker died) or waiting."""
    Q = read_queue(q)
    out = Path(Q["out"])
    rows = []
    for r in Q["runs"]:
        d = out / r["run"]
        st, gpu = "waiting", None
        if _done(d, deciding_settings(r, run_config(Q, r))):
            st = "done"
        elif (d / CLAIM).exists():
            try:
                with open(d / CLAIM) as f:
                    c = json.load(f)
            except ValueError:
                c = {}
            gpu = c.get("gpu")
            st = "failed" if (d / FAILED).exists() else ("running" if not c or _alive(c.get("pid")) else "stopped")
        rows.append(dict(run=r["run"], state=st, gpu=gpu))
    return pd.DataFrame(rows, columns=["run", "state", "gpu"]).set_index("run")


def progress(q):
    """state() with how far each run is (from the logs) and, for the runs done, minutes and main score."""
    Q = read_queue(q)
    out = Path(Q["out"])
    st = state(q)
    last = {}
    for f in sorted(out.glob("log_gpu*.txt")):
        with open(f, errors="replace") as fh:
            for line in fh:
                m = _STEP.match(line)
                if m:
                    last[m.group(1)] = f"{m.group(2)}/{m.group(3)}"
    meta = {r["run"]: r for r in Q["runs"]}
    st["step"] = [last.get(run, "") for run in st.index]
    for key in ("role", "gen", "label", "seed"):
        if any(key in r for r in Q["runs"]):
            st[key] = [meta[r].get(key) for r in st.index]
    st["model"] = [meta[r]["model"] for r in st.index]
    for run in st.index[st.state == "done"]:
        with open(out / run / SCORES) as f:
            sc = json.load(f)
        st.loc[run, "minutes"] = sc.get("minutes")
        st.loc[run, "score"] = sc.get("score")
    st["gpu"] = pd.to_numeric(st.gpu, errors="coerce").astype("Int64")
    return st


def _src_root():
    return str(Path(__file__).resolve().parents[1])


def launch(q, gpus=(0,), loader_workers=None, python=None):
    """Start one worker per GPU in the background and return. Failed runs, and runs left by a worker that died,
    are given out again. Nothing is started while workers of an earlier launch are alive."""
    Q = read_queue(q)
    out = Path(Q["out"])
    alive = [w for w in workers(q) if _alive(w["pid"])]
    if alive:
        print("workers of an earlier launch are still running: " + ", ".join(f"GPU {w['gpu']} (pid {w['pid']})" for w in alive))
        return alive
    st = state(q)
    for run in st.index[st.state.isin(["failed", "stopped"])]:
        for f in (CLAIM, FAILED):
            (out / run / f).unlink(missing_ok=True)
    todo = int((st.state != "done").sum())
    if not todo:
        print("every run of the queue is done")
        return []
    gpus = list(gpus)[:todo] or [0]
    if loader_workers is None:
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or 4
        loader_workers = max(2, min(6, cores // max(len(gpus), 1) - 1))
    entry = f"import sys; sys.path.insert(0, {_src_root()!r}); from {MARK} import main; main()"
    started = []
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
        with open(out / f"log_gpu{g}.txt", "a") as log:
            p = subprocess.Popen([python or sys.executable, "-c", entry, "--queue", str(Path(q).resolve()), "--gpu", str(g),
                                  "--loader-workers", str(loader_workers)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        _PROCS[p.pid] = p
        started.append(dict(gpu=g, pid=p.pid, started=_now(), loader_workers=loader_workers))
    with open(out / WORKERS, "w") as f:
        json.dump(started, f, indent=1)
    print(f"{todo} run(s) to do, {len(started)} worker(s): " + ", ".join(f"GPU {w['gpu']} (pid {w['pid']})" for w in started)
          + f"; logs in {out}/log_gpu<k>.txt", flush=True)
    return started


def stop(q):
    for w in workers(q):
        if _alive(w["pid"]):
            try:
                os.killpg(int(w["pid"]), signal.SIGTERM)
                print(f"stopped the worker on GPU {w['gpu']} (pid {w['pid']})")
            except ProcessLookupError:
                pass


def wait(q, every=300, first=30, table=True):
    """progress() every `every` seconds until no run is waiting or running. Interrupting it does not stop the
    workers. If every worker stopped with runs left, the end of each log is shown."""
    try:
        from IPython.display import clear_output, display
    except ImportError:
        clear_output, display = (lambda wait=False: None), print
    st, pause = None, first
    try:
        while True:
            alive = any(_alive(w["pid"]) for w in workers(q))
            st = progress(q)
            counts = ", ".join(f"{n} {s}" for s, n in st.state.value_counts().items())
            if table:
                clear_output(wait=True)
                print(_now() + "   " + counts)
                display(st.drop(columns=["model"]))
            else:
                print(f"{_now()}   {counts}", flush=True)
            if not st.state.isin(["waiting", "running"]).any():
                break
            if not alive:
                print("no worker is running; the end of each log:")
                for f in sorted(Path(read_queue(q)["out"]).glob("log_gpu*.txt")):
                    with open(f, errors="replace") as fh:
                        print(f"--- {f.name}\n" + "".join(fh.readlines()[-20:]))
                break
            time.sleep(pause)
            pause = every
    except KeyboardInterrupt:
        print("stopped watching; the workers go on")
    return st


def claim_next(Q, out, gpu):
    """The next run nobody has taken, taken by this worker (claim.json created only if it does not exist)."""
    for r in Q["runs"]:
        d = out / r["run"]
        if _done(d, deciding_settings(r, run_config(Q, r))):
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(d / CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(dict(gpu=gpu, pid=os.getpid(), host=socket.gethostname(), started=_now()), f)
        return r
    return None


# ---- one run ------------------------------------------------------------------------------------------------
def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _clean(d):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else
                float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}


def run_one(r, Q, out, opened, device, loader_workers, log, gpu):
    """Train and score one run; write its folder. Returns its scores."""
    import torch
    from .record import environment
    from .space.network import build, n_params
    from .train import data as D
    from .train.metrics import evaluate, summary
    from .train.trainer import Settings, train
    t0 = time.time()
    cfg = run_config(Q, r)
    settings = deciding_settings(r, cfg)
    d = out / r["run"]
    spec = D.DataSpec.from_config(cfg)
    s = Settings.from_config(cfg, over="Teacher" if r["role"] == "teacher" else None)
    key = (spec.pool, spec.teacher)
    if key not in opened:
        opened[key] = {}
    sets = opened[key]
    for name in [spec.train] + list(r["score_sets"]):
        if name not in sets:
            sets[name] = D.open_set(spec, name, teacher=True)
    if s.kd_weight > 0 and sets[spec.train].get("teacher") is None:
        raise RuntimeError(f"distillation needs {spec.prefix(spec.train)}_{spec.teacher}.npy")
    shape = D.input_shape(spec)
    model = build(r["model"], shape[0], spec.num_classes, shape[1:], cfg.get("Space", "encoder_weights").strip())
    (d / "config.ini").write_text(C.text(cfg))
    _write_json(d / "run.json", dict(run=r, settings=dataclasses.asdict(s), params=n_params(model),
                                     input_shape=list(shape), num_classes=spec.num_classes, environment=environment()))
    lw = loader_workers if loader_workers is not None else 3
    batches = D.loader(sets[spec.train], spec, s.batch, s.steps, workers=lw, seed=1 + s.seed, teacher=s.kd_weight > 0)
    ex = sets[spec.train]
    example = D.batch_input(torch.from_numpy(np.ascontiguousarray(ex["refl"][:8])).to(device),
                            torch.from_numpy(np.ascontiguousarray(ex["mask"][:8])).to(device), spec)
    main_prec = r["evaluate"][0]

    def eval_fn(m, step):
        res = {}
        for n in r["score_sets"]:
            res[f"{n}_{main_prec}_iou"] = summary(evaluate(m, sets[n], spec, device, main_prec), spec)[f"labels_{spec.target_name}_iou"]
        return res

    logf = (lambda line: _append(log, line)) if log else None
    result, hist, info = train(model, batches, spec, s, device, example, eval_fn=eval_fn if s.eval_every else None,
                               log=logf, name=r["run"])
    torch.save(dict(state=result.state_dict(), model=r["model"], pairs=info.get("pairs", []), folded=bool(info.get("folded")),
                    input_shape=list(shape), num_classes=spec.num_classes, run=r["run"]), d / "model.pt")
    hist.to_csv(d / "history.csv", index=False)
    row = dict(run=r["run"], model=r["model"], role=r["role"], gpu=gpu, params=n_params(result), **{k: v for k, v in info.items() if k != "pairs"})
    for n in r["score_sets"]:
        for prec in r["evaluate"]:
            sc = evaluate(result, sets[n], spec, device, prec, agreement=(prec == "fp32" and "fp16" in r["evaluate"]))
            row.update(summary(sc, spec, prefix=f"{n}_{prec}_"))
    row["score"] = fitness(row, cfg, main_prec)                    # the fitness: [Search] fitness on the search sets
    row["minutes"] = (time.time() - t0) / 60
    row = _clean(row)
    _write_json(d / SCORES, dict(row, settings=settings))
    return row


def fitness(scores, cfg, prec="fp16", sets=None):
    """Mean over the sets ([Data] search by default) of the target IoU in precision prec, scene-balanced
    ([Search] fitness = scene_mean) or over all pixels (pooled). A set where no scene has enough labelled target
    pixels for the scene-balanced IoU counts with its pooled IoU. None when no set has a value."""
    kind = cfg.get("Search", "fitness").strip()
    if kind not in ("scene_mean", "pooled"):
        raise ValueError(f"[Search] fitness = {kind}: scene_mean or pooled")
    t = cfg.get("Data", "target").strip()
    vals = []
    for n in (sets or C.get_list(cfg, "Data", "search")):
        pooled = scores.get(f"{n}_{prec}_labels_{t}_iou")
        v = scores.get(f"{n}_{prec}_labels_{t}_iou_scene_mean", pooled) if kind == "scene_mean" else pooled
        vals.append(v)
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _append(path, line):
    with open(path, "a") as f:
        f.write(f"{_now()} {line}\n")


def load_result(run_dir, device="cpu"):
    """The trained model of a run folder (folded where it was), in eval mode."""
    import torch
    from .space.network import build
    from .train.precision import fold_structure
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    shape = ck["input_shape"]
    net = build(ck["model"], shape[0], ck["num_classes"], tuple(shape[1:]), encoder_weights="none")
    if ck.get("folded"):
        fold_structure(net, [tuple(p) for p in ck["pairs"]])
    net.load_state_dict(ck["state"])
    return net.to(device).eval()


def main(argv=None):
    a = argparse.ArgumentParser(description="worker: trains the runs of a queue one after the other on one GPU")
    a.add_argument("--queue", required=True)
    a.add_argument("--gpu", type=int, required=True)
    a.add_argument("--loader-workers", type=int, default=3)
    args = a.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    import torch
    torch.backends.cudnn.benchmark = True
    Q = read_queue(args.queue)
    out = Path(Q["out"])
    log = out / f"log_gpu{args.gpu}.txt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opened = {}
    _append(log, f"worker pid {os.getpid()} on GPU {args.gpu} "
                 f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'no GPU'}), {args.loader_workers} window readers")
    while True:
        r = claim_next(Q, out, args.gpu)
        if r is None:
            break
        _append(log, f"{r['run']}: start {r['role']} {r['model']}" + (f"  {r['overrides']}" if r.get("overrides") else ""))
        try:
            row = run_one(r, Q, out, opened, device, args.loader_workers, log, args.gpu)
            (out / r["run"] / CLAIM).unlink(missing_ok=True)
            _append(log, f"{r['run']}: done in {row['minutes']:.1f} min; score {row.get('score')}")
        except Exception:                                    # noqa: BLE001 - recorded, the worker goes on
            tb = traceback.format_exc()
            (out / r["run"] / FAILED).write_text(tb)
            _append(log, f"{r['run']}: failed\n{tb}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _append(log, "no run left in the queue")


if __name__ == "__main__":
    main()

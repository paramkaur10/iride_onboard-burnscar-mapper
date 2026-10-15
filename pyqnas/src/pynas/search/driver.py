"""The search, from one configuration file, in a background process of its own.

    launch("configs/burnscar.ini", "models/pynas/search")    # returns at once; goes on when the notebook closes
    status("models/pynas/search")                             # where it is
    stop("models/pynas/search")                               # stops the driver and its workers (launch resumes)

Generation 0: random chromosomes of the space ([Space]); a chromosome tried before is skipped and an infeasible
one (hardware/constraints.py) is replaced, until [Search] population feasible candidates. Generation g > 0: the
same number of new feasible children, each from two parents drawn by binary tournament on (Pareto rank, crowding
distance) of the current population, one-point crossover ([Search] crossover), mutation ([Search] mutation),
repair to the limits. The candidates of a generation are trained in the background (runner.py, one worker per
GPU) and scored; the next population is the best [Search] population of parents and children by NSGA-II on
[Search] objectives. After the last generation, the winner is picked by the rule of [Final] pick.

The output folder:
    config.ini             the effective configuration of the search
    driver_log.txt         what the driver did
    state.json             generation and phase
    candidates.csv         every chromosome tried: generation, feasibility and why not, sizes, latency
    gen_<g>.json           the feasible candidates of generation g with their checks
    runs/<run>/            one folder per trained candidate (runner.py)
    trained.csv            every trained candidate with its scores
    generations.csv        the population after each generation, with Pareto rank and crowding distance
    winner.json            the winner and the rule that picked it
Launching a stopped search resumes it: finished parts are read back, trained runs are never trained again.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import runner
from ..hardware.constraints import feasibility
from ..space import genes
from . import nsga2

DRIVER = "driver.json"
MARK = "pynas.search.driver"


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def objectives(cfg):
    """[(name, sense)] from [Search] objectives, e.g. [('iou', 'max'), ('latency_ms', 'min')]."""
    out = []
    for item in C.get_list(cfg, "Search", "objectives"):
        name, _, sense = item.partition(":")
        sense = sense.strip() or "max"
        if sense not in ("max", "min"):
            raise ValueError(f"[Search] objectives: {item} (name:max or name:min)")
        out.append((name.strip(), sense))
    return out


def objective_matrix(table, objs):
    """Minimised objective values of a table of trained candidates ('iou' is the fitness column 'score'). An
    objective no candidate has (e.g. latency_ms with [Hardware] latency = off) is left out; a candidate missing a
    value of the others ranks behind every candidate that has them (nsga2)."""
    cols = []
    for name, sense in objs:
        col = "score" if name == "iou" else name
        v = pd.to_numeric(table[col], errors="coerce").to_numpy(float) if col in table else np.full(len(table), np.nan)
        if name != "iou" and np.isnan(v).all():
            continue
        cols.append(-v if sense == "max" else v)
    return np.stack(cols, 1) if cols else np.zeros((len(table), 0))


class Search:
    def __init__(self, cfg, out, log=print):
        self.cfg, self.out, self.log = cfg, Path(out).resolve(), log
        self.space = genes.Space.from_config(cfg)
        self.P = cfg.getint("Search", "population")
        self.G = cfg.getint("Search", "generations")
        self.seed = cfg.getint("Search", "seed")
        self.objs = objectives(cfg)
        from ..train.data import DataSpec, input_shape
        spec = DataSpec.from_config(cfg)
        self.input_shape, self.num_classes = input_shape(spec), spec.num_classes
        self.out.mkdir(parents=True, exist_ok=True)

    # -- records
    def tried(self):
        f = self.out / "candidates.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["gen", "spec", "feasible", "reason"])

    def trained(self):
        f = self.out / "trained.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["gen", "run", "spec", "score"])

    # -- candidates
    def _feasible(self, g, n, make):
        """n new feasible chromosomes from make(rng, np_rng), checked in batches; every one tried is recorded."""
        rng, nrng = random.Random(self.seed * 1000 + g), np.random.default_rng(self.seed * 1000 + g)
        seen = set(self.tried().spec)
        found, rows, tries = [], [], 0
        limit = self.cfg.getint("Search", "max_tries")
        while len(found) < n:
            batch = []
            while len(batch) < max(4, 2 * (n - len(found))) and tries < limit:
                spec = make(rng, nrng).write()
                tries += 1
                if spec not in seen:
                    seen.add(spec)
                    batch.append(spec)
            if not batch:
                pd.concat([self.tried(), pd.DataFrame(rows)], ignore_index=True).to_csv(self.out / "candidates.csv", index=False)
                why = pd.Series([r["reason"] for r in rows if r["reason"]], dtype=object).str.split(":").str[0].value_counts()
                raise RuntimeError(f"generation {g}: only {len(found)} new feasible candidates after {tries} draws "
                                   f"([Search] max_tries); reasons: {dict(why.head(5))} (all in candidates.csv)")
            names = {f"g{g}_{len(rows) + i}": s for i, s in enumerate(batch)}
            recs = feasibility(names, self.cfg, self.out / "checks" / f"gen{g}", self.input_shape, self.num_classes,
                               log=self.log)
            for name, spec in names.items():
                rec = recs[name]
                rows.append(dict(gen=g, spec=spec, feasible=rec["feasible"], reason=rec.get("reason"),
                                 params=rec.get("params"), est_mb=rec.get("est_mb"), ir_mb=rec.get("ir_mb"),
                                 compiled=rec.get("compiled"), latency_ms=rec.get("latency_ms"),
                                 layers=json.dumps(rec.get("layers")) if rec.get("layers") else None))
                if rec["feasible"] and len(found) < n:
                    found.append(rows[-1])
            self.log(f"generation {g}: {len(found)}/{n} feasible after {tries} draws")
        tried = pd.concat([self.tried(), pd.DataFrame(rows)], ignore_index=True)
        tried.to_csv(self.out / "candidates.csv", index=False)
        return found

    def _random(self, rng, nrng):
        return genes.random_chromosome(self.space, rng)

    def _child_maker(self, pop):
        F = objective_matrix(pop, self.objs)
        specs = list(pop.spec)
        pc, pm = self.cfg.getfloat("Search", "crossover"), self.cfg.getfloat("Search", "mutation")

        def make(rng, nrng):
            a, b = (genes.read(specs[i]) for i in nsga2.tournament(F, nrng, 2))
            if rng.random() < pc:
                c = rng.choice(genes.crossover(a, b, rng, self.space))
            else:
                c = a
            return genes.mutate(c, self.space, rng, pm)
        return make

    # -- training
    def _train(self, g, cands):
        runs = [runner.make_run(c["spec"], "candidate", self.cfg, gen=g) for c in cands]
        q = runner.write_queue(self.out / "runs", runs, self.cfg)
        gpus = gpu_list(self.cfg)
        self.log(f"generation {g}: training {len(runs)} candidates on GPUs {gpus}")
        runner.launch(q, gpus=gpus)
        runner.wait(q, every=120, first=20, table=False)
        Q = runner.read_queue(q)
        rows = []
        for c, r in zip(cands, runs):
            sc = runner.done_scores(Q, r) or {}
            rows.append(dict(gen=g, run=r["run"], spec=c["spec"], score=runner.fitness(sc, self.cfg) if sc else None,
                             score_pooled=runner.fitness(sc, C.with_overrides(self.cfg, {"Search.fitness": "pooled"})) if sc else None,
                             latency_ms=c.get("latency_ms"),
                             ir_mb=c.get("ir_mb"), params=c.get("params"), est_mb=c.get("est_mb"),
                             minutes=sc.get("minutes"), not_finite_steps=sc.get("not_finite_steps"),
                             folded_pairs=sc.get("folded_pairs"), trained=bool(sc),
                             **{k: v for k, v in sc.items() if k.endswith("_iou") and not k.startswith("score")}))
        failed = sum(not r["trained"] for r in rows)
        if failed:
            self.log(f"generation {g}: {failed} candidates did not train (see runs/<run>/failed.txt); they rank last")
        t = pd.concat([self.trained(), pd.DataFrame(rows)], ignore_index=True)
        t = t.drop_duplicates("run", keep="last")
        t.to_csv(self.out / "trained.csv", index=False)
        return pd.DataFrame(rows)

    # -- the whole search
    def run(self):
        (self.out / "config.ini").write_text(C.text(self.cfg))
        self._prefetch()
        pop = None
        for g in range(self.G):
            _write_json(self.out / "state.json", dict(generation=g, phase="sampling", updated=_now()))
            gfile = self.out / f"gen_{g}.json"
            if gfile.exists():
                cands = _read_json(gfile)
            else:
                cands = self._feasible(g, self.P, self._random if g == 0 else self._child_maker(pop))
                _write_json(gfile, cands)
            _write_json(self.out / "state.json", dict(generation=g, phase="training", updated=_now()))
            new = self._train(g, cands)
            both = new if pop is None else pd.concat([pop, new], ignore_index=True)
            keep = nsga2.select(objective_matrix(both, self.objs), self.P)
            pop = both.iloc[keep].reset_index(drop=True)
            front, crowd = nsga2.rank(objective_matrix(pop, self.objs))
            gens = pd.read_csv(self.out / "generations.csv") if (self.out / "generations.csv").exists() else pd.DataFrame()
            if len(gens) and "population_after" in gens:
                gens = gens[gens.population_after != g]
            layer = pop.assign(population_after=g, front=front, crowding=crowd)
            pd.concat([gens, layer], ignore_index=True).to_csv(self.out / "generations.csv", index=False)
            best = pop.loc[pd.to_numeric(pop.score, errors="coerce").idxmax()] if pop.score.notna().any() else None
            self.log(f"generation {g} done; best score {None if best is None else round(float(best.score), 4)}")
        w = pick_winner(self.trained(), self.cfg)
        _write_json(self.out / "winner.json", w)
        _write_json(self.out / "state.json", dict(generation=self.G - 1, phase="finished", updated=_now()))
        self.log(f"winner: {w.get('spec')} (score {w.get('score')}, latency {w.get('latency_ms')} ms)")
        return w

    def _prefetch(self):
        from ..space.encoders import prefetch
        names = [genes.ENCODERS[e] for e in self.space.encoders if e != "none"]
        weights = self.cfg.get("Space", "encoder_weights").strip()
        if names and weights.lower() not in ("", "none", "random"):
            for n, msg in prefetch(names, weights=weights).items():
                if msg != "ok":
                    self.log(f"pretrained weights of {n} not available: {msg}")


def pick_winner(trained, cfg):
    """[Final] pick = best_iou: the trained candidate with the best fitness; those within [Final] tie of it count
    as equal and the one with the lowest latency (then the fewest parameters) is taken."""
    rule = cfg.get("Final", "pick").strip()
    if rule != "best_iou":
        raise ValueError(f"[Final] pick = {rule}: only best_iou is defined")
    tie = cfg.getfloat("Final", "tie")
    t = trained[pd.to_numeric(trained.score, errors="coerce").notna()].copy()
    if not len(t):
        return dict(rule=rule, error="no trained candidate has a score")
    t["score"] = t.score.astype(float)
    top = t[t.score >= t.score.max() - tie]
    lat = pd.to_numeric(top.get("latency_ms"), errors="coerce") if "latency_ms" in top else pd.Series(np.nan, top.index)
    top = top.assign(_lat=lat.fillna(np.inf), _par=pd.to_numeric(top.params, errors="coerce").fillna(np.inf))
    w = top.sort_values(["_lat", "_par", "score"], ascending=[True, True, False]).iloc[0]
    out = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in w.drop(["_lat", "_par"]).items()}
    out.update(rule=rule, tie=tie, candidates_in_tie=int(len(top)), picked=_now())
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in out.items()}


def gpu_list(cfg):
    raw = cfg.get("Run", "gpus").strip().lower()
    if raw in ("", "all"):
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:                                    # noqa: BLE001
            n = 0
        return list(range(n)) or [0]
    return [int(g) for g in raw.split(",") if g.strip()]


# ---- the background process ----------------------------------------------------------------------------
def main(argv=None):
    a = argparse.ArgumentParser(description="PyNAS search driver")
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)
    out = Path(args.out)

    def log(msg):
        print(f"{_now()} {msg}", flush=True)

    try:
        Search(C.load(args.config), out, log).run()
    except Exception:                                        # noqa: BLE001 - recorded
        (out / "driver_failed.txt").write_text(traceback.format_exc())
        log("failed:\n" + traceback.format_exc())
        raise


def _alive(pid):
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            if MARK.encode() not in f.read():
                return False
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError, TypeError):
        return False


def launch(config_path, out, python=None):
    """Start the search in the background (or report that it runs). Returns the driver's pid."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = _read_json(out / DRIVER, {}) or {}
    if d.get("pid") and _alive(d["pid"]):
        print(f"the search is already running (pid {d['pid']}, started {d.get('started')})")
        return d["pid"]
    (out / "driver_failed.txt").unlink(missing_ok=True)
    src = str(Path(__file__).resolve().parents[2])
    entry = f"import sys; sys.path.insert(0, {src!r}); from {MARK} import main; main()"
    with open(out / "driver_log.txt", "a") as logf:
        p = subprocess.Popen([python or sys.executable, "-c", entry, "--config", str(Path(config_path).resolve()),
                              "--out", str(out)], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=os.getcwd(), env=dict(os.environ, PYTHONUNBUFFERED="1"))
    _write_json(out / DRIVER, dict(pid=p.pid, started=_now(), config=str(Path(config_path).resolve())))
    print(f"search started (pid {p.pid}); log: {out / 'driver_log.txt'}")
    return p.pid


def status(out):
    """Where the search is: driver, generation and phase, runs of the generation in training, best so far."""
    out = Path(out)
    st = _read_json(out / "state.json", {}) or {}
    d = _read_json(out / DRIVER, {}) or {}
    info = dict(driver="running" if d.get("pid") and _alive(d["pid"]) else "not running", **st)
    if (out / "driver_failed.txt").exists():
        info["error"] = (out / "driver_failed.txt").read_text()[-1500:]
    q = out / "runs" / "queue.json"
    runs = runner.progress(q) if q.exists() else None
    if runs is not None:
        info["runs"] = ", ".join(f"{n} {s}" for s, n in runs.state.value_counts().items())
    t = out / "trained.csv"
    if t.exists():
        tr = pd.read_csv(t)
        if tr.score.notna().any():
            b = tr.loc[tr.score.idxmax()]
            info["best"] = f"{b.score:.4f} (generation {int(b.gen)}): {b.spec}"
        info["trained"] = int(tr.score.notna().sum())
    return info, runs


def stop(out):
    """Stop the driver and the workers of the generation in training (launch resumes)."""
    out = Path(out)
    d = _read_json(out / DRIVER, {}) or {}
    if d.get("pid") and _alive(d["pid"]):
        try:
            os.killpg(int(d["pid"]), signal.SIGTERM)
            print(f"stopped the driver (pid {d['pid']})")
        except ProcessLookupError:
            pass
    q = out / "runs" / "queue.json"
    if q.exists():
        runner.stop(q)

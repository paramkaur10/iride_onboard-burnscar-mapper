"""The whole experiment on a tiny synthetic window pool: teacher (2 seeds) and its soft labels, a search (3
candidates x 2 generations), the final stage (winner, a baseline, two ablations), deployment record, test once.
Runs on the CPU in a few minutes with a real PyTorch."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pynas import config, final, runner, teacher
from pynas.search import driver


def _pool(root, size=64):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    json.dump(dict(mean=[0.1] * 7, std=[0.05] * 7, clip=1.2), open(root / "norm.json", "w"))
    for name, n, scenes in (("train", 24, ["a", "b", "c"]), ("heo_search", 6, ["d"]), ("heo_val", 6, ["e"]),
                            ("heo_test", 6, ["f"])):
        np.save(root / f"{name}_refl.npy", rng.integers(0, 3000, (n, 7, size, size)).astype(np.int16))
        mask = rng.choice(np.array([0, 2, 3, 4, 5, 6], np.uint8), (n, size, size))
        mask[:, :3] = 255
        np.save(root / f"{name}_mask.npy", mask)
        pd.DataFrame(dict(i=range(n), scene=[scenes[k % len(scenes)] for k in range(n)],
                          source=["heo" if k % 2 else "phisat2_sim" for k in range(n)],
                          gsd=[[2.5, 5.0, 10.0][k % 3] for k in range(n)])).to_csv(root / f"{name}_windows.csv", index=False)


def _config(tmp):
    pool = tmp / "pool"
    _pool(pool)
    text = f"""
[Data]
pool = {pool}
search = heo_search
val = heo_val
test = heo_test
norm = {pool / 'norm.json'}
source_share = heo: 0.5, phisat2_sim: 0.5
[Space]
encoders = none, r18
encoder_depths = 3
min_layers_plain = 1
max_layers = 2
max_stride = 16
widths = 8, 16
decoder_widths = 8, 16
[Train]
steps = 6
batch = 4
log_every = 2
final_phase = 0.5
warmup = 2
[Search]
population = 3
generations = 2
max_tries = 500
[Hardware]
check = estimate
latency = off
[Final]
seeds = 0
steps = 6
eval_every = 3
baselines = smp:Unet:resnet18:depth=3:decoder=16-8-4
[Teacher]
spec = smp:Unet:resnet18
steps = 6
warmup = 2
[Run]
gpus = 0
"""
    path = tmp / "experiment.ini"
    path.write_text(text)
    return config.load(path), path


def test_whole_experiment():
    tmp = Path(tempfile.mkdtemp())
    cfg, path = _config(tmp)
    out = tmp / "models"
    # teacher, then its soft labels
    tq = teacher.queue(cfg, out / "teacher")
    runner.launch(tq, gpus=[0])
    runner.wait(tq, every=1, first=1, table=False)
    st = runner.state(tq)
    assert (st.state == "done").all(), st
    written = teacher.write_soft_labels(tq, cfg, batch=4)
    assert set(written) == {"train", "heo_search", "heo_val"} and all(os.path.exists(p) for p in written.values())
    # search
    w = driver.Search(cfg, out / "search", log=print).run()
    tried = pd.read_csv(out / "search" / "candidates.csv")
    trained = pd.read_csv(out / "search" / "trained.csv")
    assert len(trained) == 6 and trained.score.notna().all(), trained
    assert tried.spec.is_unique
    gens = pd.read_csv(out / "search" / "generations.csv")
    assert sorted(gens.population_after.unique()) == [0, 1] and (gens.groupby("population_after").size() == 3).all()
    assert w["spec"] in set(trained.spec) and w["score"] == trained.score.max()
    for r in trained.run:                               # every candidate trained in FP16 and folded
        sc = json.load(open(out / "search" / "runs" / r / "scores.json"))
        rj = json.load(open(out / "search" / "runs" / r / "run.json"))
        assert rj["settings"]["precision"] == "fp16" and sc["folded"] is True
        assert "pynas_code_sha" in rj["environment"]
    # resuming reads everything back
    w2 = driver.Search(cfg, out / "search", log=print).run()
    assert w2["spec"] == w["spec"] and len(pd.read_csv(out / "search" / "trained.csv")) == 6
    # final stage
    fq = final.queue(cfg, out / "search", out / "final")
    runner.launch(fq, gpus=[0])
    runner.wait(fq, every=1, first=1, table=False)
    res = final.results(fq)
    assert set(res.label) == {"winner", "smp:Unet:resnet18:depth=3:decoder=16-8-4", "winner, no_kd", "winner, fp32"}, res.label
    abl = json.load(open(out / "final" / "runs" / res[res.label == "winner, fp32"].run.iloc[0] / "run.json"))
    assert abl["settings"]["precision"] == "fp32" and abl["settings"]["fold"] is False
    from pynas.space.network import n_params, n_params_used
    net = runner.load_result(out / "final" / "runs" / res.run.iloc[0])
    assert 0 < n_params_used(net, (7, 64, 64)) <= n_params(net)
    dep = final.deploy(fq, cfg, out / "onboard", "burnscar_test")
    assert dep["run"] in set(res[res.label == "winner"].run)
    assert (out / "onboard" / "burnscar_test" / "deploy.json").exists()
    # test, once
    scores, per = final.test(fq, cfg, teacher_queue=tq)
    assert {"winner", "teacher", "teacher, mean of seeds"} <= set(scores.label)
    assert (out / "final" / "test.json").exists()
    again, _ = final.test(fq, cfg, teacher_queue=tq)
    assert len(again) == len(scores)
    # round 2: finalist stage, then the final stage of its pick, reusing every run already done
    cfg2 = config.with_overrides(cfg, {"Final.finalists": "2"})
    r2 = out / "final_r2"
    q1 = final.finalist_queue(cfg2, out / "search", r2, reuse=[out / "final" / "runs"])
    assert (r2 / "finalists" / "runs" / res[res.label == "winner"].run.iloc[0] / "reused.json").exists()
    runner.launch(q1, gpus=[0])
    runner.wait(q1, every=1, first=1, table=False)
    pick, table = final.pick_finalist(q1, cfg2, out / "search", r2)
    assert len(table) == 2 and pick["spec"] in set(trained.spec) and (r2 / "finalists.json").exists()
    q2 = final.queue(cfg2, out / "search", r2, reuse=[out / "final" / "runs"])
    Q2 = runner.read_queue(q2)
    base = [r for r in Q2["runs"] if r["role"] == "baseline"]
    assert all((r2 / "runs" / r["run"] / "reused.json").exists() for r in base)     # baselines copied, not retrained
    win = [r for r in Q2["runs"] if r["label"] == "winner"]
    assert all(r["model"] == pick["spec"] and runner.done_scores(Q2, r) for r in win)  # from the finalist stage


if __name__ == "__main__":
    from _run import run
    run(dict(globals()))

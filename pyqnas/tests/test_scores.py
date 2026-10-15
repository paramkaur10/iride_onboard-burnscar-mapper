"""Scene-balanced scores and the fitness built from them."""
import pandas as pd

from pynas import config, runner
from pynas.train.data import DataSpec
from pynas.train.metrics import SCENE_MIN_PX, summary


def test_scene_mean_and_worst():
    spec = DataSpec(mask_values=[0, 2, 3], class_names=["clear", "fresh", "old"], target=1)
    rows = pd.DataFrame([dict(ref="labels", gsd=2.5, fresh_iou=0.5, fresh_precision=0.6, fresh_recall=0.7, mean_iou=0.6),
                         dict(ref="labels", gsd="all", fresh_iou=0.5, fresh_precision=0.6, fresh_recall=0.7, mean_iou=0.6)])
    rows.attrs["scene_iou"] = {"a": (0.8, 2 * SCENE_MIN_PX), "b": (0.1, SCENE_MIN_PX), "c": (0.0, SCENE_MIN_PX - 1)}
    s = summary(rows, spec, prefix="x_")
    assert abs(s["x_labels_fresh_iou_scene_mean"] - 0.45) < 1e-9          # scene c has too few fresh pixels
    assert s["x_labels_fresh_iou_worst_scene"] == 0.1 and s["x_labels_scenes_counted"] == 2
    assert s["x_labels_fresh_iou"] == 0.5


def test_fitness_scene_mean_or_pooled():
    sc = {"heo_search_fp16_labels_fresh_iou": 0.09, "heo_search_fp16_labels_fresh_iou_scene_mean": 0.48}
    assert runner.fitness(sc, config.load()) == 0.48
    assert runner.fitness(sc, config.load(overrides={"Search.fitness": "pooled"})) == 0.09
    assert runner.fitness({}, config.load()) is None
    assert runner.fitness({"heo_search_fp16_labels_fresh_iou": 0.3}, config.load()) == 0.3   # no scene counted: pooled


def test_one_standard_error_pick():
    from pynas.final import one_se_pick
    t = pd.DataFrame(dict(val_mean=[0.60, 0.59, 0.55, 0.595], val_se=[0.02, 0.01, 0.01, 0.0],
                          latency_ms=[80.0, 40.0, 20.0, None]), index=["a", "b", "c", "d"])
    pick, bar = one_se_pick(t)
    assert abs(bar - 0.58) < 1e-12 and pick == "b"        # a, b, d tie; d has no latency; b is faster than a
    t.loc["a", "val_se"] = 0.0
    assert one_se_pick(t)[0] == "a"                        # nothing else within the bar


if __name__ == "__main__":
    from _run import run
    run(dict(globals()))

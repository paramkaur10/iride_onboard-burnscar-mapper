"""The paper figures on small synthetic tables and windows (no network involved)."""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from pynas import report
from pynas.train.data import DataSpec

SPEC = DataSpec(mask_values=[0, 2, 3, 4, 5, 6], class_names=["clear", "fresh", "old", "cloud", "shadow", "water"],
                target=1, scale=10000.0)


def _set(size=32):
    rng = np.random.default_rng(0)
    rows = [dict(scene="A", gsd=g, r0=i * n, c0=j * n, n=n) for g, n in ((2.5, 40), (5.0, 80))
            for i in range(3) for j in range(2)]
    rows += [dict(scene="B", gsd=10.0, r0=0, c0=j * 160, n=160) for j in range(3)]
    win = pd.DataFrame(rows)
    mask = rng.choice(np.array([0, 2, 3, 4, 5, 6], np.uint8), (len(win), size, size))
    mask[:, :4] = 255
    return dict(refl=rng.integers(0, 4000, (len(win), 7, size, size)).astype(np.int16), mask=mask, windows=win)


def test_short_labels():
    assert report.short_label("smp:Unet:efficientnet-b0:depth=4:decoder=64-32-16-8") == "U-Net EfficientNet-B0, slim"
    assert report.short_label("smp:Unet:mobilenet_v2") == "U-Net MobileNetV2"
    assert report.short_label("winner, no_kd") == "PyNAS, no_kd" and report.short_label("winner") == "PyNAS"


def test_scene_mosaics_and_iou_in_titles():
    s = _set()
    y = report._labels(s, SPEC)
    preds = {"perfect": np.where(y < 0, 0, y).astype(np.uint8), "nothing": np.zeros_like(s["mask"])}
    fig = report.plot_scenes(s, preds, SPEC, "synthetic", max_scenes=2)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("perfect\nIoU 1.00" in t for t in titles) and any("nothing\nIoU 0.00" in t for t in titles)
    assert len(fig.axes) == 3 * 4                      # A at two resolutions, B at one; image, labels, two models
    fig.savefig(os.path.join(tempfile.mkdtemp(), "scenes.png"))


def test_scene_table_counts():
    s = _set()
    s["windows"]["role"] = np.where(s["windows"].scene == "A", "train", "test")
    y = report.labels_of(s, SPEC)
    t = report.scene_table(s, {"perfect": np.where(y < 0, 0, y).astype(np.uint8)}, SPEC)
    assert set(t.role) == {"train", "test"} and len(t) == 3            # A at two resolutions, B at one
    assert (t.iou == 1.0).all() and (t.false_alarms == 0).all() and (t.misses == 0).all()
    fig = report.plot_scenes(s, {"perfect": np.where(y < 0, 0, y).astype(np.uint8)}, SPEC, scenes=["B"],
                             tags={"B": "test"}, labels=y)
    assert fig.axes[0].get_title(loc="left").startswith("B · test")


def test_test_iou_bars_and_finalists():
    rows = [dict(label=l, seed=s, precision=p, set=n, gsd=g, iou=0.5 + 0.01 * s)
            for l in ("winner", "smp:Unet:mobilenet_v2", "teacher") for s in (0, 1) for p in ("fp16", "fp32")
            for n, gs in (("heo_test", ("2.5", "5.0", "all")), ("s2_test", ("10.0", "all"))) for g in gs
            if not (l == "teacher" and p == "fp16")]
    axes = report.plot_test_iou(pd.DataFrame(rows), labels=["winner", "smp:Unet:mobilenet_v2", "teacher"])
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["all", "2.5 m", "5 m"]
    assert len(axes[0].patches) == 3 * 3
    t = pd.DataFrame(dict(search_score=[0.68, 0.66], search_remeasured=[0.59, 0.62], val_mean=[0.67, 0.68],
                          val_se=[0.01, 0.01], latency_ms=[40.8, 55.0], within_one_se=[True, True]),
                     index=pd.Index(["a", "b"], name="spec"))
    report.plot_finalists(t)


if __name__ == "__main__":
    from _run import run
    run(dict(globals()))

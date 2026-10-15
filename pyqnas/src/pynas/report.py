"""Tables and figures for the paper, made from the folders the search and the final stage wrote.

    search_table(search_out)          every trained candidate, generation, scores, hardware numbers
    generation_summary(search_out)    per generation: candidates tried, infeasible and why, best and median fitness
    final_table(q, cfg)               per way of training: mean and standard deviation over seeds (search, val)
    test_table(scores, cfg)           per model: mean and standard deviation over seeds on every test set
    plot_search(search_out)           fitness against latency, by generation, with the final Pareto front
    plot_finalists(table)             finalists: search score, the same measured again, validation score
    plot_test_iou(per_gsd, labels)    target IoU on every test set, over all windows and per resolution
    predict_set(models, spec, s)      class maps of several models on every window of a stored set
    plot_scenes(s, preds, spec)       test scenes as mosaics: image, labels, and each model's hits and misses
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def search_table(search_out):
    t = pd.read_csv(Path(search_out) / "trained.csv")
    return t.sort_values("score", ascending=False).reset_index(drop=True)


def generation_summary(search_out):
    out = Path(search_out)
    tried = pd.read_csv(out / "candidates.csv")
    trained = pd.read_csv(out / "trained.csv")
    rows = []
    for g, c in tried.groupby("gen"):
        t = trained[trained.gen == g]
        reasons = c[~c.feasible.astype(bool)].reason.fillna("").str.split(":").str[0].value_counts()
        rows.append(dict(gen=int(g), tried=len(c), infeasible=int((~c.feasible.astype(bool)).sum()),
                         reasons="; ".join(f"{k} ({v})" for k, v in reasons.items()), trained=int(t.score.notna().sum()),
                         best=t.score.max(), median=t.score.median()))
    return pd.DataFrame(rows)


def final_table(q, cfg, prec="fp16"):
    from .final import results
    from .train.data import DataSpec
    spec = DataSpec.from_config(cfg)
    res = results(q)
    cols = [f"{n}_{prec}_labels_{spec.target_name}_iou{x}" for x in ("", "_scene_mean") for n in spec.search + spec.val]
    cols = [c for c in cols if c in res]
    t = res.groupby("label", sort=False)[cols].agg(["mean", "std"])
    t.insert(0, ("seeds", ""), res.groupby("label", sort=False).size())
    return t


def test_table(scores, cfg):
    """Mean and standard deviation over seeds of the target IoU on each test set, per model and precision."""
    from .train.data import DataSpec
    spec = DataSpec.from_config(cfg)
    cols = [f"{n}_labels_{spec.target_name}_iou{x}" for x in ("", "_scene_mean") for n in spec.test]
    if spec.group:
        cols += [f"{n}_labels_{spec.group_name}_iou" for n in spec.test]
    cols = [c for c in cols if c in scores]
    t = scores.groupby(["label", "precision"], sort=False)[cols].agg(["mean", "std"])
    t.insert(0, ("runs", ""), scores.groupby(["label", "precision"], sort=False).size())
    return t


def plot_search(search_out, ax=None):
    import matplotlib.pyplot as plt
    from .search import nsga2
    t = pd.read_csv(Path(search_out) / "trained.csv")
    t = t[t.score.notna()]
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    has_lat = "latency_ms" in t and t.latency_ms.notna().any()
    x = t.latency_ms if has_lat else t.params / 1e6
    for g, part in t.groupby("gen"):
        ax.scatter(x[part.index], part.score, s=16, alpha=0.7, label=f"generation {int(g)}")
    if has_lat:
        F = np.stack([-t.score.to_numpy(float), t.latency_ms.to_numpy(float)], 1)
        front = t.iloc[nsga2.fronts(F)[0]].sort_values("latency_ms")
        ax.plot(front.latency_ms, front.score, color="k", lw=1, label="Pareto front")
    ax.set_xlabel("predicted Myriad X latency, ms" if has_lat else "parameters, millions")
    ax.set_ylabel("fresh-burn IoU in FP16, search split")
    ax.legend(fontsize=8)
    return ax


# ---- names -------------------------------------------------------------------------------------------------
ENCODER_NAMES = {"efficientnet-b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2", "resnet50": "ResNet-50",
                 "resnet18": "ResNet-18"}


def short_label(label):
    """A readable name for a model label (smp:Unet:efficientnet-b0:depth=4:... -> 'U-Net EfficientNet-B0, slim')."""
    label = str(label)
    if label.startswith("smp:"):
        parts = label.split(":")
        arch = {"Unet": "U-Net"}.get(parts[1], parts[1])
        enc = ENCODER_NAMES.get(parts[2], parts[2])
        return f"{arch} {enc}" + (", slim" if any(q.startswith("decoder=") for q in parts[3:]) else "")
    return {"winner": "PyNAS"}.get(label, label.replace("winner", "PyNAS"))


# ---- finalists -----------------------------------------------------------------------------------------------
def plot_finalists(table, ax=None):
    """Each finalist: its search score (one run), the same split measured again (mean over seeds of the final
    training) and the validation score (mean +- standard error), against predicted latency."""
    import matplotlib.pyplot as plt
    t = table.reset_index()
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    lat = t.latency_ms.astype(float)
    ax.scatter(lat, t.search_score, marker="o", facecolors="none", edgecolors="0.4", label="search score (one run)")
    ax.scatter(lat, t.search_remeasured, marker="s", color="0.6", label="search split, measured again")
    ax.errorbar(lat, t.val_mean, yerr=t.val_se, fmt="D", color="C0", capsize=3, label="validation (mean ± s.e.)")
    for x, a, b in zip(lat, t.search_score, t.search_remeasured):
        ax.plot([x, x], [a, b], color="0.75", lw=0.8, zorder=0)
    if "within_one_se" in t:
        sel = t[t.within_one_se.astype(bool)]
        ax.scatter(sel.latency_ms.astype(float), sel.val_mean, s=140, facecolors="none", edgecolors="C3",
                   label="within one s.e. of the best")
    ax.set_xlabel("predicted Myriad X latency, ms")
    ax.set_ylabel("fresh-burn IoU in FP16")
    ax.legend(fontsize=8)
    return ax


# ---- test figures --------------------------------------------------------------------------------------------
def plot_test_iou(per_gsd, labels=None, prec="fp16", sets=None, axes=None):
    """Target IoU on every test set, over all windows ('all') and per resolution: one bar per model, the mean over
    seeds with the standard deviation. per_gsd: the second table of final.test. Onboard models in prec, the
    teacher in FP32 (the only precision it is scored in)."""
    import matplotlib.pyplot as plt
    d = per_gsd.copy()
    d["gsd"] = d.gsd.astype(str)
    d = d[(d.precision == prec) | d.label.astype(str).str.startswith("teacher")]
    d = d[d.label != "teacher, mean of seeds"]
    labels = list(labels or dict.fromkeys(d.label))
    sets = list(sets or dict.fromkeys(d.set))
    if axes is None:
        _, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 3.8), squeeze=False)
        axes = axes[0]
    w = 0.8 / len(labels)
    for ax, name in zip(axes, sets):
        part = d[d.set == name]
        gsds = ["all"] + sorted({g for g in part.gsd if g != "all"}, key=float)
        for i, lab in enumerate(labels):
            g = part[part.label == lab].groupby("gsd").iou
            mean, std = g.mean().reindex(gsds), g.std().reindex(gsds).fillna(0)
            ax.bar(np.arange(len(gsds)) + (i - (len(labels) - 1) / 2) * w, mean, w, yerr=std, capsize=2,
                   color=f"C{i}", label=short_label(lab), error_kw=dict(lw=0.8))
        ax.set_xticks(range(len(gsds)), ["all"] + [f"{float(x):g} m" for x in gsds[1:]])
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fresh-burn IoU" if ax is axes[0] else "")
    axes[0].legend(fontsize=7, ncol=1, loc="upper left")
    return axes


def predict_set(models, spec, s, device="cpu", batch=16):
    """{name: (network, 'fp16' | 'fp32')} -> {name: class index per pixel, (windows, H, W) uint8} on every window of
    the stored set s (data.open_set). fp16 runs the network as the Myriad does (precision.deploy_copy)."""
    import torch
    from .train.data import batch_input
    from .train.precision import deploy_copy
    n = len(s["refl"])
    tensor = lambda a: torch.from_numpy(np.array(a)).to(device)   # noqa: E731
    first = batch_input(tensor(s["refl"][:min(4, n)]), tensor(s["mask"][:min(4, n)]), spec)
    out = {}
    with torch.no_grad():
        for name, (net, prec) in models.items():
            net = net.to(device).eval()
            run = deploy_copy(net, first, half=True) if prec == "fp16" else net.float()
            pred = np.empty((n,) + tuple(s["mask"].shape[1:]), np.uint8)
            for k in range(0, n, batch):
                e = min(k + batch, n)
                x = batch_input(tensor(s["refl"][k:e]), tensor(s["mask"][k:e]), spec)
                pred[k:e] = run(x.half() if prec == "fp16" else x).float().argmax(1).cpu().numpy()
            out[name] = pred
    return out


CLASS_COLOURS = {"clear": "#e9e4d8", "fresh": "#d7263d", "old": "#f49d37", "cloud": "#a9c8e8", "shadow": "#6b6b6b",
                 "water": "#1d4e89"}
HIT, FALSE_ALARM, MISS = (1.0, 0.84, 0.0), (1.0, 0.0, 1.0), (0.0, 0.9, 1.0)


def _labels(s, spec):
    lut = np.full(256, -1, np.int16)
    lut[np.asarray(spec.mask_values)] = np.arange(spec.num_classes)
    return lut[np.asarray(s["mask"])]


def _mosaic(idx, win, size, step, maxpx):
    """Positions of windows idx in a mosaic (window table with r0, c0, n), shrunk by an integer step."""
    w = win.loc[idx]
    sc = size / w.n.astype(float)
    r, c = ((w.r0 - w.r0.min()) * sc).round().astype(int), ((w.c0 - w.c0.min()) * sc).round().astype(int)
    h, wd = int(r.max()) + size, int(c.max()) + size
    step = step or max(1, int(np.ceil(max(h, wd) / maxpx)))
    return r.to_numpy() // step, c.to_numpy() // step, -(-h // step), -(-wd // step), step


def _paste(canvas, tiles, r, c, step):
    for t, i, j in zip(tiles, r, c):
        t = t[::step, ::step]
        hh, ww = min(t.shape[0], canvas.shape[0] - i), min(t.shape[1], canvas.shape[1] - j)
        canvas[i:i + hh, j:j + ww] = t[:hh, :ww]
    return canvas


def plot_scenes(s, preds, spec, set_name="", scenes=None, max_scenes=3, rgb=(6, 2, 1), maxpx=900, panel=2.6):
    """One row per scene and resolution of the stored set s: the image (bands rgb, default NIR, red, green), the
    labels, and for every model of preds (predict_set) the target class: hits (yellow), false alarms (magenta) and
    misses (cyan) over the image, with the IoU of that scene and resolution in the title (over its windows, as the
    test scores count it). Windows are placed by their position in the scene (r0, c0, n of the window table); a
    set that is a sample of windows shows gaps. Scenes: the given ones, else the max_scenes with the most labelled
    target pixels. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    win = s["windows"].reset_index(drop=True)
    if not {"r0", "c0", "n"} <= set(win):
        raise ValueError("the window table has no r0, c0, n: the windows cannot be placed in their scenes")
    y = _labels(s, spec)
    f = spec.target
    tgt = pd.Series((y == f).sum((1, 2)), index=win.index)
    if scenes is None:
        scenes = list(tgt.groupby(win.scene.astype(str)).sum().sort_values(ascending=False).index[:max_scenes])
    rows = [(sc, g) for sc in scenes for g in sorted(win[win.scene.astype(str) == sc].gsd.unique())]
    names = list(preds)
    fig, axes = plt.subplots(len(rows), 2 + len(names), figsize=(panel * (2 + len(names)), panel * len(rows) + 0.6),
                             squeeze=False)
    cmap = ListedColormap([CLASS_COLOURS.get(n, "#999999") for n in spec.class_names])
    size = s["mask"].shape[-1]
    for ri, (sc, g) in enumerate(rows):
        idx = win.index[(win.scene.astype(str) == sc) & (win.gsd == g)].to_numpy()
        r, c, h, wd, step = _mosaic(idx, win, size, None, maxpx)
        refl = np.asarray(s["refl"][idx][:, list(rgb)], np.float32) / spec.scale
        img = _paste(np.full((h, wd, 3), np.nan, np.float32), refl.transpose(0, 2, 3, 1), r, c, step)
        lo, hi = np.nanpercentile(img, 2, (0, 1)), np.nanpercentile(img, 98, (0, 1))
        img = np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        lab = _paste(np.full((h, wd), -1, np.int16), y[idx], r, c, step)
        gaps = lab < 0
        img_show = np.where(gaps[..., None], 1.0, img)
        axes[ri, 0].imshow(img_show)
        axes[ri, 0].set_title(f"{str(sc)[:22]}\n{g:g} m", fontsize=8, loc="left")
        axes[ri, 1].imshow(np.ma.masked_less(lab, 0), cmap=cmap, vmin=-0.5, vmax=spec.num_classes - 0.5,
                           interpolation="nearest")
        axes[ri, 1].set_facecolor("white")
        axes[ri, 1].set_title("labels", fontsize=8)
        grey = np.where(gaps[..., None], 1.0, 0.35 + 0.5 * img.mean(-1, keepdims=True))
        for k, name in enumerate(names):
            p = preds[name][idx]
            valid = y[idx] >= 0
            tp = int(((p == f) & (y[idx] == f)).sum())
            fp = int(((p == f) & (y[idx] != f) & valid).sum())
            fn = int(((p != f) & (y[idx] == f)).sum())
            iou = tp / max(tp + fp + fn, 1)
            pm = _paste(np.full((h, wd), -1, np.int16), p.astype(np.int16), r, c, step)
            show = np.repeat(grey, 3, -1).copy()
            show[(pm == f) & (lab == f)] = HIT
            show[(pm == f) & (lab != f) & (lab >= 0)] = FALSE_ALARM
            show[(pm != f) & (lab == f)] = MISS
            axes[ri, 2 + k].imshow(show, interpolation="nearest")
            axes[ri, 2 + k].set_title(f"{name}\nIoU {iou:.2f}" if tp + fn else f"{name}\n(no labelled {spec.target_name})",
                                      fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [Patch(color=CLASS_COLOURS.get(n, "#999999"), label=n) for n in spec.class_names]
    handles += [Patch(color=HIT, label=f"{spec.target_name}: hit"), Patch(color=FALSE_ALARM, label="false alarm"),
                Patch(color=MISS, label="miss")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    if set_name:
        fig.suptitle(set_name, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return fig

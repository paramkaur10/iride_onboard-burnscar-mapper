"""Scores on stored sets: confusion matrices per window resolution (the 'gsd' column of the window table), from
which the IoU of the target class, its precision and recall, the IoU of the class group and the mean IoU, against
the labels and, where the set has them, against the teacher's classes.

precision='fp16' scores the network as the Myriad runs it (precision.deploy_copy: BatchNorm folded, half
precision); 'fp32' scores the same weights in FP32.

Besides the IoU over all pixels of a set (pooled), the scene-balanced IoU: the mean of the per-scene IoUs (scene
column of the window table) over the scenes with at least SCENE_MIN_PX labelled target pixels, so that one scene
with much of the target class cannot decide the number alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import batch_input, class_lut, teacher_logp
from .precision import deploy_copy

SCENE_MIN_PX = 500             # labelled target pixels a scene needs to count in the scene-balanced IoU


class Counts:
    """Confusion matrices (reference x prediction), one per key."""

    def __init__(self, n):
        self.n, self.m = n, {}

    def add(self, key, pred, ref):
        ok = ref >= 0
        idx = (ref[ok] * self.n + pred[ok]).flatten()
        c = torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n).cpu().numpy()
        self.m[key] = self.m.get(key, 0) + c


def scores_of(m, spec):
    """Scores of one confusion matrix."""
    m = np.asarray(m, np.float64)
    tp = np.diag(m)
    iou = tp / np.maximum(m.sum(0) + m.sum(1) - tp, 1)
    f, name = spec.target, spec.target_name
    present = m.sum(1) > 0
    out = {f"{name}_iou": iou[f], f"{name}_precision": tp[f] / max(m[:, f].sum(), 1),
           f"{name}_recall": tp[f] / max(m[f].sum(), 1), "mean_iou": float(iou[present].mean()) if present.any() else np.nan,
           f"{name}_px": int(m[f].sum())}
    if spec.group:
        sel = np.zeros(len(m), bool)
        sel[spec.group] = True
        gtp, gfp, gfn = m[np.ix_(sel, sel)].sum(), m[np.ix_(~sel, sel)].sum(), m[np.ix_(sel, ~sel)].sum()
        out[f"{spec.group_name}_iou"] = gtp / max(gtp + gfp + gfn, 1)
    return out


@torch.no_grad()
def evaluate(model, s, spec, device, precision="fp32", batch=32, agreement=False):
    """Scores of model on the stored set s (data.open_set), per resolution and over all ('all' rows), against the
    labels and the teacher. agreement: also the share of pixels with data where the FP16 network and the FP32 one
    give the same class (attrs['fp16_agreement'])."""
    n = len(s["refl"])
    first = batch_input(torch.from_numpy(np.ascontiguousarray(s["refl"][:min(4, n)])).to(device),
                        torch.from_numpy(np.ascontiguousarray(s["mask"][:min(4, n)])).to(device), spec)
    model = model.to(device).eval()
    run = deploy_copy(model, first, half=True) if precision == "fp16" else model.float()
    half = deploy_copy(model, first, half=True) if (agreement and precision != "fp16") else None
    lut = class_lut(spec, device)
    gsd = s["windows"]["gsd"].round(2).to_numpy() if "gsd" in s["windows"] else np.zeros(n)
    scenes = s["windows"]["scene"].astype(str).to_numpy() if "scene" in s["windows"] else None
    counts, per_scene, same, total = Counts(spec.num_classes), Counts(spec.num_classes), 0, 0
    for k in range(0, n, batch):
        e = min(k + batch, n)
        m = torch.from_numpy(np.ascontiguousarray(s["mask"][k:e])).to(device)
        x = batch_input(torch.from_numpy(np.ascontiguousarray(s["refl"][k:e])).to(device), m, spec)
        logits = run(x.half() if precision == "fp16" else x).float()
        pred = logits.argmax(1)
        y = lut[m.long()]
        tcls = None
        if s.get("teacher") is not None:
            t = torch.from_numpy(np.ascontiguousarray(s["teacher"][k:e])).to(device)
            tcls = teacher_logp(t).argmax(1).masked_fill(m == spec.nodata, -1)
        if half is not None:
            v = m != spec.nodata
            same += int(((half(x.half()).float().argmax(1) == pred) & v).sum())
            total += int(v.sum())
        if scenes is not None:
            for sc in np.unique(scenes[k:e]):
                sel = torch.from_numpy(scenes[k:e] == sc).to(device)
                per_scene.add(sc, pred[sel], y[sel])
        for g in np.unique(gsd[k:e]):
            sel = torch.from_numpy(gsd[k:e] == g).to(device)
            counts.add(("labels", g), pred[sel], y[sel])
            if tcls is not None:
                counts.add(("teacher", g), pred[sel], tcls[sel])
    rows = []
    for ref in ("labels", "teacher"):
        keys = sorted(g for (r, g) in counts.m if r == ref)
        for g in keys:
            rows.append(dict(ref=ref, gsd=g, **scores_of(counts.m[(ref, g)], spec)))
        if keys:
            rows.append(dict(ref=ref, gsd="all", **scores_of(sum(counts.m[(ref, g)] for g in keys), spec)))
    out = pd.DataFrame(rows)
    if per_scene.m:
        sc_scores = {sc: scores_of(m, spec) for sc, m in per_scene.m.items()}
        name = spec.target_name
        out.attrs["scene_iou"] = {sc: (float(v[f"{name}_iou"]), int(v[f"{name}_px"])) for sc, v in sc_scores.items()}
    if half is not None:
        out.attrs["fp16_agreement"] = same / max(total, 1)
    return out


def summary(scores, spec, prefix=""):
    """One row of numbers from evaluate(): <prefix><ref>_<target>_iou (all resolutions), ..._iou_worst_gsd,
    precision, recall, the group IoU, and fp16_agreement when computed."""
    out = {}
    name = spec.target_name
    for ref, g in scores.groupby("ref"):
        allr = g[g.gsd == "all"].iloc[0]
        per = g[g.gsd != "all"]
        out[f"{prefix}{ref}_{name}_iou"] = float(allr[f"{name}_iou"])
        out[f"{prefix}{ref}_{name}_iou_worst_gsd"] = float(per[f"{name}_iou"].min()) if len(per) else np.nan
        out[f"{prefix}{ref}_{name}_precision"] = float(allr[f"{name}_precision"])
        out[f"{prefix}{ref}_{name}_recall"] = float(allr[f"{name}_recall"])
        out[f"{prefix}{ref}_mean_iou"] = float(allr["mean_iou"])
        if spec.group:
            out[f"{prefix}{ref}_{spec.group_name}_iou"] = float(allr[f"{spec.group_name}_iou"])
    counted = [iou for iou, px in scores.attrs.get("scene_iou", {}).values() if px >= SCENE_MIN_PX]
    if counted:
        out[f"{prefix}labels_{name}_iou_scene_mean"] = float(np.mean(counted))
        out[f"{prefix}labels_{name}_iou_worst_scene"] = float(np.min(counted))
        out[f"{prefix}labels_scenes_counted"] = len(counted)
    if "fp16_agreement" in scores.attrs:
        out[f"{prefix}fp16_agreement"] = float(scores.attrs["fp16_agreement"])
    return out

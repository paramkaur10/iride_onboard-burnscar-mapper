"""The window pool: stored sets of windows on disk, read memory-mapped.

A stored set is a group of files with a common prefix in the [Data] pool folder:
    <name>_refl.npy      (N, bands, S, S) input values x [Data] scale (int16)
    <name>_mask.npy      (N, S, S) uint8 mask values ([Data] mask_values in class order, nodata)
    <name>_windows.csv   one row per window (columns used: source, gsd, scene)
    <name>_<teacher>.npy (N, classes, S, S) uint8 class log-probabilities of the teacher (0 = LOGP_MIN, 255 = 0)
Training draws random windows with replacement (optionally balanced by source); rotations, flips, gains, band
misregistration and noise are applied on the GPU.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data

from ..config import get_list, get_mapping, path

LOGP_MIN = -12.0


@dataclass
class DataSpec:
    """[Data] and [Augment], read once."""
    pool: str = ""
    train: str = "train"
    search: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)
    norm: dict = field(default_factory=dict)
    scale: float = 10000.0
    mask_values: list = field(default_factory=list)
    nodata: int = 255
    class_names: list = field(default_factory=list)
    class_weights: list | None = None
    target: int = 1
    group_name: str = ""
    group: list = field(default_factory=list)
    source_share: dict | None = None
    teacher: str = "teacher"
    augment: bool = True
    gain: tuple = (0.9, 1.1)
    band_gain: tuple = (0.95, 1.05)
    p_shift: float = 0.5
    noise: float = 0.003

    @property
    def num_classes(self):
        return len(self.mask_values)

    @property
    def target_name(self):
        return self.class_names[self.target]

    def prefix(self, name):
        return str(Path(self.pool) / name)

    @classmethod
    def from_config(cls, cfg):
        names = get_list(cfg, "Data", "class_names")
        masks = get_list(cfg, "Data", "mask_values", int)
        if len(names) != len(masks):
            raise ValueError("[Data] class_names and mask_values differ in length")
        target = cfg.get("Data", "target").strip()
        if target not in names:
            raise ValueError(f"[Data] target {target} is not among the class names")
        g_name, members = "", []
        graw = cfg.get("Data", "group", fallback="").strip()
        if graw:
            g_name, _, rest = graw.partition(":")
            members = [names.index(m.strip()) for m in rest.split(",") if m.strip()]
            g_name = g_name.strip()
        norm_path = path(cfg, "Data", "norm")
        rng = lambda opt: tuple(get_list(cfg, "Augment", opt, float))   # noqa: E731
        return cls(pool=path(cfg, "Data", "pool"), train=cfg.get("Data", "train").strip(),
                   search=get_list(cfg, "Data", "search"), val=get_list(cfg, "Data", "val"),
                   test=get_list(cfg, "Data", "test"), norm=json.load(open(norm_path)) if norm_path else {},
                   scale=cfg.getfloat("Data", "scale"), mask_values=masks, nodata=cfg.getint("Data", "nodata"),
                   class_names=names, class_weights=get_list(cfg, "Data", "class_weights", float) or None,
                   target=names.index(target), group_name=g_name, group=members,
                   source_share=get_mapping(cfg, "Data", "source_share") or None,
                   teacher=cfg.get("Data", "teacher").strip(), augment=cfg.getboolean("Augment", "enabled"),
                   gain=rng("gain"), band_gain=rng("band_gain"), p_shift=cfg.getfloat("Augment", "p_shift"),
                   noise=cfg.getfloat("Augment", "noise"))


def open_set(spec, name, teacher=True):
    """A stored set, memory-mapped: refl, mask, windows (the table) and, with teacher and when the arrays exist,
    teacher (the stored class log-probabilities)."""
    p = spec.prefix(name)
    out = dict(name=name, refl=np.load(f"{p}_refl.npy", mmap_mode="r"), mask=np.load(f"{p}_mask.npy", mmap_mode="r"))
    out["windows"] = pd.read_csv(f"{p}_windows.csv") if os.path.exists(f"{p}_windows.csv") else \
        pd.DataFrame(index=range(len(out["refl"])))
    if teacher and spec.teacher and os.path.exists(f"{p}_{spec.teacher}.npy"):
        out["teacher"] = np.load(f"{p}_{spec.teacher}.npy", mmap_mode="r")
    return out


def input_shape(spec):
    """(bands, S, S) of the training windows."""
    return tuple(int(v) for v in np.load(f"{spec.prefix(spec.train)}_refl.npy", mmap_mode="r").shape[1:])


class Items(torch.utils.data.Dataset):
    """Windows of a stored set: input values (bands, S, S), mask (S, S), teacher (C, S, S) or empty, index."""

    def __init__(self, s, teacher=True):
        self.refl, self.mask = s["refl"], s["mask"]
        self.teacher = s.get("teacher") if teacher else None

    def __len__(self):
        return len(self.refl)

    def __getitem__(self, i):
        r = torch.from_numpy(np.ascontiguousarray(self.refl[i]))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i]))
        t = (torch.from_numpy(np.ascontiguousarray(self.teacher[i])) if self.teacher is not None
             else torch.zeros(0, dtype=torch.uint8))
        return r, m, t, i


def loader(s, spec, batch, steps, workers=4, seed=0, teacher=True, timeout=600):
    """steps batches of windows drawn with replacement; with [Data] source_share each source is drawn that often."""
    ds = Items(s, teacher)
    g = torch.Generator()
    g.manual_seed(seed)
    if spec.source_share:
        if "source" not in s["windows"]:
            raise ValueError("[Data] source_share needs a 'source' column in the window table")
        src = s["windows"]["source"].to_numpy()
        w = np.zeros(len(src))
        for name, share in spec.source_share.items():
            sel = src == name
            if sel.any():
                w[sel] = share / sel.sum()
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), steps * batch,
                                                         replacement=True, generator=g)
    else:
        sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=steps * batch, generator=g)
    kw = dict(num_workers=workers, pin_memory=True, drop_last=True)
    if workers > 0:          # fork: the readers share the memory-mapped arrays instead of copying them
        kw.update(multiprocessing_context="fork", prefetch_factor=4, persistent_workers=False, timeout=timeout)
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, **kw)


def class_lut(spec, device):
    """Mask value -> class index (-1 for a value that is no class, e.g. nodata)."""
    lut = torch.full((256,), -1, dtype=torch.long, device=device)
    for k, v in enumerate(spec.mask_values):
        lut[v] = k
    return lut


def normalise(refl, mask, spec):
    """Stored values / scale (N, bands, S, S) to the network input: clipped, (x - mean) / std, no data at 0."""
    x = refl
    if spec.norm:
        mean = torch.as_tensor(spec.norm["mean"], dtype=torch.float32, device=refl.device).view(1, -1, 1, 1)
        std = torch.as_tensor(spec.norm["std"], dtype=torch.float32, device=refl.device).view(1, -1, 1, 1)
        x = (refl.clamp(0, spec.norm.get("clip", 1.2)) - mean) / std
    return x.masked_fill((mask == spec.nodata).unsqueeze(1), 0.0)


def batch_input(refl, mask, spec):
    """Network input from a batch of stored values (any device)."""
    return normalise(refl.float() / spec.scale, mask, spec)


def teacher_logp(t):
    """Stored uint8 log-probabilities to float."""
    return t.float() * (-LOGP_MIN / 255.0) + LOGP_MIN


def quantise_logp(logits):
    """Logits to stored uint8 log-probabilities (inverse of teacher_logp up to the quantisation)."""
    logp = F.log_softmax(logits.float(), 1).clamp(min=LOGP_MIN)
    return ((logp - LOGP_MIN) * (255.0 / -LOGP_MIN)).round().clamp(0, 255).to(torch.uint8)


def augment(refl, mask, teacher, gen, spec):
    """On the GPU, per window: one of the 8 rotations and flips (the same for image, mask and teacher), a gain for
    all bands and one per band, with probability p_shift one band moved by one pixel, a little noise. refl holds
    stored values / scale, before normalisation."""
    n, nb = refl.shape[:2]
    dev = refl.device
    k = torch.randint(0, 8, (n,), generator=gen).tolist()
    refl, mask = refl.clone(), mask.clone()
    teacher = teacher.clone() if teacher is not None else None
    for j, kk in enumerate(k):
        if kk == 0:
            continue
        r, m = refl[j].rot90(kk % 4, (-2, -1)), mask[j].rot90(kk % 4, (-2, -1))
        t = teacher[j].rot90(kk % 4, (-2, -1)) if teacher is not None else None
        if kk >= 4:
            r, m = r.flip(-1), m.flip(-1)
            t = t.flip(-1) if t is not None else None
        refl[j], mask[j] = r, m
        if t is not None:
            teacher[j] = t
    g_img = torch.empty(n, 1, 1, 1).uniform_(*spec.gain, generator=gen)
    g_band = torch.empty(n, nb, 1, 1).uniform_(*spec.band_gain, generator=gen)
    refl = refl * (g_img * g_band).to(dev)
    shift = (torch.rand(n, generator=gen) < spec.p_shift).tolist()
    bands = torch.randint(0, nb, (n,), generator=gen).tolist()
    moves = torch.randint(0, 4, (n,), generator=gen).tolist()
    for j in range(n):
        if shift[j]:
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[moves[j]]
            refl[j, bands[j]] = refl[j, bands[j]].roll((dy, dx), (-2, -1))
    if spec.noise:
        refl = refl + spec.noise * torch.randn_like(refl) * (refl > 0)
    return refl.clamp_(min=0), mask, teacher


@torch.no_grad()
def write_predictions(models, spec, name, device, batch=32, tta=True, out_name=None):
    """Class log-probabilities of one model or the mean (in probability) of several, for every window of the
    stored set `name`, written as <name>_<out_name or spec.teacher>.npy (N, classes, S, S) uint8, the format
    distillation reads. tta: averaged over the 8 rotations and flips. Written to a temporary file and renamed at
    the end. Returns the path."""
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    s = open_set(spec, name, teacher=False)
    N, _, H, W = s["refl"].shape
    final = f"{spec.prefix(name)}_{out_name or spec.teacher}.npy"
    tmp = final.replace(".npy", ".writing.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(N, spec.num_classes, H, W))
    models = [m.to(device).eval() for m in models]
    for k in range(0, N, batch):
        e = min(k + batch, N)
        x = batch_input(torch.from_numpy(np.ascontiguousarray(s["refl"][k:e])).to(device),
                        torch.from_numpy(np.ascontiguousarray(s["mask"][k:e])).to(device), spec)
        prob = 0
        for m in models:
            for kk in (range(8) if tta else [0]):
                xx = x.rot90(kk % 4, (-2, -1))
                xx = xx.flip(-1) if kk >= 4 else xx
                p = torch.softmax(m(xx).float(), 1)
                p = p.flip(-1) if kk >= 4 else p
                prob = prob + p.rot90(-(kk % 4), (-2, -1))
        prob = prob / (len(models) * (8 if tta else 1))
        out[k:e] = quantise_logp(torch.log(prob.clamp(min=1e-12))).cpu().numpy()
    out.flush()
    del out
    os.replace(tmp, final)
    return final

"""A fixed set of windows kept on disk as plain arrays, for training many small models fast.

Reading one window from the GeoTIFFs costs about 45 ms of CPU. With 8 cores that cannot keep
a GPU busy through hundreds of short trainings, so notebooks/06a_window_pool.ipynb reads a fixed
set of windows once and stores, for a set called <prefix>:

    <prefix>_refl.npy      int16  (N, 7, S, S)  reflectance x 10000, bands as in burnscar.dataset
    <prefix>_mask.npy      uint8  (N, S, S)     v3 mask values, as in data/*/masks_v3
    <prefix>_windows.csv   one row per window: i, scene, source, kind, gsd, r0, c0, n
    <prefix>_teacher.npy   uint8  (N, 6, S, S)  teacher log-probabilities, quantised (added later)

Training windows are drawn exactly as TrainWindows draws them, without augmentation; the
augmentation is done on the GPU when the windows are used (notebook 06). The .npy files open
with np.load(path, mmap_mode="r"), so nothing has to fit in memory.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import dataset as D
from . import labels as L

REFL_SCALE = 10000.0
LOGP_MIN = -12.0          # log-probabilities below this (p < 6e-6) are stored as this
N_CLASSES = len(L.TRAIN_NAMES)


def encode_refl(refl):
    """Reflectance (float) to the stored int16 (x 10000). Values above 3.27 are cut there, which
    changes nothing: the model input is clipped at 1.2 anyway. int16 rather than uint16 because
    PyTorch handles it everywhere, on the GPU too."""
    return np.clip(np.rint(np.asarray(refl, np.float32) * REFL_SCALE), 0, 32767).astype(np.int16)


def decode_refl(a):
    return np.asarray(a).astype(np.float32) / np.float32(REFL_SCALE)


def quantise_logp(logp):
    """Log-probabilities (<= 0) to uint8: 0 is LOGP_MIN, 255 is log(1) = 0. Step 0.047."""
    q = np.rint((np.clip(logp, LOGP_MIN, 0.0) - LOGP_MIN) * (255.0 / -LOGP_MIN))
    return q.astype(np.uint8)


def dequantise_logp(q):
    return np.asarray(q).astype(np.float32) * np.float32(-LOGP_MIN / 255.0) + np.float32(LOGP_MIN)


def bytes_per_window(size, with_teacher=True):
    return size * size * (7 * 2 + 1 + (N_CLASSES if with_teacher else 0))


def grid_windows(rows, gsds, size, min_valid=0.3, dec=4):
    """Windows covering whole scenes at the given ground resolutions, as in notebook 04 [12]:
    one row per window with scene, source, split, gsd, r0, c0, n and the shares of data, fresh
    and old burn. Windows with less than min_valid data are dropped. rows: scene table rows."""
    import rasterio
    from rasterio.enums import Resampling
    out = []
    for r in rows.itertuples():
        with rasterio.open(r.mask) as m:
            mk = m.read(1, out_shape=(max(m.height // dec, 1), max(m.width // dec, 1)), resampling=Resampling.nearest)
        for gsd in gsds:
            for r0, c0, n in D.grid_boxes(r.height, r.width, r.px_m, gsd, size):
                sub = mk[r0 // dec:(r0 + n) // dec, c0 // dec:(c0 + n) // dec]
                valid = float((sub != L.NODATA).mean()) if sub.size else 0.0
                if valid < min_valid:
                    continue
                out.append(dict(scene=r.scene, source=r.source, split=r.split, gsd=round(n * r.px_m / size, 2),
                                r0=r0, c0=c0, n=n, valid=round(valid, 3), fresh=round(float((sub == L.FRESH).mean()), 4),
                                old=round(float((sub == L.OLD).mean()), 4)))
    return pd.DataFrame(out)


class _Items:
    """TrainWindows or FixedWindows items as (reflectance int16, mask uint8, info) for writing."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if isinstance(self.ds, D.TrainWindows):
            d = self.ds.get(i, with_info=True)
            r0, c0, n = d["box"]
            info = dict(scene=d["scene"], source=d["source"], kind=d["kind"], gsd=round(float(d["gsd"]), 3),
                        r0=int(r0), c0=int(c0), n=int(n))
            refl, msk = d["refl"], d["msk"]
        else:
            _, _, refl, msk = self.ds.get(i, with_refl=True)
            w = self.ds.info(i)
            info = dict(scene=w["scene"], source=w["source"], kind="grid", gsd=float(w["gsd"]),
                        r0=int(w["r0"]), c0=int(w["c0"]), n=int(w["n"]))
        return encode_refl(refl), np.asarray(msk, np.uint8), info


def _collate(items):
    return (np.stack([a for a, _, _ in items]), np.stack([b for _, b, _ in items]), [c for _, _, c in items])


def write_set(ds, prefix, workers=4, batch=32, every=2000, extra=None):
    """Read every item of ds (TrainWindows or FixedWindows) once, `workers` processes in parallel,
    and write <prefix>_refl.npy, <prefix>_mask.npy and <prefix>_windows.csv. extra: columns
    (a dict of equal-length lists) added to the window table. Returns the window table."""
    n, s = len(ds), ds.size
    refl = np.lib.format.open_memmap(f"{prefix}_refl.npy", mode="w+", dtype=np.int16, shape=(n, 7, s, s))
    mask = np.lib.format.open_memmap(f"{prefix}_mask.npy", mode="w+", dtype=np.uint8, shape=(n, s, s))
    items = _Items(ds)
    try:
        from torch.utils.data import DataLoader
        batches = DataLoader(items, batch_size=batch, num_workers=workers, collate_fn=_collate)
    except ImportError:                                   # no torch: one process
        batches = (_collate([items[j] for j in range(k, min(k + batch, n))]) for k in range(0, n, batch))
    rows, k, t0, last = [], 0, time.time(), 0
    for r, m, info in batches:
        refl[k:k + len(r)] = r
        mask[k:k + len(r)] = m
        rows += info
        k += len(r)
        if every and k - last >= every:
            last = k
            print(f"    {k:,}/{n:,} windows, {(time.time() - t0) / 60:.1f} min")
    refl.flush()
    mask.flush()
    del refl, mask
    table = pd.DataFrame(rows)
    table.insert(0, "i", np.arange(len(table)))
    for col, vals in (extra or {}).items():
        table[col] = list(vals)
    table.to_csv(f"{prefix}_windows.csv", index=False)
    return table


def open_set(prefix, teacher=False):
    """The arrays of a stored set, memory-mapped, and its window table."""
    out = dict(refl=np.load(f"{prefix}_refl.npy", mmap_mode="r"), mask=np.load(f"{prefix}_mask.npy", mmap_mode="r"),
               windows=pd.read_csv(f"{prefix}_windows.csv"))
    if teacher:
        out["teacher"] = np.load(f"{prefix}_teacher.npy", mmap_mode="r")
    return out


def normalise_torch(refl, mask, norm):
    """Same as burnscar.dataset.normalise, on the GPU: stored reflectance (N, 7, S, S) and mask
    (N, S, S) tensors to the model input, no-data pixels set to 0."""
    import torch
    x = refl.float() / REFL_SCALE
    mean = torch.as_tensor(norm["mean"], dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(norm["std"], dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    x = (x.clamp(0, norm.get("clip", 1.2)) - mean) / std
    return x.masked_fill((mask == L.NODATA).unsqueeze(1), 0.0)


def write_teacher(models, prefix, norm, device, amp_dtype=None, batch=64, every=4000):
    """Average class probabilities of the teacher models for every window of a stored set,
    written as quantised log-probabilities to <prefix>_teacher.npy."""
    import torch
    from .train import autocast
    s = open_set(prefix)
    n, _, size, _ = s["refl"].shape
    out = np.lib.format.open_memmap(f"{prefix}_teacher.npy", mode="w+", dtype=np.uint8,
                                    shape=(n, N_CLASSES, size, size))
    for m in models:
        m.eval()
    t0, last = time.time(), 0
    with torch.no_grad():
        for k in range(0, n, batch):
            r = torch.from_numpy(np.ascontiguousarray(s["refl"][k:k + batch])).to(device)
            msk = torch.from_numpy(np.ascontiguousarray(s["mask"][k:k + batch])).to(device)
            x = normalise_torch(r, msk, norm)
            with autocast(device, amp_dtype):
                p = sum(m(x).float().softmax(1) for m in models) / len(models)
            out[k:k + len(r)] = quantise_logp(p.clamp_min(1e-12).log().cpu().numpy())
            done = k + len(r)
            if every and done - last >= every:
                last = done
                print(f"    {done:,}/{n:,} windows, {(time.time() - t0) / 60:.1f} min")
    out.flush()
    del out

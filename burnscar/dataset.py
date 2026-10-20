"""Training and evaluation windows, read straight from the GeoTIFFs.

A window is SIZE x SIZE pixels at a ground resolution (gsd) we pick, cut from one scene.
For a gsd coarser than the scene we read a larger area and average it down, for a finer
one a smaller area is resampled up. So one 2.5 m HEO scene gives windows from 1 m to 10 m,
and nothing is tiled to disk: change the resolution range or the augmentations and train again.

Images come out as reflectance (stored value / scale) in the band order of the model:
blue, green, red, red edge 1, red edge 2, red edge 3, NIR. Targets come out as training ids:
0 clear, 1 fresh burn, 2 old burn, 3 cloud, 4 cloud shadow, 5 water, -1 left out (ignore, no data).

The scene table with the splits, the validation and test windows and the normalisation are
made in notebooks/04_dataset.ipynb and saved in data/tables/.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import ndimage

from . import labels as L

BAND_ORDER = ["B", "G", "R", "RE1", "RE2", "RE3", "NIR"]
S2_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08"]


def band_indexes(descriptions, count):
    """1-based indexes of the 7 model bands in a file, from its band names when it has them."""
    desc = [(d or "").strip().upper() for d in descriptions]
    for names in (BAND_ORDER, S2_NAMES):
        if all(b in desc for b in names):
            return [desc.index(b) + 1 for b in names]
    if count == 7:
        return list(range(1, 8))
    if count == 8:                                   # HEO stack with PAN first
        return list(range(2, 9))
    raise ValueError(f"unknown band layout: {count} bands, names {descriptions}")


# ---- scenes, groups and splits ------------------------------------------------------
def footprint_groups(boxes, min_overlap=0.01):
    """Scenes whose lon/lat boxes (lon0, lat0, lon1, lat1) overlap by more than min_overlap of the
    smaller box end up in the same group, also through other scenes. Returns a group number per box."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(b)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(n):
        w = np.clip(np.minimum(b[i, 2], b[:, 2]) - np.maximum(b[i, 0], b[:, 0]), 0, None)
        h = np.clip(np.minimum(b[i, 3], b[:, 3]) - np.maximum(b[i, 1], b[:, 1]), 0, None)
        for j in np.nonzero(w * h > min_overlap * np.minimum(area[i], area))[0]:
            if j != i:
                parent[find(i)] = find(int(j))
    ids = {}
    return np.array([ids.setdefault(find(i), len(ids)) for i in range(n)])


def merge_groups(*labelings):
    """Items that share a label in any of the labelings end up in the same group
    (empty labels are ignored). Returns a group number per item."""
    n = len(labelings[0])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lab in labelings:
        first = {}
        for i, v in enumerate(lab):
            if v is None or (isinstance(v, str) and v == ""):
                continue
            if v in first:
                parent[find(i)] = find(first[v])
            else:
                first[v] = i
    ids = {}
    return np.array([ids.setdefault(find(i), len(ids)) for i in range(n)])


def random_group_split(groups, weight, val=0.1, test=0.1, seed=0):
    """Whole groups, in random order, go to val until val holds `val` of the total weight,
    then to test, the rest to train. Returns the split of every item."""
    g = pd.Series(np.asarray(weight, dtype=float)).groupby(np.asarray(groups)).sum()
    order = np.random.default_rng(seed).permutation(g.index.values)
    total, acc, out = float(g.sum()) or 1.0, {"val": 0.0, "test": 0.0}, {}
    for gid in order:
        if acc["val"] < val * total:
            out[gid] = "val"
            acc["val"] += g[gid]
        elif acc["test"] < test * total:
            out[gid] = "test"
            acc["test"] += g[gid]
        else:
            out[gid] = "train"
    return np.array([out[x] for x in groups])


# ---- reading windows --------------------------------------------------------------------
class SceneReader:
    """Keeps up to max_open files open in each process (every DataLoader worker gets its own)."""

    def __init__(self, max_open=64):
        self.max_open, self._open, self._bands, self._pid = max_open, {}, {}, None

    def get(self, path):
        import rasterio
        if self._pid != os.getpid():                 # new worker process: start with a clean cache
            self._open, self._pid = {}, os.getpid()
        src = self._open.pop(path, None)
        if src is None:
            if len(self._open) >= self.max_open:
                self._open.pop(next(iter(self._open))).close()
            src = rasterio.open(path)
        self._open[path] = src                       # most recently used goes last
        return src

    def bands(self, path):
        if path not in self._bands:
            src = self.get(path)
            self._bands[path] = band_indexes(src.descriptions, src.count)
        return self._bands[path]


def box_around(row, col, gsd, px, size, height, width):
    """Native pixel box (r0, c0, n) of a size x size window at gsd metres, centred as close to
    (row, col) as the scene allows. n is capped to the scene, so the gsd you get can be coarser
    than asked only when the window would be larger than the scene."""
    n = int(round(size * gsd / px))
    n = max(1, min(n, height, width))
    r0 = int(np.clip(round(row - n / 2), 0, height - n))
    c0 = int(np.clip(round(col - n / 2), 0, width - n))
    return r0, c0, n


def read_box(img_src, msk_src, r0, c0, n, size, bands, scale):
    """Read native box (r0, c0, n) resampled to size x size: reflectance (7, size, size) float32
    and mask values (size, size) uint8. Coarser than native: average for the image, most common
    value for the mask. Finer: bilinear for the image, nearest for the mask."""
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    win = Window(c0, r0, n, n)
    down = n > size
    img = img_src.read(bands, window=win, out_shape=(len(bands), size, size),
                       resampling=Resampling.average if down else Resampling.bilinear)
    msk = msk_src.read(1, window=win, out_shape=(size, size),
                       resampling=Resampling.mode if down else Resampling.nearest)
    return img.astype(np.float32) / np.float32(scale), msk


def grid_boxes(height, width, px, gsd, size):
    """Boxes (r0, c0, n) covering the whole scene at gsd, without gaps. The last row and column
    are pushed back to the edge, so they overlap the one before a little."""
    n = int(round(size * gsd / px))
    if n > min(height, width):
        return []

    def starts(total):
        s = list(range(0, total - n + 1, n))
        if s[-1] != total - n:
            s.append(total - n)
        return s

    return [(r, c, n) for r in starts(height) for c in starts(width)]


def sample_points(mask, decim, max_points=4000, rng=None):
    """Native (row, col) of fresh burn, old burn and valid pixels of a mask read at 1/decim,
    at most max_points of each. Training windows are centred on these."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for name, sel in (("fresh", mask == L.FRESH), ("old", mask == L.OLD), ("valid", mask != L.NODATA)):
        rr, cc = np.nonzero(sel)
        if len(rr) > max_points:
            k = rng.choice(len(rr), max_points, replace=False)
            rr, cc = rr[k], cc[k]
        out[name] = (np.stack([rr, cc], 1) * decim + decim // 2).astype(np.int32)
    return out


def save_points(points, path):
    np.savez_compressed(path, **{f"{s}|{k}": v for s, d in points.items() for k, v in d.items()})


def load_points(path):
    out = {}
    with np.load(path) as z:
        for key in z.files:
            s, k = key.split("|")
            out.setdefault(s, {})[k] = z[key]
    return out


# ---- augmentation and normalisation ---------------------------------------------------
def shift_edge(a, dy, dx):
    """Shift a 2-D array by whole pixels, repeating the edge instead of wrapping around."""
    h, w = a.shape
    ys = np.clip(np.arange(h) - dy, 0, h - 1)
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    return a[np.ix_(ys, xs)]


AUG_DEFAULTS = dict(gain=(0.9, 1.1),          # one factor for all bands: calibration between spacecraft
                    band_gain=(0.95, 1.05),   # and a smaller one per band
                    p_shift=0.5, max_shift=1, # bands moved by up to max_shift pixels: L1B bands are not co-registered
                    p_blur=0.3, max_blur=1.0, # Gaussian blur, sigma up to max_blur pixels
                    noise=(0.0, 0.01))        # Gaussian noise, sigma drawn from this range (reflectance)


def augment(img, msk, rng, gain=(0.9, 1.1), band_gain=(0.95, 1.05), p_shift=0.5, max_shift=1,
            p_blur=0.3, max_blur=1.0, noise=(0.0, 0.01)):
    """Random rotation and flip, then radiometric and geometric changes to the image only.
    img is reflectance (7, H, W), msk the mask values (H, W)."""
    k = int(rng.integers(4))
    img, msk = np.rot90(img, k, axes=(1, 2)), np.rot90(msk, k)
    if rng.random() < 0.5:
        img, msk = img[:, :, ::-1], msk[:, ::-1]
    img = np.array(img, dtype=np.float32)
    msk = np.array(msk)
    nodata = msk == L.NODATA
    img *= np.float32(rng.uniform(*gain)) * rng.uniform(*band_gain, size=(img.shape[0], 1, 1)).astype(np.float32)
    if max_shift and rng.random() < p_shift:
        for b in range(img.shape[0]):
            dy, dx = (int(v) for v in rng.integers(-max_shift, max_shift + 1, 2))
            if dy or dx:
                img[b] = shift_edge(img[b], dy, dx)
    if max_blur and rng.random() < p_blur:
        s = float(rng.uniform(0.3, max_blur))
        img = ndimage.gaussian_filter(img, sigma=(0, s, s)).astype(np.float32)
    if noise[1] > 0:
        img += rng.normal(0, rng.uniform(*noise), img.shape).astype(np.float32)
    img[:, nodata] = 0
    return img, msk


def normalise(img, norm, nodata=None):
    """(clip(reflectance, 0, clip) - mean) / std per band; no-data pixels set to 0."""
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]
    x = (np.clip(img, 0, norm.get("clip", 1.2)) - mean) / std
    if nodata is not None:
        x[:, nodata] = 0
    return x.astype(np.float32)


# ---- datasets -----------------------------------------------------------------------------
def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def worker_number():
    """Number of the DataLoader worker this runs in; -1 in the main process (or without torch)."""
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return -1
    info = get_worker_info()
    return -1 if info is None else int(info.id)


class TrainWindows:
    """Random training windows. Item i is (image (7, size, size) float32 normalised,
    target (size, size) int64). Every call gives a new random window, so one pass is
    `length` windows and no two epochs are the same. A window depends only on seed, i, the
    DataLoader worker and how many windows that worker has made, so the same settings
    (seed, batch size, number of workers) give the same windows in every run.

    scenes: the train rows of data/tables/scenes.csv. points: from load_points().
    source_weights: how often each source is picked. gsd_range: (lo, hi) metres per source,
    drawn on a log scale. p_fresh, p_old: chance that a window is centred on fresh or old
    burn, a number or one per source; otherwise it is centred anywhere with data ("valid").
    valid_scenes: per source, the only scenes that may give "valid" windows (a source not
    listed: all of them); burn-centred windows can come from every scene. Fresh- and
    old-centred windows pick scenes by the square root of their burn pixels, plus a floor so
    every scene is seen; "valid" windows pick the allowed scenes evenly."""

    def __init__(self, scenes, points, norm, size=256, length=4000, source_weights=None, gsd_range=None,
                 p_fresh=0.4, p_old=0.15, valid_scenes=None, augment=True, aug=None, seed=0):
        self.scenes = scenes.reset_index(drop=True)
        self.points, self.norm, self.size, self.length = points, norm, size, length
        self.do_aug, self.seed = augment, seed
        self.aug = {**AUG_DEFAULTS, **(aug or {})}
        present = sorted(self.scenes.source.unique())
        sw = source_weights or {s: 1.0 for s in present}

        def per_source(v, s):
            return float(v.get(s, 0.0)) if isinstance(v, dict) else float(v)

        self.pick = {}
        for s in present:
            rows = self.scenes[self.scenes.source == s]
            allowed = set(valid_scenes[s]) if valid_scenes and s in valid_scenes else None
            for kind in ("fresh", "old", "valid"):
                idx, w = [], []
                for j, r in rows.iterrows():
                    pts = self.points.get(r["scene"], {}).get(kind)
                    if pts is None or len(pts) == 0:
                        continue
                    if kind == "valid" and allowed is not None and r["scene"] not in allowed:
                        continue
                    idx.append(j)
                    w.append(np.sqrt(float(r[f"{kind}_px"])) if kind != "valid" else 1.0)
                if idx:
                    w = np.asarray(w, float)
                    if kind != "valid":
                        w = w + max(w.mean(), 1.0) * 0.25
                    self.pick[(s, kind)] = (np.asarray(idx), w / w.sum())
        self.sources = [s for s in present if sw.get(s, 0) > 0 and any((s, k) in self.pick for k in ("fresh", "old", "valid"))]
        p = np.array([sw[s] for s in self.sources], float)
        self.src_p = p / p.sum()
        self.p_kind = {s: (per_source(p_fresh, s), per_source(p_old, s)) for s in self.sources}
        self.gsd_range = {s: tuple((gsd_range or {}).get(s, (2.5, 10.0))) for s in self.sources}
        self.reader, self._calls, self._pid = SceneReader(), 0, None

    def __len__(self):
        return self.length

    def draw(self, rng):
        """Choose source, kind of centre, scene, centre and gsd."""
        src = self.sources[rng.choice(len(self.sources), p=self.src_p)]
        pf, po = self.p_kind[src]
        u = rng.random()
        kind = "fresh" if u < pf else ("old" if u < pf + po else "valid")
        if (src, kind) not in self.pick:                          # nothing of that kind in this source
            kind = next(k for k in ("valid", "fresh", "old") if (src, k) in self.pick)
        idx, p = self.pick[(src, kind)]
        row = self.scenes.loc[int(rng.choice(idx, p=p))]
        pts = self.points[row["scene"]][kind]
        r, c = pts[int(rng.integers(len(pts)))]
        lo, hi = self.gsd_range[src]
        gsd = log_uniform(rng, max(lo, 1e-3), hi)
        n = self.size * gsd / row["px_m"]
        r, c = r + rng.uniform(-0.35, 0.35) * n, c + rng.uniform(-0.35, 0.35) * n   # burn not always in the middle
        return row, float(r), float(c), gsd, kind

    def get(self, i, with_info=False):
        if self._pid != os.getpid():                 # a new worker counts its windows from 0
            self._pid, self._calls = os.getpid(), 0
        self._calls += 1
        rng = np.random.default_rng([self.seed, int(i), self._calls, worker_number() + 1])
        row, r, c, gsd, kind = self.draw(rng)
        r0, c0, n = box_around(r, c, gsd, row["px_m"], self.size, row["height"], row["width"])
        img, msk = read_box(self.reader.get(row["img"]), self.reader.get(row["mask"]), r0, c0, n, self.size,
                            self.reader.bands(row["img"]), row["scale"])
        if self.do_aug:
            img, msk = augment(img, msk, rng, **self.aug)
        nodata = msk == L.NODATA
        x, y = normalise(img, self.norm, nodata), L.to_train_ids(msk).astype(np.int64)
        if with_info:
            return dict(x=x, y=y, refl=img, msk=msk, scene=row["scene"], source=row["source"], kind=kind,
                        gsd=n * row["px_m"] / self.size, box=(r0, c0, n))
        return x, y

    def __getitem__(self, i):
        return self.get(i)


class FixedWindows:
    """Windows listed in a table (scene, r0, c0, n, ...), for validation and test. No augmentation.
    Item i is (image, target) like TrainWindows; info(i) gives the table row."""

    def __init__(self, windows, scenes, norm, size=256):
        self.w = windows.reset_index(drop=True)
        self.scenes = scenes.set_index("scene")
        self.norm, self.size, self.reader = norm, size, SceneReader()

    def __len__(self):
        return len(self.w)

    def info(self, i):
        return self.w.loc[i]

    def get(self, i, with_refl=False):
        w = self.w.loc[i]
        s = self.scenes.loc[w["scene"]]
        img, msk = read_box(self.reader.get(s["img"]), self.reader.get(s["mask"]), int(w["r0"]), int(w["c0"]),
                            int(w["n"]), self.size, self.reader.bands(s["img"]), s["scale"])
        nodata = msk == L.NODATA
        x, y = normalise(img, self.norm, nodata), L.to_train_ids(msk).astype(np.int64)
        return (x, y, img, msk) if with_refl else (x, y)

    def __getitem__(self, i):
        return self.get(i)

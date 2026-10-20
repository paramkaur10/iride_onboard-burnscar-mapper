"""Label masks, version 3.

A mask has one band (uint8) on the same grid as its image:

    0 clear   2 fresh burn   3 old burn   4 cloud   5 cloud shadow   6 water
    254 ignore   255 no data                                   (1 is not used)

Burns come from EFFIS polygons. For every scene we take the polygons that touch it
and look at how old each fire was on the image date:

    0 to 90 days       fresh burn
    90 to 365 days     old burn
    older, or after    not a burn: clear, or water where NDWI > 0.05

Cloud, cloud shadow and no data always win over burns. Ignore marks the pixels we
don't trust: a band of about 10 m on both sides of every burn polygon edge, and burn
pixels that are green (NDVI > 0.5) or bright (mean of blue, green, red reflectance > 0.3).
Ignore and no data are left out of training and out of the scores.

relabel_scene() and run_all() turn a v2 mask into a v3 mask: the v2 burns are taken
out and painted again with the rules above, everything else is kept from v2
(notebook 02). Notebook 03 uses compose_v3() and clean_burns() directly on HEO
products, which have no v2 mask.
"""
from __future__ import annotations

import json
import os
import sys
import re
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage

CLEAR, FRESH, OLD, CLOUD, SHADOW, WATER = 0, 2, 3, 4, 5, 6
IGNORE, NODATA = 254, 255
V2_VALUES = {CLEAR, FRESH, OLD, CLOUD, SHADOW, WATER, NODATA}
V3_VALUES = [CLEAR, FRESH, OLD, CLOUD, SHADOW, WATER, IGNORE, NODATA]
V3_NAMES = {CLEAR: "clear", FRESH: "fresh", OLD: "old", CLOUD: "cloud",
            SHADOW: "shadow", WATER: "water", IGNORE: "ignore", NODATA: "nodata"}
V3_COLORS = {CLEAR: "#d9d8d3", FRESH: "#b3261e", OLD: "#eb6834", CLOUD: "#ffffff",
             SHADOW: "#3b3b55", WATER: "#2a78d6", IGNORE: "#e87ba4", NODATA: "#000000"}
V3_COLORMAP = {k: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) + ((0,) if k == NODATA else (255,))
               for k, c in V3_COLORS.items()}

# what the model is trained on: 6 classes, ignore and no data become -1
TRAIN_IDS = {CLEAR: 0, FRESH: 1, OLD: 2, CLOUD: 3, SHADOW: 4, WATER: 5, IGNORE: -1, NODATA: -1}
TRAIN_NAMES = ["clear", "fresh", "old", "cloud", "shadow", "water"]

# the two EFFIS exports the labels use, older first (see data/README.md)
EFFIS_FILES = ["data/effis/effis_IT-FR-ES-EL_2020-01-01_2026-08-03.zip",
               "data/effis/effis_IT_2026-03-01_2026-10-05.json"]

SOURCES = ("heo", "phisat2_sim", "s2_test")


@dataclass
class RelabelConfig:
    fresh_max: float = 90.0          # days, 0 to fresh_max is fresh burn
    old_max: float = 365.0           # days, fresh_max to old_max is old burn
    ignore_max: float = 365.0        # days, old_max to ignore_max is ignore (nothing when equal to old_max)
    edge_m: float = 10.0             # ignore band on both sides of burn polygon edges, metres (0 = off)
    ndwi_threshold: float = 0.05
    ndvi_max: float | None = 0.5     # burn pixels greener than this become ignore (None = off)
    bright_max: float | None = 0.3   # burn pixels with mean B, G, R reflectance above this become ignore (None = off)

    def __post_init__(self):
        if not (0 <= self.fresh_max <= self.old_max <= self.ignore_max):
            raise ValueError("need 0 <= fresh_max <= old_max <= ignore_max")


# ---- names, dates, sources -------------------------------------------------------
_HEO_RE = re.compile(r"_(\d{8}T\d{6})_")
_PHI_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def scene_id_from_mask_path(path):
    name = os.path.basename(path)
    for suffix in ("_mask_v2.tif", "_mask.tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def source_of(path_or_scene):
    """heo, phisat2_sim or s2_test. Taken from the folder when the path has one, else from the name."""
    parts = os.path.normpath(str(path_or_scene)).split(os.sep)
    for s in SOURCES:
        if s in parts:
            return s
    return "heo" if os.path.basename(str(path_or_scene)).startswith("IMH") else "phisat2_sim"


def reflectance_scale(path_or_scene):
    """What the stored values are divided by to get reflectance: HEO 4094, PhiSat-2 and Sentinel-2 10000."""
    return 4094.0 if source_of(path_or_scene) == "heo" else 10000.0


def parse_acq_datetime(scene_id):
    """Image date from the scene name.
    HEO:                  IMH04_1CST__OPT8_20260317T123240_...
    PhiSat-2 and S2 test: effis_278540_2025-08-17T10-09-37"""
    if scene_id.startswith("IMH"):
        m = _HEO_RE.search(scene_id)
        if m:
            return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
    m = _PHI_RE.search(scene_id)
    if m:
        d, hh, mm, ss = m.groups()
        return pd.Timestamp(f"{d}T{hh}:{mm}:{ss}")
    return None


def to_train_ids(mask):
    """v3 mask values to training ids 0..5; ignore, no data and anything unexpected become -1."""
    lut = np.full(256, -1, np.int8)
    for v, t in TRAIN_IDS.items():
        lut[v] = t
    return lut[np.asarray(mask, np.uint8)]


# ---- the rules ---------------------------------------------------------------------
def age_windows(age_days, fresh_max, old_max, ignore_max):
    a = np.asarray(age_days, dtype=np.float64)
    fresh = (a >= 0) & (a <= fresh_max)
    old = (a > fresh_max) & (a <= old_max)
    ign = (a > old_max) & (a <= ignore_max)
    return fresh, old, ign


def disk(radius):
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def edge_ring(mask, radius_px):
    """Pixels within radius_px of the mask boundary, on both sides. The image border is not a boundary."""
    mask = np.asarray(mask, dtype=bool)
    if radius_px <= 0 or not mask.any():
        return np.zeros_like(mask)
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    m = radius_px + 1
    r0, c0 = max(r0 - m, 0), max(c0 - m, 0)
    r1, c1 = min(r1 + m, mask.shape[0]), min(c1 + m, mask.shape[1])
    sub, st = mask[r0:r1, c0:c1], disk(radius_px)
    out = np.zeros_like(mask)
    out[r0:r1, c0:c1] = (ndimage.binary_dilation(sub, structure=st)
                         & ~ndimage.binary_erosion(sub, structure=st, border_value=1))
    return out


def ndwi_water(green, nir, threshold=0.05):
    g, n = green.astype(np.float32), nir.astype(np.float32)
    return ((g - n) / np.maximum(g + n, 1e-6)) > threshold


def compose_v3(mask_v2, fresh_poly, old_poly, ign_poly, water=None, edge_px=0):
    """Take the burns out of a mask and paint them again from the polygons."""
    lab = mask_v2.astype(np.uint8, copy=True)
    was_burn = (lab == FRESH) | (lab == OLD)                  # 1. old burns out: clear, or water if NDWI says so
    if water is not None:
        lab[was_burn] = np.where(water[was_burn], WATER, CLEAR).astype(np.uint8)
    else:
        lab[was_burn] = CLEAR
    bg = (lab == CLEAR) | (lab == WATER)                      # 2. burns only go on clear or water
    lab[ign_poly & ~fresh_poly & ~old_poly & (lab == CLEAR)] = IGNORE
    lab[old_poly & ~fresh_poly & bg] = OLD
    lab[fresh_poly & bg] = FRESH
    if edge_px > 0:                                           # 3. band along the polygon edges
        ring = edge_ring(fresh_poly, edge_px) | edge_ring(old_poly & ~fresh_poly, edge_px)
        lab[ring & np.isin(lab, (CLEAR, FRESH, OLD))] = IGNORE
    return lab


def clean_burns(lab, b, g, r, nir, ndvi_max=None, bright_max=None):
    """Burn pixels that are clearly green or very bright become ignore. Bands in reflectance (0 to 1)."""
    lab = lab.copy()
    burn = (lab == FRESH) | (lab == OLD)
    if not burn.any():
        return lab
    drop = np.zeros_like(burn)
    if ndvi_max is not None:
        ndvi = (nir - r) / np.maximum(nir + r, 1e-6)
        drop |= ndvi > ndvi_max
    if bright_max is not None:
        drop |= (b + g + r) / 3.0 > bright_max
    lab[burn & drop] = IGNORE
    return lab


def class_counts(mask, prefix):
    vals, cnts = np.unique(mask, return_counts=True)
    d = dict(zip(vals.tolist(), cnts.tolist()))
    out = {f"{prefix}{V3_NAMES[v]}": int(d.get(v, 0)) for v in V3_VALUES}
    other = sum(c for v, c in d.items() if v not in V3_NAMES)
    if other:
        out[f"{prefix}other"] = int(other)
    return out


# ---- files ---------------------------------------------------------------------------
def load_fires(effis_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from effis_loader import load_effis
    fires = load_effis(effis_path)
    fires = fires[fires.geometry.notna() & ~fires.geometry.is_empty].copy()
    fires["geometry"] = fires.geometry.make_valid()
    _ = fires.sindex
    return fires.reset_index(drop=True)


def load_fires_merged(paths=EFFIS_FILES):
    """All exports in one table. A fire found in more than one export is taken from the
    last file in the list, so put the newest export last. Returns (merged, one table per file)."""
    parts = [load_fires(p) for p in paths]
    fires = (pd.concat([p.assign(_src=i) for i, p in enumerate(parts)], ignore_index=True)
               .sort_values("_src", kind="stable").drop_duplicates("fire_id", keep="last")
               .drop(columns="_src").reset_index(drop=True))
    _ = fires.sindex
    return fires, parts


def find_image_for_mask(mask_path):
    """data/<source>/masks_v2/<scene>_mask.tif -> data/<source>/images/<scene>_img.tif (or _s2.tif)."""
    d = os.path.dirname(mask_path)
    sid = scene_id_from_mask_path(mask_path)
    folders = [d]
    if os.path.basename(d) in ("masks", "masks_v2", "masks_v3"):
        folders.insert(0, os.path.join(os.path.dirname(d), "images"))
    for folder in folders:
        for suffix in ("_img.tif", "_s2.tif", "_phisat2.tif"):
            p = os.path.join(folder, sid + suffix)
            if p != mask_path and os.path.exists(p):
                return p
    return None


def sibling_out_path(mask_path, subdir="masks_v3"):
    """data/<source>/masks_v2/<scene>_mask.tif -> data/<source>/masks_v3/<scene>_mask.tif"""
    d, name = os.path.split(mask_path)
    name = name.replace("_mask_v2.tif", "_mask.tif")
    if os.path.basename(d) in ("masks", "masks_v2"):
        return os.path.join(os.path.dirname(d), subdir, name)
    return os.path.join(d, subdir, name)


def bgrn_band_indexes(src):
    """1-based (blue, green, red, nir) of an HEO, PhiSat-2 or Sentinel-2 file, from the band names if it has them."""
    desc = [(d or "").strip().upper() for d in src.descriptions]
    if all(k in desc for k in ("B", "G", "R", "NIR")):
        return tuple(desc.index(k) + 1 for k in ("B", "G", "R", "NIR"))
    if all(k in desc for k in ("B02", "B03", "B04", "B08")):
        return tuple(desc.index(k) + 1 for k in ("B02", "B03", "B04", "B08"))
    return {7: (1, 2, 3, 7), 8: (2, 3, 4, 8)}.get(src.count)


def green_nir_band_indexes(src):
    """1-based (green, nir). B03 is green in Sentinel-2 names but red in HEO product names,
    so Sentinel-2 names are trusted only when B08 is there too."""
    desc = [(d or "").strip().upper() for d in src.descriptions]
    if "G" in desc and "NIR" in desc:
        return desc.index("G") + 1, desc.index("NIR") + 1
    if "B03" in desc and "B08" in desc:
        return desc.index("B03") + 1, desc.index("B08") + 1
    return {7: (2, 7), 8: (3, 8)}.get(src.count)


def rasterize_bool(geoms, shape, transform):
    from rasterio.features import rasterize
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                     fill=0, dtype="uint8", all_touched=False).astype(bool)


def scene_fires(fires, crs, bounds, acq):
    """EFFIS polygons touching the scene, in the scene CRS, with age_days on the image date."""
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    ll = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    sel = fires.iloc[fires.sindex.query(box(*ll), predicate="intersects")].to_crs(crs)
    sel = sel[sel.intersects(box(*bounds))].copy()
    sel["age_days"] = (acq - sel["fire_date"]).dt.total_seconds() / 86400.0
    return sel


def write_mask(path, mask, profile, cfg):
    """Write a v3 mask with its colour table and the settings it was made with."""
    import rasterio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = dict(profile)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=NODATA,
                   compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    for k in ("photometric", "predictor"):
        profile.pop(k, None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.write_colormap(1, V3_COLORMAP)
        dst.update_tags(LABEL_VERSION="v3", FRESH_MAX_D=cfg.fresh_max, OLD_MAX_D=cfg.old_max,
                        IGNORE_MAX_D=cfg.ignore_max, EDGE_M=cfg.edge_m,
                        NDVI_MAX=cfg.ndvi_max, BRIGHT_MAX=cfg.bright_max,
                        LEGEND=json.dumps({str(k): v for k, v in V3_NAMES.items()}))


# ---- v2 -> v3 ------------------------------------------------------------------------
def relabel_scene(mask_path, fires, cfg, out_path=None, keep_arrays=True):
    import rasterio
    sid = scene_id_from_mask_path(mask_path)
    row = dict(scene=sid, source=source_of(mask_path), mask_v2=mask_path, status="ok")
    res = {"row": row}
    acq = parse_acq_datetime(sid)
    if acq is None:
        row["status"] = "skip: cannot read the image date from the name"
        return res
    row["acq"] = acq.isoformat()
    with rasterio.open(mask_path) as src:
        m2 = src.read(1)
        profile, tr, crs, bounds = src.profile.copy(), src.transform, src.crs, src.bounds
        px = float(abs(src.res[0]))
    row["px_m"] = px
    unexpected = sorted(set(np.unique(m2).tolist()) - V2_VALUES)
    if unexpected:
        row["status"] = f"skip: unexpected mask values {unexpected}"
        return res

    sf = scene_fires(fires, crs, bounds, acq)
    need_bands = ((m2 == FRESH) | (m2 == OLD)).any() or (
        len(sf) and (cfg.ndvi_max is not None or cfg.bright_max is not None))
    water, refl = None, None
    if need_bands:
        img_path = find_image_for_mask(mask_path)
        if img_path is None:
            row["warn"] = "image not found: no NDWI water for former burns, no cleaning"
        else:
            with rasterio.open(img_path) as isrc:
                idx = bgrn_band_indexes(isrc)
                if idx is None or (isrc.height, isrc.width) != m2.shape:
                    row["warn"] = "image bands or shape unexpected: no NDWI, no cleaning"
                else:
                    scale = reflectance_scale(mask_path)
                    refl = [isrc.read(i).astype(np.float32) / scale for i in idx]   # B, G, R, NIR
                    water = ndwi_water(refl[1], refl[3], cfg.ndwi_threshold)

    age = sf["age_days"].to_numpy()
    f_sel, o_sel, i_sel = age_windows(age, cfg.fresh_max, cfg.old_max, cfg.ignore_max)
    fresh_poly = rasterize_bool(sf.geometry[f_sel], m2.shape, tr)
    old_poly = rasterize_bool(sf.geometry[o_sel], m2.shape, tr)
    ign_poly = rasterize_bool(sf.geometry[i_sel], m2.shape, tr)
    row.update(n_fires_fresh=int(f_sel.sum()), n_fires_old=int(o_sel.sum()),
               n_fires_ignore=int(i_sel.sum()), n_fires_older=int((age > cfg.ignore_max).sum()),
               n_fires_future=int((age < 0).sum()))
    edge_px = int(round(cfg.edge_m / px)) if cfg.edge_m > 0 else 0
    m3 = compose_v3(m2, fresh_poly, old_poly, ign_poly, water=water, edge_px=edge_px)
    if refl is not None and (cfg.ndvi_max is not None or cfg.bright_max is not None):
        m3 = clean_burns(m3, *refl, ndvi_max=cfg.ndvi_max, bright_max=cfg.bright_max)
    row.update(class_counts(m2, "v2_"))
    row.update(class_counts(m3, "v3_"))
    row["changed_px"] = int((m2 != m3).sum())

    if out_path:
        write_mask(out_path, m3, profile, cfg)
        row["mask_v3"] = out_path
    if keep_arrays:
        res.update(m2=m2, m3=m3, fires=sf, water=water)
    return res


_FIRES, _CFG, _OUT_FN = None, None, None


def _worker(mask_path):
    try:
        out = _OUT_FN(mask_path) if _OUT_FN else None
        return relabel_scene(mask_path, _FIRES, _CFG, out_path=out, keep_arrays=False)["row"]
    except Exception as e:
        return dict(scene=scene_id_from_mask_path(mask_path), source=source_of(mask_path),
                    mask_v2=mask_path, status=f"error: {e!r}")


def run_all(mask_paths, fires, cfg, out_path_fn=None, workers=4, every=25):
    """out_path_fn=None is a dry run; out_path_fn=sibling_out_path writes masks_v3/."""
    global _FIRES, _CFG, _OUT_FN
    _FIRES, _CFG, _OUT_FN = fires, cfg, out_path_fn
    mask_paths, rows, t0 = list(mask_paths), [], time.time()
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            futs = [ex.submit(_worker, m) for m in mask_paths]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % every == 0 or i == len(mask_paths):
                    print(f"  {i}/{len(mask_paths)} scenes ({time.time() - t0:.0f}s)")
    else:
        for i, m in enumerate(mask_paths, 1):
            rows.append(_worker(m))
            if i % every == 0 or i == len(mask_paths):
                print(f"  {i}/{len(mask_paths)} scenes ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows).sort_values(["source", "scene"]).reset_index(drop=True)


def summary_table(df):
    """% of pixels per class before and after, and the number of scenes that have it, per source."""
    ok, out = df[df.status == "ok"], []
    for src_name, g in ok.groupby("source"):
        t2 = {n: g[f"v2_{n}"].sum() if f"v2_{n}" in g else 0 for n in V3_NAMES.values()}
        t3 = {n: g[f"v3_{n}"].sum() if f"v3_{n}" in g else 0 for n in V3_NAMES.values()}
        s2, s3 = sum(t2.values()) or 1, sum(t3.values()) or 1
        for n in V3_NAMES.values():
            out.append(dict(source=src_name, cls=n, v2_pct=100 * t2[n] / s2, v3_pct=100 * t3[n] / s3,
                            v2_scenes=int((g.get(f"v2_{n}", pd.Series(0)) > 0).sum()),
                            v3_scenes=int((g.get(f"v3_{n}", pd.Series(0)) > 0).sum())))
    return pd.DataFrame(out)

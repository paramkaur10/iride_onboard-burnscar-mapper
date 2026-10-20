"""
Author:      Parampuneet Thind <parampuneet.thind@uniroma1.it>
Company:     Sapienza University of Rome
App Name:    Burn-scar mapper (PyNAS)
Created:     2026-10-15
Modified:    2026-10-20
Version:     1.1.2
Python:      3.8+
Purpose:     Fresh and old burn scars, cloud, cloud shadow and water in HEO images, with a network found by
             PyNAS for the Myriad X (OpenVINO 2022.3.1 FP16 IR).
Usage:       burnscar_app.py --run_id ID --device {0,1} --model /models/<name>.xml --img-dir DIR --output-dir DIR
Notes:       model_input.json next to this script says what the app reads and how: the input layouts it accepts
             (Argotec's chips, one file per band, band-0 PAN; HEO L1C products, band files B00 (PAN) to B07; image
             stacks of 7 or 8 bands), the bands the network takes in each layout, the conversion of the stored values
             to reflectance (gain and offset per band), the no-data value and the scale (1: the network sees the
             image's own pixels; 2: the mean of 2 x 2 pixels). Output: one zip with a class map (GeoTIFF) for every
             image with burn scars and a summary (JSON) of every image; with "report" in model_input.json the summary
             also gives, for every image, the range of its stored values in each band and the metadata of its files,
             and the other files found next to the images.
"""
import os

# before any GDAL or numerical library loads: every block of an image is read once, so GDAL needs no large cache
# (its default, 5% of the machine's memory, would keep whole images), and no library may start threads of its own
os.environ.setdefault("GDAL_CACHEMAX", "32")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import shutil
import sys
import time
import zipfile

import numpy as np

VERSION = "1.1.2"
HERE = os.path.dirname(os.path.abspath(__file__))
CHIP_FILE = re.compile(r"^chip-(.+)_band-(\d+)\.tiff?$", re.IGNORECASE)
PRODUCT_FILE = re.compile(r"^(.+)\.B0([0-7])\.tiff?$", re.IGNORECASE)
TIFF_FILE = re.compile(r"\.tiff?$", re.IGNORECASE)
NODATA_OUT = 255
GEO_TAGS = (33550, 33922, 34264, 34735, 34736, 34737)
STACK_NAMES = ["B", "G", "R", "RE1", "RE2", "RE3", "NIR"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---- GeoTIFF reading and writing: rasterio, else GDAL, else tifffile --------------------------------------
def backend():
    for name in ("rasterio", "osgeo.gdal", "tifffile"):
        try:
            __import__(name)
            return name
        except ImportError:
            pass
    sys.exit("no GeoTIFF library (rasterio, GDAL or tifffile) in this environment")


class Raster:
    """One GeoTIFF of one or more bands, read by blocks of rows: only the rows asked for are decoded."""

    def __init__(self, path, lib):
        self.path, self.lib, self.whole = path, lib, None
        if lib == "rasterio":
            import rasterio
            self.ds = rasterio.open(path)
            self.height, self.width, self.count = self.ds.height, self.ds.width, self.ds.count
        elif lib == "osgeo.gdal":
            from osgeo import gdal
            self.ds = gdal.Open(path)
            if self.ds is None:
                raise IOError(f"GDAL cannot open {path}")
            self.height, self.width, self.count = self.ds.RasterYSize, self.ds.RasterXSize, self.ds.RasterCount
        else:
            import tifffile
            self.ds = tifffile.TiffFile(path)
            p = self.page = self.ds.pages[0]
            self.height, self.width = int(p.imagelength), int(p.imagewidth)
            self.spp = int(p.samplesperpixel)
            self.separate = int(p.planarconfig) == 2
            self.paged = self.spp == 1 and len(self.ds.pages) > 1 and all(
                (int(q.imagelength), int(q.imagewidth)) == (self.height, self.width) for q in self.ds.pages)
            self.count = len(self.ds.pages) if self.paged else self.spp

    def descriptions(self):
        """Band names, when the file has them."""
        try:
            if self.lib == "rasterio":
                return [d or "" for d in self.ds.descriptions]
            if self.lib == "osgeo.gdal":
                return [self.ds.GetRasterBand(i).GetDescription() or "" for i in range(1, self.count + 1)]
            tag = self.page.tags.get(42112)                     # GDAL_METADATA
            names = [""] * self.count
            if tag is not None:
                for item in re.findall(r"<Item\b[^>]*>[^<]*</Item>", str(tag.value)):
                    s = re.search(r'sample="(\d+)"', item)
                    if s and 'role="description"' in item and int(s.group(1)) < self.count:
                        names[int(s.group(1))] = re.sub(r"<[^>]+>", "", item).strip()
            return names
        except Exception:
            return [""] * self.count

    def nodata(self):
        try:
            if self.lib == "rasterio":
                return self.ds.nodata
            if self.lib == "osgeo.gdal":
                return self.ds.GetRasterBand(1).GetNoDataValue()
            tag = self.page.tags.get(42113)                     # GDAL_NODATA
            return float(str(tag.value).strip("\x00 ")) if tag is not None else None
        except Exception:
            return None

    def rows(self, bands, r0, h):
        """(len(bands), h, width) of the 1-based bands, rows r0 to r0 + h."""
        if self.lib == "rasterio":
            from rasterio.windows import Window
            return self.ds.read(list(bands), window=Window(0, r0, self.width, h))
        if self.lib == "osgeo.gdal":
            return np.stack([self.ds.GetRasterBand(b).ReadAsArray(0, r0, self.width, h) for b in bands])
        if self.whole is None:
            try:
                return self._segment_rows(bands, r0, h)
            except Exception as e:                              # an encoding the segment reader does not know
                log(f"{os.path.basename(self.path)}: read whole ({str(e)[:120]})")
                self.whole = self._read_whole()
        return self.whole[[b - 1 for b in bands], r0:r0 + h]

    def _read_whole(self):
        if self.paged:
            return np.stack([q.asarray() for q in self.ds.pages])
        a = self.page.asarray()
        if self.spp == 1:
            return a[None]
        return a if self.separate else np.moveaxis(a, -1, 0)

    def _segment_rows(self, bands, r0, h):
        """tifffile: decode only the tiles or strips that hold rows r0 to r0 + h, of the bands asked for."""
        out = np.zeros((len(bands), h, self.width), self.page.dtype)
        if self.paged:
            for j, b in enumerate(bands):
                out[j] = self._page_rows(self.ds.pages[b - 1], [0], r0, h)[0]
            return out
        if self.spp == 1 or self.separate:
            planes = [b - 1 for b in bands]
            got = self._page_rows(self.page, planes, r0, h)
            return got
        got = self._page_rows(self.page, [0], r0, h, contig=True)  # (spp, h, width)
        return got[[b - 1 for b in bands]]

    def _page_rows(self, p, planes, r0, h, contig=False):
        fh = self.ds.filehandle
        H, W = int(p.imagelength), int(p.imagewidth)
        if p.is_tiled:
            sh, sw = int(p.tilelength), int(p.tilewidth)
        else:
            sh, sw = min(int(p.rowsperstrip) or H, H), W
        ncol = -(-W // sw)
        per_plane = (-(-H // sh)) * ncol
        spp = int(p.samplesperpixel) if contig else 1
        out = np.zeros((spp if contig else len(planes), h, W), p.dtype)
        for k, plane in enumerate(planes):
            for sr in range(r0 // sh, (r0 + h - 1) // sh + 1):
                for sc in range(ncol):
                    i = plane * per_plane + sr * ncol + sc
                    if not p.databytecounts[i]:
                        continue
                    fh.seek(p.dataoffsets[i])
                    seg, idx, _ = p.decode(fh.read(p.databytecounts[i]), i, jpegtables=p.jpegtables)
                    if seg is None:
                        continue
                    seg = np.asarray(seg)
                    seg = seg.reshape(seg.shape[-3], seg.shape[-2], seg.shape[-1])      # (rows, cols, samples)
                    y0, x0 = int(idx[2]), int(idx[3])
                    a, b = max(r0, y0), min(r0 + h, y0 + seg.shape[0], H)
                    if b <= a:
                        continue
                    x1 = min(W, x0 + seg.shape[1])
                    part = seg[a - y0:b - y0, :x1 - x0]
                    if contig:
                        out[:, a - r0:b - r0, x0:x1] = np.moveaxis(part, -1, 0)
                    else:
                        out[k, a - r0:b - r0, x0:x1] = part[..., 0]
        return out

    def metadata(self, limit=300):
        """What the file says about itself: data type, no-data value, georeference and its metadata items."""
        cut = lambda v: str(v)[:limit]                               # noqa: E731
        try:
            if self.lib == "rasterio":
                d = self.ds
                return dict(dtype=str(d.dtypes[0]), nodata=d.nodata, crs=cut(d.crs), pixel=list(d.res),
                            tags={k: cut(v) for k, v in d.tags().items()}, band_tags={k: cut(v) for k, v in d.tags(1).items()},
                            descriptions=[x or "" for x in d.descriptions])
            if self.lib == "osgeo.gdal":
                d, b = self.ds, self.ds.GetRasterBand(1)
                from osgeo import gdal
                return dict(dtype=gdal.GetDataTypeName(b.DataType), nodata=b.GetNoDataValue(), crs=cut(d.GetProjection()),
                            pixel=list(d.GetGeoTransform())[1:6:4], domains=d.GetMetadataDomainList(),
                            tags={k: cut(v) for k, v in (d.GetMetadata() or {}).items()},
                            band_tags={k: cut(v) for k, v in (b.GetMetadata() or {}).items()})
            skip = (273, 279, 324, 325, 322, 323)                    # offsets and byte counts of strips and tiles
            return dict(dtype=str(self.page.dtype), tags={t.name: cut(t.value) for t in self.page.tags.values()
                                                          if t.code not in skip})
        except Exception as e:
            return dict(error=cut(e))

    def pixel_m(self):
        try:
            if self.lib == "rasterio":
                return float(abs(self.ds.res[0]))
            if self.lib == "osgeo.gdal":
                return float(abs(self.ds.GetGeoTransform()[1]))
            tag = self.page.tags.get(33550)
            return float(tag.value[0]) if tag else None
        except Exception:
            return None

    def write_like(self, path, arr, scale=1):
        """arr (uint8; the image's size, or 1/scale of it) as a compressed GeoTIFF with this file's georeference."""
        if self.lib == "rasterio":
            import rasterio
            from rasterio.transform import Affine
            prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint8",
                        crs=self.ds.crs, transform=self.ds.transform * Affine.scale(scale), nodata=NODATA_OUT,
                        compress="deflate", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(path, "w", **prof) as out:
                out.write(arr, 1)
        elif self.lib == "osgeo.gdal":
            from osgeo import gdal
            out = gdal.GetDriverByName("GTiff").Create(path, arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte,
                                                       options=["COMPRESS=DEFLATE", "TILED=YES"])
            gt = list(self.ds.GetGeoTransform())
            gt[1], gt[2], gt[4], gt[5] = gt[1] * scale, gt[2] * scale, gt[4] * scale, gt[5] * scale
            out.SetGeoTransform(gt)
            out.SetProjection(self.ds.GetProjection())
            out.GetRasterBand(1).WriteArray(arr)
            out.GetRasterBand(1).SetNoDataValue(NODATA_OUT)
            out.FlushCache()
            out = None
        else:
            import tifffile
            tags = self.page.tags
            extra = []
            for c in GEO_TAGS:
                if c in tags:
                    v = tags[c].value
                    if c == 33550 and scale != 1:                       # pixel size
                        v = (v[0] * scale, v[1] * scale) + tuple(v[2:])
                    if c == 34264 and scale != 1:                       # full transformation matrix
                        v = list(v)
                        for k in (0, 1, 4, 5):
                            v[k] *= scale
                        v = tuple(v)
                    extra.append((c, int(tags[c].dtype), tags[c].count, v, True))
            extra.append((42113, "s", 0, str(NODATA_OUT), True))
            tifffile.imwrite(path, arr, compression="zlib", tile=(256, 256), extratags=extra)

    def close(self):
        try:
            self.ds.close()
        except Exception:
            pass
        self.whole = None


class Image:
    """The bands the network takes, from one or more files: [(file, 1-based band in that file), ...] in the
    network's band order (blue, green, red, red edge 1-3, NIR)."""

    def __init__(self, iid, layout, parts, lib):
        self.id, self.layout, self.lib = iid, layout, lib
        self.files = {}
        self.parts = []
        for path, band in parts:
            if path not in self.files:
                self.files[path] = Raster(path, lib)
            self.parts.append((self.files[path], band))
        first = self.parts[0][0]
        self.height, self.width = first.height, first.width
        for r, _ in self.parts:
            if (r.height, r.width) != (self.height, self.width):
                self.close()
                raise ValueError(f"bands of different sizes ({r.height} x {r.width} and {self.height} x {self.width})")
        self.ref = first

    def rows(self, r0, h):
        """(bands, h, width) as stored, in the network's band order; a file with several bands is read once."""
        out, groups = [None] * len(self.parts), {}
        for k, (r, b) in enumerate(self.parts):
            groups.setdefault(r.path, []).append((k, b))
        for path, kb in groups.items():
            got = self.files[path].rows([b for _, b in kb], r0, h)
            for j, (k, _) in enumerate(kb):
                out[k] = got[j]
        return np.stack(out)

    def nodata_values(self, configured):
        v = {configured}
        own = self.ref.nodata()
        if own is not None and np.isfinite(own):
            v.add(own)
        return sorted(v)

    def close(self):
        for r in self.files.values():
            r.close()


# ---- finding the images ---------------------------------------------------------------------------------------
def stack_bands(names, count):
    """1-based bands of the network's 7 in an image stack, as in training: by band name when the file has the names,
    else 7 bands in the network's order, else 8 bands with PAN first."""
    up = [(n or "").strip().upper() for n in names]
    if all(n in up for n in STACK_NAMES):
        return [up.index(n) + 1 for n in STACK_NAMES]
    if count == 7:
        return list(range(1, 8))
    if count == 8:
        return list(range(2, 9))
    return None


def product_id(folder, prefix):
    name = os.path.basename(os.path.normpath(folder))
    if name.upper() in ("IMG_DATA", "IMAGE_DATA", "DATA", "IMG"):
        name = os.path.basename(os.path.dirname(os.path.normpath(folder))) or name
    return name or prefix


def find_images(img_dir, st, lib):
    """Every image in img_dir and below, in the layouts the package accepts. Returns (images, skipped, extra):
    images as (id, layout, [(file, band), ...]) in the network's band order; skipped as {file or id: reason}; extra, the
    files that are not TIFFs (up to 50)."""
    layouts = st.get("layouts", ["chips"])
    chip_bands = st.get("chip_bands", st.get("bands", [1, 2, 3, 4, 5, 6, 7]))
    product_bands = st.get("product_bands", [1, 2, 3, 4, 5, 6, 7])
    chips, products, others, skipped, extra = {}, {}, [], {}, []
    for d, subdirs, files in os.walk(img_dir, followlinks=True):
        subdirs.sort()
        for name in sorted(files):
            path = os.path.join(d, name)
            m, p = CHIP_FILE.match(name), PRODUCT_FILE.match(name)
            if m and "chips" in layouts:
                chips.setdefault(m.group(1), {})[int(m.group(2))] = path
            elif p and "product" in layouts:
                products.setdefault((d, p.group(1)), {})[int(p.group(2))] = path
            elif TIFF_FILE.search(name):
                others.append(path)
            elif len(extra) < 50:
                try:
                    extra.append(dict(file=os.path.relpath(path, img_dir), kb=round(os.path.getsize(path) / 1024, 1)))
                except OSError:
                    pass
    images = []
    for cid, files in sorted(chips.items(), key=lambda kv: (len(kv[0]), kv[0])):
        missing = [b for b in chip_bands if b not in files]
        if missing:
            skipped[f"chip-{cid}"] = f"no band file {missing}"
        else:
            images.append((f"chip-{cid}", "chips", [(files[b], 1) for b in chip_bands]))
    for (d, prefix), files in sorted(products.items()):
        iid = product_id(d, prefix)
        missing = [b for b in product_bands if b not in files]
        if missing:
            skipped[iid] = f"no band file B0{missing}"
        else:
            images.append((iid, "product", [(files[b], 1) for b in product_bands]))
    for path in others:
        name = os.path.basename(path)
        if "stack" not in layouts:
            skipped[name] = "not in a layout this package reads"
            continue
        try:
            r = Raster(path, lib)
            bands = stack_bands(r.descriptions(), r.count)
            count = r.count
            r.close()
        except Exception as e:
            skipped[name] = f"cannot be read ({str(e)[:120]})"
            continue
        if bands is None:
            skipped[name] = f"{count} band(s): not an image stack the app reads"
        else:
            images.append((TIFF_FILE.sub("", name), "stack", [(path, b) for b in bands]))
    return images, skipped, extra


# ---- the network --------------------------------------------------------------------------------------------
class Network:
    def __init__(self, xml, device, cpu_threads=1):
        from openvino.runtime import Core, get_version
        self.ov_version = get_version()
        core = Core()
        model = core.read_model(xml)
        want = "MYRIAD" if device == 1 else "CPU"
        try:
            compiled = core.compile_model(model, want) if want == "MYRIAD" else self._cpu(core, model, cpu_threads)
            self.device = want
        except Exception as e:
            if want == "CPU":
                raise
            log(f"MYRIAD not available ({str(e).splitlines()[0][:200]}); running on the CPU")
            compiled = self._cpu(core, model, cpu_threads)
            self.device = "CPU (MYRIAD not available)"
        self.cpu_threads = cpu_threads if self.device != "MYRIAD" else None
        self.request = compiled.create_infer_request()
        self.inp, self.out = compiled.input(0), compiled.output(0)
        self.shape = [int(d) for d in self.inp.shape]           # (1, bands, tile, tile)
        self.queue, self.jobs = None, 1
        if self.device == "MYRIAD":                             # several tiles in flight on the VPU
            try:
                from openvino.runtime import AsyncInferQueue
                try:
                    n = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
                except Exception:
                    n = 4
                self.jobs = max(1, min(n, 4))
                self.queue = AsyncInferQueue(compiled, self.jobs)
                self.queue.set_callback(self._done)
            except Exception as e:
                log(f"asynchronous requests not available ({e}); one tile at a time")
                self.queue, self.jobs = None, 1

    @staticmethod
    def _cpu(core, model, threads):
        """The CPU plugin with a fixed number of inference threads (the app may use 2 cores at most)."""
        for key in ("INFERENCE_NUM_THREADS", "CPU_THREADS_NUM"):
            try:
                return core.compile_model(model, "CPU", {key: str(threads)})
            except Exception:
                pass
        return core.compile_model(model, "CPU")

    @staticmethod
    def _store(logits, job):
        out, r0, c0, h, w, empty, _ = job
        cls = np.asarray(logits)[0].argmax(0).astype(np.uint8)
        cls[empty] = NODATA_OUT
        out[r0:r0 + h, c0:c0 + w] = cls[:h, :w]

    def _done(self, request, job):
        self._store(request.get_output_tensor(0).data, job)

    def submit(self, x, job):
        """Classify tile x; job = (out, r0, c0, h, w, empty, x): where its classes go."""
        if self.queue is None:
            self._store(self.request.infer({self.inp: x})[self.out], job)
        else:
            self.queue.start_async({self.inp: x}, job)

    def wait(self):
        if self.queue is not None:
            self.queue.wait_all()


def classify(img, st, net):
    """Class index per pixel of one image (uint8, NODATA_OUT where the image has no data)."""
    S = int(st.get("scale", 1))
    H, W = img.height // S, img.width // S                     # the grid the network sees
    _, C, T, _ = net.shape
    if C != len(img.parts):
        raise ValueError(f"the model takes {C} bands, the package gives {len(img.parts)}")
    gain = np.asarray(st["gain"], np.float32)[:, None, None]
    offset = np.asarray(st["offset"], np.float32)[:, None, None]
    mean = np.asarray(st["mean"], np.float32)[:, None, None]
    std = np.asarray(st["std"], np.float32)[:, None, None]
    nodata = img.nodata_values(st["nodata"])
    out = np.full((H, W), NODATA_OUT, np.uint8)
    report = bool(st.get("report"))
    samples, zeros, tops, dtype = [], np.zeros(C, np.int64), np.zeros(C, np.int64), None
    for r0 in range(0, H, T):
        h = min(T, H - r0)
        block = img.rows(r0 * S, h * S)[:, :, :W * S]          # (bands, h*S, W*S), as stored
        nod = np.zeros(block.shape[1:], bool)
        for v in nodata:
            nod |= (block == v).all(0)
        if report:                                              # every 8th pixel of every 8th row, as stored
            dtype = block.dtype
            samples.append(block[:, ::8, ::8][:, ~nod[::8, ::8]])
            zeros += (block == 0).sum((1, 2))
            if np.issubdtype(block.dtype, np.integer):
                tops += (block == np.iinfo(block.dtype).max).sum((1, 2))
        if S > 1:                                               # mean of S x S pixels; no data only where all of them are
            block = block.reshape(C, h, S, W, S).mean((2, 4), dtype=np.float32)
            nod = nod.reshape(h, S, W, S).all((1, 3))
        for c0 in range(0, W, T):
            w = min(T, W - c0)
            raw = np.zeros((C, T, T), np.float32)
            raw[:, :h, :w] = block[:, :, c0:c0 + w]
            empty = np.ones((T, T), bool)
            empty[:h, :w] = nod[:, c0:c0 + w]
            if empty.all():
                continue
            x = (np.clip(raw * gain + offset, 0.0, st["clip"]) - mean) / std
            x[:, empty] = 0.0
            x = np.ascontiguousarray(x[None], dtype=np.float32)
            net.submit(x, (out, r0, c0, h, w, empty, x))
    net.wait()
    px = img.ref.pixel_m()
    stats = None
    if report:
        a = np.concatenate(samples, 1) if samples else np.zeros((C, 0))
        stats = []
        for k in range(C):
            q = np.percentile(a[k], [0, 1, 50, 99, 100]).tolist() if a.shape[1] else [None] * 5
            stats.append(dict(band=k + 1, dtype=str(dtype), sampled_px=int(a.shape[1]), min=q[0], p1=q[1], median=q[2],
                              p99=q[3], max=q[4], zero_px=int(zeros[k]), max_value_px=int(tops[k])))
    return out, (px * S if px else None), stats


def folder_mb(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]


def main():
    ap = argparse.ArgumentParser(description="Burn scars in HEO images (PyNAS network on the Myriad X)")
    ap.add_argument("--run_id", required=True, help="id of this run (up to 8 characters)")
    ap.add_argument("--device", type=int, default=1, choices=(0, 1), help="0: CPU, 1: Myriad X")
    ap.add_argument("--model", required=True, help="the model's .xml (its .bin next to it)")
    ap.add_argument("--img-dir", required=True, help="folder with the images (chips, L1C products or image stacks)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--settings", default=os.path.join(HERE, "model_input.json"))
    ap.add_argument("--min-burn-px", type=int, default=2000, help="burn pixels (fresh + old) for an image's map to be kept")
    ap.add_argument("--max-output-mb", type=float, default=90.0, help="no more maps once the output reaches this")
    ap.add_argument("--cpu-threads", type=int, default=1, help="inference threads when the network runs on the CPU")
    a = ap.parse_args()
    t_start = time.time()
    run_id = str(a.run_id)[:8]
    st = json.load(open(a.settings))
    classes = st["classes"]
    burn = [classes.index(c) for c in ("fresh", "old") if c in classes]
    lib = backend()
    work = os.path.join(a.output_dir, f"{run_id}_burnscar")
    os.makedirs(work, exist_ok=True)
    images, skipped, extra = find_images(a.img_dir, st, lib)
    log(f"package {st.get('package', '-')}: {len(images)} images in {a.img_dir} "
        f"({', '.join(sorted(set(l for _, l, _ in images))) or 'none'}), {len(skipped)} files or images skipped; "
        f"GeoTIFF library: {lib}")
    for k, v in skipped.items():
        log(f"skipped {k}: {v}")
    if st.get("maps_valid") is False:
        log(f"conversion to reflectance not set ({st.get('conversion', '')}): the maps are not meaningful")
    net = Network(a.model, a.device, a.cpu_threads)
    log(f"model {os.path.basename(a.model)} on {net.device}, input {net.shape}, {net.jobs} request(s) in flight, "
        f"OpenVINO {net.ov_version}")
    summary = dict(app="burnscar", version=VERSION, package=st.get("package"), run_id=run_id,
                   model=os.path.basename(a.model), device=net.device, cpu_threads=net.cpu_threads,
                   requests_in_flight=net.jobs, gdal_cachemax_mb=os.environ.get("GDAL_CACHEMAX"),
                   openvino=net.ov_version, python=sys.version.split()[0], numpy=np.__version__,
                   geotiff_library=lib, classes=classes, nodata_value=NODATA_OUT,
                   conversion=st.get("conversion"), maps_valid=st.get("maps_valid", True),
                   settings={k: st.get(k) for k in ("layouts", "chip_bands", "product_bands", "gain", "offset",
                                                    "nodata", "scale")},
                   min_burn_px=a.min_burn_px, images=[], skipped=skipped)
    if st.get("report"):
        summary["other_files"] = extra
    for iid, layout, parts in images:
        t0 = time.time()
        try:
            img = Image(iid, layout, parts, lib)
        except Exception as e:
            summary["skipped"][iid] = f"cannot be read ({str(e)[:160]})"
            log(f"skipped {iid}: {e}")
            continue
        try:
            cls, px_m, stats = classify(img, st, net)
        except Exception as e:
            img.close()
            summary["skipped"][iid] = f"failed ({str(e)[:160]})"
            log(f"failed {iid}: {e}")
            continue
        counts = np.array([int(np.count_nonzero(cls == k)) for k in range(len(classes))])   # no int64 copy of the map
        rec = dict(image=iid, layout=layout, files=len(img.files), height=int(cls.shape[0]), width=int(cls.shape[1]),
                   pixel_m=px_m, pixels_with_data=int(counts.sum()), **{f"{c}_px": int(n) for c, n in zip(classes, counts)})
        if px_m:
            rec.update({f"{classes[k]}_ha": round(float(counts[k]) * px_m * px_m / 1e4, 2) for k in burn})
        burn_px = int(sum(counts[k] for k in burn))
        if stats is not None:
            rec["band_stats"] = stats
            rec["metadata"] = img.ref.metadata()
        rec["map"] = None
        if burn_px >= a.min_burn_px:
            if folder_mb(work) < a.max_output_mb:
                name = f"{run_id}_{safe(iid)}_classes.tif"
                img.ref.write_like(os.path.join(work, name), cls, int(st.get("scale", 1)))
                rec["map"] = name
            else:
                rec["map_skipped"] = f"output already at {a.max_output_mb} MB"
        img.close()
        rec["seconds"] = round(time.time() - t0, 1)
        summary["images"].append(rec)
        log(f"{iid} ({layout}): {burn_px} burn px, {rec['seconds']} s" + (f", map {rec['map']}" if rec["map"] else ""))
    summary["seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(work, f"{run_id}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    zpath = os.path.join(a.output_dir, f"{run_id}_burnscar.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(work)):
            z.write(os.path.join(work, name), arcname=name)
    shutil.rmtree(work)
    log(f"done in {summary['seconds']} s: {zpath} ({os.path.getsize(zpath) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()


# iride_onboard-burnscar-mapper

On-board burnt area segmentation for IRIDE, inference at the edge on multispectral imagery, producing burn-scar maps without downlinking raw data.

This repository builds a labelled training dataset for that segmentation task: real Sentinel-2 imagery, resampled and degraded to match the PhiSat-2 sensor, paired with a 5-class per-pixel mask (clear, burnt area, cloud, cloud shadow, water).

## Pipeline overview

## 1. Fire selection — EFFIS

Burnt-area polygons come from the [EFFIS Rapid Damage Assessment](https://forest-fire.emergency.copernicus.eu/) database, exported as GeoJSON. Each polygon carries a fire date, country, and burnt area (ha); polygons are filtered by country, date range, and a minimum area (small fires are barely visible over a 20 km scene at 4.75 m GSD, so a floor of ~100 ha is used by default).

Country filtering uses EFFIS/EU ISO codes, notably **`EL` for Greece**, not `GR`.

## 2. Sentinel-2 acquisition

For each fire, a 20.14 km × 20.14 km UTM bounding box (the PhiSat-2 swath width) is built around the polygon centroid. The Copernicus Data Space Catalog API is queried for Sentinel-2 scenes in a **post-fire window** (default: 5–60 days after the fire date), so the burn scar has fully developed and is captured, not the active fire.

**Multiple candidate dates are considered, not just the least-cloudy one.** Up to 5 candidates are ranked by scene-wide cloud cover; for each, Sentinel-2 L1C bands are downloaded at native 10 m and a preliminary cloud/shadow mask is computed. A candidate is **accepted only if the burn scar itself is not obscured** (default: ≤15% of the scar under cloud/shadow) — clouds and shadow **elsewhere** in the 20 km frame are not penalized, since they're target classes in their own right, not noise to be avoided. If no candidate meets the threshold, the least-obscured one is used as a fallback.

Bands downloaded: `B02, B03, B04, B08, B05, B06, B07` (L1C reflectance) + `SCL` (L2A scene classification) + sun zenith angles + valid-data mask.

## 3. PhiSat-2 sensor simulation

Imagery is converted from Sentinel-2-like to PhiSat-2-like characteristics using the **official simulator** from [AI4EO/orbitalAI](https://github.com/AI4EO/orbitalAI/blob/main/phisat-2/phisat-2-simulator.ipynb), fetched automatically on first run (never vendored into this repo):

1. **Reflectance → radiance** conversion, using per-band solar irradiance and Earth–Sun distance (S2A constants + analytic distance are used in place of AWS-hosted per-product metadata — negligible difference at L1C output level, since the conversion largely cancels).
2. **Synthetic PAN band** generated as a weighted combination of the multispectral bands (used internally for the simulation, dropped before export — see below).
3. **Resampling** from Sentinel-2's 10 m grid to PhiSat-2's **4.75 m** grid.
4. **Band misalignment**, replicating the small per-band spatial jitter of the real PhiSat-2 push-broom sensor.
5. **Border crop**, removing edge pixels affected by resampling/misalignment.
6. **SNR + PSF (sensor noise + optical blur)**, applied via the official mission characterization binary when available on the host platform (`phisat2_unix.bin` for Linux); falls back to an approximate Gaussian-PSF + fixed-SNR model otherwise, since the real characterization is not published for every OS/architecture.
7. **Radiance → reflectance**, back to a directly usable TOA reflectance product.

### PhiSat-2 characteristics reproduced here

| Property | Value |
|---|---|
| Ground sample distance | 4.75 m |
| Swath width | ~20.14 km |
| Spectral bands (exported) | 7: Blue, Green, Red, Red Edge 1–3, NIR |
| Processing level | L1C (TOA reflectance) |
| Sensor noise / blur | Mission SNR + PSF characterization (or documented fallback) |

## 4. Label mask

| Value | Class | Source |
|------:|-------|--------|
| 0 | Clear | Everything else |
| 2 | Burnt area | EFFIS polygons, rasterised on the final 4.75 m grid; only fires with `fire_date ≤ acquisition date` (and, by default, ≤365 days before it) are burned in |
| 3 | Cloud | [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (thick + thin), run on native 10 m R/G/NIR |
| 4 | Cloud shadow | OmniCloudMask |
| 5 | Water | Sentinel-2 SCL class 6, OR'd with NDWI > 0.05 |
| 255 | No data | Outside the valid Sentinel-2 swath |

Class 1 is intentionally unused. Where classes overlap, priority is: clear → water → **burnt (overrides water** — fresh scars have low NIR and often trip NDWI false positives) → shadow → cloud → nodata.

## 5. Output format

Per fire: `output/acquisitions/<fire_id>/<timestamp>/`
- `<fire_id>_<timestamp>_phisat2.tif` — 7-band image
- `<fire_id>_<timestamp>_mask.tif` — 1-band label mask (colormap embedded)

**Band order** (matches Sentinel-2 acquisition/wavelength order, PAN excluded from export):

| Band # | Sentinel-2 band | Role |
|---|---|---|
| 1 | B02 | Blue |
| 2 | B03 | Green |
| 3 | B04 | Red |
| 4 | B05 | Red Edge 1 |
| 5 | B06 | Red Edge 2 |
| 6 | B07 | Red Edge 3 |
| 7 | B08 | NIR |

Pixel values are **reflectance × 10000** (`uint16`, nodata = 0) — divide by 10000 to recover 0–1 reflectance. Band names are embedded in each file (readable in QGIS/GDAL).

## 6. Dataset composition

Fires are prioritised by country and capped by a total data budget (GB), tracked per country and resumable across runs (already-processed fires, identified by EFFIS `fire_id`, are never re-downloaded). The default composition targets Italy as the primary region, with a smaller, evenly-split cross-region set — southern France, Spain, and Greece — held out for testing generalisation across the Alpine, Continental, and Mediterranean climate zones represented in Italy without training on Italian data itself.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a GPU build
pip install -r requirements.txt

# fetch the official simulator code + SNR/PSF binaries (not vendored here)
git clone --depth 1 https://github.com/AI4EO/orbitalAI.git
cp orbitalAI/phisat-2/phisat2_constants.py orbitalAI/phisat-2/phisat2_utils.py .
```

Credentials (Copernicus Data Space, free): create an OAuth client at
[dataspace.copernicus.eu](https://dataspace.copernicus.eu) → Sentinel Hub → User settings → OAuth clients, then set:

```bash
# .env (gitignored)
SH_CLIENT_ID=...
SH_CLIENT_SECRET=...
```

## Known limitations

- Cloud/shadow masks are computed at 10 m and resampled alongside the imagery; they are not independently validated against ground truth.
- EFFIS polygons are satellite-derived generalisations (MODIS/VIIRS-scale); edges are approximate relative to the 4.75 m output grid.
- "Southern France" is approximated by a latitude threshold, not an administrative or climate boundary.
- Without the official per-platform SNR/PSF binary, radiometric noise/blur is a documented approximation, not the mission characterization.
EOF
echo "README.md written"</parameter>
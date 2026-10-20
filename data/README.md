# Data

Everything the notebooks read and write is in this folder, roughly 60 GB. None of it is in git apart from this file.

The data are in a private Hugging Face dataset, shared for review (the HEO images belong to the IRIDE programme and cannot be redistributed): `python scripts/download_data.py` puts them here. Not uploaded, and written again by running the notebooks in order: the training arrays of the pool (pool_v2/train_*, 33 GB) and the whole-scene pools of the reviews (pool_v2/review/, 34 GB).

## Layout

    data/
    ├── effis/                 EFFIS burnt-area polygons
    │   └── older_exports/     earlier exports, not used anymore
    ├── heo/                   IRIDE HEO
    │   ├── products_zip/      products as we received them, one zip each
    │   ├── products/          the same products unzipped
    │   ├── images/            7-band stacks used for training
    │   ├── masks_v2/          previous labels, kept for comparison
    │   └── masks_v3/          current labels
    ├── phisat2_sim/           PhiSat-2 scenes simulated from Sentinel-2
    │   ├── images/
    │   ├── masks_v2/
    │   ├── masks_v3/
    │   ├── fires_index.gpkg
    │   └── processing_index.csv
    ├── s2_test/               Sentinel-2 chips kept for testing
    │   ├── images/
    │   ├── masks_v2/
    │   └── masks_v3/
    └── tables/                csv tables written by the notebooks

An image and its mask share the scene name: `<scene>_img.tif` (`<scene>_s2.tif` in s2_test) and `<scene>_mask.tif`.

## Images

| folder | scenes | bands | pixel | size (px) | stored as |
|---|---|---|---|---|---|
| heo/images | 25 | B, G, R, RE1, RE2, RE3, NIR | 2.5 m | about 5500 x 5500 | uint16 DN, 0 = no data |
| heo/products | 10 | B00 (PAN) to B07, one file per band | 2.5 m | about 5500 x 5500 | uint16 DN |
| phisat2_sim/images | 383 | B02 to B08 | 4.75 m | 4096 x 4096 | uint16, reflectance x 10000, 0 = no data |
| s2_test/images | 3 | all 13 Sentinel-2 bands, B01 to B12 | 10 m | 256 x 256 | uint16, reflectance x 10000, 0 = no data |

The model gets 7 bands in this order: blue, green, red, red edge 1, red edge 2, red edge 3, NIR. This is where they sit in each file (band numbers start at 1):

| | blue | green | red | RE1 | RE2 | RE3 | NIR |
|---|---|---|---|---|---|---|---|
| heo/images | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| heo/products (file name) | B01 | B02 | B03 | B04 | B05 | B06 | B07 |
| phisat2_sim/images | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| s2_test/images | 2 | 3 | 4 | 5 | 6 | 7 | 8 |

To get reflectance the notebooks divide HEO values by 4094, and PhiSat-2 and Sentinel-2 values by 10000.

### HEO

The spacecraft is the first part of the product name. For example `IMH06_1CST__OPT8_20260821T123102_20260821T123104_20260821T143603_06322______O_A02` is IMH06, processing level `1CST` (L1C, `2AST` is L2A), 8-band optical, then sensing start, sensing end and processing time. HEO scenes are in UTM 32N or 33N.

`images/` has the 25 scenes we started with (18 L1C and 7 L2A, July 2025 to August 2026), one stacked file each with PAN dropped. Their original products are not in `products/`.

`products/` has 10 L1C products from August and September 2026, as delivered. Each has `IMG_DATA/` (one GeoTIFF per band), `PREVIEW/` (quicklooks), a STAC `.json` and an `.xml`. The zips have the same layout at their top level, so `unzip -q X.zip -d products/X` gives back the folder. Four of these folders had a mask from the old labelling script inside; those masks are now in `masks_v2/`.

### PhiSat-2 simulated

One scene per EFFIS fire: Sentinel-2 L1C over the fire, taken after the fire date, run through the PhiSat-2 simulator (resampling to 4.75 m, band misalignment, sensor noise and blur). See `notebooks/01_phisat2_simulation.ipynb`. `fires_index.gpkg` and `processing_index.csv` come from that run and say which fire and which Sentinel-2 acquisition each scene belongs to.

### Sentinel-2 test chips

Three Sentinel-2 L1C chips with all 13 bands. They are kept apart and only used for testing.

## Masks

One band, uint8, on the same grid as the image.

| value | class | training id |
|---|---|---|
| 0 | clear | 0 |
| 2 | fresh burn | 1 |
| 3 | old burn | 2 |
| 4 | cloud | 3 |
| 5 | cloud shadow | 4 |
| 6 | water | 5 |
| 254 | ignore (v3 only) | left out |
| 255 | no data | left out |

1 is not used.

How v3 is made:

- Burns come from the EFFIS polygons. The age of a fire is the image date minus the fire date. 0 to 90 days is fresh burn, 90 to 365 days is old burn. Older fires, and fires after the image date, are not labelled: those pixels are clear, or water where NDWI > 0.05.
- Cloud, cloud shadow and no data always win over burns.
- Ignore marks pixels we don't trust: everything within 10 m of a burn polygon edge (on both sides), and burn pixels that are green (NDVI > 0.5) or bright (mean of blue, green and red reflectance > 0.3). They are left out of training and out of the scores.
- Clouds and shadows come from OmniCloudMask. Water is NDWI > 0.05; for the PhiSat-2 scenes it also includes water from the Sentinel-2 scene classification (SCL).

v2 has the same classes, but there old burn is any fire more than 90 days old, and there is no ignore class.

## EFFIS

Burnt-area polygons from the EFFIS rapid damage assessment. The files are renamed after what is in them; the original download names are in the table.

| file | original name | polygons | fire dates | countries |
|---|---|---|---|---|
| effis_IT-FR-ES-EL_2020-01-01_2026-08-03.zip | PHISAT_5fbf994ef5774ab882bf7239ec8b32f5.zip | 25,707 | 2020-01-01 to 2026-08-03 | IT, FR, ES, EL |
| effis_IT_2026-03-01_2026-10-05.json | 7e9341ee5a4441d3ba6a226eaf9cedf5.json | 2,262 | 2026-03-01 to 2026-10-05 | IT |
| older_exports/effis_2024-06-01_2026-07-29.zip | PHISAT_83757be9667e4cb4bb9be256a8770f9b.zip | 10,586 | 2024-06-01 to 2026-07-29 | |
| older_exports/effis_2025-01-31_2026-07-27.zip | HEO_2163ec78b94341b0a7ee4b5cfd37379f.zip | 8,860 | 2025-01-31 to 2026-07-27 | |

The labels use the first two together. When a fire is in both, the newer export wins, which gives 26,824 polygons.

## Which notebook writes what

| folder | written by |
|---|---|
| phisat2_sim/images, fires_index.gpkg, processing_index.csv | notebooks/01_phisat2_simulation, then an earlier labelling script |
| heo/images of the 25 earlier scenes, all masks_v2 | an earlier labelling script (not in this repository; what it wrote is part of the data) |
| masks_v3 of phisat2_sim, s2_test and the 25 earlier HEO scenes | notebooks/02_labels_v3_from_v2 |
| heo/images and heo/masks_v3 of the products in heo/products | notebooks/03_labels_v3_from_heo_products |
| tables/ | notebooks 02 and 03 |

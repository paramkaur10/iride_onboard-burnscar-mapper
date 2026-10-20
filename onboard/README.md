# On-board application

`burnscar_app.py` runs the exported network the way the satellite does. It needs only NumPy, the OpenVINO 2022.3.1
runtime and a GeoTIFF library (rasterio, GDAL or tifffile): no PyTorch, no PyNAS.

It finds every image in the folder it is given, and in its subfolders, in three layouts:

- chips with one GeoTIFF per band, `chip-<id>_band-<k>.tif`: band-1 to band-7 are blue, green, red, red edge 1-3 and
  NIR; band-0 (PAN) is not read;
- 7-band stacks, as in `data/heo/images/` (bands found by name; a stack of 8 bands without names is read with PAN first);
- L1C product folders whose band files `*.B01.tif` to `*.B07.tif` share one grid.

For each image it reads the seven bands side by side, one row of 256 x 256 tiles at a time, so memory does not grow with
the image; converts the stored values to reflectance and normalises them as in training; runs the network on every
tile with data; and writes a class map (GeoTIFF: 0 clear, 1 fresh burn, 2 old burn, 3 cloud, 4 cloud shadow, 5 water,
255 no data) for every image with at least 2,000 burn pixels, and a JSON summary of every image, zipped as
`<run_id>_burnscar.zip`.

`model_input_l1c.json` holds the settings for HEO L1C values (value / 4094 = reflectance, as in training), the
normalisation of training and the classes.

```bash
python onboard/burnscar_app.py --run_id test --device 0 \
    --model models/onboard/burnscar_pynas_r2f4/burnscar_pynas_r2f4.xml \
    --img-dir data/heo/images --output-dir out --settings onboard/model_input_l1c.json
```

`--device 1` runs the network on the Myriad X; if the MYRIAD plugin cannot be opened, the application runs on the CPU
and says so in the summary.

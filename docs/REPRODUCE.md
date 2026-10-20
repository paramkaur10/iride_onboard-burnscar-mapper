# Reproducing the results

## 1. Environment

Python 3.12 and a CUDA GPU; the results were produced on two NVIDIA A30 (24 GB).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # first the PyTorch build for your CUDA, see https://pytorch.org
pip install -e pyqnas
```

The Myriad X checks of the search (OpenVINO 2022.3.1 Model Optimizer and compile_tool, nn-Meter) need old packages
and run in environments of their own, made once per machine:

```bash
python -c "from pynas import config; from pynas.hardware import myriad; myriad.setup(config.load('configs/burnscar.ini'))"
```

## 2. Data and trained models

```bash
hf auth login                                   # or HF_TOKEN=<token> in the environment
python scripts/download_data.py                 # everything, about 68 GB
python scripts/download_data.py --only models   # or parts: heo phisat2 s2 effis tables pools models notebooks
```

The private dataset `neet1797/iride-heo-burnscar` has the layout of the repository, so every file lands where the notebooks look for
it; the HEO products are unzipped into `data/heo/products/`. The executed notebooks go to `notebooks_executed/`.

## 3. Notebooks

Run them from `notebooks/`, in order: each reads what the earlier ones wrote. The settings of the experiment are in
`configs/burnscar.ini`, with PyNAS's defaults in `pyqnas/src/pynas/config.ini`.

- **01** simulates the PhiSat-2 scenes; it needs Copernicus Data Space credentials in `.env` and the simulator of the
  OrbitalAI challenge (see [phisat2_simulation.md](phisat2_simulation.md)). Its outputs are in the dataset, so it can
  be skipped.
- **02 to 05** make the labels, the splits and the windows, and check the HEO scenes.
- **06 to 09, 12 and 13** train and score (GPU). The trained runs are in the dataset (`models/pynas/`), so the scores,
  tables and figures can be computed again without training.

## What is not in the dataset

Written again by running the notebooks in order:

- the training arrays of the pool, `data/pool_v2/train_*` (33 GB);
- the whole-scene pools of the reviews, `data/pool_v2/review/` (34 GB).

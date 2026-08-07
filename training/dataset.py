"""
dataset.py
==========
PyTorch Dataset wrapper for the IRIDE burnt-area segmentation tile dataset.

Tiles are stored as numpy .npy pairs on disk:
  <split>/<scene_id>_<gsd>_<y>_<x>_img.npy   — (7, 256, 256) uint16 raw DN
  <split>/<scene_id>_<gsd>_<y>_<x>_mask.npy  — (256, 256)    uint8  raw mask

The Dataset reads each pair, applies preprocessing, and returns
  img  : torch.Tensor (7, 256, 256) float32, normalised to [0, 1]
  mask : torch.Tensor (256, 256)    int64,   train IDs in {-1, 0..5}

The 'scale' for each tile (10000.0 or 4094.0) is recorded in tiles_index.csv
alongside the tile filename — TileDataset looks it up from the DataFrame row
rather than hard-coding a global constant, so HEO and PhiSat-2 tiles are
handled correctly without any per-source logic at training time.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing import (
    to_reflectance,
    remap_mask_to_train_ids,
    train_augment,
    eval_transform,
    MASK_NODATA,
)


class TileDataset(Dataset):
    """
    Reads (img, mask) .npy tile pairs for a subset of tiles_index.csv rows.

    Parameters
    ----------
    tiles_root : str
        Path to the processed dataset root (e.g. "processed/dataset_v1").
        Tile files live at tiles_root/<split>/<tile_name>_img.npy.
    index_df : pd.DataFrame
        Rows from tiles_index.csv that belong to this split. Must contain
        columns: tile, split, scale.
    is_train : bool
        If True, applies full stochastic augmentation (train_augment).
        If False, applies eval_transform (normalisation only).
    aug_cfg : dict, optional
        Override kwargs forwarded to train_augment / eval_transform.
        See preprocessing.py for available keys.
    """

    def __init__(
        self,
        tiles_root: str,
        index_df: pd.DataFrame,
        is_train: bool,
        aug_cfg: dict = None,
    ):
        self.root     = tiles_root
        self.df       = index_df.reset_index(drop=True)
        self.is_train = is_train
        self.aug_cfg  = aug_cfg or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]

        # ── 1. Load raw tile pair ─────────────────────────────────────────
        base     = os.path.join(self.root, row["split"], row["tile"])
        img_dn   = np.load(base + "_img.npy")    # (7, 256, 256) uint16
        mask_raw = np.load(base + "_mask.npy")   # (256, 256)    uint8

        # ── 2. Radiometric conversion: DN → reflectance ───────────────────
        scale    = float(row["scale"])           # 10000.0 or 4094.0
        img_refl = to_reflectance(img_dn, scale) # (7, 256, 256) float32

        # ── 3. Mask remapping: raw values → contiguous train IDs ──────────
        valid_mask   = mask_raw != MASK_NODATA   # (256, 256) bool
        mask_train   = remap_mask_to_train_ids(mask_raw)  # (256, 256) int64

        # ── 4. Augmentation / normalisation ──────────────────────────────
        rng = np.random.default_rng()   # fresh generator per tile (thread-safe)
        if self.is_train:
            img, mask = train_augment(img_refl, mask_train, valid_mask, rng, self.aug_cfg)
        else:
            img, mask = eval_transform(img_refl, mask_train, valid_mask, self.aug_cfg)

        # ── 5. Convert to tensors ─────────────────────────────────────────
        return (
            torch.from_numpy(img.copy()),    # (7, 256, 256) float32
            torch.from_numpy(mask.copy()),   # (256, 256)    int64
        )
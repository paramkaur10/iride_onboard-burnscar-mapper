# training/datamodule.py
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler
from dataset import TileDataset

N_BANDS, TILE_SIZE, N_CLASSES = 7, 256, 6

class BalancedTileDataModule(pl.LightningDataModule):
    def __init__(self, tiles_root, index_csv, batch_size=16, num_workers=4,
                 nas_subset_n=None, seed=42):
        super().__init__()
        self.root, self.index_csv = tiles_root, index_csv
        self.batch_size, self.num_workers = batch_size, num_workers
        self.nas_n, self.seed = nas_subset_n, seed
        self.input_shape = (N_BANDS, TILE_SIZE, TILE_SIZE)
        self.num_classes = N_CLASSES

    def setup(self, stage=None):
        idx = pd.read_csv(self.index_csv)
        train = idx[(idx.split=="train") & (idx.gsd=="native")].reset_index(drop=True)
        val   = idx[(idx.split=="val")   & (idx.gsd=="native")].reset_index(drop=True)
        test  = idx[(idx.split=="test")  & (idx.gsd=="native")].reset_index(drop=True)

        if self.nas_n is not None:
            rng = np.random.default_rng(self.seed)
            inf_idx = train[train.informative].index.tolist()
            clr_idx = train[~train.informative].index.tolist()
            n_inf = min(len(inf_idx), int(self.nas_n * 0.8))
            n_clr = min(len(clr_idx), self.nas_n - n_inf)
            chosen = (rng.choice(inf_idx, n_inf, replace=False).tolist() +
                     rng.choice(clr_idx, n_clr, replace=False).tolist())
            train = train.loc[chosen].reset_index(drop=True)

        self.train_ds = TileDataset(self.root, train, is_train=True)
        self.val_ds   = TileDataset(self.root, val,   is_train=False)
        self.test_ds  = TileDataset(self.root, test,  is_train=False)

        rare = (train.get("frac_2",0)+train.get("frac_3",0)+
                train.get("frac_4",0)+train.get("frac_5",0)+train.get("frac_6",0)).values
        w = np.clip(np.maximum(rare, 0.02), 0, 0.3).astype(np.float64)
        w /= w.sum()
        self._sampler = WeightedRandomSampler(torch.from_numpy(w).float(),
                                              len(self.train_ds), replacement=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, sampler=self._sampler,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(self.num_workers > 0))   # ← avoids worker respawn every epoch

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0))
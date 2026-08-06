import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None, num_classes=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        store_path = f"{root_dir}/{split}"
        self.zarr_root = zarr.open(store_path, mode="r")
        self.sample_ids = sorted(self.zarr_root.keys())

        if len(self.sample_ids) == 0:
            raise RuntimeError(f"No samples found in Zarr store at {store_path}")

        if num_classes is not None:
            self.num_classes = num_classes
        else:
            # One-hot mask: number of classes = number of channels
            probe_label = self.zarr_root[self.sample_ids[0]]["label"][:]
            self.num_classes = probe_label.shape[0]

        self.classes = [f"class_{i}" for i in range(self.num_classes)]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sample = self.zarr_root[sid]

        image = sample["img"][:].astype(np.float32)
        label = sample["label"][:]  # one-hot, shape (4, H, W)

        # Collapse one-hot -> single class-index mask: (C, H, W) -> (H, W)
        label = np.argmax(label, axis=0).astype(np.int64)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None,
                 val_split=0.3, num_classes=None):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.val_split = val_split
        self.num_classes_override = num_classes

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage='fit'):
        full_dataset = SegmentationDataset(
            root_dir=self.root_dir,
            split='trainval',
            transform=self.transform,
            num_classes=self.num_classes_override,
        )

        self.class_names = full_dataset.classes
        self.num_classes = full_dataset.num_classes

        sample, _ = full_dataset[0]
        self.input_shape = tuple(sample.shape)

        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test':
            self.test_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='test',
                transform=self.transform,
                num_classes=self.num_classes,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
# -*- coding: utf-8 -*-
"""
DataModule для ETA: загрузка данных и разбиение train/val для Lightning.
"""
import torch
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from src.config import PRE_DATASETS_DIR, SEED, WINDOW_SIZE_SEC
from src.models.dataset import ETAWindowDataset


class ETADataModule(pl.LightningDataModule):
    """Датасет ETA с автоматическим train/val split (фиксированный seed для воспроизводимости)."""

    def __init__(
        self,
        preprocessed_dir: str = PRE_DATASETS_DIR,
        batch_size: int = 32,
        val_frac: float = 0.2,
        num_workers: int = 0,
        seed: int = SEED,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessed_dir = preprocessed_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.num_workers = num_workers
        self.seed = seed
        self.full_dataset = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str = None):
        self.full_dataset = ETAWindowDataset(
            preprocessed_dir=self.preprocessed_dir,
            window_size=WINDOW_SIZE_SEC,
        )
        if len(self.full_dataset) == 0:
            raise ValueError(
                "No samples in preprocessed data. Run: python run_preprocess.py"
            )
        n_val = int(len(self.full_dataset) * self.val_frac)
        n_train = len(self.full_dataset) - n_val
        gen = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(
            self.full_dataset, [n_train, n_val], generator=gen
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from src.dataset import DropletsClassificationDataset
from src.logger import get_logger

_logger = get_logger(__name__)


class DropletsClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_path: Path,
        img_size: Tuple[int, int],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_path = dataset_path
        self.img_size = (img_size,)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def dataframe(self):
        return pd.read_feather(self.dataset_path)

    @property
    def classes(self):
        return self.dataframe["name"].unique()

    @property
    def num_classes(self):
        return self.classes

    def transform_image(self, means: List[float], stds: List[float]) -> Callable[[Tensor], Dict[str, Tensor]]:
        norm = Normalize(means, stds)

        def _transform(sample: Tensor) -> Dict[str, Tensor]:
            sample = sample.float()
            sample = norm(sample)

            return sample

        return _transform

    def setup(self, stage: Optional[str] = None) -> None:
        _logger.info("Setting up the datasets. This might take a minute...")

        df = self.dataframe()

        _logger.info("Setting up the training datasets. This might take a minute...")

        train_df = df[df["train"]].sort_values(by="filepath").reset_index(drop=True)

        train_images = train_df["filepath"].apply(lambda x: self.data_dir / x).tolist()
        train_labels = train_df["name"].apply(lambda x: self.data_dir / x).tolist()

        means = df["mean"].mean().tolist()
        stds = df["std"].mean().tolist()

        self.train_dataset = DropletsClassificationDataset(
            inputs=train_images,
            targets=train_labels,
            transforms=self.transform_image(means, stds),
        )

        _logger.info("Setting up the validation datasets. This might take a minute...")
        val_df = df[df["val"]].sort_values(by="image").reset_index(drop=True)

        val_images = val_df["filepath"].apply(lambda x: self.data_dir / x).tolist()
        val_labels = val_df["name"].apply(lambda x: self.data_dir / x).tolist()

        self.val_dataset = DropletsClassificationDataset(
            inputs=val_images,
            targets=val_labels,
            transforms=self.transform_image(means, stds),
        )

        _logger.info("Setting up the test datasets. This might take a minute...")
        test_df = df[df["test"]].sort_values(by="image").reset_index(drop=True)
        self.test_weights = test_df["weight"]

        test_images = test_df["filepath"].apply(lambda x: self.data_dir / x).tolist()
        test_labels = test_df["name"].apply(lambda x: self.data_dir / x).tolist()

        self.test_dataset = DropletsClassificationDataset(
            inputs=test_images,
            targets=test_labels,
            transforms=self.transform_image(means, stds),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        _logger.info("Creating a training DataLoader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        _logger.info("Creating a test DataLoader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        _logger.info("Creating a validation DataLoader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

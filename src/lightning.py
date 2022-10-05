from __future__ import annotations

import glob
import os
from typing import Any

import albumentations as A
import albumentations.pytorch as AP
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from timm.optim import create_optimizer_v2
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from data import ImageDataset


class MyLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = timm.create_model(**config.model)
        self.model.set_grad_checkpointing(config.train.gradient_checkpointing)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits))
        accuracy = ((logits > 0).long() == labels).float().mean()
        return loss, accuracy

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, accuracy = self(*batch)
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)
        self.log("step", self.global_step)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, accuracy = self(*batch)
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)
        self.log("step", self.global_step)

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = create_optimizer_v2(self, **self.config.optim)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.epochs,
            eta_min=self.config.optim.lr * 0.01,
        )
        return [optimizer], [scheduler]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class MyLightningDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str | None = None):
        train_transform = [
            A.Resize(self.config.data.image_size, self.config.data.image_size),
            A.RandomRotate90(p=1.0),
            A.ShiftScaleRotate(rotate_limit=15, p=0.5),
            A.Cutout(
                num_holes=1,
                max_h_size=int(self.config.data.image_size * 0.3),
                max_w_size=int(self.config.data.image_size * 0.3),
                fill_value=0,
                p=0.5,
            ),
            A.Normalize(mean=0.5, std=0.5),
            AP.ToTensorV2(),
        ]
        val_transform = [
            A.Resize(self.config.data.image_size, self.config.data.image_size),
            A.Normalize(mean=0.5, std=0.5),
            AP.ToTensorV2(),
        ]

        self.train_dataset = ImageDataset(
            filenames=glob.glob(self.config.data.train.filenames),
            labels=pd.read_csv(self.config.data.train.labels, index_col="index"),
            transform=A.Compose(train_transform),
        )
        self.val_dataset = ImageDataset(
            filenames=glob.glob(self.config.data.validation.filenames),
            labels=pd.read_csv(self.config.data.validation.labels, index_col="index"),
            transform=A.Compose(val_transform),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )

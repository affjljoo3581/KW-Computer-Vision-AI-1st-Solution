from __future__ import annotations

import argparse
import os
import random

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import MyLightningDataModule, MyLightningModule


def main(
    name: str,
    config: DictConfig,
    resume_from: str | None = None,
    resume_id: str | None = None,
):
    name = f"{name}-{''.join(random.choices('0123456789abcdef', k=6))}"
    checkpoint = ModelCheckpoint(
        monitor="val/accuracy", mode="max", save_weights_only=True
    )
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_epochs=config.train.epochs,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        callbacks=[checkpoint, LearningRateMonitor("epoch")],
        logger=WandbLogger(project="kw-computer-vision-ai", name=name, id=resume_id),
    )
    trainer.fit(
        MyLightningModule(config),
        MyLightningDataModule(config),
        ckpt_path=resume_from,
    )

    module = MyLightningModule.load_from_checkpoint(
        checkpoint.best_model_path, config=config
    )
    torch.save(module.model, f"{name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    name = os.path.splitext(os.path.basename(args.config))[0]
    main(name, config, args.resume_from, args.resume_id)

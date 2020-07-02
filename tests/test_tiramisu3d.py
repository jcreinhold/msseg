#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.tests.tiramisu3d

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 02, 2020
"""

__all__ = []

from typing import *

import contextlib
import os
from os.path import join
import shutil
import tempfile
import unittest
import warnings

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

with open(os.devnull, "w") as f:
    with contextlib.redirect_stdout(f):
        import torchio
from torchio.transforms import (
    Compose,
    OneOf,
    RandomAffine,
    RandomElasticDeformation
)

import msseg
from msseg.config import ExperimentConfig
from msseg.model import Tiramisu3d
from msseg.util import n_dirname

seed_everything(1337)

msseg_dir = n_dirname(msseg.__file__, 2)
DATA_DIR = join(msseg_dir, "tests/test_data/")

default_exp_config = ExperimentConfig(
    data_params=dict(
        batch_size = 4,
        num_workers = 0,
        patch_size = (16, 16, 16),
        queue_length = 4,
        samples_per_volume = 4
    ),
    network_params=dict(
        in_channels = 1,
        out_channels = 1,
        down_blocks = (2, 2),
        up_blocks = (2, 2),
        bottleneck_layers = 2,
        growth_rate = 16,
        out_chans_first_conv = 48,
        dropout_rate = 0.2
    ),
    optim_params=dict(
        lr = 1e3,
        betas = (0.9, 0.99),
        weight_decay = 1e-7,
    )
)


class LightningTiramisu3d(pl.LightningModule):

    def __init__(self, config:ExperimentConfig=default_exp_config):
        super().__init__()
        self.config = config
        self.net = Tiramisu3d(**config.network_params)

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch['t1'][torchio.DATA]
        y = batch['label'][torchio.DATA]
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x = batch['t1'][torchio.DATA]
        y = batch['label'][torchio.DATA]
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.config.optim_params)

    def train_dataloader(self):
        subject_a = torchio.Subject(
            t1=torchio.Image(join(DATA_DIR, "img.nii.gz"), type=torchio.INTENSITY),
            label=torchio.Image(join(DATA_DIR, "mask.nii.gz"), type=torchio.LABEL)
        )
        subjects_list = [subject_a]

        spatial = OneOf(
            {RandomAffine(): 0.8, RandomElasticDeformation(): 0.2},
            p=0.75,
        )
        transforms = [spatial]
        transform = Compose(transforms)

        subjects_dataset = torchio.ImagesDataset(subjects_list, transform=transform)

        sampler = torchio.data.UniformSampler(self.config.data_params['patch_size'])
        patches_queue = torchio.Queue(
            subjects_dataset,
            self.config.data_params['queue_length'],
            self.config.data_params['samples_per_volume'],
            sampler,
            num_workers=self.config.data_params['num_workers'],
            shuffle_subjects=True,
            shuffle_patches=True)
        train_dataloader = DataLoader(
            patches_queue,
            batch_size=self.config.data_params['batch_size'])
        return train_dataloader

    def val_dataloader(self):
        subject_a = torchio.Subject(
            t1=torchio.Image(join(DATA_DIR, "img.nii.gz"), type=torchio.INTENSITY),
            label=torchio.Image(join(DATA_DIR, "mask.nii.gz"), type=torchio.LABEL)
        )
        subjects_list = [subject_a]

        subjects_dataset = torchio.ImagesDataset(subjects_list)
        sampler = torchio.data.UniformSampler(self.config.data_params['patch_size'])
        patches_queue = torchio.Queue(
            subjects_dataset,
            self.config.data_params['queue_length'],
            self.config.data_params['samples_per_volume'],
            sampler,
            num_workers=self.config.data_params['num_workers'],
            shuffle_subjects=False,
            shuffle_patches=False)
        val_dataloader = DataLoader(
            patches_queue,
            batch_size=self.config.data_params['batch_size'])
        return val_dataloader


class TestTiramisu3d(unittest.TestCase):
    def setUp(self):
        self.net = LightningTiramisu3d()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.out_dir)
        del self.net

    def test_fit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(
                default_root_dir=self.out_dir,
                fast_dev_run=True,
                progress_bar_refresh_rate=0)
            trainer.fit(self.net)


if __name__ == "__main__":
    unittest.main()

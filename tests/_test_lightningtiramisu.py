#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.tests._test_lightningtiramisu

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 03, 2020
"""

__all__ = ['LightningTiramisuTester']

import contextlib
import os
from os.path import join

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

with open(os.devnull, "w") as f:
    with contextlib.redirect_stdout(f):
        import torchio
from torchio.transforms import (
    Compose,
    OneOf,
    RandomAffine,
    RandomElasticDeformation
)

from msseg.config import ExperimentConfig
from msseg.experiment.lightningtiramisu import LightningTiramisu


class LightningTiramisuTester(LightningTiramisu):

    def __init__(self, config:ExperimentConfig, data_dir:str):
        super().__init__(config)
        self.criterion = F.binary_cross_entropy_with_logits
        self.data_dir = data_dir

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.config.optim_params)

    def train_dataloader(self):
        subject_a = torchio.Subject(
            t1=torchio.Image(join(self.data_dir, "img.nii.gz"), type=torchio.INTENSITY),
            label=torchio.Image(join(self.data_dir, "mask.nii.gz"), type=torchio.LABEL)
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
            t1=torchio.Image(join(self.data_dir, "img.nii.gz"), type=torchio.INTENSITY),
            label=torchio.Image(join(self.data_dir, "mask.nii.gz"), type=torchio.LABEL)
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


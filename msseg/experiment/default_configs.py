#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.experiment.default_configs

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 03, 2020
"""

__all__ = ['default_lightningtiramisu2d_config',
           'default_lightningtiramisu3d_config']

from msseg.config import ExperimentConfig


default_lightningtiramisu2d_config = ExperimentConfig(
    data_params=dict(
        batch_size = 16,
        num_workers = 8,
        patch_size = (3, 128, 128),
        queue_length = 100,
        samples_per_volume = 10
    ),
    lightning_params=dict(
        network_dim=2
    ),
    network_params=dict(
        in_channels = 3,
        out_channels = 1,
        down_blocks = (5, 5, 5, 5, 5),
        up_blocks = (5, 5, 5, 5, 5),
        bottleneck_layers = 5,
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


default_lightningtiramisu3d_config = ExperimentConfig(
    data_params=dict(
        batch_size = 8,
        num_workers = 8,
        patch_size = 64,
        queue_length = 100,
        samples_per_volume = 10
    ),
    lightning_params=dict(
        network_dim=3
    ),
    network_params=dict(
        in_channels = 1,
        out_channels = 1,
        down_blocks = (4, 4, 4, 4, 4),
        up_blocks = (4, 4, 4, 4, 4),
        bottleneck_layers = 4,
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


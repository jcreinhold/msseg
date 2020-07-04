#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.experiment.lightningtiramisu

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 03, 2020
"""

__all__ = ['LightningTiramisu']

from torch import Tensor

import pytorch_lightning as pl

from msseg.config import ExperimentConfig
from msseg.model import Tiramisu2d, Tiramisu3d
from msseg.util import init_weights


class LightningTiramisu(pl.LightningModule):

    def __init__(self, config:ExperimentConfig):
        super().__init__()
        self.config = config
        self.network_dim = config.lightning_params["network_dim"]
        if self._use_2d_network:
            self.net = Tiramisu2d(**config.network_params)
        elif self._use_3d_network:
            self.net = Tiramisu3d(**config.network_params)
        else:
            raise self._invalid_network_dim
        init_weights(self.net, **config.lightning_params["init_params"])

    @property
    def _use_2d_network(self):
        return self.network_dim == 2

    @property
    def _use_3d_network(self):
        return self.network_dim == 3

    @property
    def _invalid_network_dim(self):
        err_msg = f"Network dim. {self.network_dim} invalid."
        return ValueError(err_msg)

    @staticmethod
    def criterion(x:Tensor, y:Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

    def _2d_training_step(self, batch, batch_idx):
        raise NotImplementedError

    def _3d_training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, *args):
        if self._use_2d_network():
            out_dict = self._2d_training_step(*args)
        elif self._use_3d_network():
            out_dict = self._3d_training_step(*args)
        else:
            raise self._invalid_network_dim
        return out_dict

    def validation_step(self, *args):
        if self._use_2d_network():
            out_dict = self._2d_training_step(*args)
        elif self._use_3d_network():
            out_dict = self._3d_training_step(*args)
        else:
            raise self._invalid_network_dim
        return out_dict

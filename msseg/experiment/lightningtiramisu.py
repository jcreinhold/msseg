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

from pytorch_lightning.utilities.parsing import AttributeDict
from msseg.model import Tiramisu2d, Tiramisu3d
from msseg.util import init_weights


class LightningTiramisu(pl.LightningModule):

    def __init__(self, hparams:AttributeDict):
        super().__init__()
        self.hparams = hparams
        self._hparams_to_attributedict()
        self.network_dim = self.hparams.lightning_params["network_dim"]
        if self._use_2d_network:
            self.net = Tiramisu2d(**self.hparams.network_params)
        elif self._use_3d_network:
            self.net = Tiramisu3d(**self.hparams.network_params)
        else:
            raise self._invalid_network_dim
        init_weights(self.net, **self.hparams.lightning_params["init_params"])

    def _hparams_to_attributedict(self):
        if not isinstance(self.hparams, AttributeDict):
            self.hparams = AttributeDict(self.hparams)

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

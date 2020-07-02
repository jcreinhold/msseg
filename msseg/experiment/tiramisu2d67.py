#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.tiramisu2d67.py

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 02, 2020
"""

__all__ = []

from typing import *

from torch import Tensor
from torch.optim import Optimizer, AdamW

import pytorch_lightning as pl
import torchio

from msseg.config import ExperimentConfig
from msseg.model import Tiramisu2d


class Tiramisu2d67(pl.LightningModule):

    def __init__(self, config:ExperimentConfig):
        super().__init__()
        self.config = config
        self.net = Tiramisu2d(**config.network_params)

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x, y = batch
        return

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x, y = batch
        return

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), **self.config.optim_params)


if __name__ == "__main__":
    pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.tests.tiramisu3d

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 02, 2020
"""

import contextlib
import os
from os.path import join
import shutil
import tempfile
import unittest
import warnings

from pytorch_lightning import Trainer, seed_everything

with open(os.devnull, "w") as f:
    with contextlib.redirect_stdout(f):
        import torchio

import msseg
from msseg.loss import binary_combo_loss
from msseg.util import n_dirname

from _test_configs import test_lightningtiramisu3d_config
from _test_lightningtiramisu import LightningTiramisuTester

seed_everything(1337)

msseg_dir = n_dirname(msseg.__file__, 2)
DATA_DIR = join(msseg_dir, "tests/test_data/")


class LightningTiramisu3d(LightningTiramisuTester):

    def training_step(self, batch, batch_idx):
        x = batch['t1'][torchio.DATA]
        y = batch['label'][torchio.DATA]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x = batch['t1'][torchio.DATA]
        y = batch['label'][torchio.DATA]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {'val_loss': loss}


class TestTiramisu3d(unittest.TestCase):
    def setUp(self):
        self.net = LightningTiramisu3d(test_lightningtiramisu3d_config, DATA_DIR)
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

    def test_combo_loss(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.net.criterion = binary_combo_loss
            trainer = Trainer(
                default_root_dir=self.out_dir,
                fast_dev_run=True,
                progress_bar_refresh_rate=0)
            trainer.fit(self.net)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.config

configuration file setup for experiments

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 02, 2020
"""

__all__ = ['ExperimentConfig']

from typing import *

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    data_params: Dict
    optim_params: Dict
    network_params: Dict


if __name__ == "__main__":
    ExperimentConfig(optim_params={"lr": 0.1}, network_params={"dropout_rate": 0.2})

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.loss

loss functions to support lesion segmentation

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = []

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

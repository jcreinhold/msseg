#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.model.attention

grid attention blocks for gated attention networks

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 12, 2020
"""

__all__ = ['GridAttentionBlock2d',
           'GridAttentionBlock3d']

from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class GridAttentionBlock(nn.Module):

    _conv        = None
    _norm        = None
    _upsample    = None

    def __init__(self, in_channels:int, gating_channels:int, inter_channels:int):
        super().__init__()

        self.W = nn.Sequential(
            self._conv(in_channels, in_channels, 1),
            self._norm(in_channels)
        )

        self.theta = self._conv(in_channels, inter_channels, 2, stride=2, bias=False)
        self.phi = self._conv(gating_channels, inter_channels, 1)
        self.psi = self._conv(inter_channels, 1, 1)

    def _interp(self, x:Tensor, size:List[int]) -> Tensor:
        return F.interpolate(x, size=size, mode=self._upsample, align_corners=True)

    def forward(self, x:Tensor, g:Tensor) -> Tensor:
        input_size = x.shape[2:]

        theta_x = self.theta(x)
        theta_x_size = theta_x.shape[2:]

        phi_g = self.phi(g)
        phi_g = self._interp(phi_g, theta_x_size)
        theta_phi_sum = theta_x + phi_g
        f = F.relu(theta_phi_sum, inplace=True)

        psi_f = self.psi(f)
        psi_f = torch.sigmoid(psi_f)
        psi_f = self._interp(psi_f, input_size)

        y = psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class GridAttentionBlock3d(GridAttentionBlock):
    _conv = nn.Conv3d
    _norm = nn.BatchNorm3d
    _upsample = "trilinear"


class GridAttentionBlock2d(GridAttentionBlock):
    _conv = nn.Conv2d
    _norm = nn.BatchNorm2d
    _upsample = "bilinear"


if __name__ == "__main__":
    attention_block = GridAttentionBlock3d(1,1,1)
    x = torch.randn(2,1,32,32,32)
    g = torch.randn(2,1,16,16,16)
    y = attention_block(x, g)
    assert x.shape == y.shape

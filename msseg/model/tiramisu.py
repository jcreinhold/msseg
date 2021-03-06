#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.model.tiramisu

PyTorch implementation of the Tiramisu network architecture [1]
(2D) Implementation based on [2].

Changes from [2] include:
  1) removal of bias from conv layers,
  2) change zero padding to replication padding,
  3) cosmetic changes for brevity, clarity, consistency

References:
  [1] Jégou, Simon, et al. "The one hundred layers tiramisu:
      Fully convolutional densenets for semantic segmentation."
      CVPR. 2017.
  [2] https://github.com/bfortuner/pytorch_tiramisu

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['Tiramisu2d',
           'Tiramisu3d']

from typing import *

import torch
from torch import Tensor
from torch import nn

from msseg.model.dense import *


class Tiramisu(nn.Module):

    _bottleneck = None
    _conv       = None
    _denseblock = None
    _pad        = None
    _trans_down = None
    _trans_up   = None

    def __init__(self,
                 in_channels:int=3,
                 out_channels:int=1,
                 down_blocks:List[int]=(5,5,5,5,5),
                 up_blocks:List[int]=(5,5,5,5,5),
                 bottleneck_layers:int=5,
                 growth_rate:int=16,
                 out_chans_first_conv:int=48,
                 dropout_rate:float=0.2,
                 p_shakedrop:float=0.):
        super().__init__()
        assert len(down_blocks) == len(up_blocks)
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        first_kernel_size = 3
        final_kernel_size = 1
        skip_connection_channel_counts = []
        n_units = len(down_blocks)
        linear_decay_rule = lambda i: 1 - (1.0 - (p_shakedrop / n_units) * (i + 1))
        if p_shakedrop > 0.:
            ps_shakedrop = [linear_decay_rule(i) for i in range(n_units)]
        else:
            ps_shakedrop = [0. for _ in range(n_units)]

        self.firstConv = nn.Sequential(
            self._pad(first_kernel_size // 2),
            self._conv(in_channels, out_chans_first_conv,
                       first_kernel_size, bias=False))
        cur_channels_count = out_chans_first_conv

        ## Downsampling path ##
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for psd, n_layers in zip(ps_shakedrop, down_blocks):
            self.denseBlocksDown.append(self._denseblock(
                cur_channels_count, growth_rate, n_layers,
                upsample=False, dropout_rate=dropout_rate,
                p_shakedrop=psd))
            cur_channels_count += (growth_rate*n_layers)
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(self._trans_down(
                cur_channels_count, cur_channels_count,
                dropout_rate=dropout_rate))

        self.bottleneck = self._bottleneck(
            cur_channels_count, growth_rate, bottleneck_layers,
            dropout_rate=dropout_rate, p_shakedrop=p_shakedrop)
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        ## Upsampling path ##
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        up_info = zip(up_blocks,
                      skip_connection_channel_counts,
                      reversed(ps_shakedrop))

        for i, (n_layers, sccc, psd) in enumerate(up_info, 1):
            self.transUpBlocks.append(self._trans_up(
                prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + sccc
            upsample = i < len(up_blocks)  # do not upsample on last block
            self.denseBlocksUp.append(self._denseblock(
                cur_channels_count, growth_rate, n_layers,
                upsample=upsample, dropout_rate=dropout_rate,
                p_shakedrop=psd))
            prev_block_channels = growth_rate*n_layers
            cur_channels_count += prev_block_channels

        self.finalConv = self._conv(cur_channels_count, out_channels,
                                    final_kernel_size, bias=True)

    def forward(self, x:Tensor) -> Tensor:
        out = self.firstConv(x)
        skip_connections = []
        for dbd, tdb in zip(self.denseBlocksDown, self.transDownBlocks):
            out = dbd(out)
            skip_connections.append(out)
            out = tdb(out)
        out = self.bottleneck(out)
        for ubd, tub in zip(self.denseBlocksUp, self.transUpBlocks):
            skip = skip_connections.pop()
            out = tub(out, skip)
            out = ubd(out)
        out = self.finalConv(out)
        return out


class Tiramisu2d(Tiramisu):
    _bottleneck = Bottleneck2d
    _conv       = nn.Conv2d
    _denseblock = DenseBlock2d
    _pad        = nn.ReplicationPad2d
    _trans_down = TransitionDown2d
    _trans_up   = TransitionUp2d


class Tiramisu3d(Tiramisu):
    _bottleneck = Bottleneck3d
    _conv       = nn.Conv3d
    _denseblock = DenseBlock3d
    _pad        = nn.ReplicationPad3d
    _trans_down = TransitionDown3d
    _trans_up   = TransitionUp3d


if __name__ == "__main__":
    net_kwargs = dict(in_channels=1, out_channels=1,
                      down_blocks=[2,2], up_blocks=[2,2],
                      bottleneck_layers=2, p_shakedrop=0.5)
    x = torch.randn(1,1,32,32)
    net2d = Tiramisu2d(**net_kwargs)
    y = net2d(x)
    assert x.shape == y.shape
    x = torch.randn(1,1,32,32,32)
    net3d = Tiramisu3d(**net_kwargs)
    y = net3d(x)
    assert x.shape == y.shape

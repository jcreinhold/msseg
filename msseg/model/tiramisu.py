#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.tiramisu

PyTorch implementation of the Tiramisu network architecture [1]
Implementation based on [2].

References:
  [1] Jégou, Simon, et al. "The one hundred layers tiramisu:
      Fully convolutional densenets for semantic segmentation."
      CVPR. 2017.
  [2] https://github.com/bfortuner/pytorch_tiramisu

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['Tiramisu']

from typing import *

import torch
from torch import Tensor
from torch import nn

import pytorch_lightning as pl
import torchio

ACTIVATION = nn.GELU


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels:int, growth_rate:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('act', ACTIVATION())
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(dropout_rate))


class DenseBlock(nn.Module):
    def __init__(self, in_channels:int, growth_rate:int, n_layers:int,
                 upsample:bool=False, dropout_rate:float=0.2):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate, dropout_rate)
            for i in range(n_layers)])

    def forward(self, x:Tensor):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('act', ACTIVATION())
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(dropout_rate))
        self.add_module('maxpool', nn.MaxPool2d(2))


class TransitionUp(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x:Tensor, skip:Tensor):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels:int, growth_rate:int, n_layers:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers,
            upsample=True, dropout_rate=dropout_rate))


def center_crop(x:Tensor, max_height:int, max_width:int):
    _, _, h, w = x.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return x[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class Tiramisu(nn.Module):
    def __init__(self, in_channels:int=3, down_blocks:List[int]=(5,5,5,5,5),
                 up_blocks:List[int]=(5,5,5,5,5), bottleneck_layers:int=5,
                 growth_rate:int=16, out_chans_first_conv:int=48, n_classes:int=1,
                 dropout_rate:float=0.2):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        skip_connection_channel_counts = []

        ## First Convolution ##
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for n_layers in down_blocks:
            self.denseBlocksDown.append(DenseBlock(
                cur_channels_count, growth_rate, n_layers,
                dropout_rate=dropout_rate))
            cur_channels_count += (growth_rate*n_layers)
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(
                cur_channels_count, dropout_rate=dropout_rate))

        #####################
        #     Bottleneck    #
        #####################
        self.add_module('bottleneck',Bottleneck(
            cur_channels_count, growth_rate, bottleneck_layers,
            dropout_rate=dropout_rate))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for n_layers, sccc in zip(up_blocks[:-1], skip_connection_channel_counts):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + sccc
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, n_layers,
                upsample=True, dropout_rate=dropout_rate))
            prev_block_channels = growth_rate*n_layers
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False, dropout_rate=dropout_rate))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Final Convolution ##
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)

    def forward(self, x):
        out = self.firstconv(x)
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


class Tiramisu67(Tiramisu, pl.LightningModule):

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass


if __name__ == "__main__":
    net = Tiramisu67()


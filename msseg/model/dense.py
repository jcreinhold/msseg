#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.model.dense

blocks/layers for densely-connected networks

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 02, 2020
"""

__all__ = ['Bottleneck2d',
           'Bottleneck3d',
           'DenseBlock2d',
           'DenseBlock3d',
           'TransitionDown2d',
           'TransitionDown3d',
           'TransitionUp2d',
           'TransitionUp3d']

import torch
from torch import Tensor
from torch import nn

ACTIVATION = nn.GELU


class ConvLayer(nn.Sequential):

    _norm = None
    _pad  = None
    _conv = None
    _drop = None
    _maxp = None
    _ksz  = None

    def __init__(self, in_channels:int, growth_rate:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('norm', self._norm(in_channels))
        self.add_module('act', ACTIVATION())
        if self._ksz > 2:
            self.add_module('pad', self._pad(self._ksz // 2))
        self.add_module('conv', self._conv(in_channels, growth_rate, self._ksz,
                                          bias=False))
        self.add_module('drop', self._drop(dropout_rate))
        if self._maxp is not None:
            self.add_module('maxpool', self._maxp(2))


class ConvLayer2d(ConvLayer):
    _norm = nn.BatchNorm2d
    _pad  = nn.ReplicationPad2d
    _conv = nn.Conv2d
    _drop = nn.Dropout2d
    _maxp = None
    _ksz  = 3


class ConvLayer3d(ConvLayer):
    _norm = nn.BatchNorm3d
    _pad  = nn.ReplicationPad3d
    _conv = nn.Conv3d
    _drop = nn.Dropout3d
    _maxp = None
    _ksz  = 3


class DenseBlock(nn.Module):

    _layer = None

    def __init__(self, in_channels:int, growth_rate:int, n_layers:int,
                 upsample:bool=False, dropout_rate:float=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.upsample = upsample
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList([
            self._layer(ic, self.growth_rate, self.dropout_rate)
            for ic in self.in_channels_range])

    def forward(self, x:Tensor):
        if self.upsample:
            new_features = []
            # We pass all previous activations into each dense layer normally
            # but we only store each dense layer's output in the new_features array.
            # Note that all concatenation is done on the channel axis (i.e., 1)
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x

    @property
    def in_channels_range(self):
        return [self.in_channels + i * self.growth_rate for i in range(self.n_layers)]


class DenseBlock2d(DenseBlock):
    _layer = ConvLayer2d


class DenseBlock3d(DenseBlock):
    _layer = ConvLayer3d


class TransitionDown2d(ConvLayer):
    _norm = nn.BatchNorm2d
    _pad  = nn.ReplicationPad2d
    _conv = nn.Conv2d
    _drop = nn.Dropout2d
    _maxp = nn.MaxPool2d
    _ksz  = 1


class TransitionDown3d(ConvLayer):
    _norm = nn.BatchNorm3d
    _pad  = nn.ReplicationPad3d
    _conv = nn.Conv3d
    _drop = nn.Dropout3d
    _maxp = nn.MaxPool3d
    _ksz  = 1


class TransitionUp(nn.Module):

    _conv_trans = None

    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        kernel_size = 3
        _crop = None
        self.convTrans = self._conv_trans(
            in_channels, out_channels, kernel_size,
            stride=2, bias=False)

    def forward(self, x:Tensor, skip:Tensor) -> Tensor:
        out = self.convTrans(x)
        out = self._crop_to_y(out, skip)
        out = torch.cat([out, skip], 1)
        return out

    @staticmethod
    def _crop_to_y(x:Tensor, y:Tensor):
        raise NotImplementedError


class TransitionUp2d(TransitionUp):
    _conv_trans = nn.ConvTranspose2d

    @staticmethod
    def _crop_to_y(x:Tensor, y:Tensor) -> Tensor:
        _, _, max_height, max_width = y.shape
        _, _, h, w = x.size()
        w = (w - max_width) // 2
        h = (h - max_height) // 2
        return x[:, :, h:(h + max_height), w:(w + max_width)]


class TransitionUp3d(TransitionUp):
    _conv_trans = nn.ConvTranspose3d

    @staticmethod
    def _crop_to_y(x:Tensor, y:Tensor) -> Tensor:
        _, _, max_height, max_width, max_depth = y.shape
        _, _, h, w, d = x.size()
        w = (w - max_width) // 2
        h = (h - max_height) // 2
        d = (d - max_depth) // 2
        return x[:, :, d:(d + max_depth), h:(h + max_height), w:(w + max_width)]


class Bottleneck(nn.Sequential):

    _layer = None

    def __init__(self, in_channels:int, growth_rate:int, n_layers:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('bottleneck', self._layer(
            in_channels, growth_rate, n_layers,
            upsample=True, dropout_rate=dropout_rate))


class Bottleneck2d(Bottleneck):
    _layer = DenseBlock2d


class Bottleneck3d(Bottleneck):
    _layer = DenseBlock3d

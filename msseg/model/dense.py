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


class DenseLayer2d(nn.Sequential):
    def __init__(self, in_channels:int, growth_rate:int, dropout_rate:float=0.2):
        super().__init__()
        kernel_size = 3
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('act', ACTIVATION())
        self.add_module('pad', nn.ReplicationPad2d(kernel_size // 2))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size,
                                          bias=False))
        self.add_module('drop', nn.Dropout2d(dropout_rate))


class DenseLayer3d(nn.Sequential):
    def __init__(self, in_channels:int, growth_rate:int, dropout_rate:float=0.2):
        super().__init__()
        kernel_size = 3
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('act', ACTIVATION())
        self.add_module('pad', nn.ReplicationPad3d(kernel_size // 2))
        self.add_module('conv', nn.Conv3d(in_channels, growth_rate, kernel_size,
                                          bias=False))
        self.add_module('drop', nn.Dropout3d(dropout_rate))


class DenseBlock(nn.Module):

    def __init__(self, in_channels:int, growth_rate:int, n_layers:int,
                 upsample:bool=False, dropout_rate:float=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.upsample = upsample
        self.dropout_rate = dropout_rate

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            DenseLayer2d(ic, self.growth_rate, self.dropout_rate)
            for ic in self.in_channels_range])


class DenseBlock3d(DenseBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            DenseLayer3d(ic, self.growth_rate, self.dropout_rate)
            for ic in self.in_channels_range])


class TransitionDown2d(nn.Sequential):
    def __init__(self, in_channels:int, dropout_rate:float=0.2):
        super().__init__()
        kernel_size = 1
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('act', ACTIVATION())
        self.add_module('conv', nn.Conv2d(in_channels, in_channels, kernel_size,
                                          bias=False))
        self.add_module('drop', nn.Dropout2d(dropout_rate))
        self.add_module('maxpool', nn.MaxPool2d(2))


class TransitionDown3d(nn.Sequential):
    def __init__(self, in_channels:int, dropout_rate:float=0.2):
        super().__init__()
        kernel_size = 1
        self.add_module('norm', nn.BatchNorm3d(num_features=in_channels))
        self.add_module('act', ACTIVATION())
        self.add_module('conv', nn.Conv3d(in_channels, in_channels, kernel_size,
                                          bias=False))
        self.add_module('drop', nn.Dropout3d(dropout_rate))
        self.add_module('maxpool', nn.MaxPool3d(2))


class TransitionUp2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        kernel_size = 3
        self.convTrans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=2, bias=False)

    def forward(self, x:Tensor, skip:Tensor) -> Tensor:
        _, _, h, w = skip.shape
        out = self.convTrans(x)
        out = center_crop2d(out, h, w)
        out = torch.cat([out, skip], 1)
        return out


class TransitionUp3d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        kernel_size = 3
        self.convTrans = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=2, padding=0, bias=False)

    def forward(self, x:Tensor, skip:Tensor) -> Tensor:
        _, _, h, w, d = skip.shape
        out = self.convTrans(x)
        out = center_crop3d(out, h, w, d)
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck2d(nn.Sequential):
    def __init__(self, in_channels:int, growth_rate:int, n_layers:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('bottleneck', DenseBlock2d(
            in_channels, growth_rate, n_layers,
            upsample=True, dropout_rate=dropout_rate))


class Bottleneck3d(nn.Sequential):
    def __init__(self, in_channels:int, growth_rate:int, n_layers:int, dropout_rate:float=0.2):
        super().__init__()
        self.add_module('bottleneck', DenseBlock3d(
            in_channels, growth_rate, n_layers,
            upsample=True, dropout_rate=dropout_rate))


def center_crop2d(x:Tensor, max_height:int, max_width:int) -> Tensor:
    _, _, h, w = x.size()
    w = (w - max_width) // 2
    h = (h - max_height) // 2
    return x[:, :, h:(h + max_height), w:(w + max_width)]


def center_crop3d(x:Tensor, max_height:int, max_width:int, max_depth:int) -> Tensor:
    _, _, h, w, d = x.size()
    w = (w - max_width) // 2
    h = (h - max_height) // 2
    d = (d - max_depth) // 2
    return x[:, :, d:(d + max_depth), h:(h + max_height), w:(w + max_width)]


if __name__ == "__main__":
    pass

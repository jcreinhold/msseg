#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.cutmix

implementation of cutmix based on
https://github.com/hysts/pytorch_cutmix
modified for segmentation

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Oct 07, 2020
"""

__all__ = ['cutmix2d',
           'cutmix3d',
           'CutMixCollator']

import numpy as np
import torch


def cutmix2d(batch, alpha:float=1.):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets[:,:,y0:y1,x0:x1] = shuffled_targets[:,:,y0:y1,x0:x1]

    return data, targets


def cutmix3d(batch, alpha:float=1.):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_d, image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    cz = np.random.uniform(0, image_d)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    d = image_d * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))
    z0 = int(np.round(max(cz - d / 2, 0)))
    z1 = int(np.round(min(cz + d / 2, image_d)))

    data[:,:,z0:z1,y0:y1,x0:x1] = shuffled_data[:,:,z0:z1,y0:y1,x0:x1]
    targets[:,:,z0:z1,y0:y1,x0:x1] = shuffled_targets[:,:,z0:z1,y0:y1,x0:x1]

    return data, targets


class CutMixCollator:
    def __init__(self, alpha:float=1., dim:int=3):
        assert 0. < alpha
        self.alpha = alpha
        assert 1 < dim < 4
        self.dim = dim

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if self.dim == 2:
            batch = cutmix2d(batch, self.alpha)
        elif self.dim == 3:
            batch = cutmix3d(batch, self.alpha)
        else:
            raise NotImplementedError('Only 2 and 3 dimensional images are supported.')
        return batch

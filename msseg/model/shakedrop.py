#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.shakedrop

loss functions to support lesion segmentation

References:
    [4] S.A. Taghanaki et al. "Combo loss: Handling input and
        output imbalance in multi-organ segmentation." Computerized
        Medical Imaging and Graphics 75 (2019): 24-33.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Oct 07, 2020
"""

__all__ = ['ShakeDrop']

from typing import *

import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Function, Variable

Range = Tuple[float,float]


class ShakeDropFunction(Function):

    @staticmethod
    def forward(ctx:Any, x:Tensor, training:bool=True,
                p:float=0.5, alpha_range:Range=(-1.,1.)) -> Tensor:
        if training:
            dvc = x.device
            n = x.size(0)
            dims = [1 for _ in range(x.ndim-1)]
            gate = torch.tensor([0.], device=dvc).bernoulli_(1 - p)
            ctx.save_for_backward(gate)
            if gate.item() == 0.:
                alpha = torch.zeros(n, device=dvc).uniform_(*alpha_range)
                alpha = alpha.view(n, *dims).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p) * x

    @staticmethod
    def backward(ctx, grad_output:Tensor, beta_range:Range=(0.,1.)):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0.:
            dvc = grad_output.device
            n = grad_output.size(0)
            dims = [1 for _ in range(grad_output.ndim-1)]
            beta = torch.zeros(n, device=dvc).uniform_(*beta_range)
            beta = beta.view(n, *dims).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_shakedrop=1., alpha_range=(-1, 1)):
        super(ShakeDrop, self).__init__()
        self.p = p_shakedrop
        self.alpha_range = alpha_range

    def forward(self, x):
        args = (self.training, self.p, self.alpha_range)
        return ShakeDropFunction.apply(x, *args)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: layernorm.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:28:20 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    """
    A custom autograd function for layer normalization.
    This function performs the forward and backward passes for layer normalization.
    """
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        """
        The forward pass computes the mean and variance of the input,
        then normalizes the input using the computed mean and variance.
        """
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)  # Compute the mean along the channel dimension
        var = (x - mu).pow(2).mean(1, keepdim=True)  # Compute the variance along the channel dimension
        y = (x - mu) / (var + eps).sqrt()  # Normalize the input using the computed mean and variance
        ctx.save_for_backward(y, var, weight)  # Save the necessary variables for the backward pass
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)  # Add the learned scale and bias
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass computes the gradients of the input, weight, and bias.
        """
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    """
    A 2D layer normalization module.
    This module applies layer normalization along the channel dimension.
    """
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        """
        Applies the layer normalization function to the input.
        """
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        """
        Applies the GRN normalization function to the input.
        """
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

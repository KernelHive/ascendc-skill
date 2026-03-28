#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
add_layer_norm_quant
"""
import torch
import numpy as np


def add_layer_norm(**kwargs):
    input_x1 = kwargs.get('x1', {'value': None})['value']
    input_x2 = kwargs.get('x2', {'value': None})['value']
    input_gamma = kwargs.get('gamma', {'value': None})['value']
    input_beta = kwargs.get('beta', {'value': None})['value']
    input_bias = None

    epsilon = kwargs.get('epsilon', 1e-05)
    output_x = kwargs.get("additional_output", False)

    x1 = input_x1.astype(np.float32)
    x2 = input_x2.astype(np.float32)
    gamma = input_gamma.astype(np.float32)
    beta = input_beta.astype(np.float32)

    if input_bias is None:
        x = x1 + x2
    else:
        bias = input_bias.astype(np.float32)
        x = x1 + x2 + bias

    reduce_axis = -1
    input_mean = np.mean(x, reduce_axis, keepdims=True)
    input_var = np.mean(np.power((x - input_mean), 2), reduce_axis, keepdims=True)
    input_rstd = 1.0 / np.sqrt(input_var + epsilon)

    y = np.subtract(x, input_mean) * input_rstd
    y = y * gamma + beta

    if input_x1.dtype != input_x2.dtype:
        out_dtype = np.float32
    else:
        out_dtype = input_x1.dtype

    if output_x:
        return y.astype(out_dtype), input_mean, input_rstd, x.astype(out_dtype)
    else:
        return y.astype(out_dtype), input_mean, input_rstd


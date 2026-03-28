#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
add_layer_norm_quant
"""
import torch
import numpy as np


def quant_milan(output_y):
    max_y_0 = torch.max(torch.abs(output_y), axis=-1, keepdim=True)[0]
    out_scales = max_y_0 / 127.0
    x = output_y / out_scales
    y = torch.round(x).type(torch.int8)
    return y.numpy().astype(np.int8), out_scales.numpy().astype(np.float32)


def add_layer_norm_cpu(x1, x2, gamma, beta, bias, eps):
    ori_type = x1.dtype
    if ori_type != torch.float32:
        x1 = x1.type(torch.float32)
        x2 = x2.type(torch.float32)
        gamma = gamma.type(torch.float32)
        beta = beta.type(torch.float32)
        bias = bias.type(torch.float32)
    x = x1 + x2 + bias
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean(torch.pow((x - mean), 2), dim=-1, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    y = (x - mean) * rstd * gamma + beta
    if ori_type != torch.float32:
        return y, x.type(ori_type)
    else:
        return y, x


def npt2ptt(npt, dtype, shape):
    if npt is None:
        return None
    str2Type = {
        'bfloat16': torch.bfloat16,
        'int8': torch.int8,
        'float16': torch.float16,
        'float32': torch.float32
    }
    return torch.tensor(npt.view(np.int8)).view(str2Type[dtype]).reshape(shape)


def add_layer_norm_quant(x1, x2, gamma, beta, bias, scales1, scales2, y1, y2, x,
                      out_scales1, out_scales2, quant_mode="dynamic", epsilon=1e-05, additional_output=False):

    x1 = x1['value'].astype(np.float32)
    x2 = x2['value'].astype(np.float32)
    gamma = gamma['value'].astype(np.float32)
    beta = beta['value'].astype(np.float32)
    bias = bias['value'].astype(np.float32)
    scales1 = scales1['value'].astype(np.float32)
    scales2 = scales2['value'].astype(np.float32)

    dtype_x = str(x1.dtype)

    if 'float16' in dtype_x or 'bf16' in dtype_x:
        dtype_hp = np.float32
        dtype_hp_pt = torch.float32
    else:
        dtype_hp = np.float64
        dtype_hp_pt = torch.float64

    x1_tensor = torch.from_numpy(x1.astype(dtype_hp)).type(dtype_hp_pt)
    x2_tensor = torch.from_numpy(x2.astype(dtype_hp)).type(dtype_hp_pt)
    gamma_tensor = torch.from_numpy(gamma.astype(dtype_hp)).type(dtype_hp_pt)
    beta_tensor = torch.from_numpy(beta.astype(dtype_hp)).type(dtype_hp_pt)
    bias_tensor = torch.from_numpy(bias.astype(dtype_hp)).type(dtype_hp_pt)
    s1_tensor = torch.from_numpy(scales1.astype(dtype_hp)).type(dtype_hp_pt)
    s2_tensor = torch.from_numpy(scales2.astype(dtype_hp)).type(dtype_hp_pt)

    x = x1_tensor + x2_tensor + bias_tensor
    mean = torch.mean(x, dim=-1, keepdim=True)
    tensor_var = torch.mean(torch.pow((x - mean), 2), dim=-1, keepdim=True)
    rstd = 1 / torch.sqrt(tensor_var + epsilon)
    y = (x - mean) * rstd * gamma_tensor + beta_tensor

    # torch.max不支持fp64
    y = y.type(torch.float32)
    output_y1 = y * s1_tensor.type(torch.float32)
    y1, out_s1 = quant_milan(output_y1)

    output_y2 = y * s2_tensor.type(torch.float32) 
    y2, out_s2 = quant_milan(output_y2)
    gx = np.zeros(x1.shape, dtype=x1.dtype)

    return y1, y2, gx, out_s1[:1, :2, :2], out_s2[:1, :2, :2]


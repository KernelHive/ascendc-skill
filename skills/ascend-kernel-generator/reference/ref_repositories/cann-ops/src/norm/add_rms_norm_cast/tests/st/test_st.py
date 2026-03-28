#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Third-Party Packages
import subprocess
import numpy as np


def x_norm(x, eps=1e-6):
    """
    Apply the RMSNorm normalization to the input tensor.
    Args:
        x (torch.Tensor): The input tensor.usually fp32.
    Returns:
        torch.Tensor: The normalized tensor.
    """
    import torch

    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd, rstd


def rms_norm_golden(x, gamma, eps=1e-6):
    import torch

    output, rstd = x_norm(x.float(), eps)

    if gamma.dtype == torch.float16 or gamma.dtype == torch.float32:
        return output.type_as(gamma) * gamma, rstd
    else:
        return (output * gamma.to(torch.float32)).to(torch.bfloat16), rstd


def add_rms_golden_milan(x1, x2, gamma, eps=1e-6):
    import torch

    if x1.dtype == torch.float16 or x1.dtype == torch.float32:
        x = x1 + x2
    else:  # bfloat16
        if x1.shape[-1] > 12288:
            x = (x1.float() + x2.float()).to(torch.bfloat16).to(torch.float32)
        else:
            x = x1.float() + x2.float()

    y, rstd = rms_norm_golden(x, gamma, eps)
    return y, rstd, x


def add_rms_golden_torino(x1, x2, gamma, eps=1e-6):
    import torch

    x = (x1.type(torch.float32) + x2.type(torch.float32))

    y, rstd = rms_norm_golden(x, gamma, eps)
    return y, rstd, x


def get_soc_version():
    # 获取原始输出
    raw_output = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    lines = raw_output.strip().split("\n")
    if len(lines) >= 2:
        second_line = lines[1]
        fields = second_line.split()
        if len(fields) >= 5:
            result = fields[3] + fields[4]
            return result
        else:
            return None
    else:
        return None 


def add_rms_norm_cast(x1, x2, gamma, y1, y2, rstd, x, epsilon):
    import torch

    x1_data = x1['value'].astype(np.float16)
    x2_data = x2['value'].astype(np.float16)
    gamma_data = gamma['value'].astype(np.float16)

    d_len = 1
    total_len = 1
    for _d in x1_data.shape:
        total_len *= _d
    for _d in gamma_data.shape:
        d_len *= _d
    n_len = int(total_len / d_len)
    x1_data = x1_data.reshape((n_len, d_len))
    x2_data = x2_data.reshape((n_len, d_len))
    gamma_data = gamma_data.reshape((d_len))
    input_dtype = x1_data.dtype
    if x1_data.dtype == np.float16:
        dtype = "fp16"
    elif x1_data.dtype == np.float32:
        dtype = "fp32"
    else:
        dtype = "bf16"
    if dtype == 'bf16':
        x1_tensor = torch.tensor(x1_data.astype(np.float32)).to(torch.bfloat16)
        x2_tensor = torch.tensor(x2_data.astype(np.float32)).to(torch.bfloat16)
        gamma_tensor = torch.tensor(gamma_data.astype(np.float32)).to(torch.bfloat16)
    else:
        x1_tensor = torch.tensor(x1_data)
        x2_tensor = torch.tensor(x2_data)
        gamma_tensor = torch.tensor(gamma_data)

    short_soc_version = get_soc_version()
    if short_soc_version is not None and (("Ascend910B" in short_soc_version) or ("Ascend910_93" in short_soc_version)):
        y_tensor, var_tensor, x_tensor = add_rms_golden_milan(x1_tensor, x2_tensor, gamma_tensor, eps=epsilon)
        if dtype == 'bf16':
            y = y_tensor.to(torch.float32).numpy().astype(input_dtype)
            x = x_tensor.to(torch.float32).numpy().astype(input_dtype)
        else:
            y = y_tensor.numpy()
            x = x_tensor.numpy()
        rstd = var_tensor.numpy()

        y1 = y_tensor.to(torch.float32).numpy()
        return y1, y, rstd, x
    else:
        x_tensor = (x1_tensor.type(torch.float32) + x2_tensor.type(torch.float32))
        rstd_tensor = torch.rsqrt(x_tensor.pow(2).mean(-1, keepdim=True) + epsilon)
        y_tensor = x_tensor * rstd_tensor * gamma_tensor.to(torch.float32)
        if dtype == 'bf16' or dtype == 'fp16':
            y2 = y_tensor.to(torch.float32).numpy().astype(input_dtype)
            x = x_tensor.to(torch.float32).numpy().astype(input_dtype)
        else:
            y2 = y_tensor.numpy()
            x = x_tensor.numpy()
        rstd = rstd_tensor.numpy()

        y1 = y_tensor.numpy()
        return y1, y2, rstd, x

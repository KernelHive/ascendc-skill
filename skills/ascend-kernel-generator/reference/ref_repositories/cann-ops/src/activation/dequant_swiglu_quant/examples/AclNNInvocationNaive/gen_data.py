#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import torch
import tensorflow as tf
import numpy as np
def numpy_to_torch_tensor(np_array):
    if np_array.dtype == tf.bfloat16.as_numpy_dtype:
        return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(np_array)

def _dequant_swiglu_quant(x, weight_scale, activate_scale, bias, quant_scale, quant_offset, group_index, quant_mode, activate_left):
    x_dtype = x.dtype

    if len(x.shape) > 2:
        x = x.reshape(-1, x.shape[-1])
    
    if len(weight_scale.shape) == 1:
        weight_scale = weight_scale.reshape(1, -1)

    if activate_scale is not None and len(activate_scale.shape) >= 1:
        activate_scale = activate_scale.reshape(-1, 1)
    
    if quant_scale is not None and len(quant_scale.shape) == 1:
        quant_scale = quant_scale.reshape(1, -1)

    if group_index is None:
        group_index = np.array([x.shape[0]])

    if "int32" in str(x_dtype):
        weight_scale = numpy_to_torch_tensor(weight_scale)
        activate_scale = numpy_to_torch_tensor(activate_scale)
        quant_scale = numpy_to_torch_tensor(quant_scale)
        x = numpy_to_torch_tensor(x)
        res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float32)
        res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)

        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx]
            x_tensor = x[offset: (offset+groupIdx)].to(torch.float32)

            if bias is not None: # TIP:bias未适配多group
                x_tensor = torch.add(x_tensor, bias)
            res = torch.mul(x_tensor, weight_scale[g_idx].to(torch.float32))

            if activate_scale is not None:
                res = torch.mul(res, activate_scale[offset: (offset+groupIdx)].to(torch.float32))
            out = torch.chunk(res, 2, dim=-1)

            if activate_left:
                self_tensor = out[0]
                other = out[1]
            else:
                self_tensor = out[1]
                other = out[0]

            output = torch.nn.functional.silu(self_tensor) * other # 100
            if quant_scale is not None:
                output = torch.mul(output, quant_scale[g_idx].to(torch.float32))  # 100

            if quant_mode == "static":
                output = torch.add(output, quant_offset.to(torch.float32))
            if quant_mode == "dynamic":
                abs = torch.abs(output)
                max_values = torch.amax(abs, dim = -1)
                scale_out = max_values / 127
                max_values = 127 / max_values
                output = output * max_values.unsqueeze(1)
            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset: (offset+groupIdx)] = output
            res_scale[offset: (offset+groupIdx)] = scale_out
            offset = offset + groupIdx
        return res_y.to(torch.int8).numpy(), res_scale.numpy()
    else:
        tensor_x = x
        out = torch.chunk(tensor_x, 2, dim=-1)
        if activate_left:
            self_tensor = out[0]
            other = out[1]
        else:
            self_tensor = out[1]
            other = out[0]
        output = torch.nn.functional.silu(self_tensor) * other # 100
        output = torch.add(output, quant_offset.to(torch.float32))
        output = torch.clamp(output, -128, 127)
        output = torch.round(output)
        return output.to(torch.int8).numpy()

def gen_golden_data_simple():
    x = torch.randn(2, 32).to(torch.float16)
    scale = torch.tensor([1]).to(torch.float32)
    offset = torch.tensor([1]).to(torch.float32)

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    x.numpy().tofile("./input/input_x.bin")
    scale.numpy().astype(np.float32).tofile("./input/input_scale.bin")
    offset.numpy().astype(np.float32).tofile("./input/input_offset.bin")
    empty_tensor = torch.tensor([])
    out = _dequant_swiglu_quant(x, empty_tensor, empty_tensor, empty_tensor, scale, offset, None, 'static', False)
    out.tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


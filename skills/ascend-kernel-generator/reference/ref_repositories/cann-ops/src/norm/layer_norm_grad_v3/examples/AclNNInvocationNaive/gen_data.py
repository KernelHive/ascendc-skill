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
import numpy as np


def gen_golden_data_simple():
    tensor_dy = torch.tensor([2, 3, 4, 5], dtype=torch.float32).reshape(2, 2)
    tensor_x = torch.tensor([2, 3, 4, 5], dtype=torch.float32).reshape(2, 2)
    tensor_rstd = torch.tensor([4, 5], dtype=torch.float32).reshape(2, 1)
    tensor_mean = torch.tensor([2, 3], dtype=torch.float32).reshape(2, 1)
    tensor_gamma = torch.tensor([1, 1], dtype=torch.float32).reshape(2)
    normalized_shape = tensor_gamma.shape
    
    # call torch.ops.aten.native_layer_norm
    tensor_dx, tensor_dw, tensor_db = torch.ops.aten.native_layer_norm_backward(tensor_dy,
                                                                                tensor_x,
                                                                                normalized_shape,
                                                                                tensor_mean,
                                                                                tensor_rstd,
                                                                                tensor_gamma,
                                                                                tensor_gamma,
                                                                                [True, True, True])

    # convert back to numpy
    if "bfloat16" in str(tensor_dx.dtype):
        numpy_dx = tensor_dx.float().numpy()
        numpy_dw = tensor_dw.float().numpy()
        numpy_db = tensor_db.float().numpy()
    else:
        numpy_dx = tensor_dx.numpy()
        numpy_dw = tensor_dw.numpy()
        numpy_db = tensor_db.numpy()
    
    numpy_dx.tofile("./output/golden1.bin")
    numpy_dw.tofile("./output/golden2.bin")
    numpy_db.tofile("./output/golden3.bin")


if __name__ == "__main__":
    gen_golden_data_simple()


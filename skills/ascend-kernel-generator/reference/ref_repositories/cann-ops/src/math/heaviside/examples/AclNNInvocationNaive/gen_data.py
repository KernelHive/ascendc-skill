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


def add_random_zeros(tensor, zero_ratio=0.2):
    num_elements = tensor.size
    num_zeros = int(num_elements * zero_ratio)
    random_indices = np.random.choice(num_elements, num_zeros, replace=False)
    tensor.flat[random_indices] = 0
    return tensor
def gen_golden_data_simple():
    case_data = {
        'input_shape': [517, 517],
        'data_type': np.float16,
        'values_shape': [517, 517]
    }
    input_input = np.random.uniform(-100, 100, case_data['input_shape']).astype(case_data['data_type'])
    input_input = add_random_zeros(input_input, 0.2)
    input_values = np.random.uniform(-100, 100, case_data['values_shape']).astype(case_data['data_type'])
    golden = torch.heaviside(torch.from_numpy(input_input), torch.from_numpy(input_values)).numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_input.tofile("./input/input_input.bin")
    input_values.tofile("./input/input_values.bin")
    golden.tofile("./output/golden.bin")
if __name__ == "__main__":
    gen_golden_data_simple()

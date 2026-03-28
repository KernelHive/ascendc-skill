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
import torch
import os
def gen_golden_data_simple():
    dtype = "float32"
    input1 = torch.zeros(3, 4, 133, 4095, dtype=torch.float)
    golden = torch.eye(133, 4095, dtype=torch.float)
    golden = golden.unsqueeze(0).unsqueeze(0)
    golden = golden.repeat(3, 4, 1, 1)
    input1.numpy().tofile('./script/input/input0.bin')
    golden.numpy().tofile("./script/output/golden0.bin")
    
    with open("./script/output/meta", "w") as fp:
        print(dtype, file=fp)

if __name__ == "__main__":
    gen_golden_data_simple()

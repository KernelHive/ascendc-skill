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
    input1 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], dtype=torch.float32)
    input2 = torch.tensor([1, 2, 0, 1, 0, 2, 1, 0, 2], dtype=torch.int32)
    golden0 = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3],
                            [0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3],
                            [0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]], dtype=torch.float32)
    golden1 = torch.tensor([3, 6, 0, 4, 1, 7, 5, 2, 8], dtype=torch.int32)
    golden2 = torch.tensor([3, 6, 9], dtype=torch.int32)
    input1.numpy().tofile('./script/input/input0.bin')
    input2.numpy().tofile('./script/input/input1.bin')
    golden0.numpy().tofile("./script/output/golden0.bin")
    golden1.numpy().tofile("./script/output/golden1.bin")
    golden2.numpy().tofile("./script/output/golden2.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

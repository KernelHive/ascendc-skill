# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================
import torch
import torch.nn as nn
import numpy as np
import os


def gen_golden_data_simple():
    np.random.seed(0)
    src_type = np.float32
    target_type = np.int32
    dim_size = 5

    input_x = np.random.uniform(1, 5, [dim_size, dim_size]).astype(src_type)
    input_src = np.random.uniform(1, 5, [dim_size, dim_size]).astype(src_type)
    input_index = np.random.uniform(0, dim_size - 1, [dim_size, dim_size]).astype(
        target_type
    )

    reduce = "amin"
    dim = 0
    include_self = False

    input_x_cpu = torch.from_numpy(input_x)
    input_src_cpu = torch.from_numpy(input_src)
    input_index_cpu = torch.from_numpy(input_index.astype(np.int64))

    res = torch.scatter_reduce(
        input=input_x_cpu,
        dim=dim,
        index=input_index_cpu,
        src=input_src_cpu,
        reduce=reduce,
        include_self=include_self,
    )

    golden = res.numpy().astype(src_type)

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_x.tofile("./input/input_x.bin")
    input_src.tofile("./input/input_src.bin")
    input_index.tofile("./input/input_index.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

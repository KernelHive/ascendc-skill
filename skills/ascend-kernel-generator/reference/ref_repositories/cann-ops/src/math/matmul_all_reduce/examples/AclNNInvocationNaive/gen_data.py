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
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import ReduceOp

RANK_DIM = 1
RANK_M = 16384
RANK_K = 640
RANK_N = 5120


def gen_cpu_data(rank, port):
    input_x1 = torch.tensor(np.fromfile("./input/input_x1_{}.bin".format(rank), np.float16)
        .reshape([RANK_M, RANK_K]))
    input_x2 = torch.tensor(np.fromfile("./input/input_x2_{}.bin".format(rank), np.float16)
        .reshape([RANK_K, RANK_N]))
    dist.init_process_group(backend='gloo', rank=rank, world_size=RANK_DIM, init_method=f'tcp://127.0.0.1:{port}')
    print('[INFO] device_{} 构造cpu_out数据'.format(rank))
    cpu_input = input_x1.to(torch.float32)
    cpu_weight = input_x2.to(torch.float32)
    cpu_mm_out = torch.matmul(cpu_input, cpu_weight)
    dist.all_reduce(cpu_mm_out, op=dist.ReduceOp.SUM)
    np.array(cpu_mm_out.cpu()).tofile('./output/cpu_out_{}.bin'.format(rank))


def gen_cpu():
    from torch.multiprocessing import Process
    p_list = []
    port = 29500 + random.randint(0, 10000)
    mp.set_start_method("forkserver", force=True)
    for rank in range(RANK_DIM):
        p = Process(target=gen_cpu_data, args=(rank, port))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


if __name__ == "__main__":
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")

    # get x1 x2
    for rank in range(RANK_DIM):
        np.random.seed(rank)
        x1 = np.random.uniform(-3, 3, [RANK_M, RANK_K]).astype(np.float16)
        x2 = np.random.uniform(-3, 3, [RANK_K, RANK_N]).astype(np.float16)
        x1.tofile("./input/input_x1_{}.bin".format(rank))
        x2.tofile("./input/input_x2_{}.bin".format(rank))

    gen_cpu()
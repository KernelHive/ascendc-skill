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
import sys
import numpy as np

LOSS = 1e-4  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
MINIMUM = 10e-10


class Results:
    def __init__(self, result_voxels, golden_voxels, result_coors, golden_coors, 
                 result_per_voxel, golden_per_voxel, result_voxel_num):
        self.result_voxels = result_voxels
        self.golden_voxels = golden_voxels
        self.result_coors = result_coors
        self.golden_coors = golden_coors
        self.result_per_voxel = result_per_voxel
        self.golden_per_voxel = golden_per_voxel
        self.result_voxel_num = result_voxel_num


def verify_result(results):
    result_voxels = np.fromfile(results.result_voxels, dtype=np.float32) # 从bin文件读取实际运算结果
    golden_voxels = np.fromfile(results.golden_voxels, dtype=np.float32) # 从bin文件读取预期运算结果
    result_voxel_num = np.fromfile(results.result_voxel_num, dtype=np.int32)
    result_coors = np.fromfile(results.result_coors, dtype=np.int32)
    golden_coors = np.fromfile(results.golden_coors, dtype=np.int32)
    result_per_voxel = np.fromfile(results.result_per_voxel, dtype=np.int32)
    golden_per_voxel = np.fromfile(results.golden_per_voxel, dtype=np.int32)
    result_voxel_num_s = result_voxel_num[0]
    golden_voxel_num = golden_coors.shape[0] / 3
    if result_voxel_num_s != golden_voxel_num:
        print("[ERROR] result_voxel_num error")
        return False 

    result_voxels = np.reshape(result_voxels, (500, 5, 4))
    result_voxels = result_voxels[:result_voxel_num_s]

    golden_voxels = np.reshape(golden_voxels, (-1, 5, 4))

    result = np.abs(result_voxels - golden_voxels) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(result_voxels), np.abs(golden_voxels))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, LOSS) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if (np.sum(result_rtol == False) > result_voxels.size * LOSS and 
            np.sum(result_atol == False) > result_voxels.size * LOSS): 
            print("[ERROR] voxels error")
            return False

    result_coors = np.reshape(result_coors, (-1, 3))
    result_coors = result_coors[:result_voxel_num_s]
    golden_coors = np.reshape(golden_coors, (-1, 3))

    result = np.abs(result_coors - golden_coors) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(result_coors), np.abs(golden_coors))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, LOSS) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if (np.sum(result_rtol == False) > result_coors.size * LOSS and 
            np.sum(result_atol == False) > result_coors.size * LOSS): 
            print("[ERROR] coors error")
            return False

    result_per_voxel = result_per_voxel[:result_voxel_num_s]

    result = np.abs(result_per_voxel - golden_per_voxel) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(result_per_voxel), np.abs(golden_per_voxel))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, LOSS) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if (np.sum(result_rtol == False) > golden_per_voxel.size * LOSS and 
            np.sum(result_atol == False) > golden_per_voxel.size * LOSS): 
            print("[ERROR] num_points_per_voxel error")
            return False
    print("test pass")

    return True


if __name__ == '__main__':
    results = Results(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 
                      sys.argv[5], sys.argv[6], sys.argv[7])
    verify_result(results)



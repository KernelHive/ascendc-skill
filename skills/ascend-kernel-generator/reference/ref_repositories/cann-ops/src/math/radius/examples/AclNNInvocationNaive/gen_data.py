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
from dataclasses import dataclass
import numpy as np


@dataclass
class RadiusParams:
    """封装Radius操作的配置参数"""
    r: float
    max_num_neighbors: int = 32
    ignore_same_index: bool = False


def _find_neighbors_for_point(y_point: np.ndarray, y_index: int,
                              x_batch: np.ndarray, x_offset: int,
                              params: RadiusParams) -> np.ndarray:
    """为单个y点在给定的x批次中查找邻居。"""
    distances = np.linalg.norm(x_batch - y_point, axis=1)
    neighbor_indices = np.where(distances <= params.r)[0]
    neighbor_indices += x_offset

    if params.ignore_same_index:
        neighbor_indices = neighbor_indices[neighbor_indices != y_index]

    return neighbor_indices[:params.max_num_neighbors]


def radius_numpy(x: np.ndarray, y: np.ndarray, params: RadiusParams,
                 ptr_x: np.ndarray = None, ptr_y: np.ndarray = None):
    """
    用numpy实现radius_cuda算子的功能。

    :param x: 节点特征矩阵, shape为 [N, F]。
    :param y: 节点特征矩阵, shape为 [M, F]。
    :param params: Radius操作的配置参数 (RadiusParams对象)。
    :param ptr_x: 可选的批次指针。
    :param ptr_y: 可选的批次指针。
    :return: 邻接索引, shape为 [2, K]。
    """
    ans_dtype = x.dtype
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    out_vec = []

    if ptr_x is None or ptr_y is None:
        ptr_x = np.array([0, x.shape[0]])
        ptr_y = np.array([0, y.shape[0]])

    for b in range(len(ptr_x) - 1):
        x_start, x_end = ptr_x[b], ptr_x[b + 1]
        y_start, y_end = ptr_y[b], ptr_y[b + 1]
        if x_start == x_end or y_start == y_end:
            continue
        x_batch = x[x_start:x_end]
        for i in range(y_start, y_end):
            neighbors = _find_neighbors_for_point(y[i], i, x_batch, x_start, params)
            for neighbor in neighbors:
                out_vec.extend([neighbor, i])

    if not out_vec:
        return np.array([], dtype=ans_dtype).reshape(2, 0)
        
    out = np.array(out_vec, dtype=ans_dtype).reshape(-1, 2).T
    return out


if __name__ == "__main__":
    cdtype = os.getenv('COMPUTE_TYPE')
    if cdtype == 'float16':
        compute_dtype = np.float16
    elif cdtype == 'float32':
        compute_dtype = np.float32
    else:
        compute_dtype = np.int32

    x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
    y = np.random.uniform(-5, 5, [50, 2]).astype(compute_dtype)
    ptr_x = None
    ptr_y = None
    r = 1.0
    max_num_neighbors = 10
    ignore_same_index = False
    radius_params = RadiusParams(r=r, max_num_neighbors=max_num_neighbors, ignore_same_index=ignore_same_index)
    assign_index = radius_numpy(x, y, params=radius_params)
    
    for i in assign_index.shape:
        print(i, end=' ')

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.tofile("./input/input_x.bin")
    y.tofile("./input/input_y.bin")
    assign_index.tofile("./output/golden.bin")

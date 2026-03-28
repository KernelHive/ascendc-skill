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
from dataclasses import dataclass
import numpy as np


@dataclass
class BevPoolInputs:
    """封装BevPool输入的配置参数"""
    depth: np.ndarray
    feat: np.ndarray
    ranks_depth: np.ndarray
    ranks_feat: np.ndarray
    ranks_bev: np.ndarray
    interval_starts: np.ndarray
    interval_lengths: np.ndarray
    bev_feat_shape: tuple
    

def bev_pool(inputs: BevPoolInputs) -> np.ndarray:
    """
    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (N_points, ),
        ranks_feat:  (N_points, ),
        ranks_bev:   (N_points, ),
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        interval_starts: (N_pillar, )
        interval_lengths: (N_pillar, )
    Returns:
        x: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    bev_feat = np.zeros(bev_feat_shape, dtype=np.float16)

    # 遍历每一个结构
    n_pillar = len(interval_starts)
    for i in range(n_pillar):
        start = interval_starts[i]
        length = interval_lengths[i]
        end = start + length
        # 遍历当前结构中的每一个点
        for j in range(start, end):
            rank_depth = ranks_depth[j]
            rank_feat = ranks_feat[j]
            rank_bev = ranks_bev[j]

            # 从 depth 和 feat 中提取对应元素
            b, n, d, fh, fw = np.unravel_index(rank_depth, depth.shape)
            b_, n_, fh_, fw_, c = np.unravel_index(rank_feat, feat.shape)
            assert b == b_ and n == n_ and fh == fh_ and fw == fw_

            # 计算 bev_feat 中的位置
            b__, dz, dy, dx, c_ = np.unravel_index(rank_bev, bev_feat_shape)
            assert b == b__ and c == c_

            # 累加特征
            bev_feat[b, dz, dy, dx, c] += depth[b, n, d, fh, fw] * feat[b, n, fh, fw, c]

    # 调整维度为 (B, C, Dz, Dy, Dx)
    x = np.transpose(bev_feat, (0, 4, 1, 2, 3))
    return x


# 调用示例
if __name__ == "__main__":
    # 定义输入参数的形状
    B, N, D, fH, fW = 2, 4, 6, 8, 10
    D_Z, D_Y, D_X, C = 4, 6, 8, 12
    N_points, N_pillar = 20, 10

    # 生成随机输入数据
    depth = np.random.randn(B, N, D, fH, fW).astype(np.float16)
    feat = np.random.randn(B, N, fH, fW, C).astype(np.float16)

    # 保证 ranks_depth 和 ranks_feat 共享相同的 b, n, fh, fw
    b_indices = np.random.randint(0, B, N_points).astype(np.int32)
    n_indices = np.random.randint(0, N, N_points).astype(np.int32)
    fh_indices = np.random.randint(0, fH, N_points).astype(np.int32)
    fw_indices = np.random.randint(0, fW, N_points).astype(np.int32)
    d_indices = np.random.randint(0, D, N_points).astype(np.int32)
    c_indices = np.random.randint(0, C, N_points).astype(np.int32)

    ranks_depth = np.ravel_multi_index((b_indices, n_indices, d_indices, fh_indices, fw_indices),
                                        depth.shape).astype(np.int32)
    ranks_feat = np.ravel_multi_index((b_indices, n_indices, fh_indices, fw_indices, c_indices), 
                                       feat.shape).astype(np.int32)

    # 保证 ranks_bev 中的 b 和 c 与 ranks_depth 和 ranks_feat 一致
    dz_indices = np.random.randint(0, D_Z, N_points).astype(np.int32)
    dy_indices = np.random.randint(0, D_Y, N_points).astype(np.int32)
    dx_indices = np.random.randint(0, D_X, N_points).astype(np.int32)
    ranks_bev = np.ravel_multi_index((b_indices, dz_indices, dy_indices, dx_indices, c_indices), 
                                    (B, D_Z, D_Y, D_X, C)).astype(np.int32)

    bev_feat_shape = (B, D_Z, D_Y, D_X, C)
    interval_starts = np.random.randint(0, N_points // 2, N_pillar).astype(np.int32)
    interval_lengths = np.random.randint(1, N_points - interval_starts.max(), N_pillar).astype(np.int32)

    inputs = BevPoolInputs(
        depth=depth,
        feat=feat,
        ranks_depth=ranks_depth,
        ranks_feat=ranks_feat,
        ranks_bev=ranks_bev,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
        bev_feat_shape=bev_feat_shape
    )

    os.system("mkdir -p input")
    depth.tofile("./input/input_depth.bin")
    feat.tofile("./input/input_feat.bin")
    ranks_depth.tofile("./input/input_ranks_depth.bin")
    ranks_feat.tofile("./input/input_ranks_feat.bin")
    ranks_bev.tofile("./input/input_ranks_bev.bin")
    interval_starts.tofile("./input/input_interval_starts.bin")
    interval_lengths.tofile("./input/input_interval_lengths.bin")

    # 调用 bev_pool 算子
    result = bev_pool(inputs)
    
    os.system("mkdir -p output")
    result.tofile("./output/result.bin")

    print("BEV 特征形状:", result.shape)
    
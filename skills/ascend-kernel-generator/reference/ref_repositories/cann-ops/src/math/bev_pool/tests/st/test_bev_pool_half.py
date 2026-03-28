#!/usr/bin/python3
# coding=utf-8
#
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
    depth: np.ndarray
    feat: np.ndarray
    ranks_depth: np.ndarray
    ranks_feat: np.ndarray
    ranks_bev: np.ndarray
    bev_feat_shape: tuple
    interval_starts: np.ndarray
    interval_lengths: np.ndarray


def bev_pool_test(inputs: BevPoolInputs) -> np.ndarray:

    bev_feat = np.zeros(bev_feat_shape, dtype=np.float16)

    n_pillar = len(interval_starts)
    for i in range(n_pillar):
        start = interval_starts[i]
        length = interval_lengths[i]
        end = start + length
        for j in range(start, end):
            rank_depth = ranks_depth[j]
            rank_feat = ranks_feat[j]
            rank_bev = ranks_bev[j]

            # 从 depth 和 feat 中提取对应元素
            b, n, d, fh, fw = np.unravel_index(rank_depth, depth.shape)
            b_, n_, fh_, fw_, c = np.unravel_index(rank_feat, feat.shape)

            # 计算 bev_feat 中的位置
            b__, dz, dy, dx, c_ = np.unravel_index(rank_bev, bev_feat_shape)

            # 累加特征
            bev_feat[b__, dz, dy, dx, c_] += depth[b, n, d, fh, fw] * feat[b_, n_, fh_, fw_, c]

    # 调整维度为 (B, C, Dz, Dy, Dx)
    x = np.transpose(bev_feat, (0, 4, 1, 2, 3))
    return x


def calc_expect_func(**kwargs):
    depth = kwargs.get('depth', {'value': None})
    feat = kwargs.get('feat', {'value': None})
    ranks_depth = kwargs.get('ranks_depth', {'value': None})
    ranks_feat = kwargs.get('ranks_feat', {'value': None})
    ranks_bev = kwargs.get('ranks_bev', {'value': None})
    interval_starts = kwargs.get('interval_starts', {'value': None})
    interval_lengths = kwargs.get('interval_lengths', {'value': None})
    bev_feat_shape = kwargs.get('bev_feat_shape', {'value': None})

    inputs = BevPoolInputs(
        depth=depth["value"],
        feat=feat["value"],
        ranks_depth=ranks_depth["value"],
        ranks_feat=ranks_feat["value"],
        ranks_bev=ranks_bev["value"],
        bev_feat_shape=bev_feat_shape["value"],
        interval_starts=interval_starts["value"],
        interval_lengths=interval_lengths["value"]
    )
    res = bev_pool_test(inputs)
    return [res]
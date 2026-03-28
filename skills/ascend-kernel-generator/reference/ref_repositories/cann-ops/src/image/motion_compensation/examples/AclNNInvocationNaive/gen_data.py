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
class TimeRange:
    min: float
    max: float


@dataclass
class PoseRange:
    translation_min: np.ndarray   # shape (3,)
    translation_max: np.ndarray   # shape (3,)
    quaternion_min: np.ndarray    # shape (4,)  wxyz
    quaternion_max: np.ndarray    # shape (4,)  wxyz


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_coniugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ])


def motion_compensation(
    points: np.ndarray,      # [ndim, N]
    timestamps: np.ndarray,  # [N]
    time_range: TimeRange,
    pose_range: PoseRange
) -> np.ndarray:
    # 原来的实现保持不变，仅把形参替换为结构体字段
    translation_min = pose_range.translation_min
    translation_max = pose_range.translation_max
    quaternion_min = pose_range.quaternion_min
    quaternion_max = pose_range.quaternion_max
    t_min = time_range.min
    t_max = time_range.max
    f = 1.0 / (t_max - t_min) if t_max != t_min else 0.0
    
    translation_global = translation_min - translation_max
    q_max = quaternion_max
    q_min = quaternion_min
    
    q_max_coni = quaternion_coniugate(q_max)
    
    translation = np.dot(quaternion_to_rotation_matrix(q_max_coni), translation_global)
    
    q_rel = quaternion_multiply(q_max_coni, q_min)
    q_rel_norm = np.linalg.norm(q_rel)
    if q_rel_norm > 0:
        q_rel = q_rel / q_rel_norm

    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    
    d = np.dot(q_identity, q_rel).astype(np.float32)
    abs_d = abs(d)
    do_rotation = (abs_d < 1.0 - 1e-8)
    theta = np.arccos(abs_d)
    sin_theta = np.sin(theta)

    ndim, n = points.shape
    out_points = np.zeros((ndim, n), dtype=np.float32)

    for i in range(n):
        point = points[:3, i]
        timestamp = timestamps[i]
        if np.isnan(point[0]):
            out_points[:, i] = points[:, i]
            continue
        t = (t_max - timestamp) * f if t_max != t_min else 0.0
        
        translation_comp = t * translation
        
        if do_rotation:
            c1_sign = 1.0 if d >= 0 else -1.0
            c0 = np.sin((1 - t) * theta) / sin_theta
            c1 = np.sin(t * theta) / sin_theta * c1_sign
            qi = c0 * q_identity + c1 * q_rel

            rotation_matrix = quaternion_to_rotation_matrix(qi)
            rotated_point = np.dot(rotation_matrix, point)
            comp_point = rotated_point + translation_comp
        else:
            comp_point = point + translation_comp
        out_points[:3, i] = comp_point
        if ndim > 3:
            out_points[3:, i] = points[3:, i]
    
    return out_points


def random_quaternion():
    q = np.random.randn(4)
    return q / np.linalg.norm(q)


def gen_golden_data_simple():
    np.random.seed(42)

    n = 1000
    ndim = 4
    t_min = 1000
    t_max = 2000

    points = np.random.rand(ndim, n).astype(np.float32)
    points[3, :] = np.random.uniform(0, 1, n)

    nan_indices = np.random.choice(n, size=n // 100, replace=False)
    points[0, nan_indices] = np.nan

    timestamps = np.sort(np.random.randint(t_min, t_max, n)).astype(np.uint64)
    
    timestamp_min = np.array([np.min(timestamps)], dtype=np.int64)
    timestamp_max = np.array([np.max(timestamps)], dtype=np.int64)

    translation_min = np.random.randn(3).astype(np.float32)
    translation_max = np.random.randn(3).astype(np.float32)

    quaternion_min = random_quaternion().astype(np.float32)
    quaternion_max = random_quaternion().astype(np.float32)

    out_points = motion_compensation(
        points,
        timestamps,
        TimeRange(min=timestamp_min[0], max=timestamp_max[0]),
        PoseRange(
            translation_min=translation_min,
            translation_max=translation_max,
            quaternion_min=quaternion_min,
            quaternion_max=quaternion_max
        )
    )

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    points.tofile("./input/input_points.bin")
    timestamps.tofile("./input/input_timestamps.bin")
    timestamp_min.tofile("./input/input_timestamp_min.bin")
    timestamp_max.tofile("./input/input_timestamp_max.bin")
    translation_min.tofile("./input/input_translation_min.bin")
    translation_max.tofile("./input/input_translation_max.bin")
    quaternion_min.tofile("./input/input_quaternion_min.bin")
    quaternion_max.tofile("./input/input_quaternion_max.bin")
    out_points.tofile("./output/golden_out_points.bin")


if __name__ == "__main__":
    gen_golden_data_simple()


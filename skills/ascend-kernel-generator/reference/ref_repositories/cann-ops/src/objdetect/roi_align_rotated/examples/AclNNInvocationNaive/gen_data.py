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

from functools import reduce
import os
import torch
import numpy as np


def roi_align_rotated():
    x_shape = [3, 48, 32, 32]
    rois_shape = [12, 6]
    spatial_scale = 0.25
    sampling_ratio = 2
    pooled_h = 2
    pooled_w = 2
    aligned = True
    clockwise = True
    x_value = np.random.uniform(0.1, 1, x_shape).astype(np.float32)
    rois_value = np.random.uniform(0.1, 1, rois_shape).astype(np.float32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    npu_x_value = x_value.transpose(0, 2, 3, 1)
    npu_rois_value = rois_value.transpose(1, 0)
    npu_x_value.astype(np.float32).tofile("./input/input_x_output.bin")
    npu_rois_value.astype(np.float32).tofile("./input/rois.bin")
    output_type = x_value.dtype
    
    batch = x_shape[0]
    channels = x_shape[1]
    height = x_shape[2]
    width = x_shape[3]
    y_shape = [rois_shape[0], channels, pooled_h, pooled_w]
    number = reduce(lambda x1, x2: x1 * x2, y_shape)
    y = np.zeros(number).astype(np.float32)
    feature_map = x_value.reshape(-1)

    roi_offset = 0.5 if aligned else 0
    roi_batch_idx = rois_value[:, 0]
    roi_center_w = rois_value[:, 1] * spatial_scale - roi_offset
    roi_center_h = rois_value[:, 2] * spatial_scale - roi_offset
    roi_width = rois_value[:, 3] * spatial_scale
    roi_height = rois_value[:, 4] * spatial_scale
    theta = rois_value[:, 5]
    theta = -theta if clockwise else theta

    if not aligned:
        roi_width = np.maximum(roi_width, 1)
        roi_height = np.maximum(roi_height, 1)
    bin_size_h = roi_height / pooled_h
    bin_size_w = roi_width / pooled_w

    if sampling_ratio > 0:
        roi_bin_grid_h = np.ones(bin_size_h.shape).astype("int32")
        roi_bin_grid_w = np.ones(bin_size_w.shape).astype("int32")
        roi_bin_grid_h = roi_bin_grid_h * sampling_ratio
        roi_bin_grid_w = roi_bin_grid_w * sampling_ratio
    else:
        roi_bin_grid_h = np.ceil(bin_size_h).astype("int32")
        roi_bin_grid_w = np.ceil(bin_size_w).astype("int32")
    
    roi_start_h = -roi_height / 2
    roi_start_w = -roi_width / 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    count = np.maximum(roi_bin_grid_h * roi_bin_grid_w, 1)

    for index in range(number):
        pw = index % pooled_w
        ph = (index // pooled_w) % pooled_h
        c = (index // pooled_w // pooled_h) % channels
        n = index // pooled_w // pooled_h // channels
        
        start_h = roi_start_h[n]
        start_w = roi_start_w[n]
        grid_h = roi_bin_grid_h[n]
        grid_w = roi_bin_grid_w[n]
        center_h = roi_center_h[n]
        center_w = roi_center_w[n]
        size_h = bin_size_h[n]
        size_w = bin_size_w[n]
        fm_batch = int(roi_batch_idx[n])
        if 0 <= fm_batch < batch:
            val_n = count[n]
            sin_theta_n = sin_theta[n]
            cos_theta_n = cos_theta[n]
            output_val = 0

            for iy in range(grid_h):
                yy = start_h + ph * size_h + (iy + 0.5) * size_h / grid_h

                for ix in range(grid_w):
                    xx = start_w + pw * size_w +(ix + 0.5) * size_w / grid_w

                    x_val = yy * sin_theta_n + xx * cos_theta_n + center_w
                    y_val = yy * cos_theta_n - xx * sin_theta_n + center_h

                    val = bilinear_interpolate(feature_map, fm_batch, channels, height, width, y_val, x_val, c)

                    output_val += val
            
            output_val = output_val / val_n
            y[index] = output_val

    y = y.reshape(y_shape)
    y.astype(output_type).transpose(0, 2, 3, 1)
    y.tofile("./output/golden.bin")
    return y


def bilinear_interpolate(feature_map, fm_batch, channels, height, width, y_val, x_val, c):
    if y_val < -1.0:
        return 0
    if y_val > height:
        return 0
    if x_val < -1.0:
        return 0
    if x_val > width:
        return 0
    
    if y_val <= 0:
        y_val = 0
        
    if x_val <= 0:
        x_val = 0
    
    y_low = int(y_val)
    x_low = int(x_val)

    if y_low >= height - 1:
        y_high = y_low = height - 1
        y_val = y_low.astype(np.float32)
    else:
        y_high = y_low + 1
    
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x_val = x_low.astype(np.float32)
    else:
        x_high = x_low + 1
    
    ly = y_val - y_low
    lx = x_val - x_low
    hy = 1 - ly
    hx = 1 - lx

    fm_idx = (fm_batch * channels + c) * height * width
    v1 = feature_map[fm_idx + y_low * width + x_low]
    v2 = feature_map[fm_idx + y_low * width + x_high]
    v3 = feature_map[fm_idx + y_high * width + x_low]
    v4 = feature_map[fm_idx + y_high * width + x_high]

    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

    return val


if __name__ == "__main__":
    roi_align_rotated()
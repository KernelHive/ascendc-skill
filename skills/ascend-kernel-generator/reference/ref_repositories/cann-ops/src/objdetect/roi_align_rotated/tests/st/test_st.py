import copy
import math
import unittest
from functools import reduce
from typing import List

import numpy as np
import torch
EPS = 1e-8


def roi_align_rotated(input_array, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, aligned, clockwise):
    pooled_height = pooled_h
    pooled_width = pooled_w
    N, C, H, W = input_array.shape
    output_shape = [rois.shape[0], C, pooled_height, pooled_width]
    number = reduce(lambda x1, x2: x1 * x2, output_shape)
    output = np.zeros(number).astype(np.float32)
    feature_map = input_array.reshape(-1)

    roi_offset = 0.5 if aligned else 0
    roi_batch_idx = rois[:, 0]
    roi_center_w = rois[:, 1] * spatial_scale - roi_offset
    roi_center_h = rois[:, 2] * spatial_scale - roi_offset
    roi_width = rois[:, 3] * spatial_scale
    roi_height = rois[:, 4] * spatial_scale
    theta = rois[:, 5]
    theta = -theta if clockwise else theta

    if not aligned:
        roi_width = np.maximum(roi_width, 1)
        roi_height = np.maximum(roi_height, 1)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

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
        pw = index % pooled_width
        ph = (index // pooled_width) % pooled_height
        c = (index // pooled_width // pooled_height) % C
        n = index // pooled_width // pooled_height // C

        start_h = roi_start_h[n]
        start_w = roi_start_w[n]
        grid_h = roi_bin_grid_h[n]
        grid_w = roi_bin_grid_w[n]
        center_h = roi_center_h[n]
        center_w = roi_center_w[n]
        size_h = bin_size_h[n]
        size_w = bin_size_w[n]

        fm_batch = int(roi_batch_idx[n])

        if 0 <= fm_batch < N:
            val_n = count[n]
            sin_theta_n = sin_theta[n]
            cos_theta_n = cos_theta[n]
            output_val = 0

            for iy in range(grid_h):
                yy = start_h + ph * size_h + (iy + 0.5) * size_h / grid_h

                for ix in range(grid_w):
                    xx = start_w + pw * size_w + (ix + 0.5) * size_w / grid_w

                    x_val = yy * sin_theta_n + xx * cos_theta_n + center_w
                    y_val = yy * cos_theta_n - xx * sin_theta_n + center_h
                    
                    bilinear_dict = dict(C=C,
                                         H=H,
                                         W=W,
                                         y_val=y_val,
                                         x_val=x_val,
                                         c=c)
                    val = bilinear_interpolate(feature_map, fm_batch, bilinear_dict)

                    output_val += val
            
            output_val = output_val / val_n
            output[index] = output_val

    output = output.reshape(output_shape)

    return output


def bilinear_interpolate(feature_map, fm_batch, bilinear_args):
    channels, height, width, y_val, x_val, c = bilinear_args.values()

    if y_val < -1.0 or y_val > height:
        return 0
    if x_val < -1.0 or x_val > width:
        return 0
    if y_val <= 0:
        y_val = 0
    if x_val <= 0:
        x_val = 0
    
    y_low = int(y_val)
    x_low = int(x_val)

    if y_low >= height - 1:
        y_high = y_low = height - 1
        y_val = y_low
    else:
        y_high = y_low + 1
    
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x_val = x_low
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


def calc_expect_func(x, rois, y, pooled_h, pooled_w, spatial_scale, sampling_ratio, aligned, clockwise):
    res = roi_align_rotated(x["value"], rois["value"], y["value"], pooled_h["value"], pooled_w["value"], spatial_scale["value"], sampling_ratio["value"], clockwise["value"])
    return [res]
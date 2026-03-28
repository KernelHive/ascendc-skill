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
import torch
import numpy as np

def write_file(inputs, outputs):
    for input_name in inputs:
        inputs[input_name].tofile(os.path.join(input_name + ".bin"))
    for output_name in outputs:
        outputs[output_name].tofile(os.path.join(output_name + ".bin")) 


def cal_grad_value(num_point, channels, v1_ub, offset1, offset2, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                   grad_w_weight_ub, h_w_w1, h_w_w2, grad_value, top_grad_value_ub, w_weight, cp, neg_w1, neg_w2):
    ptr = offset1[cp] + offset2[cp] + base_ptr
    v1_ub[:] = value[value_ptr_offset + ptr: value_ptr_offset + ptr + channels]
    if neg_w1:
        grad_h_weight_ub[cp,:] -= v1_ub * h_w_w1[cp]
    else:
        grad_h_weight_ub[cp,:] += v1_ub * h_w_w1[cp]
    if neg_w2:
        grad_w_weight_ub[cp,:] -= v1_ub * h_w_w2[cp]
    else:
        grad_w_weight_ub[cp,:] += v1_ub * h_w_w2[cp]
    mid_ub = top_grad_value_ub[cp, :] * w_weight[cp]
    grad_value[value_ptr_offset + ptr: value_ptr_offset + ptr + channels] += mid_ub


def ms_scale_deform_att_grad(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
                             grad_value, grad_sampling_loc_out, grad_attn_weight):

    batch_size = value.shape[0]
    spatial_size = value.shape[1]
    num_heads = value.shape[2]
    channels = value.shape[3]
    num_levels = spatial_shapes.shape[0]
    num_query = sampling_loc.shape[1]
    num_point = sampling_loc.shape[5]
    print(batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point)
    value = value.flatten()
    grad_attn_weight = grad_attn_weight.flatten()
    grad_sampling_loc_out = grad_sampling_loc_out.flatten()
    grad_value = grad_value.flatten()
    for b in range(batch_size):
        for nq in range(num_query):
            for nh in range(num_heads):
                data_weight_ptr = (b * num_query * num_heads + nq * num_heads + nh) * num_levels * num_point
                data_loc_w_ptr = 2 * data_weight_ptr
                top_grad_ub = grad_output[b, nq, nh, :]
                for nl in range(num_levels):
                    level_start_id = level_start_index[nl]
                    h = spatial_shapes[nl, 0]
                    w = spatial_shapes[nl, 1]
                    qid_stride = num_heads * channels
                    data_value_ptr_init_offset = b * spatial_size * qid_stride
                    value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride
                    loc_w = sampling_loc[b, nq, nh, nl, 0, :]
                    loc_h = sampling_loc[b, nq, nh, nl, 1, :]
                    weight = attn_weight[b, nq, nh, nl, :]
                    h_im = loc_h * h - 0.5
                    w_im = loc_w * w - 0.5
                    h_low = np.floor(h_im).astype(int)
                    w_low = np.floor(w_im).astype(int)
                    h_high = h_low + 1
                    w_high = w_low + 1
                    lh = h_im - h_low
                    lw = w_im - w_low
                    hh = 1 - lh
                    hw = 1 - lw 
                    w_stride = num_heads * channels
                    h_stride = w * w_stride
                    h_low_ptr_offset = h_low * h_stride
                    h_high_ptr_offset = h_low_ptr_offset + h_stride
                    w_low_ptr_offset = w_low * w_stride
                    w_high_ptr_offset = w_low_ptr_offset + w_stride
                    base_ptr = nh * channels
                    w1 = hh * hw
                    w2 = hh * lw
                    w3 = lh * hw
                    w4 = lh * lw 
                    grad_h_weight_ub = np.zeros((num_point, channels))
                    grad_w_weight_ub = np.zeros((num_point, channels))
                    top_grad_value_ub = np.zeros((num_point, channels))
                    grad_weight_ub = np.zeros((num_point, channels))
                    for cp in range(num_point):
                        if h_im[cp] > -1 and w_im[cp] > -1 and h_im[cp] < h and w_im[cp] < w:
                            top_grad_value_ub[cp, :] = weight[cp] * top_grad_ub
                            v1_ub = np.zeros((channels,))
                            if h_low[cp] >= 0 and w_low[cp] >= 0:
                                cal_grad_value(num_point, channels, v1_ub, h_low_ptr_offset, w_low_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, hw, hh, grad_value, top_grad_value_ub, w1, cp, True, True)
                            v2_ub = np.zeros((channels,))
                            if h_low[cp] >= 0 and w_high[cp] < w:
                                cal_grad_value(num_point, channels, v2_ub, h_low_ptr_offset, w_high_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, lw, hh, grad_value, top_grad_value_ub, w2, cp, True, False)
                            v3_ub = np.zeros((channels,))
                            if h_high[cp] < h and w_low[cp] >= 0:
                                cal_grad_value(num_point, channels, v3_ub, h_high_ptr_offset, w_low_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, hw, lh, grad_value, top_grad_value_ub, w3, cp, False, True)
                            v4_ub = np.zeros((channels,))
                            if h_high[cp] < h and w_high[cp] < w:
                                cal_grad_value(num_point, channels, v4_ub, h_high_ptr_offset, w_high_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, lw, lh, grad_value, top_grad_value_ub, w4, cp, False, False)
                            val = (w1[cp] * v1_ub + w2[cp] * v2_ub + w3[cp] * v3_ub + w4[cp] * v4_ub)
                            grad_weight_ub[cp, :] = top_grad_ub * val
                    grad_sample_x_loc_ub = top_grad_value_ub * grad_w_weight_ub * w
                    grad_sample_y_loc_ub = top_grad_value_ub * grad_h_weight_ub * h
                    x = np.sum(grad_sample_x_loc_ub, axis=-1)
                    y = np.sum(grad_sample_y_loc_ub, axis=-1)
                    weight_sum = np.sum(grad_weight_ub, axis=-1)
                    grad_attn_weight[data_weight_ptr + nl * num_point: data_weight_ptr + (nl + 1) * num_point] += weight_sum
                    grad_sampling_loc_out[data_loc_w_ptr + nl * 2 * num_point: data_loc_w_ptr + nl * 2 * num_point + num_point] += x
                    grad_sampling_loc_out[data_loc_w_ptr + nl * 2 * num_point + num_point: data_loc_w_ptr + nl * 2 * num_point + 2 * num_point] += y
    grad_sampling_loc_out = grad_sampling_loc_out.reshape((batch_size, num_query, num_heads, num_levels, 2, num_point)).transpose((0, 1, 2, 3, 5, 4))
    grad_value = grad_value.reshape((batch_size, spatial_size, num_heads, channels))
    grad_attn_weight = grad_attn_weight.reshape((batch_size, num_query, num_heads, num_levels, num_point))
    return grad_value, grad_sampling_loc_out, grad_attn_weight


if __name__ == "__main__":
    bs = 1
    num_heads = 8
    channels = 2
    num_levels = 1
    num_points = 2
    num_queries = 100
    
    cpu_shapes = torch.tensor([6, 4] * num_levels).reshape(num_levels, 2).int()
    cpu_shapes_numpy = cpu_shapes.numpy()
    num_keys = sum((H * W).item() for H, W in cpu_shapes)

    cpu_value = torch.rand(bs, num_keys, num_heads, channels) * 0.01
    cpu_value_numpy = cpu_value.float().numpy()
    cpu_sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2).float()
    cpu_sampling_locations_numpy = cpu_sampling_locations.numpy()
    cpu_attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
    cpu_attention_weights_numpy = cpu_attention_weights.float().numpy()

    cpu_offset = torch.cat((cpu_shapes.new_zeros((1, )), cpu_shapes.prod(1).cumsum(0)[:-1]))
    cpu_offset_nunmpy = cpu_offset.int().numpy()
    grad_output = np.ones((bs, num_queries, num_heads, channels)).astype(np.float32)
    grad_value = np.zeros_like(cpu_value_numpy)
    grad_sample_loc = np.zeros_like(cpu_sampling_locations_numpy)
    grad_atten_weight = np.zeros_like(cpu_attention_weights_numpy)
    output1, output2, output3 = ms_scale_deform_att_grad(cpu_value_numpy, cpu_shapes_numpy, cpu_offset_nunmpy, cpu_sampling_locations_numpy,
        cpu_attention_weights_numpy, grad_output, grad_value, grad_sample_loc, grad_atten_weight)
    
    inputs_npu = {"value":cpu_value_numpy ,"value_spatial_shapes": cpu_shapes_numpy, "level_start_index":cpu_offset_nunmpy, "sample_loc": cpu_sampling_locations_numpy,
                  "attention_weight":cpu_attention_weights_numpy, "grad_output": grad_output}
    outputs_npu = {"grad_value":output1, "grad_sample_loc":output2, "grad_attn_weight":output3}
    write_file(inputs_npu, outputs_npu)

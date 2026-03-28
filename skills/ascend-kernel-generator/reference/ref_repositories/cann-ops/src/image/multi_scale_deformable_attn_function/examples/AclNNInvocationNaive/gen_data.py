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
import torch
import torch.nn.functional as F

def write_file(inputs, outputs):
    for input_name in inputs:
        inputs[input_name].tofile(os.path.join(input_name + ".bin"))
    for output_name in outputs:
        outputs[output_name].tofile(os.path.join(output_name + ".bin")) 

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()

if __name__ == "__main__":
    # 1 8 64 1 2 100
    bs = 1
    num_heads = 8
    channels = 64
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

    output_npu = multi_scale_deformable_attn_pytorch(cpu_value, cpu_shapes, cpu_sampling_locations, cpu_attention_weights).numpy()
        
    
    inputs_npu = {"value":cpu_value_numpy ,"value_spatial_shape": cpu_shapes_numpy,
                   "level_start_index":cpu_offset_nunmpy, "sample_loc": cpu_sampling_locations_numpy,
                  "attention_weight":cpu_attention_weights_numpy}
    outputs_npu = {"output":output_npu}
    write_file(inputs_npu, outputs_npu)
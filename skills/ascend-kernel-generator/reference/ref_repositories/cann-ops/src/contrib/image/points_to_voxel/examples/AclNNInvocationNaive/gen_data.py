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

import numpy as np
import os


class Inputs:
    def __init__(self, points, voxel_size, coors_range, max_points, reverse_index, max_voxels):
        self.points = points
        self.voxel_size = voxel_size
        self.coors_range = coors_range
        self.max_points = max_points
        self.reverse_index = reverse_index
        self.max_voxels = max_voxels


class ProcessArgs:
    def __init__(self, points, voxel_size, coors_range, num_points_per_voxel, 
                 coor_to_voxelidx, voxels, coors, max_points, reverse_index, max_voxels):
        self.points = points
        self.voxel_size = voxel_size
        self.coors_range = coors_range
        self.num_points_per_voxel = num_points_per_voxel
        self.coor_to_voxelidx = coor_to_voxelidx
        self.voxels = voxels
        self.coors = coors        
        self.max_points = max_points
        self.reverse_index = reverse_index
        self.max_voxels = max_voxels


def _points_to_voxel_kernel(processargs):
    n = processargs.points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (processargs.coors_range[3:] - processargs.coors_range[:3]) / processargs.voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(n):
        failed = False
        for j in range(ndim):
            c = np.floor((processargs.points[i, j] - processargs.coors_range[j]) / processargs.voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            if processargs.reverse_index:
                coor[ndim_minus_1 - j] = c
            else:    
                coor[j] = c
        if failed:
            continue
        voxelidx = processargs.coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= processargs.max_voxels:
                continue 
            voxel_num += 1
            processargs.coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            processargs.coors[voxelidx] = coor
        num = processargs.num_points_per_voxel[voxelidx]
        if num < processargs.max_points:
            processargs.voxels[voxelidx, num] = processargs.points[i]
            processargs.num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(inputs):
    if not isinstance(inputs.voxel_size, np.ndarray):
        inputs.voxel_size = np.array(inputs.voxel_size, dtype=inputs.points.dtype)
    if not isinstance(inputs.coors_range, np.ndarray):
        inputs.coors_range = np.array(inputs.coors_range, dtype=inputs.points.dtype)
    voxelmap_shape = (inputs.coors_range[3:] - inputs.coors_range[:3]) / inputs.voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if inputs.reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(inputs.max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(inputs.max_voxels, inputs.max_points, inputs.points.shape[-1]), dtype=inputs.points.dtype
    )
    coors = np.zeros(shape=(inputs.max_voxels, 3), dtype=np.int32)
    processargs = ProcessArgs(inputs.points, inputs.voxel_size, inputs.coors_range, 
                              num_points_per_voxel, coor_to_voxelidx,
                              voxels, coors, inputs.max_points, inputs.reverse_index, inputs.max_voxels) 
    voxel_num = _points_to_voxel_kernel(processargs)

    processargs.coors = processargs.coors[:voxel_num]
    processargs.voxels = processargs.voxels[:voxel_num]
    processargs.num_points_per_voxel = processargs.num_points_per_voxel[:voxel_num]
    return processargs.voxels, processargs.coors, processargs.num_points_per_voxel


if __name__ == '__main__':
    shape_x = [1000, 4]
    points = np.random.uniform(0, 200, shape_x).astype(np.float32)
    voxel_size = np.array([1.3, 1.5, 1.6]).astype(np.float32) 
    coors_range = np.array([1.1, 1.3, 1.4, 98.4, 87.6, 103.7]).astype(np.float32) 
    inputs = Inputs(points, voxel_size, coors_range, max_points=5, reverse_index=False, max_voxels=500)
    voxels, coors, num_points_per_voxel = points_to_voxel(inputs)

    points_input = points.T

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    points_input.tofile("./input/input_points.bin")
    voxels.tofile("./output/voxels.bin")
    coors.tofile("./output/coors.bin")
    num_points_per_voxel.tofile("./output/num_points_per_voxel.bin")

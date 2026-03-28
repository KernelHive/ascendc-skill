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

dtype_map = {torch.float16: np.float16, torch.float32: np.float32, torch.int32: np.int32, torch.int64: np.int64}


def sparse_flatten_indices(indices, size):
    sparse_dim = indices.shape[0]
    if sparse_dim == 1:
        return torch.squeeze(indices, dim=0)
    else:
        if not torch.numel(indices):
            return torch.zeros(indices.shape[1], dtype=indices.dtype)
        
        flatten_indices = []
        slice_size = [i for i in size][0:sparse_dim][::-1]
        slice_size = torch.cumprod(torch.tensor(slice_size), 0)
        slice_size = torch.flip(torch.cat((torch.tensor([1]), slice_size)), dims=[0])
        start_idx = slice_size.shape[0] - sparse_dim
        end_idx = slice_size.shape[0]
        slice_size = slice_size[start_idx:end_idx]
        for n in range(indices.shape[1]):
            tmp = 0
            for m in range(sparse_dim):
                tmp += indices[m][n].item() * slice_size[m]
            flatten_indices.append(tmp)
        return torch.tensor(flatten_indices, dtype=indices.dtype)

def gen_golden_data_simple():
    m = 8
    n = 13
    values_size = [n, 7, 2, 3]
    values_dtype = torch.float16
    indices_dtype = torch.int32

    indices = torch.randint(0, 10, [m, n], dtype=indices_dtype)
    values = torch.randn(values_size, dtype=values_dtype)

    max_values, _ = torch.max(indices, dim=1)
    max_indices = (max_values + 1).tolist()
    sparse_dim = indices.shape[0]
    dense_dim = values.dim() - 1
    start_idx = len(max_indices)
    for i in range(start_idx, sparse_dim + dense_dim):
        max_indices.append(values.shape[i - start_idx + 1])

    sparse_tensor_cpu = torch.sparse_coo_tensor(indices, values, max_indices)

    indices_flatten = sparse_flatten_indices(indices, sparse_tensor_cpu.shape)
    unique_len, unique_indices = torch.unique(indices_flatten, True, True)
    new_nnz = unique_len.shape[0]
    new_indices_size = [indices.shape[0], new_nnz]
    new_values_size = [i for i in values.shape]
    new_values_size[0] = new_nnz

    coalesced_tensor_cpu = sparse_tensor_cpu.coalesce()
    new_indices = coalesced_tensor_cpu.indices()
    new_values = coalesced_tensor_cpu.values()


    os.system("mkdir -p input")
    os.system("mkdir -p output")
    unique_len.numpy().astype(dtype_map[indices_dtype]).tofile("./input/input_unique_len.bin")
    unique_indices.numpy().astype(dtype_map[indices_dtype]).tofile("./input/input_unique_indices.bin")
    indices = torch.transpose(indices, 0, 1)
    indices.numpy().astype(dtype_map[indices_dtype]).tofile("./input/input_indices.bin")
    values.numpy().astype(dtype_map[values_dtype]).tofile("./input/input_values.bin")
    np.array([i for i in unique_len.shape], dtype=np.int64).tofile("./input/unique_len_shape.bin")
    np.array([i for i in unique_indices.shape], dtype=np.int64).tofile("./input/unique_indices_shape.bin")
    np.array([i for i in indices.shape], dtype=np.int64).tofile("./input/indices_shape.bin")
    np.array([i for i in values.shape], dtype=np.int64).tofile("./input/values_shape.bin")
    np.array(new_indices_size, dtype=np.int64).tofile("./input/new_indices_shape.bin")
    np.array(new_values_size, dtype=np.int64).tofile("./input/new_values_shape.bin")

    new_indices = torch.transpose(new_indices, 0, 1)
    new_indices.numpy().astype(dtype_map[indices_dtype]).tofile("./output/golden_new_indices.bin")
    new_values.numpy().astype(dtype_map[values_dtype]).tofile("./output/golden_new_values.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

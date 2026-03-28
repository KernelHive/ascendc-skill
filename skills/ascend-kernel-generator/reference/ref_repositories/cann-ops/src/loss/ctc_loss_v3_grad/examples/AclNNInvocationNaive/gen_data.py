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

def gen_golden_data_simple():
    dtype = np.float16
    num_class = 10000
    batch = 20
    max_target = 10
    max_time = 12
    alpha = 2 * max_target + 1
    grad_dtype = np.float32
    target_dtype = np.int64

    grad_out_shape = [batch]
    log_probs_shape = [max_time, batch, num_class]
    targets_shape = [batch, max_target]
    input_lengths_shape = [batch]
    target_lengths_shape = [batch]
    neg_log_likelihood_shape = [batch]
    log_alpha_shape = [batch, max_time, alpha]
    grad_out = np.random.uniform(0, 1, grad_out_shape).astype(grad_dtype)
    log_probs = np.random.uniform(-2, -1, log_probs_shape).astype(grad_dtype)
    targets = np.random.uniform(0, log_probs_shape[2], targets_shape).astype(target_dtype)
    input_lengths_min = 1
    input_lengths_max = log_probs_shape[0]
    input_lengths = np.random.uniform(input_lengths_min, input_lengths_max, input_lengths_shape).astype(target_dtype)
    input_lengths[0] = log_probs_shape[0]
    target_lengths = np.random.uniform(targets_shape[1], targets_shape[1], target_lengths_shape).astype(target_dtype)
    target_lengths[0] = targets_shape[1]
    neg_log_likelihood = np.random.uniform(0, 1, neg_log_likelihood_shape).astype(grad_dtype)
    log_alpha = np.random.uniform(0, 1, log_alpha_shape).astype(grad_dtype)

    grad_out_cpu = torch.from_numpy(grad_out)
    log_probs_cpu = torch.from_numpy(log_probs)
    targets_cpu = torch.from_numpy(targets)
    neg_log_likelihood_cpu = torch.from_numpy(neg_log_likelihood)
    log_alpha_cpu = torch.from_numpy(log_alpha)
    grad_cpu = torch.ops.aten._ctc_loss_backward(grad_out_cpu, log_probs_cpu, targets_cpu, input_lengths,
                                                 target_lengths, neg_log_likelihood_cpu, log_alpha_cpu, blank=0,
                                                 zero_infinity=False)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    grad_out.tofile("./input/input_grad_out.bin")
    log_probs.tofile("./input/input_log_probs.bin")
    targets.tofile("./input/input_targets.bin")
    input_lengths.tofile("./input/input_input_lengths.bin")
    target_lengths.tofile("./input/input_target_lengths.bin")
    neg_log_likelihood.tofile("./input/input_neg_log_likelihood.bin")
    log_alpha.tofile("./input/input_log_alpha.bin")
    grad_cpu.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


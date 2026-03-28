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


def cross_entropy_loss_grad(batch_size, num_classes, y_grad, log_softmax, 
                            target, weight, reduction, ignore_index, label_smoothing=0.0):
    log_softmax = torch.from_numpy(log_softmax.astype(np.float32))
    y_grad = torch.from_numpy(y_grad.astype(np.float32))  # grad_loss
    weight = torch.from_numpy(weight.astype(np.float32))
    target = torch.from_numpy(target.astype(np.int64))
    weight_yn = torch.gather(weight, 0, target)  # 根据target取出元素, [N]
    if ignore_index >= 0:
        ignore_mask = torch.tensor(target - ignore_index, dtype=torch.bool).float()
        
    else:
        ignore_mask = torch.ones((batch_size,))
    if reduction == "mean":
        mean_out_grad = y_grad * (1 - label_smoothing) # []
        weight_after_mask = weight_yn * ignore_mask # [N]
        weight_after_mask_sum = torch.sum(weight_after_mask, -1, keepdim=False) # []
        smooth_loss_grad = y_grad / weight_after_mask_sum * label_smoothing / num_classes # []
        loss_out_grad = mean_out_grad.unsqueeze(-1) / weight_after_mask_sum  # [N]


    elif reduction == "sum":
        sum_out_grad = y_grad * (1 - label_smoothing) # []
        smooth_loss_grad = y_grad.unsqueeze(-1) * label_smoothing / num_classes # [N]
        loss_out_grad = sum_out_grad.unsqueeze(-1)  # [N]
    else:
        none_out_grad = y_grad * (1 - label_smoothing)   # [N]
        smooth_loss_grad = y_grad * label_smoothing / num_classes   # [N]
        loss_out_grad = none_out_grad # [N]

    loss_out_grad = loss_out_grad * ignore_mask # [N]

    nll_loss_grad = loss_out_grad * weight_yn # [N]

    log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_softmax) * nll_loss_grad.unsqueeze(-1)
    predictions_grad_loss_out = torch.zeros((batch_size, num_classes)).float()
    for i in range(batch_size):
        predictions_grad_loss_out[i][target[i]] = nll_loss_grad[i]    # scalar操作，getvalue，setvalue

    predictions_grad_loss_out = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out
    return predictions_grad_loss_out.to(torch.float16)


def gen_golden_data_simple():
    batch_size = 4096
    num_classes = 1024
    data_shape = (batch_size, num_classes)
    ignore_index = -100
    label_smoothing = 0.0
    reduction = "mean"

    grad_loss = np.random.uniform(-1, 1, ()).astype(np.float16)
    log_softmax = np.random.randint(-10, 0, data_shape).astype(np.float16)
    target = np.random.randint(0, num_classes, (data_shape[0],)).astype(np.int64)
    weight = np.random.uniform(0.01, 1, (data_shape[1],)).astype(np.float32)

    golden_out = cross_entropy_loss_grad(batch_size, num_classes, grad_loss, log_softmax, 
                                         target, weight, reduction, ignore_index, label_smoothing=0.0)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    grad_loss.tofile("./input/grad_loss.bin")
    log_softmax.tofile("./input/log_softmax.bin")
    target.tofile("./input/target.bin")
    weight.tofile("./input/weight.bin")
    golden_out.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


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


def cross_entropy_loss_forward(n, c, predictions, targets, weight, ignore_index, label_smoothing, reduction):
    input_dtype = predictions.dtype
    if weight is None:
        weight = torch.ones((c,))
    predictions = predictions.to(torch.float32)
    predictions_max = torch.max(predictions, dim=1)[0].unsqueeze(-1)
    log_softmax_probs = predictions - predictions_max - \
    torch.log(torch.sum(torch.exp(predictions - predictions_max), -1, keepdim=True))  # log_softmax结果, [N,C]
    nll_loss = torch.gather(log_softmax_probs, 1, targets.reshape(-1, 1)).reshape(-1) 
    weight_yn = torch.gather(weight, 0, targets)
    loss_out = -nll_loss * weight_yn
    if ignore_index >= 0: 
        ignore_mask = (targets - ignore_index).bool().float()
    else:
        ignore_mask = torch.ones((n,))
    loss_out = loss_out * ignore_mask

    smooth_loss = -torch.sum(log_softmax_probs * weight.unsqueeze(0), -1, keepdim=False)
    if ignore_index >= 0: 
        smooth_loss = smooth_loss * ignore_mask 
    if reduction == "mean":
        weight_after_mask = weight_yn * ignore_mask
        mean_out = torch.sum(loss_out, -1, keepdim=False) / torch.sum(weight_after_mask, -1, keepdim=False)
        ret = (1 - label_smoothing) * mean_out + torch.sum(smooth_loss, -1, keepdim=False) / \
        torch.sum(weight_after_mask, -1, keepdim=False) * label_smoothing / c
    elif reduction == "sum":
        sum_out = torch.sum(loss_out, -1, keepdim=False)
        ret = (1 - label_smoothing) * sum_out + torch.sum(smooth_loss, -1, keepdim=False) * label_smoothing / c
    else:
        none_out = loss_out
        ret = (1 - label_smoothing) * none_out + smooth_loss * label_smoothing / c
    return ret.to(input_dtype), log_softmax_probs.to(input_dtype)


def gen_golden_data_simple():
    input_dtype = torch.float16
    target_dtype = torch.int64
    weight_dtype = torch.float32

    batch = 4096
    target_num = 1024
    ignore_index = -100
    label_smoothing = 0.0
    reduction = "mean"

    input_logic = torch.randn([batch, target_num]).to(input_dtype)
    target = torch.randint(low=1, high=target_num, size=(batch,)).to(target_dtype)
    weight = torch.randn(target_num).to(weight_dtype)

    loss, log_prob = cross_entropy_loss_forward(batch, target_num, input_logic, target, weight, 
                                             ignore_index, label_smoothing, reduction)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_logic.numpy().tofile("./input/input.bin")
    target.numpy().tofile("./input/target.bin")
    weight.numpy().tofile("./input/weight.bin")
    loss.numpy().tofile("./output/golden_loss.bin")
    log_prob.numpy().tofile("./output/golden_logProb.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


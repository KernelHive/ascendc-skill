#!/usr/bin/python3
# coding=utf-8
#
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

def softmax(x):
    """
    实现 softmax 函数
    """
    softmax_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_value = np.sum(exp_x, axis=-1, keepdims=True)
    exp_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x, softmax_max, sum_value


def flash_attention_score_test(query, key, value, scale_value=0.0625, head_num=576):
    scores = np.dot(query.astype(np.float32), key.transpose(0,2,1).astype(np.float32))
    scores = scores * scale_value
    attention_weights, softmax_max, softmax_sum = softmax(scores)
    output = np.dot(attention_weights, value.astype(np.float32))
    output = output.astype(np.float16) 
    return softmax_max, softmax_sum, output


def calc_expect_func(query, key, value, scale_value=0.0625, head_num=576, softmax_max=None, softmax_sum=None, attention_out=None):
    res1, res2, res3 = flash_attention_score_test(query["value"], key["value"], value["value"], scale_value, head_num)
    return [res1, res2, res3]
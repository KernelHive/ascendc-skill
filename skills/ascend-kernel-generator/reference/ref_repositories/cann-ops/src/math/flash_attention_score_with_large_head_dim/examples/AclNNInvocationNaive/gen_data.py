#!/bin/bash
#
# Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os

def softmax(x):
    """
    实现 softmax 函数
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_value = np.sum(exp_x, axis=-1, keepdims=True)
    exp_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x

def gen_golden_data_simple():
    # 定义数组的形状
    shape = (1, 2048, 576)
    # 创建query、key和value数组，所有元素的值都为3，数据类型为float16
    query = np.random.uniform(1, 10, [1, 2048, 576]).astype(np.float16)
    key = np.random.uniform(1, 10, [1, 2048, 576]).astype(np.float16)
    value = np.random.uniform(1, 10, [1, 2048, 576]).astype(np.float16)
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    scaleValue = 0.0625
    scores = np.dot(query.astype(np.float32), key.transpose(0,2,1).astype(np.float32))
    scores = scores * scaleValue
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, value.astype(np.float32))
    output = output.astype(np.float16) 
    print(output)

    query.tofile("./input/input_q.bin")
    key.tofile("./input/input_k.bin")
    value.tofile("./input/input_v.bin")
    output.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


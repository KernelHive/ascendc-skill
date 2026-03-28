#!/usr/bin/python3
# -*- coding:utf-8 -*-
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
import os
import sys
import numpy as np

LOSS = 1e-3
MINIMUM = 10e-10

def verify_result(real_result, golden):
    x1_dtype = np.int8
    x2_dtype = np.int8
    y_dtype = np.int8

    input_x1 = np.fromfile("input/input_x1.bin", dtype=x1_dtype)
    input_x2 = np.fromfile("input/input_x2.bin", dtype=x2_dtype)
    
    real_result = np.fromfile(real_result, dtype=y_dtype) 
    golden = np.fromfile(golden, dtype=y_dtype)
    golden = golden.astype(np.float32)

    result = np.abs(real_result - golden) 
    deno = np.maximum(np.abs(real_result), np.abs(golden))  
    result_atol = np.less_equal(result, LOSS) 
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS) 
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * LOSS and \
           np.sum(result_atol == False) > real_result.size * LOSS: 
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2])

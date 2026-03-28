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
import numpy as np

def gen_golden_data_simple():
    input_shape_x1 = [2, 20]
    input_shape_x2 = [2, 20]
    input_x1 = np.random.randint(0, 2, size=input_shape_x1).astype(np.bool_)
    input_x2 = np.random.randint(0, 2, size=input_shape_x2).astype(np.bool_)
    golden = np.logical_or(input_x1, input_x2).astype(np.bool_)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
    



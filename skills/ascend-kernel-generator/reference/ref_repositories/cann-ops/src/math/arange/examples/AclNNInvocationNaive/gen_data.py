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
    input_data = np.array([1, -1113, -5], dtype=np.int32)
    golden_data = np.arange(1, -1113, -5, dtype=np.int32)
    os.system("mkdir -p output")
    os.system("mkdir -p input")
    input_data.tofile("./input/input.bin")
    golden_data.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

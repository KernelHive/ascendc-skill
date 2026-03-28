#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data_simple():
    dtype = np.float16
    output_shape = [100, 100]
    input_x = np.random.uniform(1, 4, output_shape).astype(dtype)
    golden = (1.0 / input_x).astype(dtype)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

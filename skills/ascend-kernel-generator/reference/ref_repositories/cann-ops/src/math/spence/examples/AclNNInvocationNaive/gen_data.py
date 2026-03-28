
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================
import os
from math import log
import numpy as np

import tensorflow as tf



A = [4.65128586073990045278E-5, 7.31589045238094711071E-3, 1.33847639578309018650E-1, 
     8.79691311754530315341E-1, 2.71149851196553469920E0, 4.25697156008121755724E0, 
     3.29771340985225106936E0, 1.00000000000000000126E0]
B = [6.90990488912553276999E-4, 2.54043763932544379113E-2, 2.82974860602568089943E-1,
     1.41172597751831069617E0, 3.63800533345137075418E0, 5.03278880143316990390E0,
     3.54771340985225096217E0, 9.99999999999999998740E-1]
PIFS = 1.64493406684822643647


def polevlf(x, coef):  
    ans = 0  
    for c in coef:  
        ans = ans * x + c  
    return ans  


def spence(x):
    w, y, z = 0, 0, 0
    flag = 0

    if x == 1.0:
        return 0.0
    if x == 0.0:
        return PIFS
    if x > 2.0:
        x = 1.0 / x
        flag |= 2
    if x > 1.5:
        w = 1.0 / x - 1.0
        flag |= 2
    elif x < 0.5:
        w = - x
        flag |= 1
    else:
        w = x - 1.0
    y = -w * polevlf(w, A) / polevlf(w, B)
    if flag & 1:
        y = PIFS - log(x) * log(1.0 - x) - y
    if flag & 2:
        z = log(x)
        y = - 0.5 * z * z - y
    return y

def gen_golden_data_simple():
    input_x = np.random.uniform(0.0, 10.0, [4095]).astype(np.float16)
    print(input_x[:8])
    golden = tf.math.special.spence(input_x.astype(np.float32)).numpy().astype(np.float16)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

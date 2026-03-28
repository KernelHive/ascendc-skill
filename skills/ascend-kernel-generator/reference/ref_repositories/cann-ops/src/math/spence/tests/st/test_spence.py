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
from scipy import special  
import numpy as np 
import tensorflow as tf


def spence_test(x):  
    original_dtype = x.dtype  
    if original_dtype != np.float32:  
        x = tf.cast(x, tf.float32).numpy()  
    
    res = special.spence(x)  
    
    # 将结果转换回原始数据类型  
    if original_dtype != np.float32:  
        res = tf.cast(tf.convert_to_tensor(res), tf.dtypes.as_dtype(original_dtype)).numpy()  
    
    return res  


def calc_expect_func(x, y):  
    # 使用scipy的spence函数计算  
    result = spence_test(x["value"])  
    
    return [result]  
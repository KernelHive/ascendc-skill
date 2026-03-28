#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np

def gelu_test(x_np):
    # 保存原始数据类型
    original_dtype = x_np.dtype

    # 如果输入数据不是 float32 类型，则使用 TensorFlow 转换为 float32
    if original_dtype != np.float32:
        x_np = tf.cast(x_np, tf.float32).numpy()

    # 使用 NumPy 接口进行计算
    x = x_np
    golden = x / (1 + np.exp(-1.595769122 * (x + 0.0455399241 * x**3)))

    # 如果原始数据类型不是 float32，则将结果转换回原始数据类型
    if original_dtype != np.float32:
        golden = tf.cast(golden, original_dtype).numpy()

    return golden

def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    res = gelu_test(x["value"])
    return [res]

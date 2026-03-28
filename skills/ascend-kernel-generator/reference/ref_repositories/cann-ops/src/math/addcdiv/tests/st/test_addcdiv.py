#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np


def addcmul_test(input_data, x1, x2, value):
    input_data_tensor = tf.convert_to_tensor(input_data)
    x1_tensor = tf.convert_to_tensor(x1)
    x2_tensor = tf.convert_to_tensor(x2)
    value_tensor = tf.convert_to_tensor(value)
    res = input_data + x1_tensor / x2_tensor * value_tensor
    return res.numpy()


def calc_expect_func(input_data, x1, x2, value, y):
    """
    calc_expect_func
    """
    res = addcdiv_test(input_data["value"], x1['value'], x2['value'], value['value'])
    return [res]

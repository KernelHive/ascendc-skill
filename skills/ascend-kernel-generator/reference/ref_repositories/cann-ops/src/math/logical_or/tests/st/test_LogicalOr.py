# !/usr/bin/python3
# coding=utf-8
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
import tensorflow as tf


def logical_or_test(x1, x2):
    x1_tensor = tf.cast(tf.convert_to_tensor(x1), tf.bool)
    x2_tensor = tf.cast(tf.convert_to_tensor(x2), tf.bool)
    logical_or_tensor = tf.math.logical_or(x1_tensor, x2_tensor)
    re = logical_or_tensor.numpy().astype(np.bool_)
    return re


def calc_expect_func(x1, x2, y):
    res = logical_or_test(x1["value"], x2["value"])
    return [res]
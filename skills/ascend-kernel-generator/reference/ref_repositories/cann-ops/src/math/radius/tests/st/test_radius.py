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

import os
import sys
import tensorflow as tf
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
gendata_dir = os.path.abspath(os.path.join(current_dir, '../../examples/AclNNInvocationNaive'))
if gendata_dir not in sys.path:
    sys.path.insert(0, gendata_dir)
from gen_data import radius_numpy, RadiusParams


def calc_expect_func(**kwargs):
    """
    calc_expect_func
    """
    x = kwargs.get('x', {'value': None})['value']
    y = kwargs.get('y', {'value': None})['value']
    ptr_x = kwargs.get('ptr_x', {'value': None})['value']
    ptr_y = kwargs.get('ptr_y', {'value': None})['value']
    res = radius_numpy(x, y, RadiusParams(1.0, 32, False), ptr_x, ptr_y)
    return [res]

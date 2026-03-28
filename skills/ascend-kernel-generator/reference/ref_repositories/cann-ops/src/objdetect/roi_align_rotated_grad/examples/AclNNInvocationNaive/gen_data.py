#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import math
from functools import reduce

import torch
import numpy as np


def roi_align_ratated_backward():
    x_shape = [16, 48, 2, 2]
    grad_output_np = np.random.uniform(0, 5, x_shape).astype(np.float32)
    rois_shape = [16, 6]
    rois = np.random.uniform(0, 5, rois_shape).astype(np.float32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    grad_output_np_trans = grad_output_np.transpose(0, 2, 3, 1)
    grad_output_np_trans.astype(np.float32).tofile("./input/input_grad_output.bin")
    rois_trans = rois.transpose(1, 0)
    rois_trans.astype(np.float32).tofile("./input/rois_trans.bin")


if __name__ == "__main__":
    roi_align_ratated_backward()
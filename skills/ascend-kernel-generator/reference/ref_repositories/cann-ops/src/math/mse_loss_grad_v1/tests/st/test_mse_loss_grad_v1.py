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


def calc_expect_func(predict, label, dout, y):
    reduce_elts = 1.0
    for i in predict["value"].shape:
        reduce_elts *= i
    cof = (reduce_elts**(-1)) * 2.0
    sub_res = predict["value"] - label["value"]
    norm_grad = sub_res * cof
    res = norm_grad * dout["value"]
    return [res, ]
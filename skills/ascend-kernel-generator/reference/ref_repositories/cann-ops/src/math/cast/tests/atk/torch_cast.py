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

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

type_map = {
    0: torch.float32,
    1: torch.float16,
    2: torch.int8,
    3: torch.int32,
    4: torch.uint8,
    5: torch.uint32,
    6: torch.int16,
    9: torch.int64,
    12: torch.bool,
    27: torch.bfloat16
}


@register("aclnn_cast")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'cpu' or self.device == 'npu':
            input_tensor = input_data.kwargs["input"]
            torch_type = type_map[input_data.kwargs["dst_type"]]
            output = input_tensor.type(torch_type)
        return output
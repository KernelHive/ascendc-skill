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
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("aclnn_cpu_muls")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 从输入数据中获取必要的参数
        self.x = input_data.kwargs['x']
        self.value = input_data.kwargs['value']
        # 记录原始数据类型用于结果转换
        original_dtype = self.x.dtype
        # 将输入张量和值转换为float32进行计算
        if original_dtype in [torch.int16, torch.int32, torch.int64, torch.bfloat16]:
            self.x = self.x.to(torch.float32)
            self.value = float(self.value)  # 确保值为float32类型
        elif original_dtype == torch.float16:
            self.x = self.x.to(torch.float16)
            self.value = float(self.value)  # 先转为float，后续会转为float16
        # 根据设备类型执行相应的操作
        if self.device == "cpu":
            output = self.x * self.value
        elif self.device == "npu":
            output = self.x.npu() * self.value
        else:
            raise ValueError(f"Unsupported device: {self.device}. Only 'cpu' and 'npu' are supported.")
        # 将结果转回原始数据类型
        if original_dtype == torch.int16:
            output = output.to(torch.int16)
        elif original_dtype == torch.int32:
            output = output.to(torch.int32)
        elif original_dtype == torch.int64:
            output = output.to(torch.int64)
        elif original_dtype == torch.bfloat16:
            output = output.to(torch.bfloat16)
        elif original_dtype == torch.float16:
            # 对于float16，需要将value也转为float16
            self.value = torch.tensor(self.value, dtype=torch.float16, device=output.device)
            output = self.x * self.value
        return output

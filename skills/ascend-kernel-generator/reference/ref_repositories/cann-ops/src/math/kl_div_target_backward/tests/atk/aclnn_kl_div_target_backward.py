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

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


# aclnn_kl_div_target_backward     
@register("aclnn_kl_div_target_backward")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu" or self.device == "npu":
            grad_output = input_data.kwargs['grad_output']
            self_x = input_data.kwargs['self']
            target = input_data.kwargs['target']
            reduction = input_data.kwargs['reduction']
            log_target = input_data.kwargs['log_target']
            if target.numel() == 0:
                return target
            compute_dtype = grad_output.dtype
            grad_target = grad_output
            if compute_dtype == torch.bfloat16:
                grad_output = grad_output.to(torch.float)
                self_x = self_x.to(torch.float)
                target = target.to(torch.float)

            if log_target:
                grad_target = target + 1
                grad_target = grad_target - self_x
                tmp = torch.exp(target)
                grad_target = grad_target * tmp
                grad_target = grad_output * grad_target
            else:
                tmp = torch.log(target)
                grad_target = tmp + 1
                grad_target = grad_target - self_x
                grad_target = grad_output * grad_target
                grad_target = grad_target.masked_fill(target == 0, 0)

            if reduction == 1:
                target_len = max(max(grad_output.numel(), self_x.numel()), target.numel())
                grad_target = grad_target / target_len
            output = grad_target.to(compute_dtype)
        return output           
        

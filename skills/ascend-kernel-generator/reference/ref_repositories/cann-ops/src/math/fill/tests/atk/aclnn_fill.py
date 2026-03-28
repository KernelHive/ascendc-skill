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
from atk.configs.results_config import TaskResult


@register("aclnn_cpu_fill")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.dims = None
        self.value = None
        self.dtype = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 1.取参数
        # 2.封装
        # 3.执行,调用真正的标杆
        # 4. return
        if self.device == "cpu":
            dims = input_data.kwargs["dims"]
            dims_tensor = torch.zeros(dims).to(self.dtype)
            output = torch.fill_(dims_tensor, self.value)

        elif self.device == "npu":
            dims = input_data.kwargs["dims"]
            dims_tensor = torch.zeros(dims).to(self.dtype)
            output = torch.fill_(dims_tensor.npu(), self.value)

        return output

    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        """
        value_dtype = input_data.kwargs["value_dtype"]

        if value_dtype == "fp16":
            self.dtype = torch.float16
        elif value_dtype == "fp32":
            self.dtype = torch.float32
        elif value_dtype == "int8":
            self.dtype = torch.int8
        elif value_dtype == "int16":
            self.dtype = torch.int16
        elif value_dtype == "int32":
            self.dtype = torch.int32
        elif value_dtype == "int64":
            self.dtype = torch.int64
        elif value_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif value_dtype == "bool":
            self.dtype = torch.bool
        else:
            self.dtype = torch.int32

        del input_data.kwargs["value_dtype"]

        self.value = input_data.kwargs["value"]

    def get_cpp_func_signature_type(self):
        return ("aclnnStatus aclnnFillGetWorkspaceSize(const aclIntArray *dims, "
                "const aclScalar *value, const aclTensor *out, uint64_t *workspaceSize, "
                "aclOpExecutor **executor)")

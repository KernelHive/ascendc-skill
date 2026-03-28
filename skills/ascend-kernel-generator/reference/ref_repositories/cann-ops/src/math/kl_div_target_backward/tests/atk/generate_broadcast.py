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
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("broadcast")  # broadcast为注册的生成器名称，对应yaml中的generate参数
class ReduceGenerator(CaseGenerator):
    def change_shape(self, shape1, shape2):
        dim1 = len(shape1)
        dim2 = len(shape2)
        for i in range(dim2):
            shape2[dim2 - 1 - i] = shape1[dim1 - 1 - i]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        '''
        用例参数约束修改入口
        :param case_config:  生成的用例信息，可能不满足参数间约束，导致用例无效
        :return: 返回修改后符合参数间约束关系的用例，需要用例保障用例有效
        '''
        dtype = case_config.inputs[0].dtype
        case_config.inputs[1].dtype = dtype
        case_config.inputs[2].dtype = dtype
        
        grad_output = case_config.inputs[0].shape
        self_x = case_config.inputs[1].shape
        target = case_config.inputs[2].shape
        grad_output_dim_num = len(grad_output)
        self_x_dim_num = len(self_x)
        target_dim_num = len(target)
        if grad_output_dim_num >= self_x_dim_num and grad_output_dim_num >= target_dim_num:
            self.change_shape(grad_output, self_x)
            self.change_shape(grad_output, target)
        elif self_x_dim_num >= grad_output_dim_num and self_x_dim_num >= target_dim_num:
            self.change_shape(self_x, grad_output)
            self.change_shape(self_x, target)
        else:
            self.change_shape(target, self_x)
            self.change_shape(target, grad_output)
        return case_config  # 返回修改和符合参数约束的用例

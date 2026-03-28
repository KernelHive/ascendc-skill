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

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

float16 = [0, 2, 3, 4, 6, 12, 27]
float32 = [1, 2, 3, 4, 6, 9, 12, 27]
int32 = [0, 1, 2, 4, 6, 9, 12, 27]
int8 = [0, 1, 3, 4, 6, 9, 12, 27]
uint8 = [0, 1, 2, 3, 6, 9, 27]
bool8 = [0, 1, 2, 3, 4, 9, 27]
int64 = [0, 1, 2, 3, 4, 6, 12, 27]
bfloat16 = [0, 1, 2, 3, 4, 12]
int16 = [0, 1, 2, 3, 4, 9]


@GENERATOR_REGISTRY.register("reduce")
class ReduceGenerator(CaseGenerator):

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        '''
        用例参数约束修改入口
        :param case_config:  生成的用例信息，可能不满足参数间约束，导致用例无效
        :return: 返回修改后符合参数间约束关系的用例，需要用例保障用例有效
        '''
        if case_config.inputs[0].dtype == "fp32":
            if case_config.inputs[1].range_values not in float32:
                case_config.inputs[1].range_values = 1
        if case_config.inputs[0].dtype == "fp16":
            if case_config.inputs[1].range_values not in float16:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "int32":
            if case_config.inputs[1].range_values not in int32:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "int16":
            if case_config.inputs[1].range_values not in int16:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "int64":
            if case_config.inputs[1].range_values not in int64:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "int8":
            if case_config.inputs[1].range_values not in int8:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "uint8":
            if case_config.inputs[1].range_values not in uint8:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "bool":
            if case_config.inputs[1].range_values not in bool8:
                case_config.inputs[1].range_values = 0
        if case_config.inputs[0].dtype == "bf16":
            if case_config.inputs[1].range_values not in bfloat16:
                case_config.inputs[1].range_values = 0
        return case_config  # 返回修改和符合参数约束的用例
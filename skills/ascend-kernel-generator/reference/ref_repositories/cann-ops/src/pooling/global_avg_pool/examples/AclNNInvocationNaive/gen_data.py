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
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np


def gen_onnx_model(shape_x, shape_y):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)
    node_def = helper.make_node('GlobalAveragePool',
                                inputs=['x'],
                                outputs=['y']
                                )
    graph = helper.make_graph(
        [node_def],
        "test_GlobalAveragePool_case_0",
        [x],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-GlobalAveragePool_test")
    model.opset_import[0].version = 11
    model.ir_version = 6
    onnx.save(model, "./test_GlobalAveragePool_v11.onnx")


def run_mode(x):
    # 加载ONNX模型
    model_path = 'test_GlobalAveragePool_v11.onnx'  # 替换为你的ONNX模型路径
    sess = ort.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_data = x

    outputs = sess.run([output_name], {input_name: input_data})
    return outputs[0]


def gen_golden_data_simple():
    shape_x = [3, 4, 24, 2]
    shape_y = [3, 4, 1, 1]
    input_x = np.random.uniform(-1000, 1000, shape_x).astype(np.float32)
    gen_onnx_model(shape_x, shape_y)
    golden = run_mode(input_x)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
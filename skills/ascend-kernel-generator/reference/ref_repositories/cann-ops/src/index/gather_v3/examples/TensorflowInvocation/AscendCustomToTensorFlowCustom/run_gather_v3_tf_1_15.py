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
import sys
import os
import numpy as np
import tensorflow as tf
from npu_bridge import npu_init  # 显式导入模块
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

tf.compat.v1.disable_eager_execution()
tf.enable_resource_variables()
tfOpLib = tf.load_op_library(os.path.join("./../../../../../../output/libcustom_ops.so"))

#np.allclose比较函数的绝对公差参数
RELATIVE_TOL = 0.001

#np.allclose比较函数的相对公差参数
ABSOLUTE_TOL = 0.001


def sess_config():
    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    return config


# flash_attention_score 封装函数
def npu_gather_v3(x, indices, axis, batch_dims=0, negative_index_support=False):
    output = tfOpLib.gather_v3(x=x, indices=indices, axis=axis, batchDims=batch_dims, 
        negativeIndexSupport=negative_index_support)
    return output


if __name__ == '__main__':
    x_shape = [4, 2]
    y_shape = [2, 2]
    indices_shape = [2]
    axis_shape = [1]
    
    x_data = np.random.uniform(-2, 2, size=x_shape).astype(np.float16)
    indices_data = np.array([1, 0], dtype='int32')
    axis_data = np.array([1], dtype='int64')
    y_data = np.random.uniform(-2, 2, size=y_shape).astype(np.float16)
    print(x_data)
    print(indices_data)
    print(axis_data)

    x = tf.constant(x_data, tf.float16)
    axis = tf.constant(axis_data, tf.int64)
    indices = tf.constant(indices_data, tf.int32)
    
    tf_gather_result_t = tf.gather(x, indices, axis=axis[0])
    gather_v3_result_t = npu_gather_v3(x, indices, axis)
        
    with tf.compat.v1.Session(config=sess_config()) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        gather_v3_result = sess.run(gather_v3_result_t)
        print("gather_v3_result:\n", gather_v3_result)
    
    with tf.compat.v1.Session(config=sess_config()) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf_gather_result = sess.run(tf_gather_result_t)
        print("tf_gather_result:\n", tf_gather_result)
        
    # 通过np.allclose函数比较TensorFlow和Ascend C的输出是否一致
    cmp_result = np.allclose(tf_gather_result, gather_v3_result, atol=ABSOLUTE_TOL, rtol=RELATIVE_TOL)
    if cmp_result:
        print("The result of tf and ac is the same.")
    else:
        print("The result of tf and ac is different.")
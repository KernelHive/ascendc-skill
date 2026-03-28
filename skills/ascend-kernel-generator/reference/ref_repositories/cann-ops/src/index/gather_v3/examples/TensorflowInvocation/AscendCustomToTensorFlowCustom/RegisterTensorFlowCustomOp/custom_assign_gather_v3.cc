/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file custom_assign_gather_v3.cc
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_def_builder.h"

using namespace tensorflow;

// 注册TensorFlow自定义算子
REGISTER_OP("GatherV3")                                     // TensorFlow自定义算子名称
    .Input("x: T")
    .Input("indices: T1")
    .Input("axis: int64")
    .Output("y: T")
    .Attr("batchDims: int = 0")
    .Attr("negativeIndexSupport: bool = false")
    .Attr("T: {float32, float16, bfloat16, int16, int32, int64, int8, bool, uint16, uint32, uint64, uint8}")
    .Attr("T1: {int32, int64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);


// TensorFlow自定义算子的CPU实现
class GatherV3Op : public OpKernel {
public:
    explicit GatherV3Op(OpKernelConstruction* context) : OpKernel(context) {}
    // 当前算子不支持CPU设备，实现该函数以抛出异常，提示该算子不支持CPU设备
    void Compute(OpKernelContext* context) override {
        OP_REQUIRES(context, false, errors::Unimplemented("GatherV3Op is not supported on CPU"));
    }
};

// 注册TensorFlow自定义算子的CPU实现
REGISTER_KERNEL_BUILDER(Name("GatherV3").Device(DEVICE_CPU), GatherV3Op);
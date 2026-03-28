/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ge_glu_v2_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "aclnn/opdev/op_log.h"

// tools api
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define GE_OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_OUT_Y_GLU = 1;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForGeGluV2(gert::InferShapeContext* context) {
    auto x_shape = context->GetInputShape(GLU_IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto out_shape_y = context->GetOutputShape(GLU_ATTR_DIM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape_y);
    auto out_shape_y_glu = context->GetOutputShape(GLU_OUT_Y_GLU);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape_y_glu);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, split_dim_ptr);

    auto split_dim = *split_dim_ptr;

    if (split_dim < 0) {
        split_dim += x_shape->GetDimNum();
    }

    if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_shape->GetDimNum())) {
    GE_OP_LOGE("GEGLUV2", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].",
               x_shape->GetDimNum(), x_shape->GetDimNum() - 1, split_dim);
    return GRAPH_FAILED;
    }

    *out_shape_y = *x_shape;
    *out_shape_y_glu = *x_shape;
    
    // 动态shape场景split_dim_value传入-1不做处理
    auto split_dim_value = x_shape->GetDim(split_dim);
    if (split_dim_value > 0) {
        out_shape_y->SetDim(split_dim, split_dim_value / SPLIT_NUM);
        out_shape_y_glu->SetDim(split_dim, split_dim_value / SPLIT_NUM);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GeGluV2).InferShape(InferShapeForGeGluV2);
}  // namespace ops
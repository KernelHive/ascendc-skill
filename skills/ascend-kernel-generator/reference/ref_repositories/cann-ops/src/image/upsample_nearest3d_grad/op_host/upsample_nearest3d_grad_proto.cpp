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
 * \file upsample_trilinear3d_grad_proto.cpp
 * \brief
 */
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
    do {                                                      \
        std::printf("op[%s], %s", op_name, err_msg);          \
    } while (0)

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }
}  // namespace ops

using namespace ge;
namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t OUT_Y = 0;
static constexpr size_t SUPPORTED_DIM_NUM = 5;
static constexpr size_t SUPPORTED_OUTPUT_DIM_NUM = 3;
static constexpr size_t INDEX_INPUT_SIZE = 0;
static constexpr size_t INDEX_OUTPUT_SIZE = 1;
static constexpr size_t INDEX_SCALES = 2;
static constexpr size_t NOT_CHANGE_DIM = 2;

static ge::graphStatus SetUpsample3dGradInferShape(gert::InferShapeContext *context,
    const gert::Shape *grad_output_shape, gert::Shape *y_shape, const gert::ContinuousVector *input_size)
{
    auto attrs = context->GetAttrs();
    const gert::ContinuousVector *output_size = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_OUTPUT_SIZE);
    const gert::ContinuousVector *scales = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_SCALES);
    auto input_size_data = reinterpret_cast<const int64_t *>(input_size->GetData());
    auto output_size_data = reinterpret_cast<const int64_t *>(output_size->GetData());
    auto scales_data = reinterpret_cast<const float *>(scales->GetData());

    if (output_size->GetSize() != 0 && scales->GetSize() == 0) {
        OP_CHECK(output_size->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "attr::output_size dims must be 3."),
            return ge::GRAPH_FAILED);

        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            OP_CHECK(output_size_data[i] != grad_output_shape->GetDim(i + NOT_CHANGE_DIM),
                VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                    context->GetNodeName(), "attr:output_size[i] !=grad_output_shape->GetDim(i+NOT_CHANGE_DIM)"),
                return ge::GRAPH_FAILED);
        }
    } else if (output_size->GetSize() == 0 && scales->GetSize() != 0) {
        OP_CHECK(scales->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "attr::scales dims must be 3."),
            return ge::GRAPH_FAILED);

        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            int64_t tmp = int64_t(floor(input_size_data[i + NOT_CHANGE_DIM] * scales_data[i]));
            OP_CHECK(tmp != grad_output_shape->GetDim(i + NOT_CHANGE_DIM),
                VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                    context->GetNodeName(), "attr:tmp != grad_output_shape->GetDim(i+NOT_CHANGE_DIM)"),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_LOGE(context->GetNodeName(),
            "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
        return ge::GRAPH_FAILED;
    }

    *y_shape = *grad_output_shape;
    for (size_t i = 0; i < SUPPORTED_DIM_NUM; i++) {
        y_shape->SetDim(i, input_size_data[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Upsample3dGradInferShapeImpl(
    gert::InferShapeContext *context, const gert::Shape *grad_output_shape, gert::Shape *y_shape)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::ContinuousVector *input_size = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_INPUT_SIZE);
    OP_CHECK(input_size == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "get attr::input_size faild!"),
        return ge::GRAPH_FAILED);

    OP_CHECK(input_size->GetSize() != SUPPORTED_DIM_NUM,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "attr::input_size dims must be 5."),
        return ge::GRAPH_FAILED);

    return SetUpsample3dGradInferShape(context, grad_output_shape, y_shape, input_size);
}

static ge::graphStatus InferShape4Upsample3dGrad(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "begin to do InferShape4Upsample3dGrad");
    const gert::Shape *grad_output_shape = context->GetInputShape(IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, grad_output_shape);
    gert::Shape *y_shape = context->GetOutputShape(OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    auto grad_output_dim = grad_output_shape->GetDimNum();
    OP_CHECK(grad_output_dim != SUPPORTED_DIM_NUM,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Expected dim of input x should be 5."),
        return ge::GRAPH_FAILED);

    return Upsample3dGradInferShapeImpl(context, grad_output_shape, y_shape);
}

IMPL_OP_INFERSHAPE(UpsampleNearest3dGrad).InferShape(InferShape4Upsample3dGrad);
}  // namespace ops

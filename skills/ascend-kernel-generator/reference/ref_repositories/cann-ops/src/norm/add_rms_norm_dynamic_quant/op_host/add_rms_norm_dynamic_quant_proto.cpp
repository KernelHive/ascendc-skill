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
 * \file add_rms_norm_dynamic_quant.cc
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
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
}

static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int SMOOTH1_IDX = 3;
static constexpr int SMOOTH2_IDX = 4;

static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;
static constexpr int X_IDX = 2;
static constexpr int OUT_SCALE1_IDX = 3;
static constexpr int OUT_SCALE2_IDX = 4;

using namespace ge;

namespace ops {

static bool InferReduceShape(const gert::Shape *xShape, const gert::Shape *gammaShape, gert::Shape *reduceShape)
{
    size_t xDimNum = xShape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();
    reduceShape->SetDimNum(xDimNum - gammaDimNum);

    int64_t xDimValue = 0;

    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        xDimValue = xShape->GetDim(i);
        reduceShape->SetDim(i, xDimValue);
        OP_LOGI("InferShape4AddRmsNormDynamicQuant", "reduceShape[%zu] = [%zu]", i, reduceShape->GetDim(i));
    }
    return true;
}

static bool CheckOptionalShapeExisting(const gert::Shape *smoothShape)
{
    OP_CHECK(nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExisting", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetShapeSize();
    OP_CHECK((smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExisting", "Get empty smoothShape"), return false);
    return true;
}

static ge::graphStatus InferShape4AddRmsNormDynamicQuant(gert::InferShapeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShape4AddRmsNormDynamicQuant");

    // get input shapes
    const gert::Shape *x1Shape = context->GetInputShape(X1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape *gammaShape = context->GetInputShape(GAMMA_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape *y1Shape = context->GetOutputShape(Y1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape *y2Shape = context->GetOutputShape(Y2_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    gert::Shape *xShape = context->GetOutputShape(X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *outScale1Shape = context->GetOutputShape(OUT_SCALE1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outScale1Shape);
    gert::Shape *outScale2Shape = context->GetOutputShape(OUT_SCALE2_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outScale2Shape);

    *y1Shape = *x1Shape;
    *xShape = *x1Shape;

    const gert::Shape *smooth1Shape = context->GetOptionalInputShape(SMOOTH1_IDX);
    bool smooth1Exist = CheckOptionalShapeExisting(smooth1Shape);
    const gert::Shape *smooth2Shape = context->GetOptionalInputShape(SMOOTH2_IDX);
    bool smooth2Exist = CheckOptionalShapeExisting(smooth2Shape);

    OP_CHECK(smooth1Exist && (*gammaShape != *smooth1Shape),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "GammaShape is not same to smooth1Shape."),
        return GRAPH_FAILED);
    OP_CHECK(smooth2Exist && (*gammaShape != *smooth2Shape),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "GammaShape is not same to smooth2Shape."),
        return GRAPH_FAILED);

    bool isOnlyExistSmooth2 = (!smooth1Exist) && smooth2Exist;
    OP_CHECK(isOnlyExistSmooth2,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            context->GetNodeName(), "Dynamic AddRmsNormDynamicQuant Not support only have scale2."),
        return GRAPH_FAILED);
    InferReduceShape(x1Shape, gammaShape, outScale1Shape);
    if (smooth2Exist) {
        *y2Shape = *x1Shape;
        *outScale2Shape = *outScale1Shape;
    } else {
        *y2Shape = gert::Shape({1});
        *outScale2Shape = gert::Shape({1});
    }
    OP_LOGI(context->GetNodeName(), "End to do InferShape4AddRmsNormDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormDynamicQuant(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(Y1_IDX, DT_INT8);
    context->SetOutputDataType(Y2_IDX, DT_INT8);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(OUT_SCALE1_IDX, DT_FLOAT);
    context->SetOutputDataType(OUT_SCALE2_IDX, DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormDynamicQuant)
    .InferShape(InferShape4AddRmsNormDynamicQuant)
    .InferDataType(InferDataType4AddRmsNormDynamicQuant);
}  // namespace ops

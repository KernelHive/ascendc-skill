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
 * \file add_layer_norm_quant_proto.cpp
 * \brief
 */
#include <vector>
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
namespace ops {
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

static constexpr size_t X1_IDX = 0;
static constexpr size_t X2_IDX = 1;
static constexpr size_t GAMMA_IDX = 2;
static constexpr size_t BETA_IDX = 3;
static constexpr size_t BIAS_IDX = 4;
static constexpr size_t SCALE1_IDX = 5;
static constexpr size_t SCALE2_IDX = 6;
static constexpr size_t ZERO_POINT1_IDX = 7;
static constexpr size_t ZERO_POINT2_IDX = 8;

static constexpr size_t Y1_IDX = 0;
static constexpr size_t Y2_IDX = 1;
static constexpr size_t X_IDX = 2;
static constexpr size_t OUT_SCALE1_IDX = 3;
static constexpr size_t OUT_SCALE2_IDX = 4;

static constexpr size_t QUANT_MODE_IDX = 0;

using namespace ge;

namespace ops {

static bool InferReduceShape(const gert::Shape *xShape, const gert::Shape *gammaShape, gert::Shape *reduceShape)
{
    size_t xDimNum = xShape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();

    OP_LOGD("InferShape4AddLayerNormQuant", "xDimNum = %zu, gammaDimNum = %zu", xDimNum, gammaDimNum);
    reduceShape->SetDimNum(xDimNum);

    int64_t xDimValue = 0;
    int64_t gammaDimValue = 0;

    for (size_t i = 0; i < xDimNum; i++) {
        xDimValue = xShape->GetDim(i);
        if (i < xDimNum - gammaDimNum) {
            OP_LOGD("InferShape4AddLayerNormQuant", "xShape[%zu] = %zu", i, xDimValue);
            reduceShape->SetDim(i, xDimValue);
        } else {
            gammaDimValue = gammaShape->GetDim(i - xDimNum + gammaDimNum);
            OP_LOGD("InferShape4AddLayerNormQuant", "xShape[%zu] = %zu", i, xDimValue);
            OP_LOGD("InferShape4AddLayerNormQuant", "gammaShape[%zu] = %zu", i - xDimNum + gammaDimNum, gammaDimValue);
            reduceShape->SetDim(i, 1);
        }
        OP_LOGD("InferShape4AddLayerNormQuant", "reduceShape[%zu] = [%zu]", i, reduceShape->GetDim(i));
    }
    return true;
}

static inline bool CheckAllNotNull(std::initializer_list<const gert::Shape *> shapePtrList)
{
    for (auto &ptr : shapePtrList) {
        if (nullptr == ptr) {
            return false;
        }
    }
    return true;
}

static inline bool CheckDynOptInput(const gert::Shape *scale1Shape, const gert::Shape *scale2Shape,
    const gert::Shape *zeroPoint1Shape, const gert::Shape *zeroPoint2Shape)
{
    OP_LOGD("InferShape4AddLayerNormQuant", "Dynamic AddLayerNormQuant");
    OP_CHECK((nullptr != zeroPoint1Shape || nullptr != zeroPoint2Shape),
        OP_LOGE("CheckDynOptInput", "Dynamic AddLayerNormQuant Not support zeroPoints now."),
        return false);
    OP_CHECK((nullptr == scale1Shape && nullptr != scale2Shape),
        OP_LOGE("CheckDynOptInput", "Dynamic AddLayerNormQuant Not support only have scale2."),
        return false);
    return true;
}

static ge::graphStatus InferShape4AddLayerNormQuant(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4AddLayerNormQuant");

    // get input shapes
    const gert::Shape *x1Shape = context->GetInputShape(X1_IDX);
    const gert::Shape *gammaShape = context->GetInputShape(GAMMA_IDX);

    // get output shapes
    gert::Shape *y1Shape = context->GetOutputShape(Y1_IDX);
    gert::Shape *y2Shape = context->GetOutputShape(Y2_IDX);
    gert::Shape *xShape = context->GetOutputShape(X_IDX);
    gert::Shape *outScale1Shape = context->GetOutputShape(OUT_SCALE1_IDX);
    gert::Shape *outScale2Shape = context->GetOutputShape(OUT_SCALE2_IDX);

    OP_CHECK(!CheckAllNotNull({x1Shape, gammaShape, y1Shape, y2Shape, xShape, outScale1Shape, outScale2Shape}),
        OP_LOGE("AddLayerNormQuant", "Some shape is nullptr, infer failed. "),
        return GRAPH_FAILED);

    *y1Shape = *x1Shape;
    *xShape = *x1Shape;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *qms = attrs->GetAttrPointer<char>(QUANT_MODE_IDX);
    OP_CHECK(qms == nullptr,
        OP_LOGE(context->GetNodeName(), "Get required attr quantMode failed, infershape failed. "),
        return GRAPH_FAILED);
    OP_LOGD("InferShape4AddLayerNormQuant", "quantMode: %s", qms);
    std::string quantModeStr = qms;

    const gert::Shape *scale1Shape = context->GetOptionalInputShape(SCALE1_IDX);
    const gert::Shape *scale2Shape = context->GetOptionalInputShape(SCALE2_IDX);
    const gert::Shape *zeroPoint1Shape = context->GetOptionalInputShape(ZERO_POINT1_IDX);
    const gert::Shape *zeroPoint2Shape = context->GetOptionalInputShape(ZERO_POINT2_IDX);
    if (quantModeStr == "dynamic") {
        OP_CHECK(!CheckDynOptInput(scale1Shape, scale2Shape, zeroPoint1Shape, zeroPoint2Shape),
            OP_LOGE(context->GetNodeName(), "Bad opt inputs."),
            return GRAPH_FAILED);
        InferReduceShape(x1Shape, gammaShape, outScale1Shape);
        if (nullptr != scale2Shape) {
            *y2Shape = *x1Shape;
            *outScale2Shape = *outScale1Shape;
        } else {
            *y2Shape = gert::Shape({1});
            *outScale2Shape = gert::Shape({1});
        }
    } else {
        OP_LOGD("InferShape4AddLayerNormQuant", "Static AddLayerNormQuant");
        OP_CHECK((nullptr == scale1Shape),
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                context->GetNodeName(), "Static AddLayerNormQuant Not support scale1 is None."),
            return GRAPH_FAILED);
        *outScale1Shape = gert::Shape({1});
        *outScale2Shape = gert::Shape({1});
        if (nullptr != scale2Shape) {
            *y2Shape = *x1Shape;
        } else {
            *y2Shape = gert::Shape({1});
        }
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShape4AddLayerNormQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddLayerNormQuant(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4AddLayerNormQuant");

    context->SetOutputDataType(Y1_IDX, DT_INT8);
    context->SetOutputDataType(Y2_IDX, DT_INT8);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(OUT_SCALE1_IDX, DT_FLOAT);
    context->SetOutputDataType(OUT_SCALE2_IDX, DT_FLOAT);

    OP_LOGD(context->GetNodeName(), "End to do InferDataType4AddLayerNormQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddLayerNormQuant)
    .InferShape(InferShape4AddLayerNormQuant)
    .InferDataType(InferDataType4AddLayerNormQuant);
}  // namespace ops

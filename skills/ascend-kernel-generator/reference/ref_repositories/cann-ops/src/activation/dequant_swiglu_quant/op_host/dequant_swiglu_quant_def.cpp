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
 * \file dequant_swiglu_quant.cpp
 * \brief
 */
 
#include <cstdint>
#include "register/op_def_registry.h"

using namespace ge;

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)            \
    if ((ptr) == nullptr)                                    \
    {                                                        \
        std::printf("nullptr error!");                       \
        return ge::GRAPH_SUCCESS;                            \
    }                                                        \



namespace {
#define OP_CHECK(cond, log_func, return_expr) \
    do {                                      \
        if (!(cond)) {                        \
            log_func;                         \
            return_expr;                      \
        }                                     \
    } while (false)
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
static inline bool IsUnknownRank(const gert::Shape* check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
}
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_SCALE = 1;
constexpr int64_t NUM_TWO = 2;

graphStatus InferShape4DequantSwigluQuant(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "Begin to do InferShape4DequantSwigluQuant.");

  const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
  gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
  gert::Shape* scaleShape = context->GetOutputShape(OUTPUT_IDX_SCALE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

  *yShape = *xShape;
  *scaleShape = *xShape;

  OP_CHECK(IsUnknownRank(xShape),
           OP_LOGI(context->GetNodeName(), "End to do InferShape4DequantSwigluQuant, inputx is [-2]."),
           return GRAPH_SUCCESS);
  int64_t xShapeRank = static_cast<int64_t>(xShape->GetDimNum());
  int64_t lastDim = xShape->GetDim(xShapeRank - 1);
  scaleShape->SetDimNum(xShapeRank - 1);
  if (lastDim > 1) {
    yShape->SetDim(xShapeRank - 1, lastDim / NUM_TWO);
  }
  OP_LOGD(context->GetNodeName(), "End to do InferShape4DequantSwigluQuant.");
  return GRAPH_SUCCESS;
}

graphStatus InferDtype4DequantSwigluQuant(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "InferDtype4DequantSwigluQuant enter");

  context->SetOutputDataType(OUTPUT_IDX_Y, DT_INT8);
  context->SetOutputDataType(OUTPUT_IDX_SCALE, DT_FLOAT);
  OP_LOGD(context->GetNodeName(), "InferDtype4DequantSwigluQuant end");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantSwigluQuant)
    .InferShape(InferShape4DequantSwigluQuant)
    .InferDataType(InferDtype4DequantSwigluQuant);

class DequantSwigluQuant : public OpDef {
public:
    explicit DequantSwigluQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("activation_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("activate_left").AttrType(OPTIONAL).Bool(false);
        this->Attr("quant_mode").AttrType(OPTIONAL).String("static");

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DequantSwigluQuant);
}  // namespace ops
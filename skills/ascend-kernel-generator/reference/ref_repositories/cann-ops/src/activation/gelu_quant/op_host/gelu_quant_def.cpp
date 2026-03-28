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
 * @file gelu_quant_def.cpp
 */
#include "register/op_def_registry.h"
#include "gelu_quant_tiling_base.h"
namespace ops {
static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  if ((context->GetInputShape(0) == nullptr)) {
    return ge::GRAPH_FAILED;
  }

  if ((context->GetOutputShape(0) == nullptr)) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGeluQuant(gert::InferShapeContext *context)
{
    if (InferShapeCheck(context) == ge::GRAPH_FAILED) {
        OP_LOGD("GeluQuant", "[GeluQuant] InferShapeCheck failed.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* outScaleShape = context->GetOutputShape(1);
    *yShape = *xShape;
    outScaleShape->SetDimNum(xShape->GetDimNum() - 1);
    for (size_t i = 0; i < xShape->GetDimNum() - 1; ++i) {
        outScaleShape->SetDim(i, xShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGeluQuant(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGD("GeluQuant", "[GeluQuant] GeluQuantInferDataType got context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, ge::DT_INT8);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GeluQuant).InferShape(InferShapeForGeluQuant).InferDataType(InferDataTypeForGeluQuant);

class GeluQuant : public OpDef {
 public:
  explicit GeluQuant(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT,ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("input_scale")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT,ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("input_offset")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT,ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8,ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("approximate")
        .AttrType(OPTIONAL)
        .String("none");
    this->Attr("quant_mode")
        .AttrType(OPTIONAL)
        .String("dynamic");
        
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true);

    this->AICore().AddConfig("ascend910b", aicore_config);
    this->AICore().AddConfig("ascend910_93", aicore_config);
  }
};

OP_ADD(GeluQuant);
} // namespace ops

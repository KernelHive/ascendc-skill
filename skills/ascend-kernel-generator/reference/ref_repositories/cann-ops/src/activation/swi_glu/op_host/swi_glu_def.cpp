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
 * \file swi_glu.cpp
 * \brief
 */
#include <register/op_def_registry.h>
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)            \
    if ((ptr) == nullptr)                                    \
    {                                                        \
        std::printf("nullptr error!");                       \
        return ge::GRAPH_SUCCESS;                            \
    }                                                        \

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForSwiGlu(gert::InferShapeContext* context) {
    auto x_shape = context->GetInputShape(GLU_IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto y_shape = context->GetOutputShape(GLU_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, split_dim_ptr);

    auto split_dim = *split_dim_ptr;
    if (split_dim < 0) {
      split_dim += x_shape->GetDimNum();
    }
    if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_shape->GetDimNum())) {
      return GRAPH_FAILED;
    }

    *y_shape = *x_shape;
    // dynamic shape
    if (x_shape->GetDim(split_dim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (x_shape->GetDim(split_dim) < 0 || x_shape->GetDim(split_dim) % SPLIT_NUM != 0) {
        return ge::GRAPH_FAILED;
    }

    y_shape->SetDim(split_dim, x_shape->GetDim(split_dim) / SPLIT_NUM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwiGlu(gert::InferDataTypeContext *context) {
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    return ret;
}
IMPL_OP_INFERSHAPE(SwiGlu).InferShape(InferShapeForSwiGlu).InferDataType(InferDataTypeForSwiGlu);

class SwiGlu : public OpDef {
public:
    explicit SwiGlu(const char* name) : OpDef(name)
    {
      this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .AutoContiguous();
      this->Output("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Attr("dim")
          .AttrType(OPTIONAL)
          .Int(-1);
      this->AICore().AddConfig("ascend910b");

      OpAICoreConfig config_without_bf16;
      config_without_bf16.Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND});
      config_without_bf16.Output("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND});
      config_without_bf16.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true);
      this->AICore().AddConfig("ascend310p", config_without_bf16);
    }
};
OP_ADD(SwiGlu);
}
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
 * @file moe_soft_max_topk.cpp
 */
#include "moe_soft_max_topk_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  MoeSoftMaxTopkTilingData tiling;
  uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint32_t lastDim =  context->GetInputShape(0)->GetStorageShape().GetDim(1);
  const uint32_t* k = context->GetAttrs()->GetAttrPointer<uint32_t>(0);
  
  int n = *k;
  int indices_sum = 0;
  int score_sum = 0;
  while (n)
  {
    indices_sum += (1<< (2*n -1));
    score_sum += (1<< (2*n-2));
    n--;
  }
  
  tiling.set_k(*k);
  tiling.set_totalLength(totalLength);
  tiling.set_lastDim(lastDim);
  tiling.set_indicesSum(indices_sum);
  tiling.set_scoreSum(score_sum);
  tiling.set_tileNum(1);

  context->SetBlockDim(32);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(1);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static constexpr int OUTPUT_VALUES_INDEX = 0;
static constexpr int OUTPUT_INDICES_INDEX = 1;
static constexpr int INPUT_X_INDEX = 0;

static bool InferShapeForTopK(gert::InferShapeContext* context, const int64_t* k) {
  const gert::Shape *input_x_shape = context->GetInputShape(INPUT_X_INDEX);
  if (input_x_shape == nullptr) {
    return GRAPH_FAILED;
  }
  size_t dim_size = input_x_shape->GetDimNum();
  
  gert::Shape *output_values_shape = context->GetOutputShape(OUTPUT_VALUES_INDEX);
  gert::Shape *output_indices_shape = context->GetOutputShape(OUTPUT_INDICES_INDEX);

  output_values_shape->SetDimNum(dim_size);
  output_indices_shape->SetDimNum(dim_size);
  for (size_t i = 0; i < dim_size; i++) {
    if (static_cast<int64_t>(i) == 1) {
      output_values_shape->SetDim(i, *k);
      output_indices_shape->SetDim(i, *k);
      continue;
    }
    output_values_shape->SetDim(i, input_x_shape->GetDim(i));
    output_indices_shape->SetDim(i, input_x_shape->GetDim(i));
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  const int64_t *input_k = attrs->GetInt(0);
  return InferShapeForTopK(context, input_k);
  }

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType); // y
  context->SetOutputDataType(1, DT_INT32);  // indices
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MoeSoftMaxTopk : public OpDef {
 public:
  explicit MoeSoftMaxTopk(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("k").AttrType(OPTIONAL).Int(4);

    this->SetInferShape(ge::InferShape);
    this->SetInferDataType(ge::InferDataType);
    
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b")
                  .AddConfig("ascend910b");
  }
};

OP_ADD(MoeSoftMaxTopk);
}  // namespace ops

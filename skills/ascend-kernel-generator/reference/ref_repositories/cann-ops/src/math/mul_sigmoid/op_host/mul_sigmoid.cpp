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
 * @file mul_sigmoid.cpp
 */

#include "mul_sigmoid_tiling.h"

namespace optiling {

static ge::graphStatus MulSigmoidTilingFunc(gert::TilingContext* context) {
  MulSigmoidTiling tiling(context);
  auto ret = tiling.DoTiling();
  
  return ret;
}

}  // namespace optiling

namespace ge {
static ge::graphStatus MulSigmoidInferShape(gert::InferShapeContext* context) {

  const gert::Shape* input_shape0 = context->GetInputShape(0); // [row， col]
  const gert::Shape* input_shape1 = context->GetInputShape(1); // [1, col/128, 128]
  gert::Shape* output_shape = context->GetOutputShape(0); //
  int32_t dim_idx_2 = 2;
  int32_t dim_idx_3 = 3;
  int32_t size_128 = 128;
  if (input_shape1->GetDim(0) != 1) {
    std::cout << "x2 shape dim 0 cannot be larger than 1\n";
    return GRAPH_FAILED;
  }

  if (input_shape1->GetDim(1) * input_shape1->GetDim(dim_idx_2) != input_shape0->GetDim(1)) {
    std::cout << "x1 and x2 shape cannot be aligned\n";
    return GRAPH_FAILED;
  }
  
  output_shape->SetDimNum(dim_idx_3);
  output_shape->SetDim(0, input_shape0->GetDim(0));
  output_shape->SetDim(1, input_shape0->GetDim(1) / size_128);
  output_shape->SetDim(dim_idx_2, size_128);
  return GRAPH_SUCCESS;
}


static ge::graphStatus MulSigmoidInferDataType(gert::InferDataTypeContext* context) {
  auto dtype = ge::DT_FLOAT16;
  context->SetOutputDataType(0, dtype);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MulSigmoid : public OpDef {
public:
  explicit MulSigmoid(const char* name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->SetInferShape(ge::MulSigmoidInferShape);
    this->SetInferDataType(ge::MulSigmoidInferDataType);

    this->Attr("t1").AttrType(REQUIRED).Float(0);
    this->Attr("t2").AttrType(REQUIRED).Float(0);
    this->Attr("t3").AttrType(REQUIRED).Float(0);
    
    this->AICore().SetTiling(optiling::MulSigmoidTilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(MulSigmoid);
}  // namespace ops

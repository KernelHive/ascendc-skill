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
 * @file radius.cpp
 */
#include "radius_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  RadiusTilingData tiling;
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  const gert::StorageShape* y_shape = context->GetInputShape(1);
  const gert::StorageShape* ptr_x_shape = context->GetOptionalInputShape(2);
  const gert::StorageShape* ptr_y_shape = context->GetOptionalInputShape(3);
  uint32_t xSize = x_shape->GetStorageShape().GetDim(0);
  uint32_t itemLength = x_shape->GetStorageShape().GetDim(1);
  uint32_t ySize = y_shape->GetStorageShape().GetDim(0);
  
  auto attrs = context->GetAttrs();
  float r = *attrs->GetFloat(0);
  uint32_t max_num_neighbors = *attrs->GetInt(1);
  uint32_t ignore_same_index = *attrs->GetBool(2);
  uint32_t ptrXLen = 0;
  uint32_t ptrYLen = 0;
  if(ptr_x_shape != nullptr){
    ptrXLen = ptr_x_shape->GetStorageShape().GetShapeSize();
  }
  if(ptr_y_shape != nullptr){
    ptrYLen = ptr_y_shape->GetStorageShape().GetShapeSize();
  }
  context->SetBlockDim(1);
  tiling.set_ptrXLen(ptrXLen);
  tiling.set_ptrYLen(ptrYLen);
  tiling.set_xSize(xSize);
  tiling.set_ySize(ySize);
  tiling.set_itemLength(itemLength);
  tiling.set_r(r);
  tiling.set_max_num_neighbors(max_num_neighbors);
  tiling.set_ignore_same_index(ignore_same_index);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x1_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Radius : public OpDef {
public:
    explicit Radius(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ptr_x")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ptr_y")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("r").Float();
        this->Attr("max_num_neighbors").Int();
        this->Attr("ignore_same_index").Bool();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Radius);
}

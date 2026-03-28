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
 * @file stride_slice_neg_concat_v2.cpp
 */

#include "strideslice_neg_concat_v2_tiling.h"
#include "register/op_def_registry.h"
#include "iostream"
using namespace std;
namespace optiling {
const uint32_t BLOCK_DIM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    StridesliceNegConcatV2Tiling tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    uint32_t totalLength = x1_shape->GetStorageShape().GetShapeSize();
    uint32_t batch_size = x1_shape->GetStorageShape().GetDim(0);
    uint32_t height = x1_shape->GetStorageShape().GetDim(1);
    uint32_t width = x1_shape->GetStorageShape().GetDim(2);
    uint32_t channel = x1_shape->GetStorageShape().GetDim(3);
    uint32_t rowdim = batch_size * height * width;
    uint32_t tile_num_average = rowdim / BLOCK_DIM;
    uint32_t tile_num_last = tile_num_average + (rowdim % BLOCK_DIM);
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNumAverage(tile_num_average);
    tiling.set_tileNumLast(tile_num_last);
    tiling.set_tileLength(channel);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class StridesliceNegConcatV2 : public OpDef {
public:
    explicit StridesliceNegConcatV2(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(StridesliceNegConcatV2);
}
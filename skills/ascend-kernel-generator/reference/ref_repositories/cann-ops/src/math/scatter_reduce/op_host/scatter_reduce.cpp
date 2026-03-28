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
 * @file scatter_reduce.cpp
 */
#include "register/op_def_registry.h"
#include "scatter_reduce_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {

    constexpr int CORENUM = 1;
    constexpr int BLOCK_BYTES_SIZE = 32;

    ScatterReduceTilingData tiling;

    const int64_t* dimAttr = context->GetAttrs()->GetInt(0);
    const char* reduceAttr = context->GetAttrs()->GetStr(1);
    const bool* includeSelfAttr = context->GetAttrs()->GetBool(2);
    uint32_t dim = *dimAttr;
    int32_t reduction = 0;
    bool includeSelf = *includeSelfAttr;
    // "sum", "prod", "mean", "amax", "amin"
    if (strcmp(reduceAttr, "sum") == 0) {
        reduction = 0;
    } else if (strcmp(reduceAttr, "prod") == 0) {
        reduction = 1;
    } else if (strcmp(reduceAttr, "mean") == 0) {
        reduction = 2;
    } else if (strcmp(reduceAttr, "amax") == 0) {
        reduction = 3;
    } else if (strcmp(reduceAttr, "amin") == 0) {
        reduction = 4;
    }

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* src_shape = context->GetInputShape(2);

    int32_t batchSize = 1;
    for (int i = 0; i < dim; i++) batchSize *= x1_shape->GetStorageShape().GetDim(i);

    int32_t dimSizeX = x1_shape->GetStorageShape().GetDim(dim);
    int32_t dimSizeSrc = src_shape->GetStorageShape().GetDim(dim);

    int32_t strideSize = 1;
    for (int i = dim + 1; i < x1_shape->GetStorageShape().GetDimNum(); i++)
        strideSize *= x1_shape->GetStorageShape().GetDim(i);

    tiling.set_batchSize(batchSize);
    tiling.set_dimSizeX(dimSizeX);
    tiling.set_dimSizeSrc(dimSizeSrc);
    tiling.set_strideSize(strideSize);
    tiling.set_reduction(reduction);
    tiling.set_includeSelf(includeSelf);

    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (batchSize == 1 && dimSizeX == dimSizeSrc && includeSelf == false && reduction == 4 && inputBytes == 4)
    {
        int BlockSize = BLOCK_BYTES_SIZE / inputBytes;
        int BlockNum = (strideSize + BlockSize - 1)/ BlockSize;
        int coreNum = BlockNum < CORENUM ? BlockNum : CORENUM ;
        context->SetBlockDim(coreNum);
        context->SetTilingKey(1);
    } else {
        context->SetBlockDim(1);
        context->SetTilingKey(2);
    }
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


namespace ops {
class ScatterReduce : public OpDef {
public:
    explicit ScatterReduce(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->Attr("reduce").String();
        this->Attr("include_self").AttrType(OPTIONAL).Bool(true);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend310b");
    }
};

OP_ADD(ScatterReduce);
}

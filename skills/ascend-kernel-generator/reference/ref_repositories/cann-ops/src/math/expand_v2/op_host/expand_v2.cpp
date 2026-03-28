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
 * @file expand_v2.cpp
 */
#include "expand_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
constexpr uint64_t MAX_TILE_SIZE = 32 * 1024;  // 32KB

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ExpandV2TilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNum();
    uint64_t expandSize = 1;

    uint64_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize(); // 元素个数
    auto dimNum = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    auto newShapeDimNum = context->GetAttrs()->GetListInt(0)->GetSize();
    const uint64_t dimDelta = newShapeDimNum - dimNum;
    if (dimDelta < 0)
    {
        return ge::GRAPH_PARAM_INVALID;
    }
    for (size_t i = 0; i <= dimDelta; i++)
    {
        expandSize *= context->GetAttrs()->GetListInt(0)->GetData()[i];
    }

    auto dtype = context->GetInputDesc(0)->GetDataType();
    const uint64_t dtypeSize = 8; // x是int64
    const uint64_t maxTileLength = MAX_TILE_SIZE / dtypeSize; // 用B在计算

    uint64_t tileLength = totalLength <= maxTileLength ? totalLength : maxTileLength;
    uint64_t tileNum = totalLength / maxTileLength;
    uint64_t miniTileLength = totalLength % maxTileLength;
    printf("totalLength %ld maxTileLength %ld \n", totalLength, maxTileLength);

    // 文档：https://wiki.huawei.com/domains/64642/wiki/95725/WIKI202503196296522
    tiling.set_expandSize(expandSize);
    tiling.set_blockLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(tileNum);
    tiling.set_miniTileLength(miniTileLength);

    context->SetBlockDim(coreNum >= expandSize ? expandSize : coreNum);
    printf("coreNum %d\n", coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    auto newShapeDimNum = context->GetAttrs()->GetListInt(0)->GetSize();
    gert::Shape* y_shape = context->GetOutputShape(0);
    for (size_t i = 0; i < newShapeDimNum; i++)
    {
        y_shape->AppendDim(context->GetAttrs()->GetListInt(0)->GetData()[i]);
    }

    return GRAPH_SUCCESS;
}
}


namespace ops {
class ExpandV2 : public OpDef {
public:
    explicit ExpandV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("shape").ListInt();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ExpandV2);
}

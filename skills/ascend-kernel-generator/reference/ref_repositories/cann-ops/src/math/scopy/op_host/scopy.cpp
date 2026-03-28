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
 * @file scopy.cpp
 */
#include "scopy_tiling.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include <iostream>


namespace optiling {

// static variable
constexpr static uint64_t ELEMENTS_PER_BLOCK = 8;
constexpr static uint32_t ELENUM_EACH_COMPLEX = 2;

// Calculate tiling data value
static void CalTilingData(uint32_t elementNum, uint32_t* calNum, uint32_t* startOffset, uint32_t maxCoreNum)
{
    // Num of blocks
    uint32_t totalBlockNum = elementNum / ELEMENTS_PER_BLOCK;
    // Remain elements num
    uint32_t remainNum = elementNum % ELEMENTS_PER_BLOCK;

    if (totalBlockNum == 0) {
        // Use only 1 AIV core.
        calNum[0] = remainNum;
    } else if (totalBlockNum <= maxCoreNum) {
        for (uint32_t i = 0; i < totalBlockNum; i++) {
            startOffset[i] = ELEMENTS_PER_BLOCK * i;
            calNum[i] = ELEMENTS_PER_BLOCK;
        }
        calNum[totalBlockNum - 1] += remainNum;
    } else {
        uint64_t blockNumEachCore;
        uint32_t remainBlock;
        if (maxCoreNum == 0) {
            blockNumEachCore = 1;
            remainBlock = 0;
        } else {
            blockNumEachCore = totalBlockNum / maxCoreNum;
            remainBlock = totalBlockNum % maxCoreNum;
        }
        uint64_t currOffset = 0;
        uint64_t currCalNum = 0;

        for (uint32_t i = 0; i < maxCoreNum; i++) {
            if (i < remainBlock) {
                currCalNum = (blockNumEachCore + 1) * ELEMENTS_PER_BLOCK;
            } else {
                currCalNum = blockNumEachCore * ELEMENTS_PER_BLOCK;
            }
            startOffset[i] = currOffset;
            calNum[i] = currCalNum;
            currOffset += currCalNum;
        }
        calNum[maxCoreNum - 1] += remainNum;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto* attrs = context->GetAttrs();
    auto* elementNumPtr = attrs->GetAttrPointer<uint32_t>(0);
    uint32_t elementNum = static_cast<uint32_t>(*elementNumPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto vecCoreNum = ascendcPlatform.GetCoreNumAiv();

    uint32_t startOffset[40] = {0};
    uint32_t calNum[40] = {0};

    CalTilingData(elementNum, calNum, startOffset, vecCoreNum);

    ScopyTilingData tiling;
    tiling.set_n(elementNum);
    tiling.set_useCoreNum(vecCoreNum);
    tiling.set_startOffset(startOffset);
    tiling.set_calNum(calNum);

    context->SetTilingKey(0);
    context->SetBlockDim(vecCoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Scopy : public OpDef {
public:
    explicit Scopy(const char *name) : OpDef(name)
    {
        
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("n").AttrType(OPTIONAL).Int(0);
        this->Attr("incx").AttrType(OPTIONAL).Int(1);
        this->Attr("incy").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Scopy);
} // namespace ops

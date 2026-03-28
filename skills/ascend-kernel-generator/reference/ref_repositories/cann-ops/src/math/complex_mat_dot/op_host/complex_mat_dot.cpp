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
 * @file complex_mat_dot.cpp
 */
#include "complex_mat_dot_tiling.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"


namespace optiling {

// static variable
constexpr static uint32_t ELENUM_EACH_COMPLEX = 2;

// Calculate tiling data value
static void CalTilingData(uint32_t elementNum, uint32_t* calNum, uint64_t* startOffset, uint32_t maxCoreNum)
{
    uint32_t numEachCore = elementNum / maxCoreNum;  // 40 num of complex
    uint32_t remainNum = elementNum - maxCoreNum * numEachCore;

    if (numEachCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            *(calNum + i) = 1;
            *(startOffset + i) = i * ELENUM_EACH_COMPLEX;  // each row has 2*n FP32 elements
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < maxCoreNum; i++) {
            if (i < remainNum) {
                currNum = numEachCore + 1;
            } else {
                currNum = numEachCore;
            }
            *(calNum + i) = currNum;
            *(startOffset + i) = currOffset;
            currOffset += currNum * ELENUM_EACH_COMPLEX;
        }
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto* attrs = context->GetAttrs();
    auto* mPtr = attrs->GetAttrPointer<uint32_t>(0);
    auto* nPtr = attrs->GetAttrPointer<uint32_t>(1);
    uint32_t m = static_cast<uint32_t>(*mPtr);
    uint32_t n = static_cast<uint32_t>(*nPtr);
    uint32_t elementNum = m * n;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto vecCoreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t startOffset[40] = {0};
    uint32_t calNum[40] = {0};

    CalTilingData(elementNum, calNum, startOffset, vecCoreNum);

    ComplexMatDotTilingData tiling;
    tiling.set_m(m);
    tiling.set_n(n);
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
class ComplexMatDot : public OpDef {
public:
    explicit ComplexMatDot(const char *name) : OpDef(name)
    {
        this->Input("matx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Input("maty")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Attr("m").AttrType(OPTIONAL).Int(1);
        this->Attr("n").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(ComplexMatDot);
} // namespace ops

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
 * @file sub_frameworklaunch.cpp
 */
#include "sub_frameworklaunch_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t BLOCK_SIZE = 32;
// The smallest unit to which the shape needs to be aligned
constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_HALF;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    SubFrameworklaunchTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    // If the shape is not aligned, it needs to be aligned up
    uint32_t totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    /*
    Distribute all the data as evenly as possible across each core. 
    If it cannot be evenly divided, some cores will handle one more minimum unit ALIGN_NUM.
    By calculating the remainder, you can determine the number of cores that will handle one extra minimum unit, 
    as well as the number of cores that will handle one less minimum unit.
    Example: For a total data size of 1999, the aligned total data size becomes 2000, 
    the number of cores is 8, and the minimum unit of data block is 16, then:
    1. The total number of minimum unit data blocks: 2000 / 16 = 125
    2. 5 cores will be assigned 16 minimum unit data blocks: 125 % 8 = 5, which can be called large blocks.
    3. 3 cores will be assigned 15 minimum unit data blocks: 8 - 5 = 3, which can be called small blocks.
    */
    uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM;
    uint32_t tailNum = BLOCK_DIM - formerNum;
    // Calculate the data size of large and small blocks
    uint32_t formerLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t tailLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM;
    tiling.set_formerNum(formerNum);
    tiling.set_tailNum(tailNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailLength(tailLength);
    tiling.set_alignNum(ALIGN_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
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
class SubFrameworklaunch : public OpDef {
public:
    explicit SubFrameworklaunch(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend910b")
            .AddConfig("ascend310b");
    }
};
OP_ADD(SubFrameworklaunch);
} // namespace ops

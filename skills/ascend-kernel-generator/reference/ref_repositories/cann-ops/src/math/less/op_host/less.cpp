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
 * @file less.cpp
 */
#include "less_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;

const uint32_t NUM_1 = 10;
const uint32_t NUM_2 = 5;
const uint32_t NUM_3 = 6;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LessTilingData tiling;
    uint64_t ubLength = 0;
    uint32_t bigCoreDataNum = 0;
    uint32_t bigCoreLoopNum = 0;
    uint32_t bigCoreTailDataNum = 0;
    uint32_t bigprocessDataNumComputes=0;
    uint32_t tailbigprocessDataNumComputes=0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    uint32_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 1;
    auto dataType = context->GetInputDesc(0)->GetDataType();
    ge::TypeUtils::GetDataTypeLength(dataType, dataTypeLength);

    if (coreNum == 0 || BLOCK_SIZE == 0) 
    {
        return ge::GRAPH_FAILED;
    }
    uint32_t ubPartNum = 5;
    if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        ubLength = ubLength - 1024 * 8 - 256 * 6;//8k 预留空间 1k 计算对齐256字节 
        ubPartNum = NUM_1;
        context->SetTilingKey(0);
    }
    else if (dataType == ge::DT_FLOAT) {
        ubLength = ubLength - 1024 * 8 - 256 * 7;//8k 预留空间 1k 计算对齐256字节 
        context->SetTilingKey(1);
    }
    else if (dataType == ge::DT_FLOAT16) {
        ubLength = ubLength - 1024 * 8 - 256 * 6;//8k 预留空间 1k 计算对齐256字节 
    }
    else if (dataType == ge::DT_BF16) {
         ubPartNum = NUM_1;
         ubLength = ubLength - 1024 * 8 - 256 * 10;//8k 预留空间 1k 计算对齐256字节 
         context->SetTilingKey(1);
    }
    else if (dataType == ge::DT_INT32) {
        ubPartNum = NUM_2;
        ubLength = ubLength - 1024 * 8 - 256 * 7;//8k 预留空间 1k 计算对齐256字节 
        context->SetTilingKey(1);
    }
    else if (dataType == ge::DT_INT64) {
        ubPartNum = NUM_3;
        context->SetTilingKey(1);
    }
    
    ubLength = ubLength / dataTypeLength;
    uint32_t ubPartLength = ubLength / ubPartNum;
    uint32_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint32_t ubPartDataNum = ubPartBlockNum * BLOCK_SIZE;
    uint32_t inputLengthAlign32 = (((inputDataNum + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    
    if(ubPartDataNum >= inputDataNum)
    {
        coreNum=1;
    }
    else
    {
        coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }
    
    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    
    // Small chunks are calculated and sliced several times using the number of data on each core
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE ;
    uint32_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
    // Tail block calculation for small chunks of data
    uint32_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreDataNum / ubPartDataNum);
    smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;
    uint32_t smallprocessDataNumComputes= (((ubPartDataNum * dataTypeLength + 256 - 1) / 256) * 256) / dataTypeLength;//计算函数 256字节对齐
    uint32_t tailsmallprocessDataNumComputes= (((smallCoreTailDataNum * dataTypeLength + 256 - 1) / 256) * 256) / dataTypeLength;//尾块计算函数 256字节对齐
    
    uint32_t isTailBlock = 0;
    if(0 != tailBlockNum)
    {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE ;
        bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
        bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? bigCoreLoopNum : bigCoreLoopNum + 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreDataNum / ubPartDataNum);
        bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
        bigprocessDataNumComputes= (((ubPartDataNum * dataTypeLength + 256 - 1) / 256) * 256) / dataTypeLength;//计算函数 256字节对齐
        tailbigprocessDataNumComputes= (((bigCoreTailDataNum * dataTypeLength + 256 - 1) / 256) * 256) / dataTypeLength;//尾块计算函数 256字节对齐
        isTailBlock = 1;
    }

    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_ubPartDataNum(ubPartDataNum);
    tiling.set_smallCoreTailDataNum(smallCoreTailDataNum);
    tiling.set_bigCoreTailDataNum(bigCoreTailDataNum);
    tiling.set_smallCoreLoopNum(smallCoreLoopNum);
    tiling.set_bigCoreLoopNum(bigCoreLoopNum);
    tiling.set_tailBlockNum(tailBlockNum);
    tiling.set_isTailBlock(isTailBlock);
    tiling.set_bigprocessDataNumComputes(bigprocessDataNumComputes);
    tiling.set_smallprocessDataNumComputes(smallprocessDataNumComputes);
    tiling.set_tailbigprocessDataNumComputes(tailbigprocessDataNumComputes);
    tiling.set_tailsmallprocessDataNumComputes(tailsmallprocessDataNumComputes);
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Less : public OpDef {
public:
    explicit Less(const char* name) : OpDef(name) {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore() 
            .SetTiling(optiling::TilingFunc);
        this->SetInferShape(ge::InferShape);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Less);
}
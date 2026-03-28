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
 * @file equal.cpp
 */
#include "equal_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    const uint64_t BLOCK_SIZE = 32;
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        EqualTilingData tiling;
        uint64_t ubSize;
        uint32_t bigprocessDataNum_computes=0;
        uint32_t tailbigprocessDataNum_computes=0;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        auto coreNum = ascendcPlatform.GetCoreNum();
        auto socVersion = ascendcPlatform.GetSocVersion();
        if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
            return ge::GRAPH_FAILED;
        }

        uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint64_t inputLength = inputNum * typeLength;

        if (inputNum == 0){
            return ge::GRAPH_FAILED;
        }
        uint64_t inputBytes = inputLength / inputNum;

        uint64_t ubDataNumber = 6;
        if(context->GetInputDesc(0)->GetDataType() == ge::DT_INT8 || context->GetInputDesc(0)->GetDataType() == ge::DT_UINT8)
        {
            ubDataNumber=12;
        }
        
        if(context->GetInputDesc(0)->GetDataType() == ge::DT_INT32 || context->GetInputDesc(0)->GetDataType() == ge::DT_UINT32 || context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT)
        {
            ubDataNumber=5;
            ubSize=ubSize-256*7-8*1024;
        }
         
        ubSize=ubSize/typeLength;
        uint64_t tileBlockNum = (ubSize / BLOCK_SIZE ) / ubDataNumber;
        uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE);

        uint64_t inputLengthAlgin32 = (((inputNum + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

        if(tileDataNum >= inputNum)
        {
            coreNum=1;
        }
        else
        {
            // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
            coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
        }

       if (BLOCK_SIZE == 0 || coreNum == 0)
       {
            return ge::GRAPH_FAILED;
        }
        uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

        uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE ;
        uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
        uint32_t smallprocessDataNum_computes= (((tileDataNum*typeLength + 256 - 1) / 256) * 256)/typeLength;//计算函数 256字节对齐
        uint32_t tailsmallprocessDataNum_computes= (((smallTailDataNum*typeLength + 256 - 1) / 256) * 256)/typeLength;//尾块计算函数 256字节对齐
    
        everyCoreInputBlockNum += 1;
        uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE ;
        uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum; 
        bigprocessDataNum_computes= (((tileDataNum*typeLength + 256 - 1) / 256) * 256)/typeLength;//计算函数 256字节对齐
        tailbigprocessDataNum_computes= (((bigTailDataNum*typeLength + 256 - 1) / 256) * 256)/typeLength;//尾块计算函数 256字节对齐
        
        tiling.set_smallCoreDataNum((uint32_t)smallCoreDataNum);
        tiling.set_bigCoreDataNum((uint32_t)bigCoreDataNum);
        tiling.set_tileDataNum((uint32_t)tileDataNum);
        tiling.set_smallTailDataNum((uint32_t)smallTailDataNum);
        tiling.set_bigTailDataNum((uint32_t)bigTailDataNum);
        tiling.set_finalSmallTileNum((uint32_t)finalSmallTileNum);
        tiling.set_finalBigTileNum((uint32_t)finalBigTileNum);
        tiling.set_tailBlockNum((uint32_t)tailBlockNum);
        tiling.set_bigprocessDataNum_computes(bigprocessDataNum_computes);
        tiling.set_smallprocessDataNum_computes(smallprocessDataNum_computes);
        tiling.set_tailbigprocessDataNum_computes(tailbigprocessDataNum_computes);
        tiling.set_tailsmallprocessDataNum_computes(tailsmallprocessDataNum_computes);
    
        context->SetBlockDim(coreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops {
class Equal : public OpDef {
public:
    explicit Equal(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_UINT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_UINT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Equal);
}
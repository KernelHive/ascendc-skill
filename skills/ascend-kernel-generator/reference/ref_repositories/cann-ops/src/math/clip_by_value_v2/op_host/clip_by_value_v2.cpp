/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "clip_by_value_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    const uint64_t BLOCK_SIZE = 32;
    const uint64_t BUFFER_NUM = 2;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        ClipByValueV2TilingData tiling;
        uint64_t ubPartNum = 3;
        uint64_t dataTypeLength = 4;
        uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        auto dt = context->GetInputDesc(0)->GetDataType();
        if (dt == ge::DT_INT8) {
            dataTypeLength = 1;
        }
        else if (dt == ge::DT_BF16)
        {
            dataTypeLength = 2;
            ubPartNum = 6;
        }
        else if (dt == ge::DT_FLOAT16)
        {
            dataTypeLength = 2;
        }
        
        uint64_t ubLength = 0;
        uint64_t bigCoreDataNum = 0;
        uint64_t bigCoreLoopNum = 0;
        uint64_t bigCoreTailDataNum = 0;
        
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
        auto coreNum = ascendcPlatform.GetCoreNum();
        
        // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
        uint64_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint64_t inputLength = inputDataNum * dataTypeLength;
        if (coreNum == 0 || BLOCK_SIZE == 0) 
        {
            return ge::GRAPH_FAILED;
        } 
        uint64_t ubPartLength = ubLength / ubPartNum / BUFFER_NUM;
        // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
        uint64_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
        uint64_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;

        // Input data for 32B alignment
        uint64_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        
        if(ubPartDataNum >= inputDataNum)
        {
            coreNum=1;
        }
        else
        {
            // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
            coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
        }
        
        uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
        uint64_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
        
        // Small chunks are calculated and sliced several times using the number of data on each core
        uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
        uint64_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
        smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
        // Tail block calculation for small chunks of data
        uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum-1);
        smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;

        if(0 != tailBlockNum)
        {
            everyCoreInputBlockNum += 1;
            bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
            bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
            bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? bigCoreLoopNum : bigCoreLoopNum + 1;
            bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum-1);
            bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
            context->SetTilingKey(1);
        }
        else
        {
            context->SetTilingKey(0);
        }
        
        tiling.set_smallCoreDataNum(smallCoreDataNum);
        tiling.set_bigCoreDataNum(bigCoreDataNum);
        tiling.set_ubPartDataNum(ubPartDataNum);
        tiling.set_smallCoreTailDataNum(smallCoreTailDataNum);
        tiling.set_bigCoreTailDataNum(bigCoreTailDataNum);
        tiling.set_smallCoreLoopNum(smallCoreLoopNum);
        tiling.set_bigCoreLoopNum(bigCoreLoopNum);
        tiling.set_tailBlockNum(tailBlockNum);
        context->SetBlockDim(coreNum);
        
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
}


namespace ops {
class ClipByValueV2 : public OpDef {
public:
    explicit ClipByValueV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("clip_value_min")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("clip_value_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ClipByValueV2);
}

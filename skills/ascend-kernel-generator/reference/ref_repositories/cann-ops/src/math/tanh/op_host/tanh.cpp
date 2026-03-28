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
 * @file tanh.cpp
 */

#include "tanh_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TanhTilingData tiling;
        uint64_t ubSize = 0;
        uint64_t bigCoreDataNum = 0;
        uint64_t bigTileNum = 0;
        uint64_t finalBigTileNum = 0;
        uint64_t bigTailDataNum = 0;

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        auto coreNum = ascendcPlatform.GetCoreNum();
        auto socVersion = ascendcPlatform.GetSocVersion();
        if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16)
        {
            return ge::GRAPH_FAILED;
        }
        uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        if (inputNum == 0)
        {
            return ge::GRAPH_FAILED;
        }
        uint64_t inputLength = inputNum * typeLength;
        uint64_t inputBytes = inputLength / inputNum;

        if (coreNum == 0 || BLOCK_SIZE == 0)
        {
            return ge::GRAPH_FAILED;
        }
        uint64_t ubDataNumber = (typeLength == 4) ? 5 : 8;
        uint64_t ubPartLength = ubSize / ubDataNumber;

        uint64_t tileBlockNum = ubPartLength / BLOCK_SIZE;
        uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

        uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        if (inputNum <= tileDataNum)
        {
            coreNum = 1;
        }
        else
        {
            coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
            coreNum = (coreNum >= 1) ? coreNum : 1;
        }
        uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

        uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
        uint64_t smallTileNum = smallCoreDataNum / tileDataNum;
        uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

        if (0 != tailBlockNum)
        {
            everyCoreInputBlockNum += 1;
            bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
            bigTileNum = bigCoreDataNum / tileDataNum;
            finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
            bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
            bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
            context->SetTilingKey(1);
        }
        else
        {
            context->SetTilingKey(0);
        }

        tiling.set_smallCoreDataNum((uint32_t)smallCoreDataNum);
        tiling.set_bigCoreDataNum((uint32_t)bigCoreDataNum);
        tiling.set_tileDataNum((uint32_t)tileDataNum);
        tiling.set_smallTailDataNum((uint32_t)smallTailDataNum);
        tiling.set_bigTailDataNum((uint32_t)bigTailDataNum);
        tiling.set_finalSmallTileNum((uint32_t)finalSmallTileNum);
        tiling.set_finalBigTileNum((uint32_t)finalBigTileNum);
        tiling.set_tailBlockNum((uint32_t)tailBlockNum);
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
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Tanh : public OpDef
    {
    public:
        explicit Tanh(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");

            OpAICoreConfig aicConfig;
            aicConfig.Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            aicConfig.Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->AICore().AddConfig("ascend310p", aicConfig);
        }
    };
    OP_ADD(Tanh);
}

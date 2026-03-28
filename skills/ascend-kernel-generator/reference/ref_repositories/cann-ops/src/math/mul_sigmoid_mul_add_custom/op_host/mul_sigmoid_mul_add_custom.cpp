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
 * @file mul_sigmoid_mul_add_custom.cpp
 */
#include "mul_sigmoid_mul_add_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static int64_t CeilDivision(int64_t num1, int64_t num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MulSigmoidMulAddCustomTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    /* 计算输入数据总长度 */
    uint32_t totalLen = x1_shape->GetStorageShape().GetShapeSize();
    uint32_t maxTileLen = 4096;

    /* 先计算分块的大小和block大小 */
    uint32_t totalTileNum = CeilDivision(totalLen, maxTileLen);

    /* 获取系统vector数目, 如果当前TileNum小于等于核数，则只在部分核上分配任务 */
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto blockDim = (totalTileNum < aivNum) ? totalTileNum : aivNum;

    /* Tiling分块切分，分块不可整除，最后1个分块较小，存在2种分块大小 */
    uint32_t completeTileNum, partTileNum;
    uint32_t completeTileLen, partTileLen;

    completeTileNum= totalLen / maxTileLen;     
    completeTileLen  = maxTileLen;  
    if (0 == (totalLen % maxTileLen)) {
        /* 分块大小可整除 */
        partTileNum = 0;
        partTileLen = 0;
    } else {
        /* 分块大小无法整除 */
        partTileNum = 1;
        partTileLen = totalLen % maxTileLen;
    }

    /* 将分块分到不同的block上,当不整除时，前面block分配的tile会多1个 */
    uint32_t frontBlockNum, latterBlockNum;
    uint32_t tileNumInFrontBlock, tileNumInLatterBlock;
    if (0 == (totalTileNum % blockDim)) {
        frontBlockNum = blockDim;
        tileNumInFrontBlock = totalTileNum / blockDim;

        latterBlockNum = 0;
        tileNumInLatterBlock = 0;
    } else {
        tileNumInFrontBlock = totalTileNum / blockDim + 1; 
        frontBlockNum = totalTileNum % blockDim;
                
        tileNumInLatterBlock = totalTileNum / blockDim;
        latterBlockNum = blockDim - frontBlockNum; 
    }

    context->SetBlockDim(blockDim);

    /* 设置TilingData */
    tiling.set_totalLen(totalLen); 
    tiling.set_blockDim(blockDim); 
    tiling.set_completeTileNum(completeTileNum);
    tiling.set_partTileNum(partTileNum);
    tiling.set_completeTileLen(completeTileLen);
    tiling.set_partTileLen(partTileLen);
    tiling.set_totalTileNum(totalTileNum);
    tiling.set_frontBlockNum(frontBlockNum);
    tiling.set_latterBlockNum(latterBlockNum);
    tiling.set_tileNumInFrontBlock(tileNumInFrontBlock);
    tiling.set_tileNumInLatterBlock(tileNumInLatterBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;

    y_shape->SetDim(0, x1_shape->GetDim(1) * 1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MulSigmoidMulAddCustom : public OpDef {
public:
    explicit MulSigmoidMulAddCustom(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mulscalar1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mulscalar2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("addscalar3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("outputs")
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

OP_ADD(MulSigmoidMulAddCustom);
}
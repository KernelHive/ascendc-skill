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
 * @file abs.cpp
 */
#include "abs_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
constexpr int ABS_TILING_0 = 1; // 直接改变符号 :float16, bfp16, float32
constexpr int ABS_TILING_1 = 2; //int32
constexpr int ABS_TILING_2 = 3; //int64
constexpr int ABS_TILING_3 = 4; //complex64
constexpr int ABS_TILING_4 = 5; //float32

namespace optiling {
    constexpr int32_t BUFFER_NUM = 2; 
    constexpr int32_t CALC_ALIGN_NUM =256;
    constexpr int32_t ALIGN_NUM = 32;  

    uint32_t RoundUp(uint32_t a, uint32_t b)
    { 
        if(b == 0) {
            return 0;
        }
        return ((a + b - 1) / b) * b;
    }    
    uint32_t RoundDown(uint32_t a, uint32_t b)
    { 
        if(b == 0) {
            return 0;
        }        
        return (a / b) * b;
    }        
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AbsTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t dataWidth = 4;
    uint32_t ubfactor;  
    uint32_t tilingKey;
    if ((dt == ge::DT_BF16) || (dt == ge::DT_FLOAT16)) { 
            dataWidth = 2;
            ubfactor = 2; 
            tilingKey = ABS_TILING_0;  
    }
    else if ((dt == ge::DT_FLOAT)) {
        dataWidth = 4;
        ubfactor = 2; 
        tilingKey = ABS_TILING_4;    
    }
    else if ((dt == ge::DT_INT32)) {         
        dataWidth = 4;
        ubfactor = 2; 
        tilingKey = ABS_TILING_1;
    }
    else if (dt == ge::DT_INT64) {         
        dataWidth = 8;
        ubfactor = 2; 
        tilingKey = ABS_TILING_2;
    }
    else if (dt == ge::DT_COMPLEX64) {         
        dataWidth = 8;
        ubfactor = 6; 
        tilingKey = ABS_TILING_3;
    }        

    int elementsPerBlock = ALIGN_NUM / dataWidth;       
    int elementsPerRepeat = CALC_ALIGN_NUM / dataWidth;    

    int32_t totalLengthBlockAlign = RoundUp(totalLength, elementsPerBlock);
    if (elementsPerBlock == 0) {
        OP_LOGE(context->GetNodeName(), "elementsPerBlock is 0!");
        return ge::GRAPH_FAILED;
    }
    
    aivNum = (aivNum <  totalLengthBlockAlign / elementsPerBlock) ? aivNum : totalLengthBlockAlign / elementsPerBlock;
    aivNum = (aivNum >= 1) ? aivNum : 1;

    if(aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t smallBlockLength = RoundDown(totalLengthBlockAlign / aivNum, elementsPerBlock);
    if (elementsPerBlock == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t bigDataCoreNum = (totalLengthBlockAlign / elementsPerBlock) % aivNum;
    uint32_t bigBlockLength = 0;
    bigBlockLength = smallBlockLength + elementsPerBlock;

    uint32_t maxTileCalcBlock = ub_size / (CALC_ALIGN_NUM * ubfactor) / BUFFER_NUM;
    uint32_t maxTileLength =  maxTileCalcBlock * (CALC_ALIGN_NUM / dataWidth);
  
    //bigcore
    uint32_t bigTileLength = 0;
    uint32_t bigBlockLengthCalcAlign = 0;
    uint32_t bigTileNum = 0;
    uint32_t bigLasttileLength = 0;
    if(bigDataCoreNum > 0)
    {
        bigBlockLengthCalcAlign = RoundUp(bigBlockLength, elementsPerRepeat);

        (bigBlockLengthCalcAlign ) > maxTileLength ? bigTileLength = maxTileLength : bigTileLength = (bigBlockLengthCalcAlign);
        (bigTileLength >= (CALC_ALIGN_NUM * BUFFER_NUM / dataWidth)) ? bigTileLength = bigTileLength / BUFFER_NUM : bigTileLength = bigTileLength;

        bigTileNum = bigBlockLengthCalcAlign / bigTileLength;
        if (bigBlockLengthCalcAlign % bigTileLength != 0) {
            bigTileNum += 1;
        }
        bigLasttileLength = bigBlockLength - (bigTileNum -1) * bigTileLength;
    }

    //smallcore
    uint32_t smallTileLength = 0;
    uint32_t smallBlockLengthCalcAlign = 0; 
    uint32_t smallTileNum = 0;
    uint32_t smallLasttileLength = 0;

    if(bigDataCoreNum < aivNum)
    {
        smallBlockLengthCalcAlign = RoundUp(smallBlockLength, elementsPerRepeat);

        (smallBlockLengthCalcAlign ) > maxTileLength ? smallTileLength = maxTileLength : smallTileLength = (smallBlockLengthCalcAlign);
        (smallTileLength >= (CALC_ALIGN_NUM * BUFFER_NUM / dataWidth)) ? smallTileLength = smallTileLength / BUFFER_NUM : smallTileLength = smallTileLength;
        
        smallTileNum = smallBlockLengthCalcAlign / smallTileLength;
        if (smallBlockLengthCalcAlign % smallTileLength != 0) {
            smallTileNum += 1;
        }
        smallLasttileLength = smallBlockLength - (smallTileNum -1) * smallTileLength;        
    }
    
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(aivNum);
    tiling.set_bigDataCoreNum(bigDataCoreNum);
    tiling.set_smallBlockLength(smallBlockLength);
    tiling.set_bigBlockLength(bigBlockLength);
    tiling.set_smallTileNum(smallTileNum);
    tiling.set_smallTileLength(smallTileLength);
    tiling.set_smallLasttileLength(smallLasttileLength);
    tiling.set_bigTileNum(bigTileNum);
    tiling.set_bigTileLength(bigTileLength);
    tiling.set_bigLasttileLength(bigLasttileLength);
    tiling.set_dataWidth(dataWidth);
    
    std::cout << "tilingKey = " << tilingKey << std::endl;
    
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
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Abs : public OpDef {
public:
    explicit Abs(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Abs);
}

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
 * @file tril.cpp
 */

#include "tril_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    constexpr int minNum = 1;

    constexpr int keyOne = 1;
    constexpr int keyTwo = 2;
    constexpr int keyThree = 3;
    constexpr int keyFour = 4;

    constexpr int bufferFour = 4;
    constexpr int BlockSize = 32;
    constexpr int computeBatchSize = 256;
    constexpr int sizeHalf = 2;
    constexpr int VAL_ZRRO = 0;

    uint32_t typeSize = VAL_ZRRO;
    uint64_t key = keyOne;
    // buffer for queue
    uint64_t UB_SHARING_NUM = 2;
    int64_t rowLength = VAL_ZRRO;
    int64_t columnLength = VAL_ZRRO;
    int64_t matrixNum = 1, matrixSize = 1;
    int64_t diagVal = VAL_ZRRO;

    uint32_t ALIGN_NUM = VAL_ZRRO;
    uint32_t totalLengthAligned = VAL_ZRRO;
    uint64_t loopCnt = VAL_ZRRO, fullTileLength = VAL_ZRRO, lastTileLength = VAL_ZRRO;
    int32_t fullCnt = VAL_ZRRO, lastCnt = VAL_ZRRO;

    static int setShapeInfo(gert::TilingContext *context){
        const auto inputDataType = context->GetInputTensor(0)->GetDataType();

        switch (inputDataType){
            case ge::DT_FLOAT:
                typeSize = sizeof(float);
                break;
            case ge::DT_FLOAT16:
                typeSize = sizeHalf;
                break;
            default:
                typeSize = sizeof(float);
                break;
        }

        const auto inputShape = context->GetInputTensor(0)->GetOriginShape();
        // class Shape: size_t dim_num_; int64_t dims_[];
        int64_t dimSize = inputShape.GetDimNum(), i = 0;
        // The number 2 is to preserve the last two dimensions
        for (i = 0; i < dimSize - 2; i++){
            matrixNum *= inputShape.GetDim(i);
        }
        rowLength = inputShape.GetDim(i);
        i++;
        columnLength = inputShape.GetDim(i);
        matrixSize = rowLength * columnLength;

        const auto runtime_attrs = context->GetAttrs();
        const int64_t *diagPtr = runtime_attrs->GetInt(0);
        diagVal = *diagPtr;
        if (diagVal < columnLength - 1 && diagVal > -rowLength){
            // Regular
            key = keyOne;
        }else if (diagVal <= -rowLength){
            // The result is itself, TQueBind is enough
            key = keyTwo;
        }else{
            // All zero, just copyIn, Sub and copyOut
            key = keyThree;
        }
        return 0;
    }

    static int setTilingInfo(gert::TilingContext *context,uint64_t ub_size){
        loopCnt = VAL_ZRRO;
        fullTileLength = VAL_ZRRO;
        lastTileLength = VAL_ZRRO;
        fullCnt = VAL_ZRRO; 
        lastCnt = VAL_ZRRO;
        uint64_t ub_length = ((ub_size / typeSize / UB_SHARING_NUM) / ALIGN_NUM * ALIGN_NUM) - ALIGN_NUM;
        if (key == keyOne && diagVal <= 0 && columnLength % (computeBatchSize / typeSize) == 0){
            // A faster method for aligned processing only
            key = keyFour;
            // Double buffer setting
            UB_SHARING_NUM = bufferFour;
            // The result would not be the expected
            if (columnLength == 0){
                columnLength = minNum;
            }
            ub_length = ((ub_size) / typeSize / UB_SHARING_NUM) / columnLength * columnLength;
            loopCnt = (matrixSize + ub_length - 1) / ub_length;
            if (loopCnt == 1){
                fullCnt = 0;
                lastCnt = rowLength;
            }else{
                // The result would not be the expected
                if (columnLength == 0){
                    columnLength = minNum;
                }
                fullCnt = ub_length / columnLength;
                lastCnt = rowLength - fullCnt * (loopCnt - 1);
            }
            // Already aligned
            fullTileLength = fullCnt * columnLength;
            lastTileLength = lastCnt * columnLength;
        }else if (key == keyThree){
            loopCnt = (totalLengthAligned + ub_length - 1) / ub_length;
            UB_SHARING_NUM = bufferFour;
            ub_length = ((ub_size / typeSize / UB_SHARING_NUM) / ALIGN_NUM * ALIGN_NUM) - ALIGN_NUM;
            fullTileLength = ub_length;
            lastTileLength = (totalLengthAligned - fullTileLength * (loopCnt - 1) + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            if (loopCnt == 1){ fullTileLength = 0; }
        }else{
            loopCnt = (totalLengthAligned + ub_length - 1) / ub_length;
            fullTileLength = ub_length;
            lastTileLength = (totalLengthAligned - fullTileLength * (loopCnt - 1) + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            if (loopCnt == 1){ fullTileLength = 0; }
        }
        return 0;
    }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TrilTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto BLOCK_DIM = 1;
    context->SetBlockDim(BLOCK_DIM);
    
    setShapeInfo(context);

    ALIGN_NUM = BlockSize / typeSize;
    totalLengthAligned = (matrixNum * matrixSize + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    uint64_t ub_size=0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    setTilingInfo(context,ub_size);
    
    tiling.set_totalLengthAligned(totalLengthAligned);
    tiling.set_matrixNum(matrixNum);
    tiling.set_matrixSize(matrixSize);
    tiling.set_rowLength(rowLength);
    tiling.set_columnLength(columnLength);
    tiling.set_diagVal(diagVal);

    tiling.set_loopCnt(loopCnt);
    tiling.set_fullTileLength(fullTileLength);
    tiling.set_lastTileLength(lastTileLength);
    tiling.set_fullCnt(fullCnt);
    tiling.set_lastCnt(lastCnt);

    tiling.set_alignNum(ALIGN_NUM);
    tiling.set_typeSize(typeSize);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(key);
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
class Tril : public OpDef {
public:
    explicit Tril(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                    .AddConfig("ascend910b");
    }
};

OP_ADD(Tril);
}
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
 * @file triu.cpp
 */
#include "triu_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling{
    const uint32_t minNum = 1;

    const uint32_t keyOne = 1;
    const uint32_t keyTwo = 2;
    const uint32_t keyThree = 3;
    const uint32_t keyFour = 4;

    const uint32_t bufferFour = 4;
    const uint32_t BlockSize = 32;
    const uint32_t computeBatchSize = 256;
    const uint32_t sizeHalf = 2;
    const uint32_t VAL_ZRRO = 0;

    uint32_t type_Size = VAL_ZRRO;
    uint64_t key_value = keyOne;
    // buffer for queue
    uint64_t ub_Sharing_Num = 2;
    int64_t row_Length = VAL_ZRRO;
    int64_t column_Length = VAL_ZRRO;
    int64_t matrix_Num = 1, matrix_Size = 1;
    int64_t diag_Val = VAL_ZRRO;

    uint32_t align_Num = VAL_ZRRO;
    uint32_t total_Length_Aligned = VAL_ZRRO;
    uint64_t loop_Cnt = VAL_ZRRO, full_Tile_Length = VAL_ZRRO, last_Tile_Length = VAL_ZRRO;
    int32_t full_Cnt = VAL_ZRRO, last_Cnt = VAL_ZRRO;

    static int setShapeInfo(gert::TilingContext *context)
    {
        const auto inputDataType = context->GetInputTensor(0)->GetDataType();

        switch (inputDataType){
            case ge::DT_FLOAT:
                type_Size = sizeof(float);
                break;
            case ge::DT_FLOAT16:
                type_Size = sizeHalf;
                break;
            default:
                type_Size = sizeof(float);
                break;
        }

        const auto inputShape = context->GetInputTensor(0)->GetOriginShape();
        // class Shape: size_t dim_num_; int64_t dims_[];
        int64_t dimSize = inputShape.GetDimNum(), i = 0;
        // The number 2 is to preserve the last two dimensions
        for (i = 0; i < dimSize - 2; i++){
            matrix_Num *= inputShape.GetDim(i);
        }
        row_Length = inputShape.GetDim(i);
        i++;
        column_Length = inputShape.GetDim(i);
        matrix_Size = row_Length * column_Length;

        const auto runtime_attrs = context->GetAttrs();
        const int64_t *diagPtr = runtime_attrs->GetInt(0);
        diag_Val = *diagPtr;
        if (diag_Val < column_Length - 1 && diag_Val > -row_Length){
            // Regular
            key_value = keyOne;
        }else if (diag_Val <= -row_Length){
            // The result is itself, TQueBind is enough
            key_value = keyTwo;
        }else{
            // All zero, just copyIn, Sub and copyOut
            key_value = keyThree;
        }
        return 0;
    }

    static int setTilingInfo(gert::TilingContext *context,uint64_t ub_size){
        loop_Cnt = VAL_ZRRO;
        full_Tile_Length = VAL_ZRRO;
        last_Tile_Length = VAL_ZRRO;
        full_Cnt = VAL_ZRRO; 
        last_Cnt = VAL_ZRRO;
        uint64_t ub_length = ((ub_size / type_Size / ub_Sharing_Num) / align_Num * align_Num) - align_Num;
        if (key_value == keyOne && diag_Val <= 0 && column_Length % (computeBatchSize / type_Size) == 0){
            // A faster method for aligned processing only
            key_value = keyFour;
            // Double buffer setting
            ub_Sharing_Num = bufferFour;
            // The result would not be the expected
            if (column_Length == 0){
                column_Length = minNum;
            }
            ub_length = ((ub_size) / type_Size / ub_Sharing_Num) / column_Length * column_Length;
            loop_Cnt = (matrix_Size + ub_length - 1) / ub_length;
            if (loop_Cnt == 1){
                full_Cnt = 0;
                last_Cnt = row_Length;
            }else{
                // The result would not be the expected
                if (column_Length == 0){
                    column_Length = minNum;
                }
                full_Cnt = ub_length / column_Length;
                last_Cnt = row_Length - full_Cnt * (loop_Cnt - 1);
            }
            // Already aligned
            full_Tile_Length = full_Cnt * column_Length;
            last_Tile_Length = last_Cnt * column_Length;
        }else if (key_value == keyThree){
            loop_Cnt = (total_Length_Aligned + ub_length - 1) / ub_length;
            ub_Sharing_Num = bufferFour;
            ub_length = ((ub_size / type_Size / ub_Sharing_Num) / align_Num * align_Num) - align_Num;
            full_Tile_Length = ub_length;
            last_Tile_Length = (total_Length_Aligned - full_Tile_Length * (loop_Cnt - 1) + align_Num - 1) / align_Num * align_Num;
            if (loop_Cnt == 1){ full_Tile_Length = 0; }
        }else{
            loop_Cnt = (total_Length_Aligned + ub_length - 1) / ub_length;
            full_Tile_Length = ub_length;
            last_Tile_Length = (total_Length_Aligned - full_Tile_Length * (loop_Cnt - 1) + align_Num - 1) / align_Num * align_Num;
            if (loop_Cnt == 1){ full_Tile_Length = 0; }
        }
        return 0;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TriuTilingData tiling;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto coreNum = ascendcPlatform.GetCoreNum();
        auto BLOCK_DIM = 1;
        context->SetBlockDim(BLOCK_DIM);

        setShapeInfo(context);

        align_Num = BlockSize / type_Size;
        total_Length_Aligned = (matrix_Num * matrix_Size + align_Num - 1) / align_Num * align_Num;
        uint64_t ub_size=0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        setTilingInfo(context,ub_size);

        tiling.set_totalLengthAligned(total_Length_Aligned);
        tiling.set_matrixNum(matrix_Num);
        tiling.set_matrixSize(matrix_Size);
        tiling.set_rowLength(row_Length);
        tiling.set_columnLength(column_Length);
        tiling.set_diagVal(diag_Val);

        tiling.set_loopCnt(loop_Cnt);
        tiling.set_fullTileLength(full_Tile_Length);
        tiling.set_lastTileLength(last_Tile_Length);
        tiling.set_fullCnt(full_Cnt);
        tiling.set_lastCnt(last_Cnt);

        tiling.set_alignNum(align_Num);
        tiling.set_typeSize(type_Size);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(key_value);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge{
    static ge::graphStatus InferShape(gert::InferShapeContext *context){
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops{
class Triu : public OpDef{
public:
    explicit Triu(const char *name) : OpDef(name){
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

OP_ADD(Triu);
}
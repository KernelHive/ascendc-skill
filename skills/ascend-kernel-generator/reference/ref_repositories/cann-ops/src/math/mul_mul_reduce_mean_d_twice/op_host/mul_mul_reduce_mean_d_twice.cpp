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
 * @file mul_mul_reduce_mean_d_twice.cpp
 */
 
#include "mul_mul_reduce_mean_d_twice_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
const uint32_t BLOCK_DIM = 48;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MulMulReduceMeanDTwiceTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0); 
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    uint32_t former_num, former_length, tail_num, tail_length;
    uint32_t row_num = x1_shape->GetStorageShape().GetDim(0);
    uint32_t col_num = x1_shape->GetStorageShape().GetDim(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t core_num = ascendcPlatform.GetCoreNumAiv();
    core_num = (BLOCK_DIM < core_num) ? BLOCK_DIM : core_num;
    uint32_t real_core_num = (row_num > core_num) ? core_num : row_num; 
    former_num  = row_num % real_core_num; 
    if(former_num == 0) {
        former_num = real_core_num; 
    }
    former_length = (row_num + real_core_num - 1) / real_core_num; 
    tail_num = real_core_num - former_num;
    tail_length = 0;
    if (tail_num > 0) {
        tail_length = (row_num -former_length * former_num) / tail_num; 
    }

    tiling.set_size(data_sz);
    tiling.set_formerNum(former_num);
    tiling.set_formerLength(former_length);
    tiling.set_tailNum(tail_num);
    tiling.set_tailLength(tail_length);
    tiling.set_tileLength(col_num);
    context->SetBlockDim(real_core_num);

    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    uint32_t typeSize = 2; // half
    AscendC::GetMeanMaxMinTmpSize(col_num, typeSize, typeSize, false, maxValue, minValue);
    tiling.set_shareSize(minValue);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
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
    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) 
    {
        auto dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype);
        return GRAPH_SUCCESS;
    }
}

namespace ops {
class MulMulReduceMeanDTwice : public OpDef {
public:
    explicit MulMulReduceMeanDTwice(const char* name) : OpDef(name)
    {
        this->Input("mul0_input0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mul0_input1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mul1_input0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("add_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
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

OP_ADD(MulMulReduceMeanDTwice);
}
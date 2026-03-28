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
 * @file clip_by_value.cpp
 */
#include "clip_by_value_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    const uint32_t BLOCK_SIZE = 32;

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        ClipByValueTilingData tiling;
        constexpr int32_t NUM = 2;
        uint32_t sizeOfDataType;
        uint32_t totalLengthAligned;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        uint64_t ub_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        auto aivNum = ascendcPlatform.GetCoreNum();

        uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        auto dt = context->GetInputDesc(0)->GetDataType();
        if(dt == ge::DT_INT8){
            sizeOfDataType = 1;
        }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
            sizeOfDataType = 2;
        }
        else if (dt == ge::DT_INT32) {
            sizeOfDataType = 4;
        }
        else{
            sizeOfDataType = 4;
        }

        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDataType;
        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
        tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

        uint32_t block_size = tiling_size * ALIGN_NUM;
        aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
        aivNum = aivNum >= 1 ? aivNum : 1;

        uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
        uint32_t core_remain = totalLength - aivNum * core_size;

        tiling.set_totalLength(totalLength);
        tiling.set_ALIGN_NUM(ALIGN_NUM);
        tiling.set_tiling_size(tiling_size);
        tiling.set_block_size(block_size);
        tiling.set_aivNum(aivNum);
        tiling.set_core_size(core_size);
        tiling.set_core_remain(core_remain);

        context->SetBlockDim(aivNum);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}// namespace optiling

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
} // namespace ge

namespace ops 
{
    class ClipByValue : public OpDef 
    {
    public:
        explicit ClipByValue(const char* name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("clip_value_min")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("clip_value_max")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
            this->SetInferShape(ge::InferShape);
        
            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310b")
                          .AddConfig("ascend910b");
        }
    };
    
    OP_ADD(ClipByValue);
}// namespace ops

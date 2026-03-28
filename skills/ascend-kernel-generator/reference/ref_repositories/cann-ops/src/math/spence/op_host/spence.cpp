/**  
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.  
 * This file is a part of the CANN Open Software.  
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").  
 * Please refer to the License for details. You may not use this file except in compliance with the License.  
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,  
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.  
 * See LICENSE in the root of the software repository for the full text of the License.  
 */  

#include "spence_tiling.h"  
#include "register/op_def_registry.h"  
#include "tiling/platform/platform_ascendc.h"  

namespace optiling {  
const uint32_t BLOCK_SIZE = 32;  
static const uint32_t DT_SIZE1 = 1;  
static const uint32_t DT_SIZE2 = 2;  
static const uint32_t DT_SIZE3 = 4;  
static const uint32_t DT_SIZE4 = 4;  
// Define constants for magic numbers  
static const uint32_t UB_DIVISION_FACTOR = 2;  
static const uint32_t ALIGNMENT_THRESHOLD = 8;  
static const uint32_t CORE_ALIGNMENT_FACTOR = 8;  
static const uint32_t MIN_AIV_NUM = 1;  

static ge::graphStatus TilingFunc(gert::TilingContext* context) {  
    SpenceTilingData tiling;  
    int32_t NUM = 24;  
    uint32_t sizeofdatatype;  
    uint32_t totalLengthAligned;  
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());  
    auto socVersion = ascendcPlatform.GetSocVersion();  
    uint64_t ub_size;  
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();  

    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();  
    auto dt = context->GetInputDesc(0)->GetDataType();  
    if(dt == ge::DT_INT8){  
        sizeofdatatype = DT_SIZE1;  
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){  
        sizeofdatatype = DT_SIZE2;  
    }  
    else if (dt == ge::DT_INT32) {  
        sizeofdatatype = DT_SIZE3;  
    }  
    else{  
        sizeofdatatype = DT_SIZE4;  
    }  

    // Prevent division by zero for sizeofdatatype  
    if (sizeofdatatype == 0) {  
        return ge::GRAPH_FAILED;  
    }  
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;  
    
    // Prevent division by zero for BLOCK_SIZE and NUM  
    if (BLOCK_SIZE == 0 || NUM == 0) {  
        return ge::GRAPH_FAILED;  
    }  
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / UB_DIVISION_FACTOR) / NUM;  
    tiling_size = tiling_size <= ALIGNMENT_THRESHOLD ?   
                tiling_size :   
                (tiling_size / ALIGNMENT_THRESHOLD) * ALIGNMENT_THRESHOLD;  

    uint32_t block_size = tiling_size * ALIGN_NUM;  
    
    // Prevent division by zero for block_size  
    if (block_size == 0) {  
        aivNum = MIN_AIV_NUM;  
    } else {  
        aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);  
        aivNum = aivNum >= MIN_AIV_NUM ? aivNum : MIN_AIV_NUM;  
    }  

    uint32_t core_size = 0;  
    uint32_t core_remain = 0;  
    
    // Prevent division by zero for aivNum and ALIGN_NUM  
    if (aivNum == 0 || ALIGN_NUM == 0) {  
        // Default values if calculation fails  
        core_size = 0;  
        core_remain = totalLength;  
    } else {  
        uint32_t align_factor = ALIGN_NUM * CORE_ALIGNMENT_FACTOR;  
        core_size = (totalLength / aivNum) / align_factor * align_factor;  
        core_remain = totalLength - aivNum * core_size;  
    }  

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
class Spence : public OpDef {  
public:  
    explicit Spence(const char* name) : OpDef(name)  
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

        this->SetInferShape(ge::InferShape);  

        this->AICore()  
            .SetTiling(optiling::TilingFunc);  
        this->AICore().AddConfig("ascend310b")  
                    .AddConfig("ascend910b");  
    }  
};  
OP_ADD(Spence);  
}  
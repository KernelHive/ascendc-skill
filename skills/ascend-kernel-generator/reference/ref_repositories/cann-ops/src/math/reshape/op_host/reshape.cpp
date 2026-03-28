/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <vector>
#include "reshape_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


#define SET(param) tiling.set_##param(param)
#define DEBUG_OUTPUT 1

using std::max;
using std::min;
using namespace ge;

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        ReshapeTilingData tiling;
        auto dev_info = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());              // AscendC Platform Info
        uint32_t core_num = dev_info.GetCoreNum();                                                  // AI Core Number
        context->SetBlockDim(core_num);                                                
        uint64_t ub_size; dev_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);      // Unified Buffer Size
        uint32_t type_sz = GetSizeByDataType(context->GetInputTensor(0)->GetDataType());
        uint32_t dataLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        int axis = *context->GetAttrs()->GetAttrPointer<int>(0);
        int numAxes = *context->GetAttrs()->GetAttrPointer<int>(1);  
        switch(context->GetInputTensor(0)->GetDataType()) {
            case DT_UINT64:
                printf("dt=uint64\n");break;
            case DT_INT64:
                printf("dt=int64\n");break;
            case DT_INT32:
                printf("dt=int32\n");break;
            case DT_UINT32:
                printf("dt=uint32\n");break;
            case DT_FLOAT:
                printf("dt=float32\n");break;
            case DT_UINT16:
                printf("dt=uint16\n");break;
            case DT_INT16:
                printf("dt=int16\n");break;
            case DT_FLOAT16:
                printf("dt=float16\n");break;
            case DT_UINT8:
                printf("dt=uint8\n");break;
            case DT_INT8:
                printf("dt=int8\n");break;
            default:
                printf("dt=UNKNOWN: %d\n", context->GetInputTensor(0)->GetDataType());break;
        }
        if (type_sz == 0) return ge::GRAPH_FAILED;
        uint32_t mini_batch = 32 / type_sz;
        uint32_t ub_size_per_it = 0;
        switch(context->GetInputTensor(0)->GetDataType()) {
            case DT_UINT64:
            case DT_INT64:
                ub_size_per_it = 16;
                break;
            case DT_UINT32:
            case DT_INT32:
            case DT_FLOAT:
                ub_size_per_it = 8;
                break;
            case DT_UINT16:
            case DT_INT16:
            case DT_FLOAT16:
                ub_size_per_it = 4;
                break;
            case DT_UINT8:
            case DT_INT8:
                ub_size_per_it = 2;
                break;
            default:break;
        }
        printf("sizeof(data)= %u, ", type_sz);
        if(ub_size_per_it != 0) { // known data types
            context->SetTilingKey(1);
            uint32_t tileLength = ub_size / ub_size_per_it / mini_batch * mini_batch;
            uint32_t tileNumber = dataLength / tileLength;
            uint32_t reminder = (dataLength % tileLength + mini_batch - 1) / mini_batch * mini_batch;
            SET(tileLength);
            SET(tileNumber);
            SET(reminder);
            printf("UB= %u, TK= 1, tL= %u, tN= %u, r= %u\n", ub_size, tileLength, tileNumber, reminder);
        }
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
class Reshape : public OpDef {
public:
    explicit Reshape(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("num_axes").AttrType(OPTIONAL).Int(-1);
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b").AddConfig("ascend910b");
    }
};

OP_ADD(Reshape);
}

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
#include "gather_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define SET(param) tiling.set_##param(param)
#define DEBUG_OUTPUT 1

using std::max;
using std::min;
using namespace ge;

using std::vector;

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        uint64_t ub_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        auto aivNum = ascendcPlatform.GetCoreNum();
        context->SetBlockDim(aivNum);
        uint32_t xTotalLength = context->GetInputTensor(0)->GetShapeSize();
        uint32_t indicesTotalLength = context->GetInputTensor(1)->GetShapeSize();
        gert::Shape originShape = context->GetOutputShape(0)->GetOriginShape();
        int32_t shapeSize = originShape.GetDimNum();
        vector<int32_t> yShape(shapeSize);
        for(int i = 0; i < shapeSize; ++i) yShape[i] = originShape[i];
        originShape = context->GetInputTensor(1)->GetOriginShape();
        shapeSize = originShape.GetDimNum();
        vector<int32_t> indicesShape(shapeSize);
        for(int i = 0; i < shapeSize; ++i) indicesShape[i] = originShape[i];
        originShape = context->GetInputTensor(0)->GetOriginShape();
        shapeSize = originShape.GetDimNum();
        vector<int32_t> xShape(shapeSize);
        for(int i = 0; i < shapeSize; ++i) xShape[i] = originShape[i];
        int32_t batch_dims = *context->GetAttrs()->GetInt(1);
        if(batch_dims < 0) {
            batch_dims += shapeSize;
        }
        uint32_t sizeOfDataType = GetSizeByDataType(context->GetInputTensor(0)->GetDataType());
        uint32_t batchNumber = 1;
        for(int32_t i = 0; i < batch_dims; ++i) batchNumber *= xShape[i];
        uint32_t batchLength = xTotalLength / batchNumber;
        uint32_t indicesLength = indicesTotalLength / batchNumber;
        uint32_t sliceLength = batchLength / xShape[batch_dims];
        if (sizeOfDataType == 0) return GRAPH_FAILED;
        if (sliceLength * sizeOfDataType <= 32) {
            context->SetTilingKey(1);
            GatherTilingDataScalarCopy tiling;
            SET(batchNumber);
            SET(batchLength);
            SET(indicesLength);
            SET(sliceLength);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        } else { // sliceLength * sizeOfDataType > 32
            uint32_t tileLength = (ub_size / (2 * sizeOfDataType)) / (512 / sizeOfDataType) * (512 / sizeOfDataType);
            uint32_t tileNumber = sliceLength / tileLength;
            uint32_t maxLength = tileNumber * tileLength;
            uint32_t reminder = ((sliceLength % tileLength) + (32 / sizeOfDataType) - 1) / (32 / sizeOfDataType) * (32 / sizeOfDataType);
            context->SetTilingKey(0);
            GatherTilingDataWithDataCopy tiling;
            SET(batchNumber);
            SET(batchLength);
            SET(indicesLength);
            SET(sliceLength);
            SET(maxLength);
            SET(tileLength);
            SET(reminder);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        }
        return GRAPH_SUCCESS;
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
class Gather : public OpDef {
public:
    explicit Gather(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("validate_indices").AttrType(OPTIONAL).Bool(true);
        this->Attr("batch_dims").AttrType(OPTIONAL).Int(0);
        this->Attr("is_preprocessed").AttrType(OPTIONAL).Bool(false);
        this->Attr("negative_index_support").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b").AddConfig("ascend910b");
    }
};

OP_ADD(Gather);
}

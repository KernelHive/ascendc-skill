/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file arange.cpp
 */
#include "arange_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t DTYPE_SIZE2 = 2;
    const uint32_t DTYPE_SIZE4 = 4;
    const uint32_t DTYPE_SIZE8 = 8;
    #define DIVIDE_AND_ALIGN(size, split, align) \
                                    ((((size) / (split)) + ((align)-1)) & ~((align)-1))

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ArangeTilingData tiling;
        uint32_t totalLength = context->GetOutputShape(0)->GetOriginShape().GetShapeSize();
        ge::DataType dtype_out = context->GetOutputDesc(0)->GetDataType();
        uint32_t dtype_size = DTYPE_SIZE2;
        context->SetTilingKey(0);
        switch (dtype_out) {
            case ge::DataType::DT_FLOAT16:
            case ge::DataType::DT_BF16:
                dtype_size = DTYPE_SIZE2;
                break;
            case ge::DataType::DT_FLOAT:
                dtype_size = DTYPE_SIZE4;
                context->SetTilingKey(1);
                break;
            case ge::DataType::DT_INT32:
                dtype_size = DTYPE_SIZE4;
                break;
            case ge::DataType::DT_INT64:
                dtype_size = DTYPE_SIZE8;
                break;
            default:
                dtype_size = DTYPE_SIZE2; 
                break;
        }

        uint64_t ub_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        /*单次api计算大小：将ub 10等份后并按BLOCK_SIZE对齐*/
        uint64_t ub_unit_size = DIVIDE_AND_ALIGN(ub_size, 10, BLOCK_SIZE);
        uint32_t totalNum = totalLength;
        uint32_t unitNum  = ub_unit_size / dtype_size;
        uint32_t unitLoops = totalNum / unitNum;
        uint32_t tailNum = totalNum - unitNum * unitLoops;
        if( tailNum > 0 ) unitLoops += 1;

        tiling.set_dtypeSize(dtype_size);
        tiling.set_totalNum(totalNum);
        tiling.set_unitNum(unitNum);
        tiling.set_unitLoops(unitLoops);
        tiling.set_tailNum(tailNum);

        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    gert::Shape *y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, -1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops
{
    class Arange : public OpDef
    {
    public:
        explicit Arange(const char *name) : OpDef(name)
        {
            this->Input("start")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .Scalar();
            this->Input("end")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .Scalar();
            this->Input("step")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .Scalar();
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend310b")
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(Arange);
}

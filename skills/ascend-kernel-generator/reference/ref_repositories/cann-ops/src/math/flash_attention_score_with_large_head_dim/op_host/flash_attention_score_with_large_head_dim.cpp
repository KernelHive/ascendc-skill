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

#include "register/op_def_registry.h"
#include "tiling_base.h"
using namespace ge;
using namespace AscendC;
using namespace optiling::FA;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FlashAttentionScoreWithLargeHeadDimTiling* basePtr = new FlashAttentionScoreWithLargeHeadDimTiling(context);
    basePtr->DoTiling();
    delete basePtr;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    const gert::Shape* k_shape = context->GetInputShape(1);
    const gert::Shape* v_shape = context->GetInputShape(2);
    gert::Shape* s_m_shape = context->GetOutputShape(0);
    gert::Shape* s_s_shape = context->GetOutputShape(1);
    gert::Shape* o_shape = context->GetOutputShape(2);
    *s_m_shape = gert::Shape({q_shape->GetDim(0), 1, q_shape->GetDim(1),8});
    *s_s_shape = gert::Shape({q_shape->GetDim(0), 1, q_shape->GetDim(1),8});
    *o_shape = *q_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DT_FLOAT);
    context->SetOutputDataType(0, DT_FLOAT);
    context->SetOutputDataType(0, DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

}


namespace ops {
class FlashAttentionScoreWithLargeHeadDim : public OpDef {
public:
    explicit FlashAttentionScoreWithLargeHeadDim(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("head_num").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(FlashAttentionScoreWithLargeHeadDim);
}

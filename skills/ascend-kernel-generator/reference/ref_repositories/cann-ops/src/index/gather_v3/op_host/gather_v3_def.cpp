/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_v3_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

namespace ge {
// input
constexpr int64_t X_INDEX = 0;
constexpr int64_t INDICES_INDEX = 1;
constexpr int64_t AXIS_INDEX = 2;

// output
constexpr int64_t Y_INDEX = 0;

struct GatherInfo {
    int64_t axis;
    int64_t index_batch_dims;
    int64_t x_real_dim_cnt;
    int64_t rank_indices;
};

graphStatus GatherCommonInfer(gert::InferShapeContext* context, const gert::Shape* x_shape,
    const gert::Shape* indies_shape, gert::Shape* out_shape, GatherInfo& gather_info) {
    auto attrs = context->GetAttrs();
    const auto* batchdims = attrs->GetAttrPointer<int64_t>(gather_info.index_batch_dims);
    int64_t batch_dims = *batchdims;
    int64_t axis = gather_info.axis;
    int64_t x_real_dim_cnt = gather_info.x_real_dim_cnt;
    int64_t rank_indices = gather_info.rank_indices;

    gert::Shape real_x_shape = gert::Shape();
    gert::Shape real_indices_shape = gert::Shape();
    real_x_shape.SetDimNum(0);
    real_indices_shape.SetDimNum(0);
    for (int64_t i = 0; i < x_real_dim_cnt; i++) {
        real_x_shape.AppendDim(x_shape->GetDim(i));
    }
    for (int64_t i = 0; i < rank_indices; i++) {
        real_indices_shape.AppendDim(indies_shape->GetDim(i));
    }

    // Adapt the scene of scalar input, set it's shape to (1,)
    if (x_shape->IsScalar()) {
        real_x_shape.AppendDim(1);
        x_real_dim_cnt = 1;
        gather_info.x_real_dim_cnt = 1;
    }

    for (int64_t i = 0; i < axis; i++) {
        out_shape->AppendDim(real_x_shape.GetDim(i));
    }
    // real dim cnt has no existing meaning .Original shape has replace its meaning now
    for (int64_t i = batch_dims; i < rank_indices; i++) {
        out_shape->AppendDim(real_indices_shape.GetDim(i));
    }

    for (int64_t i = axis + 1; i < x_real_dim_cnt; i++) {
        out_shape->AppendDim(real_x_shape.GetDim(i));
    }
    OP_LOGD(context->GetNodeName(), "GatherV3 InferShape end.");
    return GRAPH_SUCCESS;
}

static graphStatus InferShape(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "GatherV3 InferShape begin.");
    const gert::Shape *x_shape = context->GetInputShape(X_INDEX);
    int64_t x_real_dim_cnt = x_shape->GetDimNum();
    auto indies_shape = context->GetInputShape(INDICES_INDEX);
    int64_t rank_indices = indies_shape->GetDimNum();
    const gert::Tensor *axis_tensor = context->GetInputTensor(AXIS_INDEX);
    if (axis_tensor == nullptr) {
        OP_LOGD(context->GetNodeName(), "axis_tensor is nullptr");
        return GRAPH_FAILED;
    }
    int64_t axis = 0;
    auto axisPtr = axis_tensor->GetData<int64_t>();
    if (axisPtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "axisPtr is nullptr");
    } else {
        axis = *axisPtr;
    }
    OP_LOGD(context->GetNodeName(), "infershape get axis=%ld.", axis);

    auto y_shape = context->GetOutputShape(Y_INDEX);
    y_shape->SetDimNum(0);
    
    GatherInfo gatherinfo = {axis, 0, x_real_dim_cnt, rank_indices};
    return GatherCommonInfer(context, x_shape, indies_shape, y_shape, gatherinfo);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "GatherV3 InferDataType begin.");
    const auto inputDataType = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, inputDataType);
    OP_LOGD(context->GetNodeName(), "GatherV3 InferDataType end.");
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GatherV3 : public OpDef {
public:
    explicit GatherV3(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_BOOL, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8,
                      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_BOOL, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("axis")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_BOOL, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8,
                      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_BOOL, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("batchDims")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("negativeIndexSupport")
            .AttrType(OPTIONAL)
            .Bool(false);
        
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GatherV3);
}  // namespace ops
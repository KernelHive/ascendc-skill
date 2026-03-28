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
 * @file conv2d_backprop_filter_v2.cpp
 */

#include <cstdint>
#include <cstdio>
#include "register/op_def_registry.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_CHECK(cond, log_func, return_expr)   \
    if (cond) {                                 \
        log_func;                               \
        return_expr;                            \
    }

namespace ge {
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t FROM_DEPTHWISE_ATTR_IDX = 6;
constexpr size_t kConv2dDimSizeLimit = 4;
static graphStatus InferShapeForConvBackprop(gert::InferShapeContext *context, size_t const_tensor_idx,
                                             const char *const_tensor_name, size_t dim_num) {
    const auto op_name = context->GetNodeName();
    auto y_shape = context->GetOutputShape(0);

    auto const_tensor = context->GetInputTensor(const_tensor_idx);
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
    y_shape->SetDimNum(dim_num);

    auto dtype = const_tensor->GetDataType();
    if (dtype == ge::DT_INT32) {
        auto tensor_data = const_tensor->GetData<int32_t>();
        for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
            y_shape->SetDim(idx, tensor_data[idx]);
        }
    } else if (dtype == ge::DT_INT64) {
        auto tensor_data = const_tensor->GetData<int64_t>();
        for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
            y_shape->SetDim(idx, tensor_data[idx]);
        }
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShapeForConv2DBackpropFilter(gert::InferShapeContext *context) {
    auto ret = InferShapeForConvBackprop(context, 1, "filter_size", kConv2dDimSizeLimit);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const auto runtime_attrs = context->GetAttrs();
    const auto from_depthwise = runtime_attrs->GetBool(FROM_DEPTHWISE_ATTR_IDX);
    if (from_depthwise == nullptr || !*from_depthwise) {
        return ge::GRAPH_SUCCESS;
    }

    auto y_shape = context->GetOutputShape(0);

    const auto y_desc = context->GetOutputDesc(0);
    const auto y_format = y_desc->GetOriginFormat();
    if (y_format == FORMAT_NCHW) {
        y_shape->SetDim(0, y_shape->GetDim(0) * y_shape->GetDim(1));
        y_shape->SetDim(1, 1);
    } else if (y_format == FORMAT_HWCN) {
        y_shape->SetDim(DIM_THREE, y_shape->GetDim(DIM_TWO) * y_shape->GetDim(DIM_THREE));  // 2: C dim, 3: N dim
        y_shape->SetDim(DIM_TWO, 1);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForConv2DBackpropFilterV2(gert::InferDataTypeContext *context) {
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConv2DBackpropFilterV2 enter");
    ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConv2DBackpropFilterV2 enter");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Conv2DBackpropFilterV2)
    .InferShape(InferShapeForConv2DBackpropFilter)
    .InputsDataDependency({1})
    .InferDataType(InferDataTypeForConv2DBackpropFilterV2)
    .PrivateAttr("padding", "")
    .PrivateAttr("from_depthwise", false)
    .PrivateAttr("_op_impl_mode_enum", 0L);
} // namespace ge

namespace ops {
class Conv2DBackpropFilterV2 : public OpDef {
public:
    explicit Conv2DBackpropFilterV2(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
        this->Input("filter_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("out_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04});
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1,1,1,1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "conv2d_backprop_filter_v2")
            .ExtendCfgInfo("opInterface.value", "conv2d_backprop_filter_v2")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .AddConfig("ascend910b", aicore_config);
    }
};
OP_ADD(Conv2DBackpropFilterV2);
} // namespace ops

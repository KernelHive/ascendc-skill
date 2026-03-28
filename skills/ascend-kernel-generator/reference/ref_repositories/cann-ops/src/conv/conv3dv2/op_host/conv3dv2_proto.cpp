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
 * \file conv3dv2_proto.cpp
 * \brief
 */
#include "cube_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"

namespace {
// Conv3D
constexpr size_t kConv3DPadsIdx = 1;
constexpr size_t kConv3DPaddingIdx = 6;
constexpr size_t ATTR_GROUP_INDEX = 3;
using gert::InferShapeContext;
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)      \
  if (ptr == nullptr) {                                \
    std::printf("nullptr error!");                     \
    return ge::GRAPH_FAILED;                           \
  }
}  // namespace

namespace ops {

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

static ge::graphStatus CheckConv3DV2Groups(InferShapeContext* context, const Conv3DInputShapes& shapes) {
  const auto op_name = context->GetNodeName();
  OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetAttrs());
  auto groupPtr = context->GetAttrs()->GetInt(ATTR_GROUP_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, groupPtr);
  int64_t groups = *groupPtr;
  if (groups == 1 && shapes.ic != shapes.kc) {
    OP_LOGE(context->GetNodeName(),
            "The input channel %ld must be the same as the filter channel %ld when groups is 1!",
            shapes.ic, shapes.kc);
    return ge::GRAPH_FAILED;
  }
  if (shapes.ic != shapes.kc * groups) {
    OP_LOGE(context->GetNodeName(),
            "The input channel should be equal to filter_channel*groups. input channel is %ld, filter channel is %ld, "
            "groups is: %ld.",
            shapes.ic, shapes.kc, groups);
    return ge::GRAPH_FAILED;
  }
  if (shapes.kn % groups != 0) {
    OP_LOGE(context->GetNodeName(), "The output channels %ld should be divisible by groups %ld.", shapes.kn, groups);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForConv3DV2(gert::InferShapeContext *context) {
  const auto op_name = context->GetNodeName();
  OPS_CHECK_NULL_WITH_CONTEXT(context, op_name);

  const auto x_desc = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_desc);
  const auto x_format = x_desc->GetOriginFormat();

  const auto y_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);

  const auto y_desc = context->GetOutputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_desc);
  const auto y_format = y_desc->GetOriginFormat();

  Conv3DInputShapes shapes;
  Conv3DAttrs attrs;
  if (GetConv3DXShape(context, 0UL, x_format, false, shapes) && GetConv3DFilterShape(context, 1UL, shapes) &&
      GetConv3DStridesAndDilations(context, x_format, attrs) &&
      GetConv3DPads(context, shapes, kConv3DPadsIdx, kConv3DPaddingIdx, attrs)) {
        std::printf("success to get shape and attrs");
  } else {
    CUBE_INNER_ERR_REPORT(op_name, "failed to infer shape for Conv3D");
    return ge::GRAPH_FAILED;
  }

  if (CheckConv3DV2Groups(context, shapes) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  int64_t od = (shapes.id + attrs.padf + attrs.padb - attrs.dild * (shapes.kd - 1) - 1) / attrs.strd + 1;
  int64_t oh = (shapes.ih + attrs.padu + attrs.padd - attrs.dilh * (shapes.kh - 1) - 1) / attrs.strh + 1;
  int64_t ow = (shapes.iw + attrs.padl + attrs.padr - attrs.dilw * (shapes.kw - 1) - 1) / attrs.strw + 1;

  y_shape->SetDimNum(0);
  if (y_format == ge::FORMAT_NCDHW) {
    y_shape->AppendDim(shapes.in);
    y_shape->AppendDim(shapes.kn);
    y_shape->AppendDim(od);
    y_shape->AppendDim(oh);
    y_shape->AppendDim(ow);
  } else if (y_format == ge::FORMAT_NDHWC) {
    y_shape->AppendDim(shapes.in);
    y_shape->AppendDim(od);
    y_shape->AppendDim(oh);
    y_shape->AppendDim(ow);
    y_shape->AppendDim(shapes.kn);
  } else {
    OP_LOGE(context->GetNodeName(), "The format of output y not support format %s.",
            ge::TypeUtils::FormatToSerialString(y_format).c_str());
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Conv3DV2)
    .InferShape(InferShapeForConv3DV2)
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);
}  // namespace ops

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
 * \file conv3d_backprop_input_v2_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "cube_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_log.h"
#include "register/op_impl_registry.h"
#include <sstream>

namespace ge {
template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}
}
namespace {
constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kConv3dDimSizeLimit = 5;

// Conv3DTranspose
constexpr size_t kConv3DTransposeFilterIdx = 2;
constexpr size_t kConv3DTransposePadsIdx = 1;
constexpr size_t kConv3DTransposeGroupsIdx = 3;
constexpr size_t kConv3DTransposeOutputPaddingIdx = 5;
constexpr size_t kConv3DTransposePaddingIdx = 7;

// Conv2DTranspose
constexpr size_t kConv2DTransposeFmapIdx = 1;
constexpr size_t kConv2DTransposeFilterIdx = 2;
constexpr size_t kConv2DTransposePadsIdx = 1;
constexpr size_t kConv2DTransposeGroupsIdx = 3;
constexpr size_t kConv2DTransposeOutputPaddingIdx = 5;
constexpr size_t kConv2DTransposePaddingIdx = 7;
constexpr size_t kConv2DTransposeAutoPadIdx = 8;

// Deconvolution
constexpr size_t kDeconvolutionFmapIdx = 0;
constexpr size_t kDeconvolutionFilterIdx = 1;
constexpr size_t kDeconvolutionPadsIdx = 1;
constexpr size_t kDeconvolutionGroupsIdx = 3;
constexpr size_t kDeconvolutionPaddingIdx = 6;
constexpr size_t kDeconvolutionAutoPadIdx = 7;

using gert::InferShapeContext;
using ge::Format;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_HWCN;
using ge::GRAPH_FAILED;
using ge::graphStatus;
}  // namespace
namespace ops {

template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}


static graphStatus InferShapeForConvBackprop(InferShapeContext *context, size_t const_tensor_idx,
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
    return GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static bool GetConv3DTransposeOutputPadding(InferShapeContext *context, Format x_format,
                                            int64_t output_padding[3]) {  // 3: DHW
  const auto runtime_attrs = context->GetAttrs();
  const auto output_padding_list =
      runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv3DTransposeOutputPaddingIdx);

  const auto output_padding_data = reinterpret_cast<const int64_t *>(output_padding_list->GetData());
  size_t idx = 0;
  if (x_format == FORMAT_NCDHW) {
    output_padding[idx++] = output_padding_data[kDDimNCDHWIdx];
    output_padding[idx++] = output_padding_data[kHDimNCDHWIdx];
    output_padding[idx++] = output_padding_data[kWDimNCDHWIdx];
  } else {
    // FORMAT_NDHWC, already checked in GetConv3DXShape, else is enough
    output_padding[idx++] = output_padding_data[kDDimNDHWCIdx];
    output_padding[idx++] = output_padding_data[kHDimNDHWCIdx];
    output_padding[idx++] = output_padding_data[kWDimNDHWCIdx];
  }

  return true;
}

static bool CheckOutputAllZero(const gert::Shape *shape) {
  size_t dim_num = shape->GetDimNum();
  for (size_t idx = 0; idx < dim_num; ++idx) {
    if (shape->GetDim(idx) != 0) {
      return false;
    }
  }

  return true;
}

static graphStatus InferShapeForConv3DTranspose(InferShapeContext *context) {
  auto ret = InferShapeForConvBackprop(context, 0, "input_size", kConv3dDimSizeLimit);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  auto y_shape = context->GetOutputShape(0);

  if (CheckOutputAllZero(y_shape)) {
    const auto x_desc = context->GetInputDesc(0);
    const auto x_format = x_desc->GetOriginFormat();

    Conv3DInputShapes shapes;
    Conv3DAttrs attrs;
    int64_t output_padding[3];  // 3: DHW
    if (GetConv3DXShape(context, 1UL, x_format, false, shapes) &&
        GetConv3DFilterShape(context, kConv3DTransposeFilterIdx, shapes) &&
        GetConv3DStridesAndDilations(context, x_format, attrs) &&
        GetConvGroups(context, kConv3DTransposeGroupsIdx, attrs) &&
        GetConv3DPads(context, shapes, kConv3DTransposePadsIdx, kConv3DTransposePaddingIdx, attrs) &&
        GetConv3DTransposeOutputPadding(context, x_format, output_padding)) {
      int64_t output_d = attrs.strd * (shapes.id - 1) + output_padding[0] + ((shapes.kd - 1) * attrs.dild + 1) -
                         (attrs.padf + attrs.padb);
      int64_t output_h = attrs.strh * (shapes.ih - 1) + output_padding[1] + ((shapes.kh - 1) * attrs.dilh + 1) -
                         (attrs.padu + attrs.padd);
      // 2: w dim
      int64_t output_w = attrs.strw * (shapes.iw - 1) + output_padding[2] + ((shapes.kw - 1) * attrs.dilw + 1) -
                         (attrs.padl + attrs.padr);

      y_shape->SetDimNum(0);
      if (x_format == FORMAT_NCDHW) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(shapes.kc * attrs.groups);
        y_shape->AppendDim(output_d);
        y_shape->AppendDim(output_h);
        y_shape->AppendDim(output_w);
      } else if (x_format == FORMAT_NDHWC) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(output_d);
        y_shape->AppendDim(output_h);
        y_shape->AppendDim(output_w);
        y_shape->AppendDim(shapes.kc * attrs.groups);
      } else {
        return false;
      }
      return ge::GRAPH_SUCCESS;
    }

    return  ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForConv3DTransposeV2(gert::InferDataTypeContext *context) {
  auto xDataType = context->GetInputDataType(1);
  ge::graphStatus ret = context->SetOutputDataType(0, xDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Conv3DTransposeV2)
    .InferShape(InferShapeForConv3DTranspose)
    .InferDataType(InferDataTypeForConv3DTransposeV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);
}  // namespace ops
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
 * \file convolution_backprop_infer_fns.cc
 * \brief
 */
#include "cube_util.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_log.h"
#include "register/op_impl_registry.h"

namespace {
#define OP_CHECK(cond, log_func, return_expr) \
  if (cond)                                   \
  {                                           \
    log_func;                                 \
    return_expr;                              \
  }
  
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)


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
namespace ge {
using namespace ops;
static graphStatus InferShapeForConvBackprop(InferShapeContext *context, size_t const_tensor_idx,
                                             const char *const_tensor_name, size_t dim_num) {
  const auto op_name = context->GetNodeName();
  auto y_shape = context->GetOutputShape(0);
  OP_CHECK(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "y shape"),
    return GRAPH_FAILED);

  auto const_tensor = context->GetInputTensor(const_tensor_idx);
  OP_CHECK(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name),
        return GRAPH_FAILED);
  size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
  OP_CHECK(const_tensor_dim_num != dim_num,
        CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid", const_tensor_name, const_tensor_dim_num),
        return GRAPH_FAILED);
  y_shape->SetDimNum(dim_num);

  auto dtype = const_tensor->GetDataType();
  if (dtype == ge::DT_INT32) {
    auto tensor_data = const_tensor->GetData<int32_t>();
    OP_CHECK(tensor_data == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor data", const_tensor_name),
          return GRAPH_FAILED);
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else if (dtype == ge::DT_INT64) {
    auto tensor_data = const_tensor->GetData<int64_t>();
    OP_CHECK(tensor_data == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor data", const_tensor_name),
          return GRAPH_FAILED);
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else {
    CUBE_INNER_ERR_REPORT(op_name, "tensor %s not support dtype %s", const_tensor_name,
                          ge::TypeUtils::DataTypeToAscendString(dtype).GetString());
    return GRAPH_FAILED;
  }

  OP_LOGD(context->GetNodeName(), "y_shape: %s", ge::Shape2String(*y_shape).c_str());
  return ge::GRAPH_SUCCESS;
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

static bool GetConv2DTransposeOutputPadding(InferShapeContext *context, Format x_format,
                                            int64_t output_padding[2]) {  // 2: HW
  const auto runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
  const auto output_padding_list =
      runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv2DTransposeOutputPaddingIdx);
  OP_LOGE_IF(output_padding_list == nullptr, false, context->GetNodeName(), "failed to get output_padding attrs");
  OP_LOGE_IF(output_padding_list->GetSize() != kConv2dDimSizeLimit, false, context->GetNodeName(),
             "The output_padding should be 4d, actual dim num: %zu", output_padding_list->GetSize());

  const auto output_padding_data = reinterpret_cast<const int64_t *>(output_padding_list->GetData());
  size_t idx = 0;
  if (x_format == ge::FORMAT_NCHW) {
    output_padding[idx++] = output_padding_data[kHDimNCHWIdx];
    output_padding[idx++] = output_padding_data[kWDimNCHWIdx];
  } else {
    // FORMAT_NHWC, already checked in GetConv2DXShape, else is enough
    output_padding[idx++] = output_padding_data[kHDimNHWCIdx];
    output_padding[idx++] = output_padding_data[kWDimNHWCIdx];
  }

  return true;
}

static graphStatus InferShapeForConv2DTranspose(InferShapeContext *context) {
  auto ret = InferShapeForConvBackprop(context, 0, "input_size", kConv2dDimSizeLimit);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  auto y_shape = context->GetOutputShape(0);
  OP_CHECK(y_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y shape is null"), return ge::GRAPH_FAILED);

  if (CheckOutputAllZero(y_shape)) {
    const auto x_desc = context->GetInputDesc(1);
    OP_CHECK(x_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x desc is null"), return ge::GRAPH_FAILED);
    const auto x_format = x_desc->GetOriginFormat();

    Conv2DInputShapes shapes;
    Conv2DAttrs attrs;
    int64_t output_padding[2];  // 2: HW
    const std::vector<size_t> idx_vec = {kConv2DTransposePadsIdx, kConv2DTransposePaddingIdx,
                                         kConv2DTransposeAutoPadIdx};
    if (GetConv2DXShape(context, kConv2DTransposeFmapIdx, x_format, shapes) &&
        GetConv2DFilterShape(context, kConv2DTransposeFilterIdx, shapes) &&
        GetConv2DStridesAndDilations(context, x_format, attrs, false) &&
        GetConv2DGroups(context, kConv2DTransposeGroupsIdx, attrs) &&
        GetConv2DPads(context, shapes, idx_vec, attrs) &&
        GetConv2DTransposeOutputPadding(context, x_format, output_padding)) {
      int64_t output_h = attrs.str_h * (shapes.ih - 1) + output_padding[0] + ((shapes.kh - 1) * attrs.dil_h + 1) -
                         (attrs.pad_u + attrs.pad_d);
      // 2: w dim
      int64_t output_w = attrs.str_w * (shapes.iw - 1) + output_padding[1] + ((shapes.kw - 1) * attrs.dil_w + 1) -
                         (attrs.pad_l + attrs.pad_r);

      y_shape->SetDimNum(0);
      if (x_format == ge::FORMAT_NCHW) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(shapes.kc * attrs.groups);
        y_shape->AppendDim(output_h);
        y_shape->AppendDim(output_w);
      } else if (x_format == ge::FORMAT_NHWC) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(output_h);
        y_shape->AppendDim(output_w);
        y_shape->AppendDim(shapes.kc * attrs.groups);
      } else {
        OP_LOGE(context->GetNodeName(), "The format of output y not support format %s.",
                ge::TypeUtils::FormatToAscendString(x_format).GetString());
        return false;
      }

      OP_LOGD(context->GetNodeName(), "y_shape: %s", ge::Shape2String(*y_shape).c_str());
      return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}


IMPL_OP_INFERSHAPE(Conv2DTransposeV2)
    .InferShape(InferShapeForConv2DTranspose)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("auto_pad", "NOTSET")
    .PrivateAttr("output_shape", std::vector<int64_t>{})
    .PrivateAttr("_op_impl_mode_enum", -1L);
}

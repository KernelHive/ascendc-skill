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
 * \file conv2d_backprop_input_v2_proto.cpp
 * \brief
 */

#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include <cstdio>

#define CUBE_INNER_ERR_REPORT(ignore, ...) \
    do { \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } while(0)
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK(cond, log_func, return_expr) \
  if (cond) {                                 \
    log_func;                                 \
    return_expr;                              \
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
  
static graphStatus InferShapeForConvBackprop(InferShapeContext *context, size_t const_tensor_idx,
                                             const char *const_tensor_name, size_t dim_num) {
  OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"),
    return GRAPH_FAILED);
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

static graphStatus InferShapeForConv2DBackpropInput(InferShapeContext *context) {
  return InferShapeForConvBackprop(context, 0, "input_size", kConv2dDimSizeLimit);
}

IMPL_OP_INFERSHAPE(Conv2DBackpropInputV2)
    .InferShape(InferShapeForConv2DBackpropInput)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", -1L);
}  // namespace ops

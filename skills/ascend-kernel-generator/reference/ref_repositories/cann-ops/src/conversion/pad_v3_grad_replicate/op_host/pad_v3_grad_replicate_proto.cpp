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
 * \file pad_v3_grad_replicate_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t INDEX_PADDINGS = 1;
constexpr size_t INDEX_Y = 0;
constexpr size_t INDEX_PADDINGS_CONTIGUOUS = 1;
constexpr size_t PAIR = 2;
// tools api
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                  \
  if ((ptr) == nullptr) {                                                                          \
      const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();  \
      std::printf("op[%s], %s is nullptr, error!", name, #ptr);                                    \
      return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace

namespace ops {
template <typename T>
static ge::graphStatus PadV3GradInfershape(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                           const gert::Tensor* paddings_tensor, gert::Shape* y_shape) {
  const T* paddings_value = paddings_tensor->GetData<T>();
  const size_t paddings_num = static_cast<size_t>(paddings_tensor->GetShapeSize());
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const bool* paddings_contiguous = attrs->GetAttrPointer<bool>(INDEX_PADDINGS_CONTIGUOUS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, paddings_contiguous);
  // input shape check
  size_t input_dim_size = static_cast<size_t>(x_shape->GetDimNum());
  // infer by paddings_contiguous
  y_shape->SetDimNum(input_dim_size);
  int64_t index_cof = 1;
  size_t index_offset = input_dim_size;
  if (*paddings_contiguous) {
    index_cof = static_cast<int64_t>(PAIR);
    index_offset = 1;
  }
  for (size_t i = 0; i < input_dim_size; ++i) {
    auto pad_front = static_cast<size_t>(paddings_value[index_cof * i]);
    auto pad_end = static_cast<size_t>(paddings_value[index_cof * i + index_offset]);
    y_shape->SetDim(i, x_shape->GetDim(i) - pad_front - pad_end);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4PadV3Grad(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(INDEX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  gert::Shape* y_shape = context->GetOutputShape(INDEX_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  const gert::Tensor* paddings_tensor = context->GetInputTensor(INDEX_PADDINGS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, paddings_tensor);
  ge::DataType paddings_dtype = paddings_tensor->GetDataType();
  switch (paddings_dtype) {
    case ge::DT_INT32: {
      return PadV3GradInfershape<int32_t>(context, x_shape, paddings_tensor, y_shape);
    }
    case ge::DT_INT64: {
      return PadV3GradInfershape<int64_t>(context, x_shape, paddings_tensor, y_shape);
    }
    default:
      return ge::GRAPH_FAILED;
  }
}

IMPL_OP_INFERSHAPE(PadV3GradReplicate)
    .InferShape(InferShape4PadV3Grad)
    .InputsDataDependency({INDEX_PADDINGS});
}  // namespace ops
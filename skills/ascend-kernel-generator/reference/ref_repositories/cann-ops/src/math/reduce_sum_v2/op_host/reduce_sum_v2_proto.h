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
 * \file reduce_sum_v2_proto.h
 * \brief
 */
#ifndef REDUCE_SUM_V2_PROTO_H
#define REDUCE_SUM_V2_PROTO_H
#include "register/op_def_registry.h"
#include "error_util.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
    do {                                                      \
        std::printf("op[%s], %s", op_name, err_msg);          \
    } while (0)   

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }                                     

// tools api
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

template <typename T1, typename T2>
bool IsDimValid(const T1 shape_size, const T2 dim_value) {
  int64_t minimum_num = static_cast<int64_t>(shape_size) * (-1);
  int64_t maximum_num = static_cast<int64_t>(shape_size) - 1;

  return static_cast<int64_t>(dim_value) >= minimum_num && static_cast<int64_t>(dim_value) <= maximum_num;
}

std::vector<int64_t> ToVector(const gert::Shape& shape) {
  size_t shape_size = shape.GetDimNum();
  std::vector<int64_t> shape_vec(shape_size, 0);

  for (size_t i = 0; i < shape_size; i++) {
    shape_vec[i] = shape.GetDim(i);
  }
  return shape_vec;
}

std::string ToString(const gert::Shape& shape) {
  return ge::DebugString(ToVector(shape));
}

namespace ops {
template <typename T>
ge::graphStatus ReduceDimsWithKeepDims(const gert::Shape* x_shape, const T* axes_dims, int32_t axes_size,
                                       gert::Shape* output_shape) {
  T dim_num = x_shape->GetDimNum();
  const bool is_scalar = x_shape->GetDimNum() == 0;
  dim_num = is_scalar ? 1 : dim_num;
  *output_shape = *x_shape;
  for (int32_t i = 0; i < axes_size; i++) {
    OP_CHECK(!IsDimValid(dim_num, axes_dims[i]), OP_LOGE("reduce", "axes_dims is invalid"), return ge::GRAPH_FAILED);
    if (is_scalar) {
      // no need to update output shape, when input is scalar
      continue;
    }
    T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
    output_shape->SetDim(dim, 1);
  }
  OP_LOGD("ReduceDimsWithKeepDims", "after reduce output shape is %s.", ToString(*output_shape).c_str());
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDimsWithoutKeepDims(const gert::Shape* x_shape, const T* axes_dims, int32_t axes_size,
                                          gert::Shape* output_shape) {
  T dim_num = x_shape->GetDimNum();
  output_shape->SetDimNum(0);
  for (T j = 0; j < dim_num; j++) {
    bool reduce_flag = false;
    for (int32_t i = 0; i < axes_size; i++) {
      OP_CHECK(!IsDimValid(dim_num, axes_dims[i]), OP_LOGE("reduce", "axes_dims is invalid"), return ge::GRAPH_FAILED);
      T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
      if (dim == j) {
        reduce_flag = true;
        break;
      }
    }
    if (!reduce_flag) {
      output_shape->AppendDim(x_shape->GetDim(j));
    }
  }

  OP_LOGD("ReduceDimsWithoutKeepDims", "after reduce output shape is %s.", ToString(*output_shape).c_str());
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDims(const gert::Shape* x_shape, const gert::Tensor* axes_tensor, int32_t axes_size,
                           const bool keep_dims, gert::Shape* output_shape) {
  const T* axes_dims = axes_tensor->GetData<T>();
  if (keep_dims) {
    return ReduceDimsWithKeepDims<T>(x_shape, axes_dims, axes_size, output_shape);
  }
  return ReduceDimsWithoutKeepDims<T>(x_shape, axes_dims, axes_size, output_shape);
}
}  // namespace ops
#endif

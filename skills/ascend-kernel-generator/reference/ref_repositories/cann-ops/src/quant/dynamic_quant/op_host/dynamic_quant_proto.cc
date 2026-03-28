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
 * \file dynamic_quant_proto.cc
 * \brief
 */
#include "register/op_def_registry.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

#define OP_CHECK(cond, log_func, return_expr) \
  if (cond) {                                 \
    log_func;                                 \
    return_expr;                              \
  }

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)        \
  do {                                                               \
    std::printf("op[%s], %s", op_name, err_msg);                     \
  } while (0)

using namespace ge;
namespace ops {
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;
static constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT = 2;

static ge::graphStatus CheckComputeNodeNumMerged(gert::InferShapeContext* context) {
  // check input and output number
  if (context->GetComputeNodeOutputNum() != OUTPUT_NUM_DYNAMIC_QUANT) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context) {
  if (context == nullptr || CheckComputeNodeNumMerged(context) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }

  for (uint32_t i = 0; i < OUTPUT_NUM_DYNAMIC_QUANT; ++i) {
    if (context->GetOutputShape(i) == nullptr) {
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantInferShape(gert::InferShapeContext* context) {
  if (InferShapeCheck(context) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }

  const gert::Shape* xShape = context->GetInputShape(0);
  gert::Shape* yShape = context->GetOutputShape(0);
  gert::Shape* scaleShape = context->GetOutputShape(1);
  *yShape = *xShape;
  scaleShape->SetDimNum(xShape->GetDimNum() - 1);
  for (uint32_t i = 0; i < xShape->GetDimNum() - 1; ++i) {
    scaleShape->SetDim(i, xShape->GetDim(i));
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantInferDataType(gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }
  OP_LOGD(context->GetNodeName(), "DynamicQuantInferDataType begin");
  ge::DataType yDtype = ge::DT_INT8;
  auto* attrs = context->GetAttrs();
  if (attrs != nullptr) {
    const int32_t *pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
    if (pDstDtype != nullptr) {
      int32_t dstDtype = *pDstDtype;
      OP_CHECK(dstDtype != DTYPE_INT8 && dstDtype != DTYPE_INT4,
              VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DynamicQuant", "attr dst_type only support 2(int8) and 29(int4)"),
              return ge::GRAPH_FAILED);
      yDtype = static_cast<ge::DataType>(dstDtype);
    }
  }
  context->SetOutputDataType(0, yDtype);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  OP_LOGD(context->GetNodeName(), "DynamicQuantInferDataType end");
  return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DynamicQuant).InferShape(DynamicQuantInferShape).InferDataType(DynamicQuantInferDataType);
}  // namespace ops
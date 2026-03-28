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
 * \file group_quant_proto.cc
 * \brief
 */
#include <numeric>
#include "register/op_impl_registry.h"

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

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

using namespace ge;
namespace ops {
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;

static ge::graphStatus InferDataType4GroupQuant(gert::InferDataTypeContext* context) {
  OP_CHECK(context == nullptr,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT("GroupQuant", "InferDataTypeContext is nullptr"),
           return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "InferDataType4GroupQuant begin");

  ge::DataType yDtype = ge::DT_INT8;
  const int32_t *pDstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
  if (pDstDtype != nullptr) {
    int32_t dstDtype = *pDstDtype;
    OP_CHECK(dstDtype != DTYPE_INT8 && dstDtype != DTYPE_INT4,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT("GroupQuant", "attr dst_type only support 2(int8) and 29(int4)"),
             return ge::GRAPH_FAILED);
    yDtype = static_cast<ge::DataType>(dstDtype);
  }
  context->SetOutputDataType(0, yDtype);

  OP_LOGD(context->GetNodeName(), "InferDataType4GroupQuant end");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GroupQuant(gert::InferShapeContext* context) {
  OP_CHECK(context == nullptr,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT("GroupQuant", "InferShapeContext is nullptr"),
           return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "InferShape4GroupQuant begin");

  const gert::Shape* xShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

  gert::Shape* yShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

  *yShape = *xShape;

  OP_LOGD(context->GetNodeName(), "InferShape4GroupQuant end.");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupQuant).InferDataType(InferDataType4GroupQuant)
                              .InferShape(InferShape4GroupQuant);
}  // namespace ops
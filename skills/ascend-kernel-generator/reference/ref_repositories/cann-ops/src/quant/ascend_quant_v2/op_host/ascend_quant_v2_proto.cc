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
 * \file ascend_quant_v2_proto.cc
 * \brief
 */
#include "register/op_def_registry.h"

using namespace ge;

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

// tools api
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

namespace ops {
constexpr size_t g_AttrDstType = 2;

static graphStatus InferShapeForAscendQuantV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForAscendQuantV2");
  const gert::Shape* inputXShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inputXShape);

  gert::Shape* outputShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, outputShape);

  *outputShape = *inputXShape;
  OP_LOGD(context->GetNodeName(), "End to do InferShapeForAscendQuantV2");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForAscendQuantV2(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAscendQuantV2");
  const int32_t *dstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(g_AttrDstType);
  ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
  context->SetOutputDataType(0, outDtype);
  OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAscendQuantV2");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendQuantV2).InferShape(InferShapeForAscendQuantV2)
                      .InferDataType(InferDataTypeForAscendQuantV2);
}  // namespace ops
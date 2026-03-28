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
 * \file aclnn_embedding_bag.cpp
 * \brief
 */

#include "embedding_bag_l0.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

namespace l0op {

OP_TYPE_REGISTER(EmbeddingBag);

static const std::initializer_list<op::DataType> WEIGHT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_BF16,
                                                                              op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> INT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_INT64};

//根据dtype判断算子是否支持aicore
static inline bool IsAiCoreSupport(const aclTensor* weight, const aclTensor* indices,
                                   const aclTensor* offsets, const aclTensor* perSampleWeights) {
  if (perSampleWeights != nullptr && perSampleWeights->GetDataType() != weight->GetDataType()) {
    return false;
  }
  return op::CheckType(weight->GetDataType(), WEIGHT_DTYPE_SUPPORT_LIST) &&
         op::CheckType(indices->GetDataType(), INT_DTYPE_SUPPORT_LIST) &&
         op::CheckType(offsets->GetDataType(), INT_DTYPE_SUPPORT_LIST);
}

static op::Shape GetOutPutShape(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                                bool includeLastOffset, aclOpExecutor* executor) {
  op::Shape outputShape;
  int64_t offsetSize = offsets->GetViewShape().GetDim(0);
  if (includeLastOffset) {
    offsetSize -= 1;
  }
  outputShape.AppendDim(offsetSize);
  outputShape.AppendDim((weight->GetViewShape())[1]);
  return outputShape;
}

const std::tuple<aclTensor*,aclTensor*,aclTensor*,aclTensor*> EmbeddingBag(const aclTensor* weight,
                                                                           const aclTensor* indices,
                                                                           const aclTensor* offsets,
                                                                           bool scaleGradByFreq,
                                                                           const std::string &modeStr,
                                                                           bool sparse,
                                                                           const aclTensor* perSampleWeights,
                                                                           bool includeLastOffset,
                                                                           int64_t paddingIdx,
                                                                           aclOpExecutor *executor) {
  if(!IsAiCoreSupport(weight, indices, offsets, perSampleWeights)) {
    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>((aclTensor*)nullptr, (aclTensor*)nullptr,
                                                                      (aclTensor*)nullptr, (aclTensor*)nullptr);
  }
  //申请output_tensor的Tensor
  auto outputShape = GetOutPutShape(weight, indices, offsets, includeLastOffset, executor);
  auto outputTensor = executor->AllocTensor(outputShape, weight->GetDataType());

  // 申请offset2bag的Tensor
  op::Shape offset2bagShape;
  offset2bagShape.AppendDim(indices->GetViewShape().GetDim(0));
  auto offset2bag = executor->AllocTensor(offset2bagShape, indices->GetDataType(), op::Format::FORMAT_ND);

  // 申请bagSize的Tensor
  auto bagSize = executor->AllocTensor(offsets->GetViewShape(), offsets->GetDataType(), op::Format::FORMAT_ND);
  if (includeLastOffset) {
    op::Shape bagSizeShape;
    bagSizeShape.AppendDim(offsets->GetViewShape().GetDim(0) -1);
    bagSize = executor->AllocTensor(bagSizeShape, offsets->GetDataType(), op::Format::FORMAT_ND);
  }

  // 申请maxIndices的Tensor
  aclTensor* maxIndices;
  if (modeStr == "max") {
    auto maxIndicesShape = GetOutPutShape(weight, indices, offsets, includeLastOffset, executor);
    maxIndices = executor->AllocTensor(maxIndicesShape, offsets->GetDataType(), op::Format::FORMAT_ND);
  }else {
    maxIndices = executor->AllocTensor(offsets->GetViewShape(), offsets->GetDataType(), op::Format::FORMAT_ND);
    if (includeLastOffset) {
        op::Shape maxIndicesShape;
        maxIndicesShape.AppendDim(offsets->GetViewShape().GetDim(0) - 1);
        maxIndices = executor->AllocTensor(maxIndicesShape, offsets->GetDataType(), op::Format::FORMAT_ND);
    }
  }
  if (outputTensor == nullptr || offset2bag == nullptr ||
      bagSize == nullptr || maxIndices == nullptr) {
    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>((aclTensor*)nullptr, (aclTensor*)nullptr,
                                                                      (aclTensor*)nullptr, (aclTensor*)nullptr);
  }

  L0_DFX(EmbeddingBag, weight, indices, offsets, scaleGradByFreq, modeStr, sparse, perSampleWeights,
         includeLastOffset);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(EmbeddingBag,
                                         OP_INPUT(weight, indices, offsets, perSampleWeights),
                                         OP_OUTPUT(outputTensor, offset2bag, bagSize, maxIndices),
                                         OP_ATTR(modeStr, scaleGradByFreq, sparse, includeLastOffset, paddingIdx));
  if (ret !=  ACL_SUCCESS) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingBagAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
      return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr, nullptr);
  }
  return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(outputTensor, offset2bag, bagSize, maxIndices);
}
}//l0op

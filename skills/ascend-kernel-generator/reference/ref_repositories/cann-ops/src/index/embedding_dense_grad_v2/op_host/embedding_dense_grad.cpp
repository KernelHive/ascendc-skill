/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file embedding_Dense_grad.cpp
 * \brief
 */

#include "embedding_dense_grad.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"

namespace l0op {

OP_TYPE_REGISTER(EmbeddingDenseGrad);
OP_TYPE_REGISTER(EmbeddingDenseGradV2);

const aclTensor *EmbeddingDenseGradV2(const aclTensor *grad, const aclTensor *sortIndices,
                                      const aclTensor *posIdx, uint64_t numWeights, uint64_t paddingIdx,
                                      bool scaleGradByFreq, aclOpExecutor *executor)
{
  L0_DFX(EmbeddingDenseGradV2,  grad, sortIndices, posIdx, numWeights, paddingIdx, scaleGradByFreq);
  // shape推导
  auto gradOutputShape = grad->GetViewShape();
  int64_t gradOutputShapeLen = gradOutputShape.GetDimNum();
  int64_t outShapeLen = 2;
  op::Shape outShape;
  outShape.SetDimNum(outShapeLen);
  outShape.SetDim(0, numWeights);
  outShape.SetDim(1, gradOutputShape.GetDim(gradOutputShapeLen - 1));

  // 创建输出Tensor
  auto out = executor->AllocTensor(outShape, grad->GetDataType());
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(EmbeddingDenseGradV2,
                                         OP_INPUT(grad, sortIndices, posIdx),
                                         OP_OUTPUT(out),
                                         OP_ATTR(numWeights, paddingIdx, scaleGradByFreq));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingDenseGradV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor *EmbeddingDenseGrad(const aclTensor *grad, const aclTensor *indices, uint64_t numWeights,
                                    uint64_t paddingIdx, bool scaleGradByFreq, aclOpExecutor *executor) {
  L0_DFX(EmbeddingDenseGrad, grad, indices, numWeights, paddingIdx, scaleGradByFreq);
  // shape推导
  auto gradOutputShape = grad->GetViewShape();
  int64_t gradOutputShapeLen = gradOutputShape.GetDimNum();
  int64_t outShapeLen = 2;
  op::Shape outShape;
  outShape.SetDimNum(outShapeLen);
  outShape.SetDim(0, numWeights);
  outShape.SetDim(1, gradOutputShape.GetDim(gradOutputShapeLen - 1));

  // 创建输出Tensor
  auto out = executor->AllocTensor(outShape, grad->GetDataType());
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }
  // 调用device的EmbeddingDenseGrad算子
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(EmbeddingDenseGrad,
                                         OP_INPUT(grad, indices),
                                         OP_OUTPUT(out),
                                         OP_ATTR(numWeights, paddingIdx, scaleGradByFreq));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingDenseGradAiCcore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
} // l0op

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
 * \file embedding_dense_grad.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_EMBEDDING_DENSE_GRAD_H_
#define OP_API_INC_LEVEL0_EMBEDDING_DENSE_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *EmbeddingDenseGrad(const aclTensor *grad, const aclTensor *indices, uint64_t numWeights,
                                    uint64_t paddingIdx, bool scaleGradByFreq, aclOpExecutor *executor);
    
const aclTensor *EmbeddingDenseGradV2(const aclTensor *grad, const aclTensor *sortIndices, const aclTensor *posIdx,
                                      uint64_t numWeights, uint64_t paddingIdx,
                                      bool scaleGradByFreq, aclOpExecutor *executor);
} // l0op

#endif // OP_API_INC_LEVEL0_EMBEDDING_DENSE_GRAD_H_

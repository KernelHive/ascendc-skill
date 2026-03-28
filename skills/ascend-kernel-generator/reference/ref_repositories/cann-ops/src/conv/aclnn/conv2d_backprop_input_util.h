/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ConvolutionBackwardInputTensor {
  const aclTensor *gradOutput;
  const aclTensor *input;
  const aclTensor *weight;
};

struct ConvolutionBackwardParams {
  const aclIntArray *biasSizes;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  const bool transposed;
  const aclIntArray *outputPadding;
  const int64_t groups;
  const aclBoolArray *outputMask;
  const int8_t cubeMathType;
};

const aclTensor *CalculateConv2DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                              ConvolutionBackwardParams &params, aclOpExecutor *executor);

#ifdef __cplusplus
}
#endif
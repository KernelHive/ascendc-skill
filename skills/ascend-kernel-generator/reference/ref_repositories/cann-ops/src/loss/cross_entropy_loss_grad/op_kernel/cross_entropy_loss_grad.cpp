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
 * \file cross_entropy_loss_grad.cpp
 * \brief
 */

#include "cross_entropy_loss_grad_weight_not_none.h"
#include "cross_entropy_loss_grad_weight_none.h"

using namespace AscendC;
using namespace CrossEntropyLossGrad;
extern "C" __global__ __aicore__ void cross_entropy_loss_grad(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                              GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                              GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
  // 10: bf16, 不存在weight
  // 11: bf16, 存在weight
  // 20: fp16, 不存在weight
  // 21: fp16, 存在weight
  // 30: fp32, 不存在weight
  // 31: fp32, 存在weight
  if (TILING_KEY_IS(10)) {
    CrossEntropyLossGradWeightNone<bfloat16_t> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if (TILING_KEY_IS(11)) {
    CrossEntropyLossGradWeightNotNone<bfloat16_t> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
  else if (TILING_KEY_IS(20)) {
    CrossEntropyLossGradWeightNone<half> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if (TILING_KEY_IS(21)) {
    CrossEntropyLossGradWeightNotNone<half> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
  else if (TILING_KEY_IS(30)) {
    CrossEntropyLossGradWeightNone<float> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if (TILING_KEY_IS(31)) {
    CrossEntropyLossGradWeightNotNone<float> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
}
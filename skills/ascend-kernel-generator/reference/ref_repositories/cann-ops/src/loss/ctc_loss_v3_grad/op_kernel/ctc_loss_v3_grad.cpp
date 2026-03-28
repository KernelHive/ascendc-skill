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
 * \file ctc_loss_v3_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v3_grad.h"
using namespace CTCLossV3GradNS;

extern "C" __global__ __aicore__ void ctc_loss_v3_grad(GM_ADDR grad_out, GM_ADDR log_probs, GM_ADDR targets,
                                                       GM_ADDR input_lengths, GM_ADDR target_lengths,
                                                       GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR grad,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    CTCLossV3Grad<DTYPE_GRAD, DTYPE_TARGETS> op;
    TPipe pipe;
    op.Init(&tilingData, grad_out, log_probs, targets,
            input_lengths, target_lengths, neg_log_likelihood, log_alpha, grad, workspace, &pipe);
    op.Process();
}
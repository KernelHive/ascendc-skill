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
 * \file add_layer_norm_grad.cpp
 * \brief
 */
#include "add_layer_norm_grad_cut_d.h"
#include "add_layer_norm_grad_cut_n.h"

using namespace AscendC;

/* ******************************************************************************
 * @brief: AddLayerNormGrad
 * @param Dy: shape [B, H]
 * @param X1: shape [B, H]
 * @param X2: shape [B, H]
 * @param Rstd: shape [B, 1]
 * @param Mean: shape [B, 1]
 * @param Gamma: shape [H]
 * @param DSum: shape [H]
 * @return PdX: shape [B, H]
 * @return PdGamma: shape [H]
 * @return PdBeta: shape [H]
 ****************************************************************************** */
extern "C" __global__ __aicore__ void add_layer_norm_grad(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd,
    GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum, GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
#define INIT_CUT_N_PROCESS                                                                           \
    op.Init(dy, x_1, x_2, rstd, mean, gamma, dsum, d_x, d_gamma, d_beta, tiling_data, usrWorkspace); \
    op.CutNProcess()
#define INIT_CUT_D_PROCESS                                                                           \
    op.Init(dy, x_1, x_2, rstd, mean, gamma, dsum, d_x, d_gamma, d_beta, tiling_data, usrWorkspace); \
    op.CutDProcess()

    if (TILING_KEY_IS(10)) {
        KernelAddLayerNormGrad<float, 10> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(11)) {
        KernelAddLayerNormGrad<float, 11> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(20)) {
        KernelAddLayerNormGrad<half, 20> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(21)) {
        KernelAddLayerNormGrad<half, 21> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(40)) {
        KernelAddLayerNormGradLarge<float, 40> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(41)) {
        KernelAddLayerNormGradLarge<float, 41> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(50)) {
        KernelAddLayerNormGradLarge<half, 50> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(51)) {
        KernelAddLayerNormGradLarge<half, 51> op;
        INIT_CUT_D_PROCESS;
    } else {
#if __CCE_AICORE__ == 220
        if (TILING_KEY_IS(30)) {
            KernelAddLayerNormGrad<bfloat16_t, 30> op;
            INIT_CUT_N_PROCESS;
        } else if (TILING_KEY_IS(31)) {
            KernelAddLayerNormGrad<bfloat16_t, 31> op;
            INIT_CUT_N_PROCESS;
        } else if (TILING_KEY_IS(60)) {
            KernelAddLayerNormGradLarge<bfloat16_t, 60> op;
            INIT_CUT_D_PROCESS;
        } else if (TILING_KEY_IS(61)) {
            KernelAddLayerNormGradLarge<bfloat16_t, 61> op;
            INIT_CUT_D_PROCESS;
        }
#endif
    }
}

#ifndef __CCE_KT_TEST__
void add_layer_norm_grad_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *dy, uint8_t *x1, uint8_t *x2,
    uint8_t *rstd, uint8_t *mean, uint8_t *gamma, uint8_t *dsum, uint8_t *d_x, uint8_t *d_gamma, uint8_t *d_beta,
    uint8_t *workspace, uint8_t *tiling)
{
    add_layer_norm_grad<<<blockDim, l2ctrl, stream>>>(
        dy, x1, x2, rstd, mean, gamma, dsum, d_x, d_gamma, d_beta, workspace, tiling);
}
#endif
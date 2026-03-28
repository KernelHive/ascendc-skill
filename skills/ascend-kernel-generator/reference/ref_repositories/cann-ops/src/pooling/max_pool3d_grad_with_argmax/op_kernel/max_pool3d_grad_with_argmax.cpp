/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file max_pool3d_grad_with_argmax.cpp
 * \brief
 */

#include "max_pool3d_grad_with_argmax_nosplit.h"
#include "max_pool3d_grad_with_argmax_splitd.h"
#include "max_pool3d_grad_with_argmax_splith.h"
#include "max_pool3d_grad_with_argmax_splitw.h"
#include "max_pool3d_grad_with_argmax_normal.h"
#include "max_pool3d_grad_with_argmax_scatter.h"
#include "max_pool3d_grad_with_argmax_scatter_overlap.h"
#include "max_pool3d_grad_with_argmax_cutk_d.h"
#include "max_pool3d_grad_with_argmax_cutk_dh.h"
#include "max_pool3d_grad_with_argmax_cutk_dhw.h"

using namespace MaxPool3DGradWithArgmax;

#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.Process();                                        \
    } while (0)

#define GENERAL_OP_IMPL_CUTNC(templateClass, ...)            \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.ProcessCutNc();                                        \
    } while (0)

extern "C" __global__ __aicore__ void max_pool3d_grad_with_argmax(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    if (g_coreType == AIC) {
        return;
    }

    // 100000 = 1, splitD=0 splitH=0 splitW=0 splitKernel=0 type=float(0)
    // 100001 = 1, splitD=0 splitH=0 splitW=0 splitKernel=0 type=half(1)
    // 100002 = 1, splitD=0 splitH=0 splitW=0 splitKernel=0 type=bfloat(2)
    // 110000 = 1, splitD=1 splitH=0 splitW=0 splitKernel=0 type=float(0)
    // 110001 = 1, splitD=1 splitH=0 splitW=0 splitKernel=0 type=half(1)
    // 110002 = 1, splitD=1 splitH=0 splitW=0 splitKernel=0 type=bfloat(2)
    // 111000 = 1, splitD=1 splitH=1 splitW=0 splitKernel=0 type=float(0)
    // 111001 = 1, splitD=1 splitH=1 splitW=0 splitKernel=0 type=half(1)
    // 111002 = 1, splitD=1 splitH=1 splitW=0 splitKernel=0 type=bfloat(2)
    // 111100 = 1, splitD=1 splitH=1 splitW=1 splitKernel=0 type=float(0)
    // 111101 = 1, splitD=1 splitH=1 splitW=1 splitKernel=0 type=half(1)
    // 111102 = 1, splitD=1 splitH=1 splitW=1 splitKernel=0 type=bfloat(2)

    TPipe pipe;
    // The percentile determines if overlap occurs
    if (TILING_KEY_IS(0)) { // Normal Kernel
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxNormal, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxNormal, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(2)) { // Scatter Kernel
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatter, DTYPE_X, DTYPE_X, int32_t, DTYPE_X);
    } else if (TILING_KEY_IS(102)) {
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatterOverlap, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y);
    } else if (TILING_KEY_IS(1)) { // CutK Kernel, no cut
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(21)) { // CutK Kernel, cut do
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(31)) { // CutK Kernel, cut do, kd, ho
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(41)) { // CutK Kernel, cut do, kd, ho, kh, wo
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(51)) { // CutK Kernel, cut do, kd
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(61)) { // CutK Kernel, cut do, kd, ho, kh
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, false);
    } else if (TILING_KEY_IS(71)) { // CutK Kernel, cut do, kd, ho, kh, wo, kw
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKDHW, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(101)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(121)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(131)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(141)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(151)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(161)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(171)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKDHW, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else // Russian kernel
    if (TILING_KEY_IS(100000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxNoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxNoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxNoSplit<float,float> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(100001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxNoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxNoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxNoSplit<half,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(200001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxNoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxNoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxNoSplit<float,half> op(tilingData);
        op.Init(grad, x, argmax, y, nullptr);
        op.Process();
    } else if (TILING_KEY_IS(100002)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxNoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxNoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxNoSplit<float,bfloat16_t> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(110000)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitD<float,float> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(110001)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitD<half,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(210001)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitD<float,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(110002)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitD<float,bfloat16_t> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(111000)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitH<float,float> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(111001)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitH<half,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(211001)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitH<float,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(111002)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitH<float,bfloat16_t> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(111100)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitWTilingData *__restrict tilingData = &tilingDataIn;	
        KernelMaxPool3DGradWithArgmaxSplitW<float,float> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(111101)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitWTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitW<half,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(211101)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitWTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitW<float,half> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    } else if (TILING_KEY_IS(111102)) {
		GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradWithArgmaxSplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DGradWithArgmaxSplitWTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DGradWithArgmaxSplitW<float,bfloat16_t> op(tilingData);
        op.Init(grad, x, argmax, y, userWS);
        op.Process();
    }

    return;
}
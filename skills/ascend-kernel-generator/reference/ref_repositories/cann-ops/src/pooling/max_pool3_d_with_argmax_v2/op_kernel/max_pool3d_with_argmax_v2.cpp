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
 * \file max_pool3d_with_argmax_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "max_pool3d_with_argmax_v2_nosplit.h"
#include "max_pool3d_with_argmax_v2_splitd.h"
#include "max_pool3d_with_argmax_v2_splith.h"
#include "max_pool3d_with_argmax_v2_splitw.h"
#include "max_pool3d_with_argmax_v2_huge_kernel.h"
#include "max_pool3d_with_argmax_v2_no_expand_indices.h"
#include "max_pool3d_with_argmax_big_kernel.h"

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {

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
    // 111110 = 1, splitD=1 splitH=1 splitW=1 splitKernel=1 type=float(0)
    // 111111 = 1, splitD=1 splitH=1 splitW=1 splitKernel=1 type=half(1)
    // 111112 = 1, splitD=1 splitH=1 splitW=1 splitKernel=1 type=bfloat(2)
    TPipe pipe;
    if (TILING_KEY_IS(100000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2NoSplit<float,float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2NoSplit<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100002)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoSplitTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2NoSplit<float,bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<float,float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110002)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<float,bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitH<float,float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitH<half,half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111002)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitHTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitH<float,bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111100)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitWTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitW<float,float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111101)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitWTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitW<half,half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111102)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitWTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitW<float,bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111110)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2HugeKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2HugeKernelTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2HugeKernel<float,float> op(tilingData);
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111111)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2HugeKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2HugeKernelTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2HugeKernel<float,half> op(tilingData);
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111112)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2HugeKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2HugeKernelTilingData *__restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2HugeKernel<float,bfloat16_t> op(tilingData);
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe);
        op.Process();
    } else if (TILING_KEY_IS(300001UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData *__restrict tilingData = &tilingDataIn;
#if ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_DISABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_DISABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(300002UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData *__restrict tilingData = &tilingDataIn;
#if ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_ENABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_ENABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(311110)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData *__restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<float, float, false> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 0);
        op.Process();
    } else if (TILING_KEY_IS(311111)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData *__restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<half, half, false> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 1);
        op.Process();
    } else if (TILING_KEY_IS(311112)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData *__restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<bfloat16_t, float, false> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 2);
        op.Process();
    }
    
    return;
}
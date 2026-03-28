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
 * \file avg_pool3_d.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

#include "avg_pool3d_ndhwc_split_c.h"
#include "avg_pool3d_ndhwc_multi_w.h"
#include "avg_pool3d_ndhwc_split_w.h"
#include "avg_pool3d_ncdhw_reduce_d.h"

#define DISPATCH_OP_IMPL(KernelImpl, ...)                                                                              \
  do {                                                                                                                 \
    KernelImpl<__VA_ARGS__> op;                                                                                        \
    TPipe tPipe;                                                                                                       \
    op.Init(x, y, workspace, &tilingData, &tPipe);                                                                     \
    op.Process();                                                                                                      \
  } while (0)


extern "C" __global__ __aicore__ void avg_pool3_d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(10)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dSplitC, float, 1);
  } else if (TILING_KEY_IS(11)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dSplitC, half, 1);
#if __CCE_AICORE__ >= 220
  } else if (TILING_KEY_IS(12)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dSplitC, bfloat16_t, 1);
#endif
  } else if (TILING_KEY_IS(20)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dSplitW, float, 1);
  } else if (TILING_KEY_IS(21)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dSplitW, half, 1);
#if __CCE_AICORE__ >= 220
  } else if (TILING_KEY_IS(22)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dSplitW, bfloat16_t, 1);
#endif
  } else if (TILING_KEY_IS(30)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dMultiW, float, 1);
  } else if (TILING_KEY_IS(31)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dMultiW, half, 1);
#if __CCE_AICORE__ >= 220
  } else if (TILING_KEY_IS(32)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dMultiW, bfloat16_t, 1);
#endif
  } else if (TILING_KEY_IS(40)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dReduceD, float, 1);
  } else if (TILING_KEY_IS(41)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dReduceD, half, 1);
#if __CCE_AICORE__ >= 220
  } else if (TILING_KEY_IS(42)) {
    DISPATCH_OP_IMPL(KernelAvgPool3dReduceD, bfloat16_t, 1);
#endif
  }
}

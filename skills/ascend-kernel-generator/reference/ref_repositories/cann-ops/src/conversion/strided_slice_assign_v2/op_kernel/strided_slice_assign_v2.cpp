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
 * \file strided_slice_assign_v2.cpp
 * \brief
 */
#include "strided_slice_assign_v2.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void strided_slice_assign_v2(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin,
                                                              GM_ADDR end, GM_ADDR strides, GM_ADDR axes,
                                                              GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelStridedSliceAssignV2<DTYPE_VAR> op(&pipe);
        op.Init(var, input_value, begin, end, strides, axes, var_out, &tilingData);
        op.Process();
    }
}

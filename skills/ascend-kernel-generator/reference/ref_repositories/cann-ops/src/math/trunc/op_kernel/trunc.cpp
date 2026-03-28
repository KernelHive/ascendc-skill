/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "kernel_operator.h"
#include "copy_kl.h"
#include "sync_kl.h"
#include "trunc_.cpp"
#include "trunc_f32.cpp"
using namespace AscendC;
extern "C" __global__ __aicore__ void trunc(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe; // Pipeline
    GET_TILING_DATA(tiling_data, tiling); // Tiling data

    uint32_t Len;
    uint32_t fNum, fLen, tLen;
    {
        bool is_former = GetBlockIdx() < tiling_data.fNum;
        Len = tiling_data.Len;
        fNum = tiling_data.fNum;
        fLen = tiling_data.fLen;
        tLen = tiling_data.tLen;
    }

     // 初始化&计算
    if constexpr(std::is_same_v<DTYPE_INPUT_X, float>){
        KernelTruncF32 op;
        op.Init(&pipe,
            input_x, output_y, 
            fLen, fNum, tLen, 
            Len // 核内切分参数
            
        );
        op.Process();
    }else{
        KernelTrunc<DTYPE_INPUT_X, DTYPE_OUTPUT_Y> op;
        op.Init(&pipe,
            input_x, output_y, 
            fLen, fNum, tLen, 
            Len // 核内切分参数
            
        );
        op.Process();
    }
}

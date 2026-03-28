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
 * \file mse_loss_grad_v2.cpp
 * \brief
 */
#include <cstdint>
#include "kernel_operator.h"
#if __CCE_AICORE__ == 200
    #include "mse_loss_grad_v2_310p.h"
#else
    #include "mse_loss_grad_v2.h"
#endif
using namespace AscendC;

extern "C" __global__ __aicore__ void mse_loss_grad_v2(GM_ADDR predict, GM_ADDR label, GM_ADDR dout,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
#define INIT_AND_PROCESS \
    op.Init(predict, label, dout, y, tiling_data.cof, \
            tiling_data.totalLength, tiling_data.tileNum, \
            tiling_data.blockLength, tiling_data.padLength, tiling_data.usedDb); \
    op.Process()

#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(1)) {
        KernelMseLossGrad310P<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelMseLossGrad310P<half> op;
        INIT_AND_PROCESS;
    }
#else
    if (TILING_KEY_IS(1)) {
        KernelMseLossGrad910<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)){
        KernelMseLossGrad910<half> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        KernelMseLossGrad910<bfloat16_t> op;
        INIT_AND_PROCESS;
    }
#endif
}

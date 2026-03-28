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
 * \file mse_loss_v2.cpp
 * \brief
 */

#include "mse_loss_v2_none.h"
#include "mse_loss_v2_sum.h"
#include "mse_loss_v2_mean.h"


#define INIT_AND_PROCESS(mode, dtype) \
        MSELossV2##mode<dtype> op(&pipe, &tiling_data); \
        op.Init(output, input, target, usrWorkspace); \
        op.Process()


extern "C" __global__ __aicore__ void mse_loss_v2(GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    using namespace MSELossV2Namespace;
    if (TILING_KEY_IS(11)) {
        INIT_AND_PROCESS(None, float);
    } else if (TILING_KEY_IS(12)) {
        INIT_AND_PROCESS(None, half);
    } else if (TILING_KEY_IS(21)) {
        INIT_AND_PROCESS(Sum, float);
    } else if (TILING_KEY_IS(22)) {
        INIT_AND_PROCESS(Sum, half);
    } else if (TILING_KEY_IS(31)) {
        INIT_AND_PROCESS(Mean, float);
    } else if (TILING_KEY_IS(32)) {
        INIT_AND_PROCESS(Mean, half);
#if __CCE_AICORE__ != 200
    } else if (TILING_KEY_IS(13)) {
        INIT_AND_PROCESS(None, bfloat16_t);
    } else if (TILING_KEY_IS(23)) {
        INIT_AND_PROCESS(Sum, bfloat16_t);
    } else if (TILING_KEY_IS(33)) {
        INIT_AND_PROCESS(Mean, bfloat16_t);
#endif
    }
}
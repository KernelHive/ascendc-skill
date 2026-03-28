/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file ge_glu_grad_v2.cpp
 * \brief
 */
#if __CCE_AICORE__ == 200
#include "ge_glu_grad_v2_fp16_310p.h"
#include "ge_glu_grad_v2_fp32_310p.h"
#else
#include "ge_glu_grad_v2_tanh_bfp16.h"
#include "ge_glu_grad_v2_tanh_fp16.h"
#include "ge_glu_grad_v2_tanh_fp32.h"
#include "ge_glu_grad_v2_erf_bfp16.h"
#include "ge_glu_grad_v2_erf_fp16.h"
#include "ge_glu_grad_v2_erf_fp32.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void ge_glu_grad_v2(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, GM_ADDR workspace,
                                                     GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(201)) {
        GeGluGradV2For310P::GeGluGradV2FP16By310p op(dy, x, gelu, dx, userWS, &tilingData);
        op.Init();
        op.Process();
    } else if (TILING_KEY_IS(202)) {
        GeGluGradV2For310P::GeGluGradV2FP16By310p op(dy, x, gelu, dx, userWS, &tilingData);
        op.Init();
        op.Process();
    } else if (TILING_KEY_IS(301)) {
        GeGluGradV2For310P::GeGluGradV2FP32By310p op(dy, x, gelu, dx, userWS, &tilingData);
        op.Init();
        op.Process();
    } else if (TILING_KEY_IS(302)) {
        GeGluGradV2For310P::GeGluGradV2FP32By310p op(dy, x, gelu, dx, userWS, &tilingData);
        op.Init();
        op.Process();
    }
#else
    /* Tanh */
    if (TILING_KEY_IS(101)) {
        GeGluGradV2Tanh::GeGluGradV2TanhBFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(102)) {
        GeGluGradV2Tanh::GeGluGradV2TanhBFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(201)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(202)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(301)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP32 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(302)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP32 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(103)) {
        GeGluGradV2Tanh::GeGluGradV2TanhBFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(203)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(303)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP32 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process(true);
        return;
    }

    /* Erf */
    if (TILING_KEY_IS(701)) {
        GeGluGradV2Erf::GeGluGradV2ErfBFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(702)) {
        GeGluGradV2Erf::GeGluGradV2ErfBFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(801)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(802)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(901)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP32 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(902)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP32 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process();
    }
    if (TILING_KEY_IS(703)) {
        GeGluGradV2Erf::GeGluGradV2ErfBFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(803)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP16 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(903)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP32 op(dy, x, gelu, dx, &tilingData);
        op.Init();
        op.Process(true);
        return;
    }
#endif
}

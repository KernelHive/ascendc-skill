/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file apply_adam_wv2.cpp
 */

#include "apply_adam_wv2_fp.h"
#include "apply_adam_wv2_b16.h"
#include "apply_adam_wv2_mix_dtype.h"

using namespace ApplyAdamWV2;
extern "C" __global__ __aicore__ void apply_adam_w_v2(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq,
    GM_ADDR grad, GM_ADDR step, GM_ADDR maxGradNorm, GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(101)) {
        ApplyAdamWV2B16<bfloat16_t, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        ApplyAdamWV2B16<bfloat16_t, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        ApplyAdamWV2B16<half, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(104)) {
        ApplyAdamWV2B16<half, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(105)) {
        ApplyAdamWV2Fp<float, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        ApplyAdamWV2Fp<float, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(107)) {
        ApplyAdamWV2MixType<float, half, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(108)) {
        ApplyAdamWV2MixType<float, half, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(109)) {
        ApplyAdamWV2MixType<float, bfloat16_t, float> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(110)) {
        ApplyAdamWV2MixType<float, bfloat16_t, int64_t> op;
        op.Init(var, expAvg, expAvgSq, grad, step, maxGradNorm, userWS, &tilingData);
        op.Process();
    }
}

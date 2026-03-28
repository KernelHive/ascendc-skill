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
 * \file foreach_non_finite_check_and_unscale.cpp
 * \brief
 */
#include "foreach_non_finite_check_and_unscale_n_d.h"

using namespace ForeachNonFiniteCheckAndUnscale;

extern "C" __global__ __aicore__ void foreach_non_finite_check_and_unscale(GM_ADDR scaled_grads, GM_ADDR found_inf,
                                                                           GM_ADDR inv_scale, GM_ADDR workspace,
                                                                           GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        ForeachNonFiniteCheckAndUnscaleND<float> op;
        op.Init(scaled_grads, found_inf, inv_scale, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachNonFiniteCheckAndUnscaleND<half> op;
        op.Init(scaled_grads, found_inf, inv_scale, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
#if (__CCE_AICORE__ > 200)
        ForeachNonFiniteCheckAndUnscaleND<bfloat16_t> op;
        op.Init(scaled_grads, found_inf, inv_scale, userWS, &tilingData);
        op.Process();
#endif
    }
}
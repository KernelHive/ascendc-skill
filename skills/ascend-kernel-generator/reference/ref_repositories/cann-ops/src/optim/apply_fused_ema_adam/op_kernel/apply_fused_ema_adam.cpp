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
 * \file apply_fused_ema_adam.cpp
 * \brief
 */

#include "apply_fused_ema_adam_f32.h"
#include "apply_fused_ema_adam_f_bf16.h"

using namespace FusedEmaAdam;
extern "C" __global__ __aicore__ void apply_fused_ema_adam(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR s,
                                                           GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
                                                           GM_ADDR s_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWorkSpace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
    if (TILING_KEY_IS(102)) {
        FusedEmaAdamF32<float> op;
        op.Init(grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        FusedEmaAdamF16<half> op;
        op.Init(grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100)) {
        FusedEmaAdamF16<bfloat16_t> op;
        op.Init(grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, tiling_data, &pipe);
        op.Process();
    }
}
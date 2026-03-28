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
 * @file apply_adam_wv2_tiling.h
 */
#ifndef TILING_RUNTIME_APPLY_FUSED_EMA_ADAM_H_
#define TILING_RUNTIME_APPLY_FUSED_EMA_ADAM_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyFusedEmaAdamTilingData)
    TILING_DATA_FIELD_DEF(float, lr);
    TILING_DATA_FIELD_DEF(float, beta1);
    TILING_DATA_FIELD_DEF(float, beta2);
    TILING_DATA_FIELD_DEF(float, eps);
    TILING_DATA_FIELD_DEF(float, emaDecay);
    TILING_DATA_FIELD_DEF(float, weightDecay);
    TILING_DATA_FIELD_DEF(uint64_t, mode);
    TILING_DATA_FIELD_DEF(uint64_t, biasCorrection);

    TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, coreCalcNum);
    TILING_DATA_FIELD_DEF(uint64_t, loopNum);
    TILING_DATA_FIELD_DEF(uint64_t, coreCalcMax);
    TILING_DATA_FIELD_DEF(uint64_t, frontCalcExtra);
    TILING_DATA_FIELD_DEF(uint64_t, tailCalcExtra);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyFusedEmaAdam, ApplyFusedEmaAdamTilingData)

}  // namespace optiling
#endif // TILING_RUNTIME_APPLY_FUSED_EMA_ADAM_H_
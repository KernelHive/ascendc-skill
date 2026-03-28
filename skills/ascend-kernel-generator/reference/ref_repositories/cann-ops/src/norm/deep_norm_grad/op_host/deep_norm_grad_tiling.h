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
 * \file deep_norm_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_GRAD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_GRAD_H_
#include "register/tilingdata_base.h"

namespace optiling {
template <typename T1, typename T2>
inline static T1 CeilDiv(const T1 dividend, const T2 divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}
BEGIN_TILING_DATA_DEF(DeepNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, nDimNum);
TILING_DATA_FIELD_DEF(uint32_t, dDimNum);
TILING_DATA_FIELD_DEF(uint32_t, nDealPerCore);
TILING_DATA_FIELD_DEF(uint32_t, nDealLastCore);
TILING_DATA_FIELD_DEF(uint32_t, mergeNCount);
TILING_DATA_FIELD_DEF(uint32_t, cutDTime);
TILING_DATA_FIELD_DEF(uint32_t, cutDPerTime);
TILING_DATA_FIELD_DEF(uint32_t, cutDLastTime);
TILING_DATA_FIELD_DEF(uint32_t, alpha);
TILING_DATA_FIELD_DEF(uint32_t, fixedOutputFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeepNormGrad, DeepNormGradTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_GRAD_H_

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
 * \file foreach_non_finite_check_and_unscale_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_

#include <vector>
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_COUNT = 256;
constexpr uint16_t MAX_CORE_COUNT = 64;
constexpr uint16_t MAX_CORE_COUNT_REGBASE = 80;
struct ForeachNonFiniteCheckAndUnscaleCompileInfo {
    uint32_t coreNum;
};

BEGIN_TILING_DATA_DEF(ForeachNonFiniteCheckAndUnscaleTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, scaledGradsUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, reduceTempValUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorEndOffsetList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ForeachNonFiniteCheckAndUnscale, ForeachNonFiniteCheckAndUnscaleTilingData)
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_

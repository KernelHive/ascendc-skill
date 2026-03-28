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
 * \file non_finite_check_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_NON_FINITE_CHECK_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_NON_FINITE_CHECK_TILING_H

#include <vector>
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_COUNT = 256;
constexpr uint16_t MAX_CORE_COUNT = 64;

struct NonFiniteCheckCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

enum class NonFiniteCheckTilingKey : uint64_t { KEY_FLOAT16 = 101, KEY_BF16 = 201, KEY_FLOAT = 301 };

BEGIN_TILING_DATA_DEF(NonFiniteCheckTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, maxProcCount);
    TILING_DATA_FIELD_DEF(uint32_t, tempValUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonFiniteCheck, NonFiniteCheckTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_NON_FINITE_CHECK_TILING_H

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
 * \file foreach_reduce_tiling_def.h
 * \brief
 */
 
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_DEF_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_DEF_H_

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 50;
constexpr uint16_t MAX_CORE_CONT = 50;
struct ForeachNormCompileInfo {
    uint32_t coreNum;
};
BEGIN_TILING_DATA_DEF(ForeachReduceTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorEndOffsetList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_TENSOR_CONT, tensorMiddleCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_TENSOR_CONT, tensorMiddleStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, coreMiddleOffsetList);
    TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalTensorCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachNorm, ForeachReduceTilingData)
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_REDUCE_DEF_H_

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
 * \file foreach_tiling_def.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_H_

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 50;
constexpr uint16_t MAX_CORE_CONT = 50;
struct ForeachCompileInfo {
    uint64_t coreNum;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t sysWorkspaceSize;
};

BEGIN_TILING_DATA_DEF(ForeachCommonTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachPowScalarAndTensor, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachReciprocal, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachExpm1, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog1p, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSin, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLerpScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachZeroInplace, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachTanh, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLerpList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachErf, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachErfc, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachCosh, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSinh, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAsin, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAcos, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachTan, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAtan, ForeachCommonTilingData)

REGISTER_TILING_DATA_CLASS(ForeachAbs, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachCopy, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSign, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachExp, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSigmoid, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMaximumList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSubList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMulList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcdivList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcdivScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcdivScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachAddcmulScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachCos, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog2, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachLog10, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMaximumScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMaximumScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMinimumScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMinimumList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachMinimumScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachNeg, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachPowList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachPowScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachPowScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachRoundOffNumber, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSqrt, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSubScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachSubScalarList, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivScalar, ForeachCommonTilingData)
REGISTER_TILING_DATA_CLASS(ForeachDivScalarList, ForeachCommonTilingData)
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_H_

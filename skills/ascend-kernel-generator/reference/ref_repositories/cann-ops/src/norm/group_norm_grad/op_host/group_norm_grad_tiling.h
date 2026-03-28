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
 * \file group_norm_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_GROUPNORMGRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_GROUPNORMGRAD_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, Tiling_key);                // 0
TILING_DATA_FIELD_DEF(uint32_t, N);                         // 1
TILING_DATA_FIELD_DEF(uint32_t, C);                         // 2
TILING_DATA_FIELD_DEF(uint32_t, HXW);                       // 3
TILING_DATA_FIELD_DEF(uint32_t, G);                         // 4
TILING_DATA_FIELD_DEF(uint32_t, NXG);                       // 5
TILING_DATA_FIELD_DEF(uint32_t, C_G);                       // 6
TILING_DATA_FIELD_DEF(uint32_t, task_num_per_core);         // 7
TILING_DATA_FIELD_DEF(uint32_t, task_num_per_tail_core);    // 8
TILING_DATA_FIELD_DEF(uint32_t, tail_core);                 // 9
TILING_DATA_FIELD_DEF(uint32_t, mode1_ub_cap_C_num);        // 10
TILING_DATA_FIELD_DEF(uint32_t, mode1_ub_iter_C_num);       // 11
TILING_DATA_FIELD_DEF(uint32_t, mode1_ub_tail_C_num);       // 12
TILING_DATA_FIELD_DEF(uint32_t, mode2_ub_capacity_ele);     // 13
TILING_DATA_FIELD_DEF(uint32_t, mode2_ub_iteration_num);    // 14
TILING_DATA_FIELD_DEF(uint32_t, mode2_ub_tail_num);         // 15
TILING_DATA_FIELD_DEF(uint32_t, workSpaceSize);             // 16
TILING_DATA_FIELD_DEF(uint32_t, stage2CoreUsed);            // 17
TILING_DATA_FIELD_DEF(uint32_t, castEleNum);                // 18
TILING_DATA_FIELD_DEF(uint32_t, tailCastNum);               // 19
TILING_DATA_FIELD_DEF(uint32_t, coreBatchParts);            // 20
TILING_DATA_FIELD_DEF(uint32_t, coreBatchPartsTailRepeat);  // 21
TILING_DATA_FIELD_DEF(uint32_t, repeatTime4Stage2);         // 22
TILING_DATA_FIELD_DEF(bool, dx_is_require);                 // 23
TILING_DATA_FIELD_DEF(bool, dgamma_is_require);             // 24
TILING_DATA_FIELD_DEF(bool, dbeta_is_require);              // 25
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormGrad, GroupNormGradTilingData)
struct GroupNormGradCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

struct TilingCalculationParameters {
    uint32_t tilingKey = -1;
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hxw = 0;
    uint32_t g = 0;
    uint32_t nxg = 0;
    uint32_t channelPerGroup = 0;
    uint32_t taskNumPerCore = 0;
    uint32_t taskNumPerTailCore = 0;
    uint32_t tailCore = 0;
    uint32_t mode0UbCapGNum = 0;
    uint32_t mode1UbCapCNum = 0;
    uint32_t mode1UbIterCNum = 0;
    uint32_t mode1UbTailCNum = 0;
    uint32_t mode2UbCapacityEle = 0;
    uint32_t mode2UbIterationNum = 0;
    uint32_t mode2UbTailNum = 0;
    uint32_t workSpaceSize = 0;
    uint32_t stage2CoreUsed = 0;
    uint32_t castEleNum = 0;
    uint32_t tailCastNum = 0;
    uint32_t coreBatchParts = 0;
    uint32_t coreBatchPartsTailRepeat = 0;
    uint32_t repeatTime4Stage2 = 0;
    uint32_t coreNumUsed = 0;
    bool dxIsRequire = true;
    bool dgammaIsRequire = true;
    bool dbetaIsRequire = true;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_GROUPNORMGRAD_H
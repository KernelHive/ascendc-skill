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
 * \file group_norm_swish_grad_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SWISH_GRAD_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SWISH_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormSwishGradTilingData)
TILING_DATA_FIELD_DEF(uint64_t, Tiling_key);                // 0
TILING_DATA_FIELD_DEF(uint64_t, N);                         // 1
TILING_DATA_FIELD_DEF(uint64_t, C);                         // 2
TILING_DATA_FIELD_DEF(uint64_t, HXW);                       // 3
TILING_DATA_FIELD_DEF(uint64_t, G);                         // 4
TILING_DATA_FIELD_DEF(uint64_t, NXG);                       // 5
TILING_DATA_FIELD_DEF(uint64_t, C_G);                       // 6
TILING_DATA_FIELD_DEF(uint64_t, task_num_per_core);         // 7
TILING_DATA_FIELD_DEF(uint64_t, task_num_per_tail_core);    // 8
TILING_DATA_FIELD_DEF(uint64_t, tail_core);                 // 9
TILING_DATA_FIELD_DEF(uint64_t, mode1_ub_cap_C_num);        // 10
TILING_DATA_FIELD_DEF(uint64_t, mode1_ub_iter_C_num);       // 11
TILING_DATA_FIELD_DEF(uint64_t, mode1_ub_tail_C_num);       // 12
TILING_DATA_FIELD_DEF(uint64_t, mode2_ub_capacity_ele);     // 13
TILING_DATA_FIELD_DEF(uint64_t, mode2_ub_iteration_num);    // 14
TILING_DATA_FIELD_DEF(uint64_t, mode2_ub_tail_num);         // 15
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);             // 16
TILING_DATA_FIELD_DEF(uint64_t, stage2CoreUsed);            // 17
TILING_DATA_FIELD_DEF(uint64_t, castEleNum);                // 18
TILING_DATA_FIELD_DEF(uint64_t, tailCastNum);               // 19
TILING_DATA_FIELD_DEF(uint64_t, coreBatchParts);            // 20
TILING_DATA_FIELD_DEF(uint64_t, coreBatchPartsTailRepeat);  // 21
TILING_DATA_FIELD_DEF(uint64_t, repeatTime4Stage2);         // 22
TILING_DATA_FIELD_DEF(uint64_t, dgamma_is_require);         // 23
TILING_DATA_FIELD_DEF(uint64_t, dbeta_is_require);          // 24
TILING_DATA_FIELD_DEF(float, swish_scale);                  // 25
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormSwishGrad, GroupNormSwishGradTilingData)
struct GroupNormSwishGradCompileInfo {
  int32_t totalCoreNum = 0;
  uint64_t sysWorkspaceSize = 0;
  uint64_t ubSizePlatForm = 0;
};

struct GroupNormSwishGradTilingCalculationParameters {
  uint64_t tilingKey = 100;
  uint64_t n = 0;
  uint64_t c = 0;
  uint64_t hxw = 0;
  uint64_t g = 0;
  uint64_t nxg = 0;
  uint64_t channelPerGroup = 0;
  uint64_t taskNumPerCore = 0;
  uint64_t taskNumPerTailCore = 0;
  uint64_t tailCore = 0;
  uint64_t mode0UbCapGNum = 0;
  uint64_t mode1UbCapCNum = 0;
  uint64_t mode1UbIterCNum = 0;
  uint64_t mode1UbTailCNum = 0;
  uint64_t mode2UbCapacityEle = 0;
  uint64_t mode2UbIterationNum = 0;
  uint64_t mode2UbTailNum = 0;
  uint64_t workSpaceSize = 0;
  uint64_t stage2CoreUsed = 0;
  uint64_t castEleNum = 0;
  uint64_t tailCastNum = 0;
  uint64_t coreBatchParts = 0;
  uint64_t coreBatchPartsTailRepeat = 0;
  uint64_t repeatTime4Stage2 = 0;
  uint64_t coreNumUsed = 0;
  uint64_t dgammaIsRequire = 1;
  uint64_t dbetaIsRequire = 1;
  float swishScale = 1.0;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SWISH_GRAD_TILING_H

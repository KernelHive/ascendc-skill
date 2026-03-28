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
 * \file group_norm_swish_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SWISH_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SWISH_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormSwishTilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, activateSwish);
TILING_DATA_FIELD_DEF(float, swishScale);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeCAlign);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
TILING_DATA_FIELD_DEF(int64_t, numPerGroup);
TILING_DATA_FIELD_DEF(int64_t, groupPerCore);
TILING_DATA_FIELD_DEF(int64_t, groupLastCore);
TILING_DATA_FIELD_DEF(int64_t, groupPerCoreAlign);
TILING_DATA_FIELD_DEF(int64_t, numPerLoop);
TILING_DATA_FIELD_DEF(int64_t, loopTimes);
TILING_DATA_FIELD_DEF(int64_t, loopTimesAlign);
TILING_DATA_FIELD_DEF(int64_t, numTailLoop);
TILING_DATA_FIELD_DEF(int32_t, totalCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormSwish, GroupNormSwishTilingData)

struct GroupNormSwishCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int32_t is310P = 0;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_Swish_TILING_H_
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
 * \file reduce_sum_v2_tiling.h
 * \brief
 */

#ifndef REDUCE_SUM_V2_TILING_H
#define REDUCE_SUM_V2_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"

namespace optiling {
// ub
BEGIN_TILING_DATA_DEF(ReduceSumV2UbInfos)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerATimes)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerA)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, tailA)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerRTimes)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerR)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, tailR)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerA1Times)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerA1)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, tailA1)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, formerRealTimes)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, tailRealData)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(ReduceSumV2UbInfosOp, ReduceSumV2UbInfos)

// block
BEGIN_TILING_DATA_DEF(ReduceSumV2BlockInfos)
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerCoreNum)
    TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerUnitDataLen)
    TILING_DATA_FIELD_DEF(uint64_t, tailUnitDataLen)
    TILING_DATA_FIELD_DEF(uint64_t, tailRealDataLen)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(ReduceSumV2BlockInfosOp, ReduceSumV2BlockInfos)

// process
BEGIN_TILING_DATA_DEF(ReduceSumV2Process)
    TILING_DATA_FIELD_DEF(uint64_t, A)
    TILING_DATA_FIELD_DEF(uint64_t, R)
    TILING_DATA_FIELD_DEF(uint64_t, A1)
    TILING_DATA_FIELD_DEF(uint64_t, pattern)
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint64_t, inBufferSize)
    TILING_DATA_FIELD_DEF(uint64_t, cacheBufferSize)
    TILING_DATA_FIELD_DEF(uint64_t, workspaceSize)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2UbInfos, ubInfos)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2BlockInfos, blockA)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2BlockInfos, blockR)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2BlockInfos, blockA1)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(ReduceSumV2ProcessOp, ReduceSumV2Process)

// base
BEGIN_TILING_DATA_DEF(ReduceSumV2TilingData)
    TILING_DATA_FIELD_DEF(uint64_t, processNum)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2Process, process0)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2Process, process1)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2Process, process2)
    TILING_DATA_FIELD_DEF_STRUCT(ReduceSumV2Process, process3)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ReduceSumV2, ReduceSumV2TilingData)

struct ReduceSumV2CompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t sysWorkspaceSize = 0;
};
}
#endif // REDUCE_SUM_V2_TILING_H
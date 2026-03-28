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
 * \file reduce_sum_v2.cpp
 * \brief
 */

#include "reduce_sum_v2_ar.h"
#include "reduce_sum_v2_ara.h"
using namespace AscendC;

template<uint8_t processId>
__aicore__ inline void GetProcessTilingData(const ReduceSumV2TilingData &tilingData, ReduceSumV2Process &processTiling)
{
    if constexpr (processId == 0) {
        processTiling = tilingData.process0;
    } else if constexpr (processId == 1) {
        processTiling = tilingData.process1;
    } else if constexpr (processId == 2) {
        processTiling = tilingData.process2;
    } else {
        processTiling = tilingData.process3;
    }
}

template<uint8_t processId, uint8_t processNum>
__aicore__ inline void DoReduceSumAR(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ReduceSumV2TilingData &tilingData, TPipe *tpipe)
{
    ReduceSumV2Process processTiling;
    GetProcessTilingData<processId>(tilingData, processTiling);
    if (GetBlockIdx() < processTiling.usedCoreNum) {
        if constexpr (processId == 0 && processId == processNum - 1) {
            ReduceSumV2ARKernel<DTYPE_X, DTYPE_X, processId, processNum> op(x, y, workspace, processTiling, tpipe);
            op.Process();
        } else {
            ReduceSumV2ARKernel<float, DTYPE_X, processId, processNum> op(x, y, workspace, processTiling, tpipe);
            op.Process();
        }
    }
}

template<uint8_t processId, uint8_t processNum>
__aicore__ inline void DoReduceSumARA(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ReduceSumV2TilingData &tilingData, TPipe *tpipe)
{
    ReduceSumV2Process processTiling;
    GetProcessTilingData<processId>(tilingData, processTiling);
    if (GetBlockIdx() < processTiling.usedCoreNum) {
        if constexpr (processId == 0 && processId == processNum - 1) {
            ReduceSumV2ARAKernel<DTYPE_X, DTYPE_X, processId, processNum> op(x, y, workspace, processTiling, tpipe);
            op.Process();
        } else if constexpr (processId == 0) {
            ReduceSumV2ARAKernel<DTYPE_X, float, processId, processNum> op(x, y, workspace, processTiling, tpipe);
            op.Process();
        } else if constexpr (processId > 0 && processId == processNum - 1) {
            ReduceSumV2ARAKernel<float, DTYPE_X, processId, processNum> op(x, y, workspace, processTiling, tpipe);
            op.Process();
        } else {
            ReduceSumV2ARAKernel<float, float, processId, processNum> op(x, y, workspace, processTiling, tpipe);
            op.Process();
        }
    }
}


extern "C" __global__ __aicore__ void reduce_sum_v2(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe tpipe;
    if (TILING_KEY_IS(0)) {
        DoReduceSumAR<0, 1>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(1)) {
        DoReduceSumARA<0, 1>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(10)) {
        DoReduceSumARA<0, NUM_2>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumAR<1, NUM_2>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(11)) {
        DoReduceSumARA<0, NUM_2>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<1, NUM_2>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(110)) {
        DoReduceSumARA<0, NUM_3>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<1, NUM_3>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumAR<NUM_2, NUM_3>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(111)) {
        DoReduceSumARA<0, NUM_3>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<1, NUM_3>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<NUM_2, NUM_3>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(1110)) {
        DoReduceSumARA<0, NUM_4>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<1, NUM_4>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<NUM_2, NUM_4>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumAR<NUM_3, NUM_4>(x, y, workspace, tilingData, &tpipe);
    } else if (TILING_KEY_IS(1111)) {
        DoReduceSumARA<0, NUM_4>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<1, NUM_4>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<NUM_2, NUM_4>(x, y, workspace, tilingData, &tpipe);
        tpipe.Reset();
        SyncAll();
        DoReduceSumARA<NUM_3, NUM_4>(x, y, workspace, tilingData, &tpipe);
    }
}
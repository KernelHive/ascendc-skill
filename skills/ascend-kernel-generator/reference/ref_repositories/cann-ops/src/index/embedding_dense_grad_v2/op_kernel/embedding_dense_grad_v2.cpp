/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file embedding_dense_grad_v2.cpp
 */
#include "embedding_dense_grad_v2.h"
#include "embedding_dense_grad_v2_scale.h"
#include "embedding_dense_grad_v2_determinist.h"
#include "embedding_dense_grad_v2_small_dim.h"

namespace AscendC {
__aicore__ inline void InitWorkspace(const EmbeddingDenseGradV2TilingData &tiling, GM_ADDR workSpace)
{
    GlobalTensor<float> indexCountGm;
    uint64_t scaleRowNum = 0;
    float initParam = 0.0;
    uint64_t blockIdx = GetBlockIdx();
    uint64_t formerCoreRowNumLoops = blockIdx < tiling.scaleTiling.formerCoreRowRepTime ? blockIdx : tiling.scaleTiling.formerCoreRowRepTime;
    uint64_t tailCoreRowNumLoops = blockIdx < tiling.scaleTiling.formerCoreRowRepTime ? 0 : blockIdx - tiling.scaleTiling.formerCoreRowRepTime;
    uint64_t addrOffset = tiling.scaleTiling.formerCoreRowNum * formerCoreRowNumLoops + tiling.scaleTiling.tailCoreRowNum * tailCoreRowNumLoops;
    if (blockIdx >= tiling.scaleTiling.formerCoreRowRepTime) {
        scaleRowNum = tiling.scaleTiling.tailCoreRowNum;
    } else {
        scaleRowNum = tiling.scaleTiling.formerCoreRowNum;
    }
    indexCountGm.SetGlobalBuffer((__gm__ float*)workSpace + addrOffset);
    InitOutput(indexCountGm, scaleRowNum, initParam);
}

struct ProcessInputs {
    GM_ADDR grad;
    GM_ADDR sortIndices;
    GM_ADDR posIdx;
    GM_ADDR backProps;
    GM_ADDR workSpace;
};

template <bool isSmallDim, bool isDetermin>
__aicore__ inline void ProcessAndScale(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps, GM_ADDR workSpace, 
                                       const EmbeddingDenseGradV2TilingData &tilingData, TPipe &tpipe)
{
    InitWorkspace(tilingData, workSpace);
    SyncAll();
    if constexpr(isSmallDim) {
        AscendC::EmbeddingDenseGradV2SmallDimKernel<float> op(grad, sortIndices, backProps, workSpace, tilingData, tpipe);
        op.Process();
    } else if constexpr(isDetermin) {
        EmbeddingDenseGradV2DeterministKernel<float> determinOp(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
        determinOp.Process();
    } else {
        EmbeddingDenseGradV2Kernel<float> op(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
        op.Process();
    }
    SyncAll();
    tpipe.Destroy();
    TPipe pipe;
    EmbeddingDenseGradV2ScaleKernel<float> scaleOp(backProps, workSpace, tilingData, pipe);
    scaleOp.Process();
}
}

extern "C" __global__ __aicore__ void embedding_dense_grad_v2(GM_ADDR grad, GM_ADDR sortIndices, 
    GM_ADDR posIdx, GM_ADDR backProps, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
    if (TILING_KEY_IS(0)) {
        AscendC::EmbeddingDenseGradV2Kernel<float> op(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::ProcessAndScale<false, false>(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
    } else if (TILING_KEY_IS(10)) {
        AscendC::EmbeddingDenseGradV2DeterministKernel<float> determinOp(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
        determinOp.Process();
    } else if (TILING_KEY_IS(11)) {
        AscendC::ProcessAndScale<false, true>(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
    } else if (TILING_KEY_IS(100)) {
        AscendC::EmbeddingDenseGradV2SmallDimKernel<float> op(grad, sortIndices, backProps, workSpace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        AscendC::ProcessAndScale<true, false>(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
    }
}
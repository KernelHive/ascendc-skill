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
 * @file matmul_all_reduce.h
 */

#ifndef MC2_MM_ALL_REDUCE_H
#define MC2_MM_ALL_REDUCE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_all_reduce_common.h"
#include "matmul_all_reduce_compute.h"
#include "matmul_all_reduce_tiling.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulKernelAllReduce(GM_ADDR aAddr, GM_ADDR bGM, GM_ADDR cAddr, GM_ADDR computeResAddr,
    GM_ADDR biasGM, TCubeTiling& tiling, AllReduceRCSTiling& cfg, AscendC::Hccl<AscendC::HCCL_SERVER_TYPE_AICPU> &hccl,
    uint32_t tileCnt, AscendC::HcclHandle &handleId)
{
    if (g_coreType == AIV) {
        return;
    }
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (int i=0; i< tileCnt; i++) {
            CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
            CrossCoreWaitFlag(0x8);
        }
        return;
    }

    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    auto aOffset = tiling.M *  tiling.Ka * sizeof(A_T);
    auto cOffset = tiling.M *  tiling.N  * sizeof(C_T);

    // 处理带C‘场景
    uint32_t indexC = 0;
    uint8_t enAtomicC = 0;

    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm;
    // AllReduce需要提前计算一次C矩阵的Offset地址
    mm.Init(tiling, cfg);
    mm.InitGlobalBTensor(bGM, biasGM);

    for (uint32_t i = 0; i < tileCnt; i++) {
        mm.InitGlobalATensor(aAddr, aOffset, computeResAddr, cOffset);
        mm.Compute(indexC, enAtomicC);
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
        CrossCoreWaitFlag(0x8);
        hccl.Commit(handleId);
        aAddr += aOffset;
        cAddr += cOffset;
        computeResAddr += cOffset;
    }
    mm.End();
}
}
#endif // MC2_MM_ALL_REDUCE_H

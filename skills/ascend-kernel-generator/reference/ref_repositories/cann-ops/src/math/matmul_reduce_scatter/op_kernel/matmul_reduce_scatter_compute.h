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
 * @file matmul_reduce_scatter_compute.h
 */

#ifndef MC2_MATMUL_COMPUTE_H
#define MC2_MATMUL_COMPUTE_H

#include "matmul_reduce_scatter_tiling.h"
#include "matmul_reduce_scatter_common.h"
#include "matmul_reduce_scatter_block.h"

namespace AscendC {

constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, false, true);

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
class MatmulCompute {
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulCompute() {}
    __aicore__ inline void Init(TCubeTiling& tiling, ReduceScatterRCSTiling& cfg);
    __aicore__ inline void InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM);
    __aicore__ inline void InitGlobalATensor(GM_ADDR aGM, uint32_t aSize, GM_ADDR cGM, uint32_t cSize);
    __aicore__ inline void Compute(uint32_t index=0, uint8_t enAtomic=0);
    __aicore__ inline void End();

    __aicore__ inline void SetOrgShapeAlign();
    __aicore__ inline void SetSingleCoreShape();

    MatmulBlock block;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM)
{
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), block.tiling.Kb * block.tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block.tiling.N);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::InitGlobalATensor(GM_ADDR aGM, uint32_t aSize, GM_ADDR cGM, uint32_t cSize)
{
    // MC2的计算流中默认B矩阵不变，GM地址无需偏移
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), aSize);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), cSize);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::Init(TCubeTiling& tiling, ReduceScatterRCSTiling& cfg)
{
    // MatmulImpl初始化
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, GetTPipePtr());
    // MatmulBlock初始化
    block.Init(tiling, cfg);
    SetOrgShapeAlign();
    LocalTensor<uint8_t> mmFormatUb;
    mm.SetLocalWorkspace(mmFormatUb);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::SetOrgShapeAlign()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        auto alignKa = AlignUp(block.tiling.Ka, SHAPE_ALIGNED_SIZE);
        auto alignKb = AlignUp(block.tiling.Kb, SHAPE_ALIGNED_SIZE);
        auto alignM = AlignUp(block.tiling.M, SHAPE_ALIGNED_SIZE);
        auto alignN = AlignUp(block.tiling.N, SHAPE_ALIGNED_SIZE);
        mm.SetOrgShape(alignM, alignN, alignKa, alignKb, block.cfg.rankN);
    } else if (A_TYPE::format == CubeFormat::NZ) {
        auto alignKa = AlignUp(block.tiling.Ka, SHAPE_ALIGNED_SIZE);
        auto alignM = AlignUp(block.tiling.M, SHAPE_ALIGNED_SIZE);
        mm.SetOrgShape(alignM, block.tiling.N, alignKa, block.tiling.Kb, block.cfg.rankN);
    } else if (B_TYPE::format == CubeFormat::NZ) {
        auto alignKb = AlignUp(block.tiling.Kb, SHAPE_ALIGNED_SIZE);
        auto alignN = AlignUp(block.tiling.N, SHAPE_ALIGNED_SIZE);
        mm.SetOrgShape(block.tiling.M, alignN, block.tiling.Ka, alignKb, block.cfg.rankN);
    }
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::SetSingleCoreShape()
{
    if (block.mBlockIndex == (block.mBlockCnt - 1) && block.nBlockIndex == (block.nBlockCnt - 1)) {
        // 最后一块是尾块
        mm.SetSingleShape(block.mBaseTail, block.nBaseTail, block.tiling.singleCoreK);
    } else if (block.mBlockIndex == (block.mBlockCnt - 1)) {
        // M方向的尾块
        mm.SetSingleShape(block.mBaseTail, block.tiling.baseN, block.tiling.singleCoreK);
    } else if (block.nBlockIndex == (block.nBlockCnt - 1)) {
        // N方向的尾块
        mm.SetSingleShape(block.tiling.baseM, block.nBaseTail, block.tiling.singleCoreK);
    } else {
        mm.SetSingleShape(block.tiling.baseM, block.tiling.baseN, block.tiling.singleCoreK);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::Compute(uint32_t index, uint8_t enAtomic)
{
    // 每次block循环开始前需要计算初始blockIndex
    block.UpdateBlockCnt(index);
    for (uint32_t i = 0; i < block.blockCnt; i++) {
        if (block.blockIndex < block.totalBlockCnt) {
            block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            SetSingleCoreShape();
            mm.SetTensorA(aGlobal[block.offset.offsetA], block.isTransposeA);
            mm.SetTensorB(bGlobal[block.offset.offsetB], block.isTransposeB);
            if (block.tiling.isBias) {
                mm.SetBias(biasGlobal[block.offset.offsetBias]);
            }
            mm.Iterate();
            mm.GetTensorC(cGlobal[block.offset.offsetC], enAtomic);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
        }
        block.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::End()
{
    mm.End();
}
}
#endif // MC2_MATMUL_COMPUTE_H

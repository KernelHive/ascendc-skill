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
 * @file matmul_all_reduce_common.h
 */

#ifndef MC2_ALLREDUCE_COMM_H
#define MC2_ALLREDUCE_COMM_H

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

#include "lib/hccl/hccl.h"

namespace AscendC {
// 代码多数据类型支持
using A_DTYPE = DTYPE_X1;
using B_DTYPE = DTYPE_X1;
using C_DTYPE = DTYPE_Y;
using BIAS_DTYPE = DTYPE_Y;

using namespace matmul;
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void CalcOffsetA(int blockIdx, int usedCoreNum, TCubeTiling &param, uint64_t &offsetA, 
    int32_t isTransposeAIn, int mCoreIndx, int nCoreIndx, int subKindx)
{
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        offsetA = isTransposeAIn > 0 
            ? mCoreIndx * param.singleCoreM + subKindx * param.M * param.singleCoreK
            : mCoreIndx * param.Ka * param.singleCoreM + subKindx * param.singleCoreK;
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        offsetA = subKindx * param.singleCoreK * param.M + mCoreIndx * param.singleCoreM * BLOCK_CUBE;
    } else {
        ASSERT(false && "Data format of A matrix should be ND or NZ.");
    }
}

template <class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void CalcOffsetB(int blockIdx, int usedCoreNum, TCubeTiling &param, uint64_t &offsetB, 
    int32_t isTransposeBIn, int mCoreIndx, int nCoreIndx, int subKindx)
{
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        offsetB = isTransposeBIn > 0
            ? subKindx * param.singleCoreK + nCoreIndx * param.Ka * param.singleCoreN
            : subKindx * param.singleCoreK * param.N + nCoreIndx * param.singleCoreN;
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        offsetB = isTransposeBIn > 0
            ? nCoreIndx * param.singleCoreN * 16
            : param.Ka * nCoreIndx * param.singleCoreN + subKindx * param.singleCoreK * BLOCK_CUBE;
    } else {
        ASSERT(false && "Data format of B matrix should be ND or NZ.");
    }
}

template <class C_TYPE, class BIAS_TYPE>
__aicore__ inline void CalcOffsetCAndBias(TCubeTiling &param, uint64_t &offsetC, uint64_t &offsetBias, 
    int mCoreIndx, int nCoreIndx)
{
    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
        offsetC = mCoreIndx * param.N * param.singleCoreM + nCoreIndx * param.singleCoreN;
    } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
        offsetC = param.M * nCoreIndx * param.singleCoreN + mCoreIndx * param.singleCoreM * BLOCK_CUBE;
    } else {
        ASSERT(false && "Data format of C matrix should be ND or ND_ALIGN or NZ.");
    }

    if constexpr (BIAS_TYPE::format == CubeFormat::ND) {
        offsetBias = nCoreIndx * param.singleCoreN;
    } else {
        ASSERT(false && "Data format of BIAS should be ND.");
    }
}

__aicore__ inline void CalcTailBlocks(TCubeTiling &param, int mCoreIndx, int nCoreIndx, int subKindx)
{
    // 尾块M
    int gmUseM = param.M - mCoreIndx * param.singleCoreM;
    param.singleCoreM = gmUseM < param.singleCoreM ? gmUseM : param.singleCoreM;

    // 尾块N
    int gmUseN = param.N - nCoreIndx * param.singleCoreN;
    param.singleCoreN = gmUseN < param.singleCoreN ? gmUseN : param.singleCoreN;

    // 尾块K
    int gmUseK = param.Ka - subKindx * param.singleCoreK;
    param.singleCoreK = gmUseK < param.singleCoreK ? gmUseK : param.singleCoreK;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void CalcGMOffset(int blockIdx, int usedCoreNum, TCubeTiling &param, uint64_t &offsetA, uint64_t &offsetB,
    uint64_t &offsetC, uint64_t &offsetBias, int32_t isTransposeAIn, int32_t isTransposeBIn)
{
    auto temp0 = Ceil(param.M, param.singleCoreM);
    auto temp2 = Ceil(param.Ka, param.singleCoreK);
    auto divideKcoreNum = usedCoreNum / temp2;

    auto mCoreIndx = (blockIdx % divideKcoreNum) % temp0;
    auto nCoreIndx = (blockIdx % divideKcoreNum) / temp0;
    auto subKindx = blockIdx / divideKcoreNum;

    CalcOffsetA<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(blockIdx, usedCoreNum, param, offsetA, isTransposeAIn, mCoreIndx, nCoreIndx, subKindx);
    CalcOffsetB<B_TYPE, C_TYPE, BIAS_TYPE>(blockIdx, usedCoreNum, param, offsetB, isTransposeBIn, mCoreIndx, nCoreIndx, subKindx);
    CalcOffsetCAndBias<C_TYPE, BIAS_TYPE>(param, offsetC, offsetBias, mCoreIndx, nCoreIndx);
    CalcTailBlocks(param, mCoreIndx, nCoreIndx, subKindx);
}

__aicore__ __inline__ GM_ADDR GetTailA(GM_ADDR aGM, TCubeTiling& tiling, uint32_t size)
{
    return aGM + (tiling.M * tiling.Ka) * sizeof(A_DTYPE) * size;
}
__aicore__ __inline__ GM_ADDR GetTailC(GM_ADDR cGM, TCubeTiling& tiling, uint32_t size)
{
    return cGM + (tiling.M * tiling.N) * sizeof(C_DTYPE) * size;
}

}
#endif // MC2_ALLREDUCE_COMM_H

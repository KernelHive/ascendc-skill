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
 * \file lin_space_with_big_shape_and_cast.h
 * \brief
 */
#ifndef LINSPACE_WITH_BIG_SHAPE_AND_CAST_H
#define LINSPACE_WITH_BIG_SHAPE_AND_CAST_H

#include "lin_space_base.h"

namespace LinSpace {
using namespace AscendC;

template <typename T1, typename T2>
class LinSpaceWithBigShapeAndCast : public LinSpaceBase<T2> {
public:
    __aicore__ inline LinSpaceWithBigShapeAndCast(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                GM_ADDR workspace, const LinSpaceTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyInReverse();
    __aicore__ inline void ComputeAndOut(const int64_t &loopNum, const int64_t &loopTail);
    __aicore__ inline void ComputeReverseAndOut(const int64_t &loopNum, const int64_t &loopTail);
    __aicore__ inline void CopyOut(const int64_t offset, const int64_t &outLen);
    __aicore__ inline void CopyOutReverse(const int64_t offset, const int64_t &outLen);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();
    __aicore__ inline void ProcessPerCoreReverse();

    constexpr static int32_t matrixSize = 256;
    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t POWER_BASE_NUM = 2;
    constexpr static int32_t outSize = 16 * 1024;
    constexpr static int32_t maxOutNum = outSize / sizeof(T2);
    constexpr static int64_t reverseScalar = -1.0;
    constexpr static int64_t blockSize = 32;
    constexpr static int64_t elementPerBlock = blockSize / sizeof(T1);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueMatrix;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;
    TBuf<QuePosition::VECCALC> castBuf;
    GlobalTensor<T1> outputGm;
    GlobalTensor<T2> gmAssist;
    GlobalTensor<T2> gmAssistReverse;

    int32_t blockIdx = 0;
    T2 blockOffset = 0;
    int64_t gmOutOffset = 0;
    // tiling params
    LinSpaceTilingData m_tilingData;
    RoundMode retR = RoundMode::CAST_TRUNC;
};

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num,
                                                                 GM_ADDR output, GM_ADDR workspace,
                                                                 const LinSpaceTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    outputGm.SetGlobalBuffer((__gm__ T1 *)output);
    gmAssist.SetGlobalBuffer((__gm__ T2 *)this->assistGm, matrixSize);
    gmAssistReverse.SetGlobalBuffer((__gm__ T2 *)this->assistGmReverse, matrixSize);
    this->ParseTilingData(tilingData, m_tilingData);

    pipe.InitBuffer(inQueueMatrix, 1, matrixSize * sizeof(T2));
    pipe.InitBuffer(outQueue, bufferNum, maxOutNum * sizeof(T1));
    pipe.InitBuffer(castBuf, maxOutNum * sizeof(T2));

    if (sizeof(T1) == sizeof(int16_t) && sizeof(T2) == sizeof(float)) {
        retR = RoundMode::CAST_ROUND;
    }
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::Process()
{
    if (m_tilingData.num == 0 || blockIdx >= m_tilingData.realCoreNum) {
        return;
    }

    // load matrix
    #if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        OOMCheckAddrRange(gmAssist.GetPhyAddr(), 2 * matrixSize * sizeof(T2));
    #endif

    if (blockIdx < m_tilingData.realCoreNum / POWER_BASE_NUM) {
        blockOffset = m_tilingData.scalar * blockIdx * m_tilingData.numPerCore + m_tilingData.start;
        ProcessPerCore();
    } else if (blockIdx == m_tilingData.realCoreNum - 1) {  // process last core
        blockOffset = m_tilingData.stop;
        ProcessLastCore();
    } else {
        blockOffset =
            m_tilingData.stop - m_tilingData.scalar * (m_tilingData.num - (blockIdx + 1) * m_tilingData.numPerCore);
        ProcessPerCoreReverse();
    }
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::ProcessPerCore()
{
    gmOutOffset = blockIdx * m_tilingData.numPerCore;
    CopyIn();
    ComputeAndOut(m_tilingData.outerLoopNum, m_tilingData.outerLoopNumTail);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::ProcessLastCore()
{
    gmOutOffset = m_tilingData.num;
    CopyInReverse();
    ComputeReverseAndOut(m_tilingData.outerTailLoopNum, this->CeilDiv(m_tilingData.outerTailLoopNumTail, elementPerBlock) * elementPerBlock);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::ProcessPerCoreReverse()
{
    gmOutOffset = blockIdx * m_tilingData.numPerCore + m_tilingData.numPerCore;
    CopyInReverse();
    ComputeReverseAndOut(m_tilingData.outerLoopNum, m_tilingData.outerLoopNumTail);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::CopyIn()
{
    LocalTensor<T2> ubAssist = inQueueMatrix.AllocTensor<T2>();
    DataCopy(ubAssist, gmAssist, m_tilingData.matrixLen);
    inQueueMatrix.EnQue(ubAssist);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::CopyInReverse()
{
    LocalTensor<T2> ubAssist = inQueueMatrix.AllocTensor<T2>();
    DataCopy(ubAssist, gmAssistReverse, m_tilingData.matrixLen);
    inQueueMatrix.EnQue(ubAssist);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::ComputeAndOut(
    const int64_t &loopNum, const int64_t &loopTail)
{
    LocalTensor<T2> ubAssist = inQueueMatrix.DeQue<T2>();
    LocalTensor<T1> outLocal = outQueue.AllocTensor<T1>();
    LocalTensor<T2> ubNeedCast = castBuf.Get<T2>();

    Muls(ubNeedCast, ubAssist, T2(m_tilingData.scalar), m_tilingData.matrixLen);
    Adds(ubNeedCast, ubNeedCast, blockOffset, m_tilingData.matrixLen);
    inQueueMatrix.FreeTensor(ubAssist);

    for (int64_t idx = 1; idx <= maxOutNum / matrixSize / POWER_BASE_NUM; idx *= POWER_BASE_NUM) {
        Adds(ubNeedCast[idx * matrixSize], ubNeedCast, T2(m_tilingData.scalar * matrixSize * idx), matrixSize * idx);
    }

    if (loopNum == 1 && loopTail > 0) {
        Cast(outLocal, ubNeedCast, retR, loopTail);
        outQueue.EnQue(outLocal);
        CopyOut(gmOutOffset, loopTail);
    } else if (loopNum > 0) {
        Cast(outLocal, ubNeedCast, retR, maxOutNum);
        outQueue.EnQue(outLocal);
        CopyOut(gmOutOffset, maxOutNum);
    }

    for (int64_t idx = 1; idx < loopNum; idx++) {
        LocalTensor<T1> outLocal = outQueue.AllocTensor<T1>();
        if (idx == loopNum - 1) {
            Adds(ubNeedCast, ubNeedCast, T2(m_tilingData.scalar * maxOutNum), loopTail);
            Cast(outLocal, ubNeedCast, retR, loopTail);
            outQueue.EnQue(outLocal);
            CopyOut(gmOutOffset + idx * maxOutNum, loopTail);
        } else {
            Adds(ubNeedCast, ubNeedCast, T2(m_tilingData.scalar * maxOutNum), maxOutNum);
            Cast(outLocal, ubNeedCast, retR, maxOutNum);
            outQueue.EnQue(outLocal);
            CopyOut(gmOutOffset + idx * maxOutNum, maxOutNum);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::ComputeReverseAndOut(
    const int64_t &loopNum, const int64_t &loopTail)
{
    LocalTensor<T2> ubAssist = inQueueMatrix.DeQue<T2>();
    LocalTensor<T1> outLocal = outQueue.AllocTensor<T1>();
    LocalTensor<T2> ubNeedCast = castBuf.Get<T2>();

    Muls(ubNeedCast[maxOutNum - matrixSize], ubAssist, T2(m_tilingData.scalar), m_tilingData.matrixLen);
    Adds(ubNeedCast[maxOutNum - matrixSize], ubNeedCast[maxOutNum - matrixSize], blockOffset, m_tilingData.matrixLen);
    inQueueMatrix.FreeTensor(ubAssist);

    for (int64_t idx = 1; idx <= maxOutNum / matrixSize / POWER_BASE_NUM; idx *= POWER_BASE_NUM) {
        Adds(ubNeedCast[maxOutNum - idx * matrixSize * POWER_BASE_NUM],
            ubNeedCast[maxOutNum - idx * matrixSize],
            T2(m_tilingData.scalar * matrixSize * idx * reverseScalar),
            matrixSize * idx);
    }

    if (loopNum == 1 && loopTail > 0) {
        Cast(outLocal[maxOutNum - loopTail], ubNeedCast[maxOutNum - loopTail], retR, loopTail);
        outQueue.EnQue(outLocal);
        CopyOutReverse(gmOutOffset - loopTail, loopTail);
    } else if (loopNum > 0) {
        Cast(outLocal[maxOutNum - maxOutNum], ubNeedCast[maxOutNum - maxOutNum], retR, maxOutNum);
        outQueue.EnQue(outLocal);
        CopyOutReverse(gmOutOffset - maxOutNum, maxOutNum);
    }

    for (int64_t idx = 1; idx < loopNum; idx++) {
        LocalTensor<T1> outLocal = outQueue.AllocTensor<T1>();
        if (idx == loopNum - 1) {
            Adds(ubNeedCast[maxOutNum - loopTail],
                ubNeedCast[maxOutNum - loopTail],
                T2(m_tilingData.scalar * maxOutNum * reverseScalar),
                loopTail);
            Cast(outLocal[maxOutNum - loopTail], ubNeedCast[maxOutNum - loopTail], retR, loopTail);
            outQueue.EnQue(outLocal);
            CopyOutReverse(gmOutOffset - idx * maxOutNum - loopTail, loopTail);
        } else {
            Adds(ubNeedCast, ubNeedCast, T2(m_tilingData.scalar * maxOutNum * reverseScalar), maxOutNum);
            Cast(outLocal, ubNeedCast, retR, maxOutNum);
            outQueue.EnQue(outLocal);
            CopyOutReverse(gmOutOffset - idx * maxOutNum - maxOutNum, maxOutNum);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::CopyOut(const int64_t offset, const int64_t &outLen)
{
    LocalTensor<T1> outLocal = outQueue.DeQue<T1>();
    DataCopy(outputGm[offset], outLocal, outLen);
    outQueue.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceWithBigShapeAndCast<T1, T2>::CopyOutReverse(const int64_t offset, const int64_t &outLen)
{
    LocalTensor<T1> outLocal = outQueue.DeQue<T1>();
    DataCopy(outputGm[offset], outLocal[maxOutNum - outLen], outLen);
    outQueue.FreeTensor(outLocal);
}

}  // namespace LinSpace

#endif  // LINSPACE_WITH_BIG_SHAPE_AND_CAST_H
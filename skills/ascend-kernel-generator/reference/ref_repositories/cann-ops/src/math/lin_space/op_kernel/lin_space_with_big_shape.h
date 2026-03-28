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
 * \file lin_space_with_big_shape.h
 * \brief
 */
#ifndef LINSPACE_WITH_BIG_SHAPE_H
#define LINSPACE_WITH_BIG_SHAPE_H

#include "lin_space_base.h"

namespace LinSpace {
using namespace AscendC;

template <typename T>
class LinSpaceWithBigShape : public LinSpaceBase<T> {
public:
    __aicore__ inline LinSpaceWithBigShape(){};
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
    constexpr static int32_t bufferNum = 3;
    constexpr static int64_t POWER_BASE_NUM = 2;
    constexpr static int32_t outSize = 16 * 1024;
    constexpr static int32_t maxOutNum = outSize / sizeof(T);
    constexpr static int64_t reverseScalar = -1.0;
    constexpr static int64_t blockSize = 32;
    constexpr static int64_t elementPerBlock = blockSize / sizeof(T);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueMatrix;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> gmAssist;
    GlobalTensor<T> gmAssistReverse;

    int32_t blockIdx = 0;
    T blockOffset = 0;
    int64_t gmOutOffset = 0;
    // tiling params
    LinSpaceTilingData m_tilingData;
};

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                                     GM_ADDR workspace, const LinSpaceTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    outputGm.SetGlobalBuffer((__gm__ T *)output);
    gmAssist.SetGlobalBuffer((__gm__ T *)this->assistGm, matrixSize);
    gmAssistReverse.SetGlobalBuffer((__gm__ T *)this->assistGmReverse, matrixSize);
    this->ParseTilingData(tilingData, m_tilingData);

    pipe.InitBuffer(inQueueMatrix, 1, matrixSize * sizeof(T));
    pipe.InitBuffer(outQueue, bufferNum, outSize);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::Process()
{
    if (m_tilingData.num == 0 || blockIdx >= m_tilingData.realCoreNum) {
        return;
    }
    // load matrix
    #if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        OOMCheckAddrRange(gmAssist.GetPhyAddr(), 2 * matrixSize * sizeof(T));
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

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::ProcessPerCore()
{
    gmOutOffset = blockIdx * m_tilingData.numPerCore;
    CopyIn();
    ComputeAndOut(m_tilingData.outerLoopNum, m_tilingData.outerLoopNumTail);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::ProcessLastCore()
{
    gmOutOffset = m_tilingData.num;
    CopyInReverse();
    ComputeReverseAndOut(m_tilingData.outerTailLoopNum, this->CeilDiv(m_tilingData.outerTailLoopNumTail, elementPerBlock) * elementPerBlock);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::ProcessPerCoreReverse()
{
    gmOutOffset = blockIdx * m_tilingData.numPerCore + m_tilingData.numPerCore;
    CopyInReverse();
    ComputeReverseAndOut(m_tilingData.outerLoopNum, m_tilingData.outerLoopNumTail);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::CopyIn()
{
    LocalTensor<T> ubAssist = inQueueMatrix.AllocTensor<T>();
    DataCopy(ubAssist, gmAssist, m_tilingData.matrixLen);
    inQueueMatrix.EnQue(ubAssist);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::CopyInReverse()
{
    LocalTensor<T> ubAssist = inQueueMatrix.AllocTensor<T>();
    DataCopy(ubAssist, gmAssistReverse, m_tilingData.matrixLen);
    inQueueMatrix.EnQue(ubAssist);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::ComputeAndOut(const int64_t &loopNum, const int64_t &loopTail)
{
    LocalTensor<T> ubAssist = inQueueMatrix.DeQue<T>();
    LocalTensor<T> outLocalBase = outQueue.AllocTensor<T>();

    Muls(outLocalBase, ubAssist, T(m_tilingData.scalar), m_tilingData.matrixLen);
    Adds(outLocalBase, outLocalBase, blockOffset, m_tilingData.matrixLen);

    inQueueMatrix.FreeTensor(ubAssist);

    for (int64_t idx = 1; idx <= maxOutNum / matrixSize / POWER_BASE_NUM; idx *= POWER_BASE_NUM) {
        Adds(outLocalBase[idx * matrixSize], outLocalBase, T(m_tilingData.scalar * matrixSize * idx), idx * matrixSize);
    }

    for (int64_t idx = 1; idx < loopNum; idx++) {
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        if (idx == loopNum - 1) {
            Adds(outLocal, outLocalBase, T(m_tilingData.scalar * maxOutNum * idx), loopTail);
            outQueue.EnQue(outLocal);
            CopyOut(gmOutOffset + idx * maxOutNum, loopTail);
        } else {
            Adds(outLocal, outLocalBase, T(m_tilingData.scalar * maxOutNum * idx), maxOutNum);
            outQueue.EnQue(outLocal);
            CopyOut(gmOutOffset + idx * maxOutNum, maxOutNum);
        }
    }

    if (loopNum == 1 && loopTail > 0) {
        outQueue.EnQue(outLocalBase);
        CopyOut(gmOutOffset, loopTail);
    } else if (loopNum > 0) {
        outQueue.EnQue(outLocalBase);
        CopyOut(gmOutOffset, maxOutNum);
    }
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::ComputeReverseAndOut(const int64_t &loopNum, const int64_t &loopTail)
{
    LocalTensor<T> ubAssist = inQueueMatrix.DeQue<T>();
    LocalTensor<T> outLocalBase = outQueue.AllocTensor<T>();

    Muls(outLocalBase[maxOutNum - matrixSize], ubAssist, T(m_tilingData.scalar), m_tilingData.matrixLen);
    Adds(outLocalBase[maxOutNum - matrixSize], outLocalBase[maxOutNum - matrixSize], blockOffset, m_tilingData.matrixLen);

    inQueueMatrix.FreeTensor(ubAssist);

    for (int64_t idx = 1; idx <= maxOutNum / matrixSize / POWER_BASE_NUM; idx *= POWER_BASE_NUM) {
        Adds(outLocalBase[maxOutNum - idx * matrixSize * POWER_BASE_NUM],
            outLocalBase[maxOutNum - idx * matrixSize],
            T(m_tilingData.scalar * matrixSize * idx * reverseScalar),
            matrixSize * idx);
    }

    for (int64_t idx = 1; idx < loopNum; idx++) {
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        if (idx == loopNum - 1) {
            Adds(outLocal[maxOutNum - loopTail], outLocalBase[maxOutNum - loopTail], T(m_tilingData.scalar * maxOutNum * idx * reverseScalar), loopTail);
            outQueue.EnQue(outLocal);
            CopyOutReverse(gmOutOffset - idx * maxOutNum - loopTail, loopTail);
        } else {
            Adds(outLocal, outLocalBase, T(m_tilingData.scalar * maxOutNum * idx * reverseScalar), maxOutNum);
            outQueue.EnQue(outLocal);
            CopyOutReverse(gmOutOffset - idx * maxOutNum - maxOutNum, maxOutNum);
        }
    }

    if (loopNum == 1 && loopTail > 0) {
        outQueue.EnQue(outLocalBase);
        CopyOutReverse(gmOutOffset - loopTail, loopTail);
    } else if (loopNum > 0) {
        outQueue.EnQue(outLocalBase);
        CopyOutReverse(gmOutOffset - maxOutNum, maxOutNum);
    }
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::CopyOut(const int64_t offset, const int64_t &outLen)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopy(outputGm[offset], outLocal, outLen);
    outQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LinSpaceWithBigShape<T>::CopyOutReverse(const int64_t offset, const int64_t &outLen)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopy(outputGm[offset], outLocal[maxOutNum - outLen], outLen);
    outQueue.FreeTensor(outLocal);
}

}  // namespace LinSpace

#endif  // LINSPACE_WITH_BIG_SHAPE_H
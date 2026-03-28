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
 * \file lin_space_half_and_float.h
 * \brief
 */
#ifndef LINSPACE_HALF_AND_FLOAT_H
#define LINSPACE_HALF_AND_FLOAT_H

#include "lin_space_base.h"

namespace LinSpace {
using namespace AscendC;

template <typename T>
class LinSpaceHalfAndFloat : public LinSpaceBase<T> {
public:
    __aicore__ inline LinSpaceHalfAndFloat(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output, GM_ADDR workspace,
                                const LinSpaceTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyInReverse();
    __aicore__ inline void Compute(const int64_t &processNum, const int64_t &loopNum, const int64_t &loopTail);
    __aicore__ inline void ComputeReverse(const int64_t &processNum, const int64_t &loopNum, const int64_t &loopTail);
    __aicore__ inline void CopyOut(const int64_t &outLen);
    __aicore__ inline void CopyOutReverse(const int64_t &outLen);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessPerCoreReverse();
    __aicore__ inline void ProcessLastCore();

    constexpr static int64_t matrixSize = 256;
    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t POWER_BASE_NUM = 2;
    constexpr static int32_t outSize = 16 * 1024;
    constexpr static int32_t outNum = outSize / sizeof(T);
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
__aicore__ inline void LinSpaceHalfAndFloat<T>::Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                                     GM_ADDR workspace, const LinSpaceTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    outputGm.SetGlobalBuffer((__gm__ T *)output);
    gmAssist.SetGlobalBuffer((__gm__ T *)this->assistGm, matrixSize);
    gmAssistReverse.SetGlobalBuffer((__gm__ T *)this->assistGmReverse, matrixSize);
    this->ParseTilingData(tilingData, m_tilingData);

    pipe.InitBuffer(inQueueMatrix, 1, matrixSize * sizeof(T));
    pipe.InitBuffer(outQueue, bufferNum, outSize);

    gmOutOffset = blockIdx * m_tilingData.numPerCore;
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::Process()
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
__aicore__ inline void LinSpaceHalfAndFloat<T>::ProcessPerCore()
{
    CopyIn();
    Compute(m_tilingData.numPerCore, m_tilingData.innerLoopNum, m_tilingData.innerLoopTail);
    CopyOut(m_tilingData.numPerCore);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::ProcessLastCore()
{
    int64_t tailAlignNum = this->CeilDiv(m_tilingData.tailNum, elementPerBlock) * elementPerBlock;
    gmOutOffset = m_tilingData.num - tailAlignNum;  // must be bigger than two cores
    CopyInReverse();
    ComputeReverse(tailAlignNum, m_tilingData.innerTailLoopNum,
                   this->CeilDiv(m_tilingData.innerTailLoopTail, elementPerBlock) * elementPerBlock);
    CopyOutReverse(tailAlignNum);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::ProcessPerCoreReverse()
{
    CopyInReverse();
    ComputeReverse(m_tilingData.numPerCore, m_tilingData.innerLoopNum, m_tilingData.innerLoopTail);
    CopyOutReverse(m_tilingData.numPerCore);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::CopyIn()
{
    LocalTensor<T> ubAssist = inQueueMatrix.AllocTensor<T>();
    DataCopy(ubAssist, gmAssist, m_tilingData.matrixLen);
    inQueueMatrix.EnQue(ubAssist);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::CopyInReverse()
{
    LocalTensor<T> ubAssist = inQueueMatrix.AllocTensor<T>();
    DataCopy(ubAssist, gmAssistReverse, matrixSize);
    inQueueMatrix.EnQue(ubAssist);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::Compute(
    const int64_t &processNum, const int64_t &loopNum, const int64_t &loopTail)
{
    LocalTensor<T> ubAssist = inQueueMatrix.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    Muls(outLocal, ubAssist, T(m_tilingData.scalar), m_tilingData.matrixLen);
    Adds(outLocal, outLocal, blockOffset, m_tilingData.matrixLen);

    for (int64_t idx = 1; idx <= loopNum; idx *= POWER_BASE_NUM) {
        Adds(outLocal[idx * matrixSize], outLocal, T(m_tilingData.scalar * matrixSize * idx), matrixSize * idx);
    }

    if (loopTail > 0) {
        Adds(outLocal[processNum - loopTail], outLocal, T(m_tilingData.scalar * (processNum - loopTail)), loopTail);
    }
    outQueue.EnQue(outLocal);
    inQueueMatrix.FreeTensor(ubAssist);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::ComputeReverse(
    const int64_t &processNum, const int64_t &loopNum, const int64_t &loopTail)
{
    LocalTensor<T> ubAssist = inQueueMatrix.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    Muls(outLocal[outNum - matrixSize], ubAssist, T(m_tilingData.scalar), matrixSize);
    Adds(outLocal[outNum - matrixSize], outLocal[outNum - matrixSize], blockOffset, matrixSize);

    for (int64_t idx = 1; idx <= loopNum; idx *= POWER_BASE_NUM) {
        Adds(outLocal[outNum - idx * matrixSize * POWER_BASE_NUM],
            outLocal[outNum - idx * matrixSize],
            T(m_tilingData.scalar * matrixSize * idx * reverseScalar),
            matrixSize * idx);
    }

    if (loopTail > 0) {
        Adds(outLocal[outNum - processNum],
            outLocal[outNum - loopTail],
            T(m_tilingData.scalar * (processNum - loopTail) * reverseScalar),
            loopTail);
    }
    outQueue.EnQue(outLocal);
    inQueueMatrix.FreeTensor(ubAssist);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::CopyOut(const int64_t &outLen)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopy(outputGm[gmOutOffset], outLocal, outLen);
    outQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LinSpaceHalfAndFloat<T>::CopyOutReverse(const int64_t &outLen)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopy(outputGm[gmOutOffset], outLocal[outNum - outLen], outLen);
    outQueue.FreeTensor(outLocal);
}

}  // namespace LinSpace

#endif  // LINSPACE_HALF_AND_FLOAT_H
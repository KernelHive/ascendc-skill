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
 * @file complex_mat_dot_aiv.h
 */
#ifndef COMPLEX_MAT_DOT_AIV_H
#define COMPLEX_MAT_DOT_AIV_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ELENUM_EACH_COMPLEX = 2;

namespace ComplexMatDot {

struct ComplexMatDotKernelParam {
    GM_ADDR matx;
    GM_ADDR maty;
    GM_ADDR result;
    uint32_t m;
    uint32_t n;
    uint64_t offset;
    uint32_t calNumPerCore;
};

template <typename T>
class ComplexMatDotAIV {
public:
    __aicore__ inline ComplexMatDotAIV(){};
    __aicore__ inline void Init(ComplexMatDotKernelParam kernelParam);
    __aicore__ inline void Process();
    __aicore__ inline void SingleIteration(uint64_t offset, uint64_t dataCount, LocalTensor<uint32_t> offsetLocal);
    __aicore__ inline void SingleIterationAligned(uint64_t offset, uint32_t dataCount,
                                                  LocalTensor<uint32_t> offsetLocal);
    __aicore__ inline void GenOffsetData();
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint64_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount, LocalTensor<uint32_t> offsetLocal);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataCount, uint32_t isAligned);

private:
    TPipe pipe;

    GlobalTensor<T> matxGM;
    GlobalTensor<T> matyGM;
    GlobalTensor<T> resultGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> xMatQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> yMatQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outMatQueue;

    TBuf<TPosition::VECCALC> offsetBuf;

    uint32_t calNum;

    uint64_t startOffset;
    uint64_t maxDataCount;
};

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::Init(ComplexMatDotKernelParam kernelParam)
{
    matxGM.SetGlobalBuffer((__gm__ T *)(kernelParam.matx), kernelParam.m * kernelParam.n);
    matyGM.SetGlobalBuffer((__gm__ T *)(kernelParam.maty), kernelParam.m * kernelParam.n);
    resultGM.SetGlobalBuffer((__gm__ T *)(kernelParam.result), kernelParam.m * kernelParam.n);

    calNum = kernelParam.calNumPerCore;

    startOffset = kernelParam.offset;
    maxDataCount = 27 * 1024 / 4;  // 27kb / 4b

    // ub 192kb
    pipe.InitBuffer(xMatQueue, 2, maxDataCount * sizeof(T));    // 54kb
    pipe.InitBuffer(yMatQueue, 2, maxDataCount * sizeof(T));    // 54kb
    pipe.InitBuffer(outMatQueue, 2, maxDataCount * sizeof(T));  // 54kb

    uint64_t offsetLen = maxDataCount * 4;
    pipe.InitBuffer(offsetBuf, offsetLen);  // 27kb

    return;
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::Process()
{
    SetAtomicNone();
    GenOffsetData();

    // cal offset
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();

    uint32_t repeatTimes = calNum * 2 / static_cast<uint32_t>(maxDataCount);
    uint32_t remainNum = calNum * 2 % static_cast<uint32_t>(maxDataCount);
    uint64_t currOffset = startOffset;

    if (repeatTimes > 0) {
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < repeatTimes; i++) {
            SingleIteration(currOffset, maxDataCount, offsetLocal);
            currOffset += maxDataCount;
        }
    }

    if (remainNum > 0) {
        AscendC::PipeBarrier<PIPE_V>();
        SingleIterationAligned(currOffset, remainNum, offsetLocal);
    }

    AscendC::PipeBarrier<PIPE_V>();
    return;
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::GenOffsetData()
{
    uint32_t complexCount = maxDataCount / ELENUM_EACH_COMPLEX;
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();

    for (uint32_t i = 0; i < complexCount; i++) {
        offsetLocal.SetValue(ELENUM_EACH_COMPLEX * i, sizeof(T) * i);
        offsetLocal.SetValue(ELENUM_EACH_COMPLEX * i + 1, sizeof(T) * (i + complexCount));
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::SingleIteration(uint64_t offset, uint64_t dataCount,
                                                            LocalTensor<uint32_t> offsetLocal)
{
    CopyIn(offset, dataCount);
    Compute(dataCount, offsetLocal);
    CopyOut(offset, dataCount, 1);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::SingleIterationAligned(uint64_t offset, uint32_t dataCount,
                                                                   LocalTensor<uint32_t> offsetLocal)
{
    uint32_t dataCountAligned = (dataCount + 7) / 8 * 8;
    CopyInPad(offset, dataCount);
    Compute(dataCountAligned, offsetLocal);
    CopyOut(offset, dataCount, 0);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::CopyIn(uint64_t offset, uint32_t dataCount)
{
    LocalTensor<T> xMatLocal = xMatQueue.AllocTensor<T>();
    LocalTensor<T> yMatLocal = yMatQueue.AllocTensor<T>();
    DataCopy(xMatLocal, matxGM[offset], dataCount);
    DataCopy(yMatLocal, matyGM[offset], dataCount);
    xMatQueue.EnQue<T>(xMatLocal);
    yMatQueue.EnQue<T>(yMatLocal);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::CopyInPad(uint64_t offset, uint32_t dataCount)
{
    LocalTensor<T> xMatLocal = xMatQueue.AllocTensor<T>();
    LocalTensor<T> yMatLocal = yMatQueue.AllocTensor<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>((dataCount) * sizeof(float)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(xMatLocal, matxGM[offset], copyParams, padParams);
    DataCopyPad(yMatLocal, matyGM[offset], copyParams, padParams);
    xMatQueue.EnQue<T>(xMatLocal);
    yMatQueue.EnQue<T>(yMatLocal);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::Compute(uint32_t dataCount, LocalTensor<uint32_t> offsetLocal)
{
    LocalTensor<T> xMatLocal = xMatQueue.DeQue<T>();
    LocalTensor<T> yMatLocal = yMatQueue.DeQue<T>();
    LocalTensor<T> outMatLocal = outMatQueue.AllocTensor<T>();

    uint32_t complexNum = dataCount / 2;
    uint32_t alignedComplexNum = (complexNum + 7) / 8 * 8;

    uint32_t maxComplexNum = (27 * 1024 / sizeof(float)) / 2;  // 27kb
    uint32_t realOffset = 0;
    uint32_t imagOffset = (maxComplexNum + 7) / 8 * 8;

    uint64_t rsvdCnt = 64;
    uint16_t repeatTimes = (alignedComplexNum * 2 + 63) / 64;

    // Rx
    GatherMask(outMatLocal[realOffset], xMatLocal, 1, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    // Ix
    GatherMask(outMatLocal[imagOffset], xMatLocal, 2, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // Ry
    GatherMask(xMatLocal[realOffset], yMatLocal, 1, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    // Iy
    GatherMask(xMatLocal[imagOffset], yMatLocal, 2, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // Rx * Ry
    Mul(yMatLocal[realOffset], xMatLocal[realOffset], outMatLocal[realOffset], alignedComplexNum);
    // Ix * Iy
    Mul(yMatLocal[imagOffset], xMatLocal[imagOffset], outMatLocal[imagOffset], alignedComplexNum);
    PipeBarrier<PIPE_V>();
    // Rx * Ry - Ix * Iy
    Sub(yMatLocal[realOffset], yMatLocal[realOffset], yMatLocal[imagOffset], alignedComplexNum);

    // Rx * Iy
    Mul(outMatLocal[realOffset], outMatLocal[realOffset], xMatLocal[imagOffset], alignedComplexNum);
    // Ix * Ry
    Mul(outMatLocal[imagOffset], outMatLocal[imagOffset], xMatLocal[realOffset], alignedComplexNum);
    PipeBarrier<PIPE_V>();
    // Rx * Iy + Ix * Ry
    Add(yMatLocal[imagOffset], outMatLocal[realOffset], outMatLocal[imagOffset], alignedComplexNum);

    PipeBarrier<PIPE_V>();
    // restore position
    Gather(outMatLocal, yMatLocal, offsetLocal, 0, imagOffset * 2);

    PipeBarrier<PIPE_V>();
    outMatQueue.EnQue<T>(outMatLocal);
    xMatQueue.FreeTensor(xMatLocal);
    yMatQueue.FreeTensor(yMatLocal);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::CopyOut(uint64_t offset, uint32_t dataCount, uint32_t isAligned)
{
    LocalTensor<T> outMatLocal = outMatQueue.DeQue<T>();
    if (isAligned) {
        DataCopy(resultGM[offset], outMatLocal, dataCount);
    } else {
        DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T) * dataCount), 0, 0};
        DataCopyPad(resultGM[offset], outMatLocal, copyParams);
    }
    outMatQueue.FreeTensor(outMatLocal);
}

}  // namespace ComplexMatDot
#endif  // COMPLEX_MAT_DOT_AIV_H
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
 * \file group_norm_swish_base.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_BASE_H
#define GROUP_NORM_SWISH_BASE_H

#include "kernel_operator.h"

namespace GroupNormSwish {
using namespace AscendC;

constexpr int32_t reduceNum = 4096;
constexpr int32_t doubleRNum = 8192;
constexpr int32_t oneBlockNum = 8;
constexpr int32_t oneBlockSize = 32;

template <typename T1, typename T2>

class GroupNormSwishBase {
public:
    __aicore__ inline GroupNormSwishBase(){};

protected:
    __aicore__ inline void InitGlobal(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                      const GroupNormSwishTilingData *tilingData, TPipe *pipeIn);
    __aicore__ inline void InitLocal(const int32_t gammaSize, const int32_t outMeanSize, const int32_t meanBufSize);
    __aicore__ inline void InitLocalB32(const int32_t gammaSize, const int32_t outMeanSize, const int32_t meanBufSize);
    __aicore__ inline void CopyInX(const int64_t xOffset, const int64_t copyNum);
    __aicore__ inline void CopyOutY(const int64_t yOffset, const int64_t copyNum);
    __aicore__ inline void CastMeanAndRstd(const int64_t copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t copyNum);
    __aicore__ inline void CopyOutMeanAndRstdWithOffset(const int64_t offset, const int64_t copyNum);
    __aicore__ inline void CopyOutWithOutPadT1(const GlobalTensor<T1> &dstGM, const LocalTensor<T1> &srcUB,
                                               const int64_t copyNum);
    __aicore__ inline void CopyOutWithOutPadT2(const GlobalTensor<T2> &dstGM, const LocalTensor<T2> &srcUB,
                                               const int64_t copyNum);
    __aicore__ inline void ComputeSwishB16(const int64_t calcNum);
    __aicore__ inline void CopyInGammaAndBeta(const int64_t gmOffset, const int64_t localOffset, const int64_t copyNum);
    __aicore__ inline void CastGammaAndBeta(const LocalTensor<float> &gammaLocal, const LocalTensor<float> &betaLocal,
                                            const int64_t offset);
    __aicore__ inline void ReduceSumCustomSmall(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                                const LocalTensor<float> &workLocal, const int32_t count);
    __aicore__ inline void ReduceSumCustom(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                           const LocalTensor<float> &workLocal, const int32_t count);
    __aicore__ inline void TwoPassSumOneLoop(const int64_t num);
    __aicore__ inline void TwoPassSumMulLoop(const int64_t loopTimesBegin, const int64_t loopTimesEnd);

protected:
    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return b == 0 ? 0 : (a + b - 1) / b;
    };
    __aicore__ inline int64_t CeilRem(int64_t a, int64_t b)
    {
        if (b == 0) {
            return a;
        }
        return a % b == 0 ? b : a % b;
    };
    __aicore__ inline int64_t GetMin(int64_t a, int64_t b)
    {
        return a > b ? b : a;
    };
    __aicore__ inline RoundMode GetRoundMode()
    {
#if __CCE_AICORE__ == 220
        return RoundMode::CAST_RINT;
#else
        return RoundMode::CAST_NONE;
#endif
    };

protected:
    TPipe *pipe;
    // Queue
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    // Tbuf
    TBuf<QuePosition::VECCALC> x1Buf32;
    TBuf<QuePosition::VECCALC> x2Buf32;
    TBuf<QuePosition::VECCALC> meanBuf32;
    TBuf<QuePosition::VECCALC> rstdBuf32;
    // Global Tensor
    GlobalTensor<T1> xGm, yGm;
    GlobalTensor<T2> gammaGm, betaGm;
    GlobalTensor<T2> meanGm, rstdGm;
    // Local Tensor
    LocalTensor<float> meanOut;
    LocalTensor<float> rstdOut;
    LocalTensor<float> x1Ub32;
    LocalTensor<float> x2Ub32;
    LocalTensor<float> meanUb;
    LocalTensor<float> rstdUb;
    // local data
    float c[2] = {0, 0};
    float sum[2] = {0, 0};
    int64_t blockIdx = 0;
    int64_t usedBlock = 0;
    uint32_t blockSize = 32;
    // TilingData
    const GroupNormSwishTilingData *tiling;
    float numRec;
    constexpr static int64_t elementsPerBlockT1 = oneBlockSize / sizeof(T1);
    constexpr static int64_t elementsPerBlockT2 = oneBlockSize / sizeof(T2);
    constexpr static float scalarOne = 1.0;
};

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::InitGlobal(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                              GM_ADDR mean, GM_ADDR rstd,
                                                              const GroupNormSwishTilingData *tilingData, TPipe *pipeIn)
{
    tiling = tilingData;
    blockIdx = GetBlockIdx();
    usedBlock = GetBlockNum();
    numRec = float(1.0) / float(tiling->numPerGroup);
    pipe = pipeIn;

    int64_t xGMOffset = blockIdx * tiling->groupPerCore * tiling->numPerGroup;
    xGm.SetGlobalBuffer((__gm__ T1 *)x + xGMOffset);
    gammaGm.SetGlobalBuffer((__gm__ T2 *)gamma);
    betaGm.SetGlobalBuffer((__gm__ T2 *)beta);
    yGm.SetGlobalBuffer((__gm__ T1 *)y + xGMOffset);
    meanGm.SetGlobalBuffer((__gm__ T2 *)mean);
    rstdGm.SetGlobalBuffer((__gm__ T2 *)rstd);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::InitLocal(const int32_t inQueSize, const int32_t outQueSize,
                                                             const int32_t bufSize)
{
    pipe->InitBuffer(inQueueGamma, 1, inQueSize * sizeof(float));
    pipe->InitBuffer(inQueueBeta, 1, inQueSize * sizeof(float));
    pipe->InitBuffer(inQueueX, 2, tiling->numPerLoop * sizeof(T1));
    pipe->InitBuffer(outQueueY, 2, tiling->numPerLoop * sizeof(T1));
    pipe->InitBuffer(outQueueMean, 1, outQueSize * sizeof(float));
    pipe->InitBuffer(outQueueRstd, 1, outQueSize * sizeof(float));

    pipe->InitBuffer(x1Buf32, tiling->numPerLoop * sizeof(float));
    pipe->InitBuffer(x2Buf32, tiling->numPerLoop * sizeof(float));
    pipe->InitBuffer(meanBuf32, bufSize * sizeof(float));
    pipe->InitBuffer(rstdBuf32, bufSize * sizeof(float));

    x1Ub32 = x1Buf32.Get<float>();
    x2Ub32 = x2Buf32.Get<float>();
    meanUb = meanBuf32.Get<float>();
    rstdUb = rstdBuf32.Get<float>();
    meanOut = outQueueMean.AllocTensor<float>();
    rstdOut = outQueueRstd.AllocTensor<float>();
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::InitLocalB32(const int32_t inQueSize, const int32_t outQueSize,
                                                                const int32_t bufSize)
{
    pipe->InitBuffer(inQueueGamma, 1, inQueSize * sizeof(float));
    pipe->InitBuffer(inQueueBeta, 1, inQueSize * sizeof(float));
    pipe->InitBuffer(inQueueX, 2, tiling->numPerLoop * sizeof(float));
    pipe->InitBuffer(outQueueY, 2, tiling->numPerLoop * sizeof(float));
    pipe->InitBuffer(outQueueMean, 1, outQueSize * sizeof(float));
    pipe->InitBuffer(outQueueRstd, 1, outQueSize * sizeof(float));

    pipe->InitBuffer(x2Buf32, tiling->numPerLoop * sizeof(float));
    pipe->InitBuffer(meanBuf32, bufSize * sizeof(float));
    pipe->InitBuffer(rstdBuf32, bufSize * sizeof(float));

    x2Ub32 = x2Buf32.Get<float>();
    meanUb = meanBuf32.Get<float>();
    rstdUb = rstdBuf32.Get<float>();
    meanOut = outQueueMean.AllocTensor<float>();
    rstdOut = outQueueRstd.AllocTensor<float>();
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyInX(const int64_t xOffset, const int64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX.AllocTensor<T1>();
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
    DataCopyPad(xUb, xGm[xOffset], {1, static_cast<uint16_t>(copyNum * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
#endif
#else
    int64_t copyNumAlign = CeilDiv(copyNum, elementsPerBlockT1) * elementsPerBlockT1;
    DataCopy(xUb, xGm[xOffset], copyNumAlign);
#endif
    inQueueX.EnQue(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyOutY(const int64_t yOffset, const int64_t copyNum)
{
    LocalTensor<T1> yUb = outQueueY.DeQue<T1>();
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
    DataCopyPad(yGm[yOffset], yUb, {1, static_cast<uint16_t>(copyNum * sizeof(T1)), 0, 0});
#endif
#else
    CopyOutWithOutPadT1(yGm[yOffset], yUb, copyNum);
#endif
    outQueueY.FreeTensor(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyOutWithOutPadT1(const GlobalTensor<T1> &dstGM,
                                                                       const LocalTensor<T1> &srcUB,
                                                                       const int64_t copyNum)
{
    if (copyNum % elementsPerBlockT1 == 0) {
        DataCopy(dstGM, srcUB, copyNum);
    } else {
        int64_t copyNumAlign = CeilDiv(copyNum, elementsPerBlockT1) * elementsPerBlockT1;
        int64_t copyNumTail = copyNumAlign - elementsPerBlockT1;
        for (int64_t i = copyNum; i < copyNumAlign; i++) {
            srcUB(i) = 0;
        }
        if (copyNum > elementsPerBlockT1) {
            DataCopy(dstGM, srcUB, copyNum);
        }
        SetAtomicAdd<T1>();
        DataCopy(dstGM[copyNumTail], srcUB[copyNumTail], elementsPerBlockT1);
        SetAtomicNone();
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyOutWithOutPadT2(const GlobalTensor<T2> &dstGM,
                                                                       const LocalTensor<T2> &srcUB,
                                                                       const int64_t copyNum)
{
    if (copyNum % elementsPerBlockT2 == 0) {
        DataCopy(dstGM, srcUB, copyNum);
    } else {
        int64_t copyNumAlign = CeilDiv(copyNum, elementsPerBlockT2) * elementsPerBlockT2;
        int64_t copyNumTail = copyNumAlign - elementsPerBlockT2;
        for (int64_t i = copyNum; i < copyNumAlign; i++) {
            srcUB(i) = 0;
        }
        if (copyNum > elementsPerBlockT2) {
            DataCopy(dstGM, srcUB, copyNum);
        }
        SetAtomicAdd<T2>();
        DataCopy(dstGM[copyNumTail], srcUB[copyNumTail], elementsPerBlockT2);
        SetAtomicNone();
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CastMeanAndRstd(const int64_t copyNum)
{
    if constexpr (std::is_same_v<T2, float>) {
        outQueueMean.EnQue(meanOut);
        outQueueRstd.EnQue(rstdOut);
    } else {
        LocalTensor<T2> meanOutT2 = meanOut.template ReinterpretCast<T2>();
        LocalTensor<T2> rstdOutT2 = rstdOut.template ReinterpretCast<T2>();
#ifndef __CCE_KT_TEST__
        Cast(meanOutT2, meanOut, RoundMode::CAST_RINT, copyNum);
#endif
        outQueueMean.EnQue(meanOutT2);
#ifndef __CCE_KT_TEST__
        Cast(rstdOutT2, rstdOut, RoundMode::CAST_RINT, copyNum);
#endif
        outQueueRstd.EnQue(rstdOutT2);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::ComputeSwishB16(const int64_t calcNum)
{
    if (tiling->activateSwish) {
        Muls(x2Ub32, x1Ub32, -tiling->swishScale, calcNum);
        pipe_barrier(PIPE_V);
        Exp(x2Ub32, x2Ub32, calcNum);
        pipe_barrier(PIPE_V);
        Adds(x2Ub32, x2Ub32, scalarOne, calcNum);
        pipe_barrier(PIPE_V);
        Div(x1Ub32, x1Ub32, x2Ub32, calcNum);
        pipe_barrier(PIPE_V);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyOutMeanAndRstd(const int64_t copyNum)
{
    CopyOutMeanAndRstdWithOffset(0, copyNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyOutMeanAndRstdWithOffset(const int64_t offset,
                                                                                const int64_t copyNum)
{
    LocalTensor<T2> meanOutT2 = outQueueMean.DeQue<T2>();
    LocalTensor<T2> rstdOutT2 = outQueueRstd.DeQue<T2>();
#if __CCE_AICORE__ == 220
    uint16_t blockCount = 1;
    uint16_t blockLen = copyNum * sizeof(T2);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
#ifndef __CCE_KT_TEST__
    DataCopyPad(meanGm[blockIdx * tiling->groupPerCore + offset], meanOutT2, dataCopyParams);
#endif
#ifndef __CCE_KT_TEST__
    DataCopyPad(rstdGm[blockIdx * tiling->groupPerCore + offset], rstdOutT2, dataCopyParams);
#endif
#else
    CopyOutWithOutPadT2(meanGm[blockIdx * tiling->groupPerCore + offset], meanOutT2, copyNum);
    CopyOutWithOutPadT2(rstdGm[blockIdx * tiling->groupPerCore + offset], rstdOutT2, copyNum);
#endif
    outQueueMean.FreeTensor(meanOutT2);
    outQueueRstd.FreeTensor(rstdOutT2);
    meanOut = outQueueMean.AllocTensor<float>();
    rstdOut = outQueueRstd.AllocTensor<float>();
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CopyInGammaAndBeta(const int64_t gmOffset, const int64_t localOffset,
                                                                      const int64_t copyNum)
{
#if __CCE_AICORE__ == 220
    if constexpr (std::is_same_v<T2, float>) {
        LocalTensor<T2> gammaLocal = inQueueGamma.AllocTensor<T2>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(gammaLocal, gammaGm[gmOffset], {1, static_cast<uint16_t>(copyNum * sizeof(T2)), 0, 0},
                    {false, 0, 0, 0});
#endif
        inQueueGamma.EnQue(gammaLocal);
        LocalTensor<T2> betaLocal = inQueueBeta.AllocTensor<T2>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(betaLocal, betaGm[gmOffset], {1, static_cast<uint16_t>(copyNum * sizeof(T2)), 0, 0},
                    {false, 0, 0, 0});
#endif
        inQueueBeta.EnQue(betaLocal);
    } else {
        LocalTensor<T2> gammaLocal = inQueueGamma.AllocTensor<T2>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(gammaLocal[localOffset], gammaGm[gmOffset], {1, static_cast<uint16_t>(copyNum * sizeof(T2)), 0, 0},
                    {false, 0, 0, 0});
#endif
        inQueueGamma.EnQue(gammaLocal);
        LocalTensor<T2> betaLocal = inQueueBeta.AllocTensor<T2>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(betaLocal[localOffset], betaGm[gmOffset], {1, static_cast<uint16_t>(copyNum * sizeof(T2)), 0, 0},
                    {false, 0, 0, 0});
#endif
        inQueueBeta.EnQue(betaLocal);
    }
#else
    int64_t copyNumAlign = CeilDiv(copyNum, elementsPerBlockT2) * elementsPerBlockT2;
    if constexpr (std::is_same_v<T2, float>) {
        LocalTensor<T2> gammaLocal = inQueueGamma.AllocTensor<T2>();
        DataCopy(gammaLocal, gammaGm[gmOffset], copyNumAlign);
        inQueueGamma.EnQue(gammaLocal);
        LocalTensor<T2> betaLocal = inQueueBeta.AllocTensor<T2>();
        DataCopy(betaLocal, betaGm[gmOffset], copyNumAlign);
        inQueueBeta.EnQue(betaLocal);
    } else {
        LocalTensor<T2> gammaLocal = inQueueGamma.AllocTensor<T2>();
        DataCopy(gammaLocal[localOffset], gammaGm[gmOffset], copyNumAlign);
        inQueueGamma.EnQue(gammaLocal);
        LocalTensor<T2> betaLocal = inQueueBeta.AllocTensor<T2>();
        DataCopy(betaLocal[localOffset], betaGm[gmOffset], copyNumAlign);
        inQueueBeta.EnQue(betaLocal);
    }
#endif
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::CastGammaAndBeta(const LocalTensor<float> &gammaLocal,
                                                                    const LocalTensor<float> &betaLocal,
                                                                    const int64_t offset)
{
    LocalTensor<T2> gammaLocalb16 = gammaLocal.template ReinterpretCast<T2>();
    LocalTensor<T2> betaLocalb16 = betaLocal.template ReinterpretCast<T2>();
#ifndef __CCE_KT_TEST__
    Cast(gammaLocal, gammaLocalb16[offset], RoundMode::CAST_NONE, offset);
#endif
#ifndef __CCE_KT_TEST__
    Cast(betaLocal, betaLocalb16[offset], RoundMode::CAST_NONE, offset);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void
GroupNormSwishBase<T1, T2>::ReduceSumCustomSmall(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                                 const LocalTensor<float> &workLocal, const int32_t count)
{
    int32_t repeat = count / 64;
    int32_t tailNum = count % 64;
    if (repeat > 0) {
        WholeReduceSum(workLocal, srcLocal, 64, repeat, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }
    if (tailNum != 0) {
        WholeReduceSum(workLocal[repeat], srcLocal[count - tailNum], tailNum, 1, 1, 1, 8);
        pipe_barrier(PIPE_V);
        WholeReduceSum(dstLocal, workLocal, repeat + 1, 1, 1, 1, 8);
    } else {
        pipe_barrier(PIPE_V);
        WholeReduceSum(dstLocal, workLocal, repeat, 1, 1, 1, 8);
    }
}

template <typename T1, typename T2>
__aicore__ inline void
GroupNormSwishBase<T1, T2>::ReduceSumCustom(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                            const LocalTensor<float> &workLocal, const int32_t count)
{
    if (count == doubleRNum) {
        BlockReduceSum(workLocal, srcLocal, 128, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        BlockReduceSum(workLocal, workLocal, 16, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        WholeReduceSum(dstLocal, workLocal, 64, 2, 1, 1, 8);
        pipe_barrier(PIPE_V);
    } else if (count <= reduceNum) {
        dstLocal(1) = 0;
        ReduceSumCustomSmall(dstLocal, srcLocal, workLocal, count);
        pipe_barrier(PIPE_V);
    } else {
        BlockReduceSum(workLocal, srcLocal, 64, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        BlockReduceSum(workLocal, workLocal, 8, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        WholeReduceSum(dstLocal, workLocal, 64, 1, 1, 1, 8);
        pipe_barrier(PIPE_V);
        ReduceSumCustomSmall(dstLocal[1], srcLocal[reduceNum], workLocal, count - reduceNum);
        pipe_barrier(PIPE_V);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::TwoPassSumOneLoop(const int64_t num)
{
    float y, t;
    float n_a, n_b;
    float mean_a, mean_b;
    float sigmaB;

    sum[0] = (meanUb(0) + meanUb(1));
    sum[1] = (rstdUb(0) + rstdUb(1)) / doubleRNum;
    for (int64_t i = 1; i < num - 1; i++) {
        n_a = float(i) / float(i + 1);
        n_b = 1 - n_a;
        mean_b = (meanUb(2 * i) + meanUb(2 * i + 1)) / doubleRNum;
        sigmaB = (rstdUb(2 * i) + rstdUb(2 * i + 1)) / doubleRNum;
        mean_a = float(sum[0]) / float(i * doubleRNum);
        sum[1] = sum[1] * n_a + n_b * sigmaB + n_a * n_b * (mean_a - mean_b) * (mean_a - mean_b);

        y = meanUb(i * 2) - c[0];
        t = sum[0] + y;
        c[0] = (t - sum[0]) - y;
        sum[0] = t;
        y = meanUb(i * 2 + 1) - c[0];
        t = sum[0] + y;
        c[0] = (t - sum[0]) - y;
        sum[0] = t;
    }
    n_a = float((num - 1) * doubleRNum) / float(tiling->numPerGroup);
    n_b = 1 - n_a;
    mean_b = (meanUb(2 * (num - 1)) + meanUb(2 * (num - 1) + 1)) / tiling->numTailLoop;
    sigmaB = (rstdUb(2 * (num - 1)) + rstdUb(2 * (num - 1) + 1)) / tiling->numTailLoop;
    mean_a = float(sum[0]) / float(tiling->numPerGroup - tiling->numTailLoop);
    sum[1] = sum[1] * n_a + n_b * sigmaB + n_a * n_b * (mean_a - mean_b) * (mean_a - mean_b);

    y = meanUb((num - 1) * 2) - c[0];
    t = sum[0] + y;
    c[0] = (t - sum[0]) - y;
    sum[0] = t;
    y = meanUb(2 * (num - 1) + 1) - c[0];
    t = sum[0] + y;
    c[0] = (t - sum[0]) - y;
    sum[0] = t;
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishBase<T1, T2>::TwoPassSumMulLoop(const int64_t loopTimesBegin,
                                                                     const int64_t loopTimesEnd)
{
    float y, t;
    float n_a, n_b;
    float mean_b, sigmaB, mean_a;
    int64_t num = loopTimesEnd - loopTimesBegin;

    for (int64_t i = 0; i < num - 1; i++) {
        n_a = float(i + loopTimesBegin) / float(i + 1 + loopTimesBegin);
        n_b = 1 - n_a;
        mean_b = meanUb(2 * i);
        sigmaB = (rstdUb(2 * i) + rstdUb(2 * i + 1)) / doubleRNum;
        mean_a = (i + loopTimesBegin) == 0 ? 0 : sum[0] / float((i + loopTimesBegin)) / doubleRNum;
        sum[1] = sum[1] * n_a + n_b * sigmaB + n_a * n_b * (mean_a - mean_b) * (mean_a - mean_b);

        y = mean_b * doubleRNum - c[0];
        t = sum[0] + y;
        c[0] = (t - sum[0]) - y;
        sum[0] = t;
    }

    if (loopTimesEnd == tiling->loopTimes) {
        n_a = float(tiling->numPerGroup - tiling->numTailLoop) / float(tiling->numPerGroup);
        mean_b = meanUb(2 * (num - 1));
        sigmaB = (rstdUb(2 * (num - 1)) + rstdUb(2 * (num - 1) + 1)) / tiling->numTailLoop;
        y = mean_b * tiling->numTailLoop - c[0];
        mean_a = tiling->numPerGroup - tiling->numTailLoop == 0 ?
                     0 :
                     sum[0] / float(tiling->numPerGroup - tiling->numTailLoop);
    } else {
        n_a = float(loopTimesEnd - 1) / float(loopTimesEnd);

        mean_b = meanUb(2 * (num - 1));
        sigmaB = (rstdUb(2 * (num - 1)) + rstdUb(2 * (num - 1) + 1)) / doubleRNum;
        y = mean_b * doubleRNum - c[0];
        mean_a = sum[0] / float((loopTimesEnd - 1) * doubleRNum);
    }
    n_b = 1 - n_a;
    sum[1] = sum[1] * n_a + n_b * sigmaB + n_a * n_b * (mean_a - mean_b) * (mean_a - mean_b);
    t = sum[0] + y;
    c[0] = (t - sum[0]) - y;
    sum[0] = t;
}

} // namespace GroupNormSwish

#endif // GROUP_NORM_SWISH_BASE_H
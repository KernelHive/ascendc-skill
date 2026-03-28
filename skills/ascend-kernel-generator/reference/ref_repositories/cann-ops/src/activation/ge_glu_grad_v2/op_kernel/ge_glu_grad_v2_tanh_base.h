/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ge_glu_grad_v2_tanh_base.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_TANH_BASE_H_
#define GE_GLU_GRAD_V2_TANH_BASE_H_

#include "kernel_operator.h"

namespace GeGluGradV2Tanh {
using namespace AscendC;

constexpr int32_t NO_DB_BUFFER = 1;
constexpr int32_t DB_BUFFER = 2;
constexpr int32_t BFP16_TEMP_BUF_CNT = 5;
constexpr int32_t FP16_TEMP_BUF_CNT = BFP16_TEMP_BUF_CNT;
constexpr int32_t FP32_TEMP_BUF_CNT = 4;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TPS_REPEAT_SIZE = 512;

// const vaiable
constexpr float NEG_ONE = -1.0;
constexpr float POS_ONE = 1.0;
constexpr float COEFFICIENT_A1 = -0.0713548162726002527220;
constexpr float COEFFICIENT_A2 = -1.5957691216057308;
constexpr float COEFFICIENT_A3 = -0.21406444881780074632901625683959062;
constexpr float COEFFICIENT_A4 = -1.5957691216057307117597842397375274738;
constexpr int64_t DUP_COUNT = 64;
constexpr uint64_t DIV_MASK = 64;

template <typename T>
class GeGluGradV2TanhBase {
public:
    __aicore__ inline GeGluGradV2TanhBase(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
                                          const GeGluGradV2TilingData* tilingDataPtr) {
        approximate = tilingDataPtr->approximate;
        activateLeft = static_cast<bool>(tilingDataPtr->activateLeft);
        maxProcCount = tilingDataPtr->maxProcCount;
        valueM = tilingDataPtr->valueM;
        needCoreNum = tilingDataPtr->needCoreNum;
        loopNumPerCore = tilingDataPtr->loopNumPerCore;
        tailCoreIndex = tilingDataPtr->tailCoreIndex;
        tailUbLoopNum = tilingDataPtr->tailUbLoopNum;
        groupNum = tilingDataPtr->groupNum;

        dyGm.SetGlobalBuffer((__gm__ T*)dy);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        geluGm.SetGlobalBuffer((__gm__ T*)gelu);
        dxGm.SetGlobalBuffer((__gm__ T*)dx);
        BaseInit();
    };

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a + b - 1) / b);
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlignA2B(T1 a, T2 b) {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : CeilDiv(a, b) * b);
    };

    template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
              void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
    __aicore__ inline void ProcessLessEqual(CLS_NAME* objPtr);
    template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
              void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
    __aicore__ inline void ProcessGreater(CLS_NAME* objPtr);
    template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
              void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
    __aicore__ inline void ProcessPerf(CLS_NAME* objPtr);

    __aicore__ inline void ComputeGeluGrad(LocalTensor<float>& y, LocalTensor<float>& dy, LocalTensor<float>& x,
                                           const int64_t& realProcCount);
    __aicore__ inline void CopyInDyAndGelu(const int64_t& gmOffset, const int64_t& dataCount,
                                           const int64_t& blockCount);
    __aicore__ inline void CopyInX(const int64_t& gmOffset, const int64_t& dataCount, const int64_t& blockCount);

    __aicore__ inline void CopyInXPerf(const int64_t& gmOffset, const int64_t& dataCount);
    template <typename T2>
    __aicore__ inline void SplitXLeftAndRight(LocalTensor<T2> dst1, LocalTensor<T2> dst2, LocalTensor<T2> src,
                                              const int64_t& nBatch);
    template <typename T2>
    __aicore__ inline void TransposeX(LocalTensor<T2>& dst, LocalTensor<T2>& src, const int64_t& nBatch);
    template <typename T2>
    __aicore__ inline void CopySplitTensor(LocalTensor<T2>& dst1, LocalTensor<T2>& dst2, LocalTensor<T2>& src,
                                           const int64_t& nBatch);
    template <typename T2>
    __aicore__ inline void TransposeXBack(LocalTensor<T2>& dst1, LocalTensor<T2>& dst2, LocalTensor<T2>& src1,
                                          LocalTensor<T2>& src2, const int64_t& nBatch);

    __aicore__ inline void CopyOutLeft(const int64_t& gmOffset, const int64_t& dataCount, const int64_t& blockCount);
    __aicore__ inline void CopyOutRight(const int64_t& gmOffset, const int64_t& dataCount, const int64_t& blockCount);

    __aicore__ inline void CopyOutDXPerf(const int64_t& gmOffset, const int64_t& dataCount);
    template <typename T2>
    __aicore__ inline void ConcatDXLeftAndRight(LocalTensor<T2> dst, LocalTensor<T2> src1, LocalTensor<T2> src2,
                                                const int64_t& nBatch);
    template <typename T2>
    __aicore__ inline void TransposeDX(LocalTensor<T2>& dst1, LocalTensor<T2>& dst2, LocalTensor<T2>& src1,
                                       LocalTensor<T2>& src2, const int64_t& nBatch);
    template <typename T2>
    __aicore__ inline void CopyConcatTensor(LocalTensor<T2>& dst, LocalTensor<T2>& src1, LocalTensor<T2>& src2,
                                            const int64_t& nBatch);
    template <typename T2>
    __aicore__ inline void TransposeDXBack(LocalTensor<T2>& dst, LocalTensor<T2>& src, const int64_t& nBatch);

    template <typename T2>
    __aicore__ inline LocalTensor<T2> GetTempBuf(const int32_t index);

private:
    __aicore__ inline void BaseInit();

protected:
    TPipe pipe;
    int32_t blockIdx = 0;

    GlobalTensor<T> dyGm, xGm, geluGm;
    GlobalTensor<T> dxGm;

    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueX1;
#if defined(ORIG_DTYPE_DY) && ORIG_DTYPE_DY != DT_FLOAT
    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueX2;
#else
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX2;
#endif
    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueDY;
    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueGelu;
    TQue<QuePosition::VECOUT, NO_DB_BUFFER> outQueueDX1;
    TQue<QuePosition::VECOUT, NO_DB_BUFFER> outQueueDX2;

    TBuf<QuePosition::VECCALC> resultTempBuf;

    int32_t perBlockCount = 0;
    int32_t dtypeSize = 0;
    int32_t tpsWidth = 0;

    // tilingParams
    int32_t approximate = 1;
    bool activateLeft = false;
    int64_t maxProcCount = 0;
    int64_t valueM = 0;
    int64_t needCoreNum = 0;
    int64_t loopNumPerCore = 0;
    int64_t tailCoreIndex = 0;
    int64_t tailUbLoopNum = 0;
    int64_t groupNum = 0;
};

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::BaseInit() {
    blockIdx = GetBlockIdx();
    dtypeSize = sizeof(T);
    perBlockCount = BLOCK_SIZE / dtypeSize;
}

template <typename T>
template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
          void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
__aicore__ inline void GeGluGradV2TanhBase<T>::ProcessLessEqual(CLS_NAME* objPtr) {
    int64_t loopNum = loopNumPerCore;
    if (blockIdx < tailCoreIndex) {
        loopNum += 1;
    }
    int64_t idx = 0;
    for (; idx < loopNum; idx++) {
        int64_t tempOffset = (needCoreNum * idx + blockIdx) * groupNum * valueM;
        int64_t realProcCount = CeilAlignA2B(valueM, perBlockCount) * groupNum;
        CopyInDyAndGelu(tempOffset, valueM, groupNum);
        CopyInX(2 * tempOffset, valueM, groupNum);
        (objPtr->*funComputeLeftHalf)(realProcCount);
        CopyOutLeft(2 * tempOffset, valueM, groupNum);
        (objPtr->*funComputeRightHalf)(realProcCount);
        CopyOutRight(2 * tempOffset, valueM, groupNum);
    }
    if (blockIdx == tailCoreIndex && tailUbLoopNum > 0) {
        int64_t tempOffset = (needCoreNum * idx + blockIdx) * groupNum * valueM;
        int64_t realProcCount = CeilAlignA2B(valueM, perBlockCount) * tailUbLoopNum;
        CopyInDyAndGelu(tempOffset, valueM, tailUbLoopNum);
        CopyInX(2 * tempOffset, valueM, tailUbLoopNum);
        (objPtr->*funComputeLeftHalf)(realProcCount);
        CopyOutLeft(2 * tempOffset, valueM, tailUbLoopNum);
        (objPtr->*funComputeRightHalf)(realProcCount);
        CopyOutRight(2 * tempOffset, valueM, tailUbLoopNum);
    }
}

template <typename T>
template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
          void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
__aicore__ inline void GeGluGradV2TanhBase<T>::ProcessGreater(CLS_NAME* objPtr) {
    int64_t loopNum = loopNumPerCore;
    if (blockIdx < tailCoreIndex) {
        loopNum += 1;
    }
    int64_t modCount = valueM % maxProcCount;
    modCount = modCount ? modCount : maxProcCount;
    for (int64_t idx = 0; idx < loopNum; idx++) {
        int64_t mIndex = (needCoreNum * idx + blockIdx) / groupNum;
        int64_t mIndexSub = (needCoreNum * idx + blockIdx) % groupNum;
        int64_t tempOffset = mIndex * valueM + mIndexSub * maxProcCount;
        int64_t tempXOffset = 2 * mIndex * valueM + mIndexSub * maxProcCount;
        if (mIndexSub + 1 == groupNum) {
            CopyInDyAndGelu(tempOffset, modCount, 1);
            CopyInX(tempXOffset, modCount, 1);
            (objPtr->*funComputeLeftHalf)(CeilAlignA2B(modCount, perBlockCount));
            CopyOutLeft(tempXOffset, modCount, 1);
            (objPtr->*funComputeRightHalf)(CeilAlignA2B(modCount, perBlockCount));
            CopyOutRight(tempXOffset, modCount, 1);
        } else {
            CopyInDyAndGelu(tempOffset, maxProcCount, 1);
            CopyInX(tempXOffset, maxProcCount, 1);
            (objPtr->*funComputeLeftHalf)(maxProcCount);
            CopyOutLeft(tempXOffset, maxProcCount, 1);
            (objPtr->*funComputeRightHalf)(maxProcCount);
            CopyOutRight(tempXOffset, maxProcCount, 1);
        }
    }
}

template <typename T>
template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
          void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
__aicore__ inline void GeGluGradV2TanhBase<T>::ProcessPerf(CLS_NAME* objPtr) {
    int64_t loopNum = loopNumPerCore;
    if (blockIdx < tailCoreIndex) {
        loopNum += 1;
    }
    int64_t idx = 0;
    for (; idx < loopNum; idx++) {
        int64_t tempOffset = (needCoreNum * idx + blockIdx) * groupNum * valueM;
        int64_t realProcCount = CeilAlignA2B(groupNum * valueM, perBlockCount);
        CopyInDyAndGelu(tempOffset, valueM * groupNum, 1);
        CopyInXPerf(2 * tempOffset, 2 * valueM * groupNum);
        (objPtr->*funComputeLeftHalf)(realProcCount);
        (objPtr->*funComputeRightHalf)(realProcCount);
        CopyOutDXPerf(2 * tempOffset, 2 * valueM * groupNum);
    }
    if (blockIdx == tailCoreIndex && tailUbLoopNum > 0) {
        int64_t tempOffset = (needCoreNum * idx + blockIdx) * groupNum * valueM;
        int64_t realProcCount = CeilAlignA2B(tailUbLoopNum * valueM, perBlockCount);
        CopyInDyAndGelu(tempOffset, valueM * tailUbLoopNum, 1);
        CopyInXPerf(2 * tempOffset, 2 * valueM * tailUbLoopNum);
        (objPtr->*funComputeLeftHalf)(realProcCount);
        (objPtr->*funComputeRightHalf)(realProcCount);
        CopyOutDXPerf(2 * tempOffset, 2 * valueM * tailUbLoopNum);
    }
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::ComputeGeluGrad(LocalTensor<float>& y, LocalTensor<float>& dy,
                                                               LocalTensor<float>& x, const int64_t& realProcCount) {
    LocalTensor<float> g1 = GetTempBuf<float>(1);
    LocalTensor<float> g2 = GetTempBuf<float>(2);
    LocalTensor<float> t5 = GetTempBuf<float>(3);

    // compute g1 = 1.0 / (exp(x * (x^2 * a1 + a2)) + 1)
    Mul(g2, x, x, realProcCount);
    Muls(g1, g2, COEFFICIENT_A1, realProcCount);
    Adds(g1, g1, COEFFICIENT_A2, realProcCount);
    Mul(g1, g1, x, realProcCount);
    Exp(g1, g1, realProcCount);
    Adds(g1, g1, POS_ONE, realProcCount);
    Duplicate(t5, POS_ONE, realProcCount);
    Div(g1, t5, g1, realProcCount);

    // compute g2 = x^2 * a3 + a4
    Muls(g2, g2, COEFFICIENT_A3, realProcCount);
    Adds(g2, g2, COEFFICIENT_A4, realProcCount);

    // compute (x * (g1 - 1) * g2 + 1) * g1 * dy
    Adds(t5, g1, NEG_ONE, realProcCount);
    Mul(t5, t5, x, realProcCount);
    Mul(t5, t5, g2, realProcCount);
    Adds(t5, t5, POS_ONE, realProcCount);
    Mul(t5, t5, g1, realProcCount);
    Mul(y, t5, dy, realProcCount);
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyInDyAndGelu(const int64_t& gmOffset, const int64_t& dataCount,
                                                               const int64_t& blockCount) {
    int64_t ubOffset = 0;
#if defined(ORIG_DTYPE_DY) && ORIG_DTYPE_DY == DT_BF16
    LocalTensor<T> ubGelu = inQueueGelu.AllocTensor<float>().ReinterpretCast<T>();
    LocalTensor<T> ubDY = inQueueDY.AllocTensor<float>().ReinterpretCast<T>();
    ubOffset = maxProcCount;
#else
    LocalTensor<T> ubGelu = inQueueGelu.AllocTensor<T>();
    LocalTensor<T> ubDY = inQueueDY.AllocTensor<T>();
#endif

    struct DataCopyParams copyInParams(blockCount, 0, 0, 0);
    if (dataCount % perBlockCount == 0) {
        copyInParams.blockLen = dataCount / perBlockCount;
        DataCopy(ubGelu[ubOffset], geluGm[gmOffset], copyInParams);
        DataCopy(ubDY[ubOffset], dyGm[gmOffset], copyInParams);
    } else {
        copyInParams.blockLen = dataCount * dtypeSize;
        struct DataCopyPadParams padParams = {true, 0, 0, 1};
        padParams.rightPadding = CeilAlignA2B(dataCount, perBlockCount) - dataCount;
        DataCopyPad(ubGelu[ubOffset], geluGm[gmOffset], copyInParams, padParams);
        DataCopyPad(ubDY[ubOffset], dyGm[gmOffset], copyInParams, padParams);
    }
    inQueueGelu.EnQue(ubGelu);
    inQueueDY.EnQue(ubDY);
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyInX(const int64_t& gmOffset, const int64_t& dataCount,
                                                       const int64_t& blockCount) {
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
    struct DataCopyParams copyInParams(blockCount, 0, 0, 0);
    int64_t x1GmOffset = activateLeft ? gmOffset + valueM : gmOffset;
    int64_t x2GmOffset = activateLeft ? gmOffset : gmOffset + valueM;
    if (dataCount % perBlockCount == 0) {
        copyInParams.blockLen = dataCount / perBlockCount;
        copyInParams.srcStride = copyInParams.blockLen;
        DataCopy(ubX1, xGm[x1GmOffset], copyInParams);
        DataCopy(ubX2, xGm[x2GmOffset], copyInParams);
    } else {
        copyInParams.blockLen = dataCount * dtypeSize;
        copyInParams.srcStride = copyInParams.blockLen;
        struct DataCopyPadParams padParams = {true, 0, 0, 1};
        padParams.rightPadding = CeilAlignA2B(dataCount, perBlockCount) - dataCount;
        DataCopyPad(ubX1, xGm[x1GmOffset], copyInParams, padParams);
        DataCopyPad(ubX2, xGm[x2GmOffset], copyInParams, padParams);
    }

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyInXPerf(const int64_t& gmOffset, const int64_t& dataCount) {
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
    LocalTensor<T> t0 = GetTempBuf<T>(0);
    int32_t nBatch = CeilDiv(dataCount / (2 * valueM), TPS_REPEAT_SIZE / dtypeSize);
    struct DataCopyParams copyInParams(1, 0, 0, 0);
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    set_flag(PIPE_MTE3, PIPE_MTE2, eventID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, eventID1);
    if (dataCount % perBlockCount == 0) {
        copyInParams.blockLen = dataCount / perBlockCount;
        DataCopy(t0, xGm[gmOffset], copyInParams);
    } else {
        copyInParams.blockLen = dataCount * dtypeSize;
        struct DataCopyPadParams padParams = {true, 0, 0, 1};
        padParams.rightPadding = CeilAlignA2B(dataCount, perBlockCount) - dataCount;
        DataCopyPad(t0, xGm[gmOffset], copyInParams, padParams);
    }
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventID2);
    wait_flag(PIPE_MTE2, PIPE_V, eventID2);

#if defined(ORIG_DTYPE_DY) && ORIG_DTYPE_DY != DT_FLOAT
    SplitXLeftAndRight(ubX1.template ReinterpretCast<half>(), ubX2.template ReinterpretCast<half>(),
                       t0.template ReinterpretCast<half>(), nBatch);
#else
    SplitXLeftAndRight(ubX1.template ReinterpretCast<float>(), ubX2.template ReinterpretCast<float>(),
                       t0.template ReinterpretCast<float>(), nBatch);
#endif

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::SplitXLeftAndRight(LocalTensor<T2> dst1, LocalTensor<T2> dst2,
                                                                  LocalTensor<T2> src, const int64_t& nBatch) {
    LocalTensor<T2> t0 = src;
    LocalTensor<T2> t1 = t0[maxProcCount];
    LocalTensor<T2> t2 = GetTempBuf<T2>(2);

    tpsWidth = BLOCK_SIZE / sizeof(T2);
    TransposeX(t2, t0, nBatch);
    CopySplitTensor(t0, t1, t2, nBatch);
    TransposeXBack(dst1, dst2, t0, t1, nBatch);
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::TransposeX(LocalTensor<T2>& dst, LocalTensor<T2>& src,
                                                          const int64_t& nBatch) {
    __ubuf__ T2* srcAddr = (__ubuf__ T2*)src.GetPhyAddr();
    __ubuf__ T2* dstAddr = (__ubuf__ T2*)dst.GetPhyAddr();
    int64_t coefficient = tpsWidth * nBatch * 2 * valueM;
    __ubuf__ T2* srcLocalList[16];
    for (int32_t i = 0; i < 16; i++) {
        srcLocalList[i] = srcAddr + coefficient * i;
    }
    __ubuf__ T2* dstLocalList[16];
    for (int32_t i = 0; i < 16; i++) {
        dstLocalList[i] = dstAddr + tpsWidth * i;
    }

    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = nBatch * 2 * valueM;
    transDataParams.srcRepStride = 1;
    transDataParams.dstRepStride = 16;

    TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopySplitTensor(LocalTensor<T2>& dst1, LocalTensor<T2>& dst2,
                                                               LocalTensor<T2>& src, const int64_t& nBatch) {
    struct DataCopyParams copyParams(tpsWidth * nBatch, 0, 0, 0);
    copyParams.blockLen = 16 * valueM / tpsWidth;
    copyParams.srcStride = copyParams.blockLen;
    int64_t x1SrcOffset = activateLeft ? 16 * valueM : 0;
    int64_t x2SrcOffset = activateLeft ? 0 : 16 * valueM;

    PipeBarrier<PIPE_V>();
    DataCopy(dst1, src[x1SrcOffset], copyParams);
    DataCopy(dst2, src[x2SrcOffset], copyParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::TransposeXBack(LocalTensor<T2>& dst1, LocalTensor<T2>& dst2,
                                                              LocalTensor<T2>& src1, LocalTensor<T2>& src2,
                                                              const int64_t& nBatch) {
    __ubuf__ T2* src1Addr = (__ubuf__ T2*)src1.GetPhyAddr();
    __ubuf__ T2* src2Addr = (__ubuf__ T2*)src2.GetPhyAddr();
    __ubuf__ T2* dst1Addr = (__ubuf__ T2*)dst1.GetPhyAddr();
    __ubuf__ T2* dst2Addr = (__ubuf__ T2*)dst2.GetPhyAddr();
    __ubuf__ T2 *srcList1[16], *srcList2[16];
    __ubuf__ T2 *dstList1[16], *dstList2[16];
    int64_t coefficient1 = tpsWidth * nBatch * valueM;
    if (tpsWidth == 8) {
        for (int32_t i = 0; i < 8; i++) {
            srcList1[i] = src1Addr + 16 * i;
            srcList1[i + 8] = src1Addr + 16 * i + 8;
            srcList2[i] = src2Addr + 16 * i;
            srcList2[i + 8] = src2Addr + 16 * i + 8;
        }
        int64_t coefficient2 = 64 * nBatch * valueM;
        for (int32_t i = 0; i < 8; i++) {
            dstList1[2 * i] = dst1Addr + coefficient1 * i;
            dstList1[2 * i + 1] = dst1Addr + coefficient1 * i + coefficient2;
            dstList2[2 * i] = dst2Addr + coefficient1 * i;
            dstList2[2 * i + 1] = dst2Addr + coefficient1 * i + coefficient2;
        }
    } else {
        for (int32_t i = 0; i < 16; i++) {
            srcList1[i] = src1Addr + 16 * i;
            srcList2[i] = src2Addr + 16 * i;
        }
        for (int32_t i = 0; i < 16; i++) {
            dstList1[i] = dst1Addr + coefficient1 * i;
            dstList2[i] = dst2Addr + coefficient1 * i;
        }
    }

    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = nBatch * valueM;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : 16;
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;
    TransDataTo5HDImpl(dstList1, srcList1, transDataParams);
    TransDataTo5HDImpl(dstList2, srcList2, transDataParams);
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyOutLeft(const int64_t& gmOffset, const int64_t& dataCount,
                                                           const int64_t& blockCount) {
    LocalTensor<T> outLocalLeft = outQueueDX1.DeQue<T>();
    struct DataCopyParams copyOutParams(blockCount, 0, 0, 0);
    int64_t x1GmOffset = activateLeft ? gmOffset + valueM : gmOffset;
    if (dataCount % perBlockCount == 0) {
        copyOutParams.blockLen = dataCount / perBlockCount;
        copyOutParams.dstStride = copyOutParams.blockLen;
        DataCopy(dxGm[x1GmOffset], outLocalLeft, copyOutParams);
    } else {
        copyOutParams.blockLen = dataCount * dtypeSize;
        copyOutParams.dstStride = copyOutParams.blockLen;
        DataCopyPad(dxGm[x1GmOffset], outLocalLeft, copyOutParams);
    }
    outQueueDX1.FreeTensor(outLocalLeft);
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyOutRight(const int64_t& gmOffset, const int64_t& dataCount,
                                                            const int64_t& blockCount) {
    LocalTensor<T> outLocalRight = outQueueDX2.DeQue<T>();
    struct DataCopyParams copyOutParams(blockCount, 0, 0, 0);
    int64_t x2GmOffset = activateLeft ? gmOffset : gmOffset + valueM;
    if (dataCount % perBlockCount == 0) {
        copyOutParams.blockLen = dataCount / perBlockCount;
        copyOutParams.dstStride = copyOutParams.blockLen;
        DataCopy(dxGm[x2GmOffset], outLocalRight, copyOutParams);
    } else {
        copyOutParams.blockLen = dataCount * dtypeSize;
        copyOutParams.dstStride = copyOutParams.blockLen;
        DataCopyPad(dxGm[x2GmOffset], outLocalRight, copyOutParams);
    }
    outQueueDX2.FreeTensor(outLocalRight);
}

template <typename T>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyOutDXPerf(const int64_t& gmOffset, const int64_t& dataCount) {
    LocalTensor<T> ubDX1 = outQueueDX1.DeQue<T>();
    LocalTensor<T> ubDX2 = outQueueDX2.DeQue<T>();
    LocalTensor<T> t0 = GetTempBuf<T>(0);
    int32_t nBatch = CeilDiv(dataCount / (2 * valueM), TPS_REPEAT_SIZE / dtypeSize);

#if defined(ORIG_DTYPE_DY) && ORIG_DTYPE_DY != DT_FLOAT
    ConcatDXLeftAndRight(t0.template ReinterpretCast<half>(), ubDX1.template ReinterpretCast<half>(),
                         ubDX2.template ReinterpretCast<half>(), nBatch);
#else
    ConcatDXLeftAndRight(t0.template ReinterpretCast<float>(), ubDX1.template ReinterpretCast<float>(),
                         ubDX2.template ReinterpretCast<float>(), nBatch);
#endif

    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventID1);
    wait_flag(PIPE_V, PIPE_MTE3, eventID1);
    struct DataCopyParams copyOutParams(1, 0, 0, 0);
    if (dataCount % perBlockCount == 0) {
        copyOutParams.blockLen = dataCount / perBlockCount;
        DataCopy(dxGm[gmOffset], t0, copyOutParams);
    } else {
        copyOutParams.blockLen = dataCount * dtypeSize;
        DataCopyPad(dxGm[gmOffset], t0, copyOutParams);
    }

    outQueueDX1.FreeTensor(ubDX1);
    outQueueDX2.FreeTensor(ubDX2);
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::ConcatDXLeftAndRight(LocalTensor<T2> dst, LocalTensor<T2> src1,
                                                                    LocalTensor<T2> src2, const int64_t& nBatch) {
    LocalTensor<T2> t0 = dst;
    LocalTensor<T2> t1 = t0[maxProcCount];
    LocalTensor<T2> t2 = GetTempBuf<T2>(2);

    TransposeDX(t0, t1, src1, src2, nBatch);
    CopyConcatTensor(t2, t0, t1, nBatch);
    TransposeDXBack(t0, t2, nBatch);
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::TransposeDX(LocalTensor<T2>& dst1, LocalTensor<T2>& dst2,
                                                           LocalTensor<T2>& src1, LocalTensor<T2>& src2,
                                                           const int64_t& nBatch) {
    __ubuf__ T2* src1Addr = (__ubuf__ T2*)src1.GetPhyAddr();
    __ubuf__ T2* src2Addr = (__ubuf__ T2*)src2.GetPhyAddr();
    __ubuf__ T2* dst1Addr = (__ubuf__ T2*)dst1.GetPhyAddr();
    __ubuf__ T2* dst2Addr = (__ubuf__ T2*)dst2.GetPhyAddr();
    __ubuf__ T2 *srcList1[16], *srcList2[16];
    int64_t coefficient = tpsWidth * nBatch * valueM;
    for (int32_t i = 0; i < 16; i++) {
        srcList1[i] = src1Addr + coefficient * i;
        srcList2[i] = src2Addr + coefficient * i;
    }
    __ubuf__ T2 *dstList1[16], *dstList2[16];
    for (int32_t i = 0; i < 16; i++) {
        dstList1[i] = dst1Addr + tpsWidth * i;
        dstList2[i] = dst2Addr + tpsWidth * i;
    }

    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = nBatch * valueM;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : 16;

    TransDataTo5HDImpl(dstList1, srcList1, transDataParams);
    TransDataTo5HDImpl(dstList2, srcList2, transDataParams);
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::CopyConcatTensor(LocalTensor<T2>& dst, LocalTensor<T2>& src1,
                                                                LocalTensor<T2>& src2, const int64_t& nBatch) {
    struct DataCopyParams copyParams(tpsWidth * nBatch, 0, 0, 0);
    copyParams.blockLen = 16 * valueM / tpsWidth;
    copyParams.dstStride = copyParams.blockLen;
    int64_t x1DstOffset = activateLeft ? 16 * valueM : 0;
    int64_t x2DstOffset = activateLeft ? 0 : 16 * valueM;

    PipeBarrier<PIPE_V>();
    DataCopy(dst[x1DstOffset], src1, copyParams);
    DataCopy(dst[x2DstOffset], src2, copyParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
template <typename T2>
__aicore__ inline void GeGluGradV2TanhBase<T>::TransposeDXBack(LocalTensor<T2>& dst, LocalTensor<T2>& src,
                                                               const int64_t& nBatch) {
    __ubuf__ T2* srcAddr = (__ubuf__ T2*)src.GetPhyAddr();
    __ubuf__ T2* dstAddr = (__ubuf__ T2*)dst.GetPhyAddr();
    __ubuf__ T2 *srcList[16], *dstList[16];
    int64_t coefficient1 = tpsWidth * nBatch * 2 * valueM;
    if (tpsWidth == 8) {
        for (int32_t i = 0; i < 8; i++) {
            srcList[i] = srcAddr + 16 * i;
            srcList[i + 8] = srcAddr + 16 * i + 8;
        }
        int64_t coefficient2 = 64 * nBatch * 2 * valueM;
        for (int32_t i = 0; i < 8; i++) {
            dstList[2 * i] = dstAddr + coefficient1 * i;
            dstList[2 * i + 1] = dstAddr + coefficient1 * i + coefficient2;
        }
    } else {
        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = srcAddr + 16 * i;
        }
        for (int32_t i = 0; i < 16; i++) {
            dstList[i] = dstAddr + coefficient1 * i;
        }
    }

    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = nBatch * 2 * valueM;
    transDataParams.srcRepStride = 16;
    transDataParams.dstRepStride = 1;

    TransDataTo5HDImpl(dstList, srcList, transDataParams);
}

template <typename T>
template <typename T2>
__aicore__ inline LocalTensor<T2> GeGluGradV2TanhBase<T>::GetTempBuf(const int32_t index) {
    return resultTempBuf.Get<float>()[maxProcCount * index].ReinterpretCast<T2>();
}

}  // namespace GeGluGradV2Tanh

#endif  // GE_GLU_GRAD_V2_TANH_BASE_H_

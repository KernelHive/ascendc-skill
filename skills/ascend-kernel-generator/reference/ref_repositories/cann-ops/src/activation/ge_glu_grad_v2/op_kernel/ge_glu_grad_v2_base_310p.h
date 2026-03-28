/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file ge_glu_grad_v2_base_310p.h
 * \brief
 */

#ifndef GE_GLU_GRAD_V2_BASE_310P_H_
#define GE_GLU_GRAD_V2_BASE_310P_H_

#include <functional>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GeGluGradV2For310P {
using namespace AscendC;

constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

constexpr int32_t NO_DB_BUFFER = 1;
constexpr int32_t BLOCK_SIZE = 32;

// const vaiable
constexpr float NEG_ONE = -1.0;
constexpr float POS_ONE = 1.0;
constexpr float SCALER_TWO = 2.0;
constexpr float SCALER_HALF = 0.5;
constexpr float COEFFICIENT_1 = 0.044715;
constexpr float COEFFICIENT_2 = 0.7978846;     // equals np.sqrt(2 / np.pi)
constexpr float COEFFICIENT_3 = 0.0535161122;  // equals 0.5 * np.sqrt(2 / np.pi) * 3 * COEFFICIENT_1
constexpr float COEFFICIENT_4 = 0.3989422804;  // equals 0.5 * np.sqrt(2 / np.pi)
constexpr float MAX_TANH = 8.8;
constexpr float MIN_TANH = -8.8;

template <typename T>
class GeGluGradV2Base310p {
public:
    __aicore__ inline GeGluGradV2Base310p(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
                                          GM_ADDR workspace, const GeGluGradV2TilingData* tilingDataPtr) {
        approximate = tilingDataPtr->approximate;
        activateLeft = static_cast<bool>(tilingDataPtr->activateLeft);
        maxProcCount = tilingDataPtr->maxProcCount;
        valueM = tilingDataPtr->valueM;
        needCoreNum = tilingDataPtr->needCoreNum;
        loopNumPerCore = tilingDataPtr->loopNumPerCore;
        tailCoreIndex = tilingDataPtr->tailCoreIndex;
        tailUbLoopNum = tilingDataPtr->tailUbLoopNum;
        groupNum = tilingDataPtr->groupNum;
        valueN = tilingDataPtr->valueN;

        dyGm.SetGlobalBuffer((__gm__ T*)dy);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        geluGm.SetGlobalBuffer((__gm__ T*)gelu);
        dxGm.SetGlobalBuffer((__gm__ T*)dx);
        workspace_ = workspace;
        output_ = dx;
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
    __aicore__ inline void ComputeGeluGrad(LocalTensor<float>& y, LocalTensor<float>& x1, LocalTensor<float>& x2,
                                           const int64_t& realProcCount);
    __aicore__ inline void ComputeTanh(LocalTensor<float>& x, const int64_t& realProcCount);
    __aicore__ inline void CopyInDyAndGelu(const int64_t& gmOffset, const int64_t& dataCount,
                                           const int64_t& blockCount);
    __aicore__ inline void CopyInX(const int64_t& gmOffset, const int64_t& dataCount, const int64_t& blockCount);
    __aicore__ inline void CopyOutLeft(const int64_t& gmOffset, const int64_t& dataCount, const int64_t& blockCount);
    __aicore__ inline void CopyOutRight(const int64_t& gmOffset, const int64_t& dataCount, const int64_t& blockCount);
    __aicore__ inline void CustomDuplicate(LocalTensor<T> &outLocal, const uint64_t addr, uint64_t mask[2], const uint64_t repeatTimes, const uint64_t repeatStride);
    template <typename U>
    __aicore__ inline void MemSetZero(GlobalTensor<U> gmTensor, int64_t size);
    __aicore__ inline void SyncAllCore();

    __aicore__ inline void BaseInit();

protected:
    TPipe pipe;
    int32_t blockIdx = 0;

    GlobalTensor<T> dyGm, xGm, geluGm;
    GlobalTensor<T> dxGm;
    GlobalTensor<int32_t> syncGlobal_;
    LocalTensor<float> localUbBuf5;

    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueX1;
    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueX2;
    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueDY;
    TQue<QuePosition::VECIN, NO_DB_BUFFER> inQueueGelu;
    TQue<QuePosition::VECOUT, NO_DB_BUFFER> outQueueDX1;
    TQue<QuePosition::VECOUT, NO_DB_BUFFER> outQueueDX2;
    TQue<QuePosition::VECIN, 1> syncWorkQueue_;

    TBuf<QuePosition::VECCALC> resultTempBuf1;
    TBuf<QuePosition::VECCALC> resultTempBuf2;
    TBuf<QuePosition::VECCALC> resultTempBuf3;
    TBuf<QuePosition::VECCALC> resultTempBuf4;

    int32_t perBlockCount = 0;
    GM_ADDR workspace_;
    GM_ADDR output_;

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
    int64_t valueN = 0;
};

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::BaseInit() {
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_SIZE / sizeof(T);

    // set workspace as 0, each core handle workspace 32bytes
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t*)workspace_, GetBlockNum() * DEFAULT_SYNCALL_NEED_SIZE);
    MemSetZero<int32_t>(syncGlobal_, GetBlockNum() * DEFAULT_SYNCALL_NEED_SIZE);
    // set workspace for sync
    pipe.InitBuffer(syncWorkQueue_, 1, GetBlockNum() * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));

    uint64_t cleanNum = (2 * valueM * valueN + needCoreNum - 1) / needCoreNum;
    uint64_t tailNum = 2 * valueM * valueN -  cleanNum * (needCoreNum - 1);
    GlobalTensor<T> gmOutput_;
    gmOutput_.SetGlobalBuffer((__gm__ T*)output_ + (GetBlockIdx() * cleanNum));

    if (blockIdx < needCoreNum){
        if (blockIdx == needCoreNum - 1) {
            cleanNum = tailNum;
        }
        MemSetZero<T>(gmOutput_, cleanNum);
    }

    SyncAllCore();
}

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::SyncAllCore()
{
    // multi-core sync
    LocalTensor<int32_t> workLocal = syncWorkQueue_.AllocTensor<int32_t>();
    SyncAll(syncGlobal_, workLocal);
    syncWorkQueue_.FreeTensor(workLocal);
}


template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::CustomDuplicate(LocalTensor<T> &outLocal, const uint64_t addr, uint64_t mask[2], const uint64_t repeatTimes, const uint64_t repeatStride){
    T inputVal(0.0);
    if (repeatTimes <= 255 && repeatStride <= 255) {
        Duplicate(outLocal[addr], inputVal, mask, (uint8_t)repeatTimes, 1, (uint8_t)repeatStride);
    } else if (repeatStride <= 255) {
        uint8_t curRepatTimes = 255;
        int64_t duplicateLoop = (repeatTimes + 255 - 1) / 255;
        uint8_t tail = (repeatTimes % 255 == 0) ? 255 : repeatTimes % 255;
        for (uint64_t i = 0; i < duplicateLoop; ++i){
            if (i == duplicateLoop - 1){
                curRepatTimes = tail;
            }
            Duplicate(outLocal[addr + 32/sizeof(T) * repeatStride * 255 * i],
                    inputVal, mask, curRepatTimes, 1, repeatStride);
        }
        
    } else if (repeatTimes <= 255){
        for (int64_t i = 0; i < repeatTimes; ++i){
             Duplicate(outLocal[addr + 32/sizeof(T) * repeatStride * i],
                inputVal, mask, 1, 1, 0);
        }
    }
}

template <typename T>
template <typename U>
__aicore__ inline void GeGluGradV2Base310p<T>::MemSetZero(GlobalTensor<U> gmTensor, int64_t size)
{
    if (g_coreType == AIC) {
        return;
    }
    int64_t int16Size = (size * sizeof(U) + sizeof(int16_t) - 1) / sizeof(int16_t);
    LocalTensor<int16_t> popBuffer;
    bool ret = PopStackBuffer<int16_t, TPosition::LCM>(popBuffer);
    uint32_t maxBurstSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(int16_t);
    uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    uint32_t round = int16Size / popSize;
    uint32_t tail = int16Size % popSize;
    uint32_t roundSize = round != 0 ? popSize : 0;
    DuplicateImpl<int16_t>((__ubuf__ int16_t *)popBuffer.GetPhyAddr(), 0, popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    uint32_t comOffset = 0;
    // compute the main block
    for (int index = 0; index < round; ++index) {
        DataCopyUB2GMImpl((__gm__ int16_t *)gmTensor.GetPhyAddr() + comOffset,
            (__ubuf__ int16_t *)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((roundSize * sizeof(int16_t) + ONE_BLK_SIZE - 1) / (ONE_BLK_SIZE)), 0, 0});
        comOffset += roundSize;
    }
    // compute the tail block
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyUB2GMImpl((__gm__ int16_t *)gmTensor.GetPhyAddr() + comOffset,
            (__ubuf__ int16_t *)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((tail * sizeof(int16_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE), 0, 0});
    }
}

template <typename T>
template <typename CLS_NAME, void (CLS_NAME::*funComputeLeftHalf)(const int64_t&),
          void (CLS_NAME::*funComputeRightHalf)(const int64_t&)>
__aicore__ inline void GeGluGradV2Base310p<T>::ProcessLessEqual(CLS_NAME* objPtr) {
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
__aicore__ inline void GeGluGradV2Base310p<T>::ProcessGreater(CLS_NAME* objPtr) {
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
__aicore__ inline void GeGluGradV2Base310p<T>::ComputeGeluGrad(LocalTensor<float>& y, LocalTensor<float>& x1,
                                                           LocalTensor<float>& x2, const int64_t& realProcCount) {
    LocalTensor<float> tempBuf3 = resultTempBuf3.Get<float>();
    LocalTensor<float> tempBuf4 = resultTempBuf4.Get<float>();

    // compute tempBuf4 = np.sqrt(2 / np.pi) * (x2 + 0.044715 * np.power(x2, 3))
    Mul(tempBuf4, x2, x2, realProcCount);
    PipeBarrier<PIPE_V>();
    Mul(tempBuf4, tempBuf4, x2, realProcCount);
    PipeBarrier<PIPE_V>();
    Muls(tempBuf4, tempBuf4, COEFFICIENT_1, realProcCount);
    PipeBarrier<PIPE_V>();
    Add(tempBuf4, tempBuf4, x2, realProcCount);
    PipeBarrier<PIPE_V>();
    Muls(tempBuf4, tempBuf4, COEFFICIENT_2, realProcCount);
    PipeBarrier<PIPE_V>();

    // compute tempBuf3 = x2 * (1 - np.tanh(tempBuf4) * np.tanh(tempBuf4))
    ComputeTanh(tempBuf4, realProcCount);  // tempBuf4 equals np.tanh(tempBuf4)
    PipeBarrier<PIPE_V>();
    Mul(tempBuf3, tempBuf4, tempBuf4, realProcCount);
    PipeBarrier<PIPE_V>();
    Muls(tempBuf3, tempBuf3, NEG_ONE, realProcCount);
    PipeBarrier<PIPE_V>();
    Adds(tempBuf3, tempBuf3, POS_ONE, realProcCount);
    PipeBarrier<PIPE_V>();
    Mul(tempBuf3, tempBuf3, x2, realProcCount);
    PipeBarrier<PIPE_V>();

    // compute x2 = 0.5 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(x2, 2))
    Mul(x2, x2, x2, realProcCount);
    PipeBarrier<PIPE_V>();
    Muls(x2, x2, COEFFICIENT_3, realProcCount);
    PipeBarrier<PIPE_V>();
    Adds(x2, x2, COEFFICIENT_4, realProcCount);
    PipeBarrier<PIPE_V>();

    Mul(x2, tempBuf3, x2, realProcCount);  // compute x2 = tempBuf3 * x2
    PipeBarrier<PIPE_V>();

    // compute tempBuf4 = 0.5 * (1 + np.tanh(tempBuf4))
    Adds(tempBuf3, tempBuf4, POS_ONE, realProcCount);
    PipeBarrier<PIPE_V>();
    Muls(tempBuf3, tempBuf3, SCALER_HALF, realProcCount);
    PipeBarrier<PIPE_V>();
    Add(x2, x2, tempBuf3, realProcCount);
    PipeBarrier<PIPE_V>();
    Mul(y, x1, x2, realProcCount);
}

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::ComputeTanh(LocalTensor<float>& x, const int64_t& realProcCount) {
    LocalTensor<float> tempBuf3 = resultTempBuf3.Get<float>();
    // compute tanh = (e^2x - 1) / (e^2x + 1)
    Mins(x, x, MAX_TANH, realProcCount);
    PipeBarrier<PIPE_V>();
    Maxs(x, x, MIN_TANH, realProcCount);
    PipeBarrier<PIPE_V>();
    Muls(x, x, SCALER_TWO, realProcCount);
    PipeBarrier<PIPE_V>();
    Exp(x, x, realProcCount);
    PipeBarrier<PIPE_V>();
    Adds(tempBuf3, x, POS_ONE, realProcCount);
    PipeBarrier<PIPE_V>();
    Adds(x, x, NEG_ONE, realProcCount);
    PipeBarrier<PIPE_V>();
    Div(x, x, tempBuf3, realProcCount);
}

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::CopyInDyAndGelu(const int64_t& gmOffset, const int64_t& dataCount,
                                                           const int64_t& blockCount) {
    int64_t ubOffset = 0;
    LocalTensor<T> ubGelu = inQueueGelu.AllocTensor<T>();
    LocalTensor<T> ubDY = inQueueDY.AllocTensor<T>();

    struct DataCopyParams copyInParams(blockCount, 0, 0, 0);
    if (dataCount % perBlockCount == 0) {
        copyInParams.blockLen = dataCount * sizeof(T) / BLOCK_SIZE;
        DataCopy(ubGelu[ubOffset], geluGm[gmOffset], copyInParams);
        DataCopy(ubDY[ubOffset], dyGm[gmOffset], copyInParams);

        inQueueGelu.EnQue(ubGelu);
        inQueueDY.EnQue(ubDY);
        return;
    }
    // 非对齐情况搬运
    auto copyLen = CeilAlignA2B(dataCount, perBlockCount);
    for (auto i = 0; i < blockCount; i++) {
        // dst按32B整数倍存放数据，src按dataCount偏移搬运数据
        auto dstOffset = ubOffset + CeilAlignA2B(dataCount, perBlockCount) * i;
        auto srcOffset = gmOffset + (dataCount * i);
        DataCopy(ubGelu[dstOffset], geluGm[srcOffset], copyLen);
        DataCopy(ubDY[dstOffset], dyGm[srcOffset], copyLen);
    }

    inQueueGelu.EnQue(ubGelu);
    inQueueDY.EnQue(ubDY);
}

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::CopyInX(const int64_t& gmOffset, const int64_t& dataCount,
                                                   const int64_t& blockCount) {
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
    struct DataCopyParams copyInParams(blockCount, 0, 0, 0);
    int64_t ubOffset = 0;
    if (dataCount % perBlockCount == 0) {
        copyInParams.blockLen = dataCount * sizeof(T) / BLOCK_SIZE;
        copyInParams.srcStride = copyInParams.blockLen;
        if (activateLeft) {
            DataCopy(ubX1[ubOffset], xGm[gmOffset + valueM], copyInParams);
            DataCopy(ubX2[ubOffset], xGm[gmOffset], copyInParams);
        } else {
            DataCopy(ubX1[ubOffset], xGm[gmOffset], copyInParams);
            DataCopy(ubX2[ubOffset], xGm[gmOffset + valueM], copyInParams);
        }
        inQueueX1.EnQue(ubX1);
        inQueueX2.EnQue(ubX2);
        return;
    }

    // 非对齐情况搬运
    auto copyLen = CeilAlignA2B(dataCount, perBlockCount);
    for (auto i = 0; i < blockCount; i++) {
        // dst按32B整数倍存放数据，src按dataCount偏移搬运数据
        auto dstOffset = ubOffset + CeilAlignA2B(dataCount, perBlockCount) * i;
        auto srcOffset = gmOffset + (dataCount * 2 * i);
        if (activateLeft) {
            DataCopy(ubX1[dstOffset], xGm[srcOffset + valueM], copyLen);
            DataCopy(ubX2[dstOffset], xGm[srcOffset], copyLen);
        } else {
            DataCopy(ubX1[dstOffset], xGm[srcOffset], copyLen);
            DataCopy(ubX2[dstOffset], xGm[srcOffset + valueM], copyLen);
        }
    }

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::CopyOutLeft(const int64_t& gmOffset, const int64_t& dataCount,
                                                       const int64_t& blockCount) {
    LocalTensor<T> outLocalLeft = outQueueDX1.DeQue<T>();
    struct DataCopyParams copyOutParams(blockCount, 0, 0, 0);
    if (dataCount % perBlockCount == 0) {
        copyOutParams.blockLen = dataCount * sizeof(T) / BLOCK_SIZE;
        copyOutParams.dstStride = copyOutParams.blockLen;
        if (activateLeft) {
            DataCopy(dxGm[gmOffset + valueM], outLocalLeft, copyOutParams);
        } else {
            DataCopy(dxGm[gmOffset], outLocalLeft, copyOutParams);
        }
        outQueueDX1.FreeTensor(outLocalLeft);
        return;
    }

    // 非对齐情况尾部补0
    T inputVal(0.0);
    uint64_t padNum = perBlockCount - (dataCount % perBlockCount);
    uint64_t maskLow = ((1 << padNum) - 1) << (perBlockCount - padNum);
    uint64_t mask[2] = {maskLow, 0};
    uint64_t dstRepeatStride = CeilDiv(dataCount, perBlockCount);

    // 对最后一个block块填充
    uint64_t padStartAddr = (CeilDiv(dataCount, perBlockCount) - 1) * perBlockCount;
    PipeBarrier<PIPE_V>();
    CustomDuplicate(outLocalLeft, padStartAddr, mask, (uint64_t)blockCount, dstRepeatStride);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);

    auto copyLen = CeilAlignA2B(dataCount, perBlockCount);
    SetAtomicAdd<T>();
    for (auto i = 0; i < blockCount; i++) {
        // dst按(2 * dataCount)存放数据，src按32B整数倍偏移搬运数据
        auto dstOffset = gmOffset + (dataCount * 2 * i);
        auto srcOffset = CeilAlignA2B(dataCount, perBlockCount);
        if (activateLeft) {
            DataCopy(dxGm[dstOffset + valueM], outLocalLeft[i * srcOffset], copyLen);
        } else {
            DataCopy(dxGm[dstOffset], outLocalLeft[i * srcOffset], copyLen);
        }
    }
    SetAtomicNone();
    outQueueDX1.FreeTensor(outLocalLeft);
}

template <typename T>
__aicore__ inline void GeGluGradV2Base310p<T>::CopyOutRight(const int64_t& gmOffset, const int64_t& dataCount,
                                                        const int64_t& blockCount) {
    LocalTensor<T> outLocalRight = outQueueDX2.DeQue<T>();
    struct DataCopyParams copyOutParams(blockCount, 0, 0, 0);
    if (dataCount % perBlockCount == 0) {
        copyOutParams.blockLen = dataCount * sizeof(T) / BLOCK_SIZE;
        copyOutParams.dstStride = copyOutParams.blockLen;
        if (activateLeft) {
            DataCopy(dxGm[gmOffset], outLocalRight, copyOutParams);
        } else {
            DataCopy(dxGm[gmOffset + valueM], outLocalRight, copyOutParams);
        }
        outQueueDX2.FreeTensor(outLocalRight);
        return;
    }

    // 非对齐情况尾部补0
    T inputVal(0.0);
    uint64_t padNum = perBlockCount - (dataCount % perBlockCount);
    uint64_t maskLow = ((1 << padNum) - 1) << (perBlockCount - padNum);
    uint64_t mask[2] = {maskLow, 0};
    uint64_t dstRepeatStride = CeilDiv(dataCount, perBlockCount);

    // 对最后一个block块填充
    uint64_t padStartAddr = (CeilDiv(dataCount, perBlockCount) - 1) * perBlockCount;
    PipeBarrier<PIPE_V>();
    CustomDuplicate(outLocalRight, padStartAddr, mask, (uint64_t)blockCount, dstRepeatStride);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);

    auto copyLen = CeilAlignA2B(dataCount, perBlockCount);
    SetAtomicAdd<T>();
    for (auto i = 0; i < blockCount; i++) {
        // dst按(2 * dataCount)存放数据，src按32B整数倍偏移搬运数据
        auto dstOffset = gmOffset + (dataCount * 2 * i);
        auto srcOffset = CeilAlignA2B(dataCount, perBlockCount);
        if (activateLeft) {
            DataCopy(dxGm[dstOffset], outLocalRight[i * srcOffset], copyLen);
        } else {
            DataCopy(dxGm[dstOffset + valueM], outLocalRight[i * srcOffset], copyLen);
        }
    }
    SetAtomicNone();

    outQueueDX2.FreeTensor(outLocalRight);
}

}  // namespace GeGluGradV2For310P

#endif  // GE_GLU_GRAD_V2_BASE_310P_H_

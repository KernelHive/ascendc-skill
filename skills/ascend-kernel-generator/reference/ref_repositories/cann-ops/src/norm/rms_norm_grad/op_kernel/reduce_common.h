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
 * \file reduce_common.h
 * \brief
 */
#ifndef REDUCE_COMMON_H_
#define REDUCE_COMMON_H_
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr int32_t HALf_INTERVAL = 2;
constexpr float ZERO = 0;

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(const LocalTensor<float> &dstLocal,
    const LocalTensor<float> &srcLocal, const LocalTensor<float> &addSumLocal, const uint32_t elemNum,
    const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    uint32_t elemIdx = 0;
    for (; elemIdx + ELEM_PER_REP_FP32 <= numLastDim; elemIdx += ELEM_PER_REP_FP32) {
        Add(addSumLocal,
            srcLocal[elemIdx],
            addSumLocal,
            elemNum,
            repeat,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(addSumLocal,
            srcLocal[elemIdx],
            addSumLocal,
            tailCount,
            repeat,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);  // set mask = 64
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float,false>(dstLocal,addSumLocal,MASK_PLACEHOLDER,repeat,1,1,ELEM_PER_BLK_FP32);
    }
#else
    WholeReduceSum<float, false>(dstLocal, addSumLocal, MASK_PLACEHOLDER, repeat, 1, 1, ELEM_PER_BLK_FP32);
#endif
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(const LocalTensor<float> &dstLocal,
    const LocalTensor<float> &srcLocal, const LocalTensor<float> &addSumLocal, const uint32_t numLastDimAligned,
    const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    uint32_t repeatTimes = repeat / MAX_REP_NUM;
    if (repeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal, srcLocal, addSumLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex],
                srcLocal[repIndex * numLastDimAligned],
                addSumLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32,
                numLastDim,
                tailCount,
                MAX_REP_NUM,
                repStride);
        }
        if (repTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex],
                srcLocal[repIndex * numLastDimAligned],
                addSumLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32,
                numLastDim,
                tailCount,
                repTailNum,
                repStride);
        }
    }
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim, require D < 255 * 8.
 * size of addSumLocal: (N, 64)
 */
__aicore__ inline void ReduceSumMultiN(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
    const LocalTensor<float> &addSumLocal, const uint32_t numRow, const uint32_t numCol, const uint32_t numColAlign)
{
    const uint32_t tailCount = numCol % ELEM_PER_REP_FP32;
    const uint32_t repeat = numRow;
    const uint8_t repStride = numColAlign / ELEM_PER_BLK_FP32;
    Duplicate(addSumLocal, ZERO, numRow * ELEM_PER_REP_FP32);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSumForSmallReduceDim(dstLocal, srcLocal, addSumLocal, numColAlign, numCol, tailCount, repeat, repStride);
}

__aicore__ inline int32_t findPowerTwo(int32_t count)
{
    // find max power of 2 no more than count (32 bit)
    count |= count >> 1;  // Set the first digit of count's binary to 1
    count |= count >> 2;
    count |= count >> 4;
    count |= count >> 8;
    count |= count >> 16;
    return (count + 1) >> 1;
}

__aicore__ inline void ReduceSumHalfInterval(
    const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(srcLocal, srcLocal, srcLocal[bodyCount], tailCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(srcLocal, srcLocal, srcLocal[bodyCount], bodyCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float,false>(dstLocal,srcLocal,MASK_PLACEHOLDER,1,0,1,0);
    }
#else
    WholeReduceSum<float, false>(dstLocal, srcLocal, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float> &srcLocal, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(srcLocal, srcLocal, srcLocal[bodyCount], tailCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(srcLocal, srcLocal, srcLocal[bodyCount], bodyCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float,false>(srcLocal,srcLocal,MASK_PLACEHOLDER,1,0,1,0);
    }
#else
    WholeReduceSum<float, false>(srcLocal, srcLocal, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    return srcLocal.GetValue(0);
}
#endif  // REDUCE_COMMON_H_
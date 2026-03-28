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
 * \file rms_norm_base.h
 * \brief
 */
#ifndef RMS_NORM_BASE_H_
#define RMS_NORM_BASE_H_
#include "kernel_operator.h"
#include "reduce_common.h"

using namespace AscendC;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#define bfloat16_t int16_t
#endif
constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t NUM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr int32_t FLOAT_BTYPE_SIZE = 4;
constexpr int32_t NUM_PER_BLK_FP16 = 16;
constexpr int32_t CONTINUE_STRIDE = 8;
constexpr int32_t BLOCK_SIZE = 32;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr float MINUS_HALF = -0.5;
constexpr uint32_t ZERO_UINT = 0;
constexpr float ONE = 1;
constexpr int32_t SECOND_LOOP = 2;
constexpr int32_t MAX_REAPEAT = 255;
constexpr int32_t DIM_NUM = 2;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using true_type = IntegralConstant<bool, true>;
using false_type = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public false_type {};
template <typename Tp>
struct IsSame<Tp, Tp> : public true_type {};

template <typename T, typename T_GAMMA>
class KernelRmsNormBase {
#define IS_X_FP32 (IsSame<T, float>::value)
#define IS_GAMMA_FP32 (IsSame<T_GAMMA, float>::value)
#define IS_MIX_DTYPE ((!IS_X_FP32) && IS_GAMMA_FP32)
};

__aicore__ inline void ReduceSumHalfIntervalToRepeat(
    const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local, int32_t count, int32_t left)
{
    // count need smaller than 255 repeat
    if (likely(count > NUM_PER_BLK_FP32)) {
        int32_t bodyCount = count - left;
        int32_t tailCount = left;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        while (bodyCount > SECOND_LOOP * NUM_PER_BLK_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        bodyCount = bodyCount / HALf_INTERVAL;
        Add(dst_local, src_local, src_local[bodyCount], bodyCount);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ReduceSumFP32(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
    const LocalTensor<float> &work_local, int32_t count)
{
    // count need smaller than 255 repeat
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    AscendC::PipeBarrier<PIPE_V>();
    if (likely(repeatTimes > 0)) {
        Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        RepeatReduceSum<float, false>(dst_local, work_local, 1, 1, 0, 0, 1, 1);
    }
#else
    WholeReduceSum<float, false>(dst_local, work_local, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumCustom(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
    const LocalTensor<float> &work_local, int32_t count)
{
    ReduceSumFP32(dst_local, src_local, work_local, count);
}
__aicore__ inline void ReduceSumFP32ToBlock(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
    const LocalTensor<float> &work_local, int32_t count)
{
    // count need smaller than 255 repeat
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONCE_VECTOR_SIZE / BLOCK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    AscendC::PipeBarrier<PIPE_V>();
    if (likely(repeatTimes > 0)) {
        Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    BlockReduceSum(dst_local, work_local, 1, mask, 1, 1, DEFAULT_REPEAT_STRIDE);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BlockReduceSumFP32(
    const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local, int32_t count)
{
    // count need multiple of 8
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t dstAddr = repeatTimes * 8;
    int32_t srcAddr = repeatTimes * NUM_PER_REP_FP32;
    if (likely(repeatTimes > 0)) {
        BlockReduceSum(dst_local, src_local, repeatTimes, NUM_PER_REP_FP32, 1, 1, DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (tailCount != 0) {
        BlockReduceSum(dst_local[dstAddr], src_local[srcAddr], 1, tailCount, 1, 1, DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U &dstTensor, const R &srcTensor, const uint32_t count)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    DataCopyParams copyParams;
    copyParams.blockLen = count * sizeof(T);
    copyParams.blockCount = 1;
    if constexpr (IsSame<U, AscendC::LocalTensor<T>>::value) {
        DataCopyPadParams padParams;
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    } else {
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }
#else
    // only support count greater than 32byte
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        if constexpr (IsSame<U, AscendC::LocalTensor<T>>::value) {
            int32_t num = AlignUp(count, numPerBlock);
            DataCopy(dstTensor, srcTensor, num);
        } else {
            if (count < numPerBlock) {
                DataCopy(dstTensor, srcTensor, numPerBlock);
            } else {
                int32_t num = count / numPerBlock * numPerBlock;
                DataCopy(dstTensor, srcTensor, num);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
                    srcTensor.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
            }
        }
    }
#endif
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    const LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, const uint32_t numRow, const uint32_t numCol)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    DataCopyParams copyParams;
    copyParams.blockLen = numCol * sizeof(T);
    copyParams.blockCount = numRow;
    DataCopyPadParams padParams;
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
#endif
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    const GlobalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t numRow, const uint32_t numCol)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    DataCopyParams copyParams;
    copyParams.blockLen = numCol * sizeof(T);
    copyParams.blockCount = numRow;
    DataCopyPad(dstTensor, srcTensor, copyParams);
#endif
}

__aicore__ inline void RoundFloat2Int8(LocalTensor<int8_t> &dstTensor, LocalTensor<float> &srcTensor, int32_t size)
{
    Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
    AscendC::PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
}

__aicore__ inline uint32_t ROUND_UP(uint32_t x, uint32_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}

#endif  // RMS_NORM_BASE_H_

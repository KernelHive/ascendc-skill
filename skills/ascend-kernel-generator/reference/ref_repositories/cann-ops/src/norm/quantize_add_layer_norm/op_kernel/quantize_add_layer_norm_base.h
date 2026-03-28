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
 * \file quantize_add_layer_norm_base.h
 * \brief
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_BASE_H_
#define QUANTIZE_ADD_LAYER_NORM_BASE_H_

#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

using namespace AscendC;
static constexpr float ZERO = 0;

#if __CCE_AICORE__ == 220
#define OUTPUT_MEAN_RSTD 1
#define SUPPORT_BF16 1
#else
#define OUTPUT_MEAN_RSTD 0
#define SUPPORT_BF16 0
#endif

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

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(const R<T> &dst, const S<T> &src, const uint32_t len, const uint32_t count = 1,
    const DataCopyPadParams &padParams = {})
{
#if __CCE_AICORE__ == 220
    DataCopyParams copyParams;
    copyParams.blockLen = len * sizeof(T);
    copyParams.blockCount = count;
    if constexpr (IsSame<R<T>, AscendC::LocalTensor<T>>::value) {
        DataCopyPad(dst, src, copyParams, padParams);
    } else {
        DataCopyPad(dst, src, copyParams);
    }
#else
    auto elementCount = len * count;
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (elementCount % numPerBlock == 0) {
        DataCopy(dst, src, elementCount);
    } else {
        if constexpr (IsSame<R<T>, AscendC::LocalTensor<T>>::value) {
            auto num = AlignUp(elementCount, numPerBlock);
            DataCopy(dst, src, num);
        } else {
            int32_t num = elementCount / numPerBlock * numPerBlock;
            DataCopy(dst, src, num);
            if (elementCount != num) {
                event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                SetFlag<HardEvent::MTE3_S>(eventMTE3S);
                WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    auto tensorValue = src.GetValue(elementCount - numPerBlock + i);
                    src.SetValue(i, tensorValue);
                }
                event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventSMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
                DataCopy(dst[elementCount - numPerBlock], src, numPerBlock);
            }
        }
    }
#endif
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float> &src_local, int32_t count)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
    assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
    float value = 0.0;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        if (likely(repeatTimes > 0)) {
            AscendCUtils::SetMask<float>(elementNumPerRep);
            ReduceSum(src_local, src_local, src_local, count);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            uint64_t acc_val = GetAccVal();
            value = *reinterpret_cast<float *>(&acc_val);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            ReduceSum(src_local[bodyCount], src_local[bodyCount], src_local[bodyCount], count);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            uint64_t acc_val = GetAccVal();
            value += *reinterpret_cast<float *>(&acc_val);
        }
    }
#else
    ReduceSum(src_local, src_local, src_local, count);
    value = src_local.GetValue(0);
#endif
    return value;
}

__aicore__ inline void ReduceSumShort(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
    const LocalTensor<float> &tmp_local, int32_t align_len, int32_t data_len, int32_t repeat)
{
    int32_t elementNum = ONE_BLK_SIZE / sizeof(float);
    int32_t maxRepeat = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    int32_t tailCount = data_len % elementNum;
    uint32_t index = 0;
    uint8_t repStride = align_len / ONE_BLK_FLOAT_NUM;

    int32_t repeatTimes = repeat / elementNum;
    int32_t bodyCount = repeatTimes * elementNum;
    int32_t repeatTail = repeat % elementNum * elementNum;

    Duplicate<float>(tmp_local, ZERO, repeat * elementNum);
    PipeBarrier<PIPE_V>();
    for (index = 0; index + elementNum <= data_len; index += elementNum) {
        Add(tmp_local, tmp_local, src_local[index], elementNum, repeat, {1, 1, 1, 1, 1, repStride});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmp_local, tmp_local, src_local[index], tailCount, repeat, {1, 1, 1, 1, 1, repStride});
    }
    PipeBarrier<PIPE_V>();
    if (repeatTimes != 0) {
        BlockReduceSum<float>(dst_local, tmp_local, repeatTimes, maxRepeat, 1, 1, elementNum);
    }
    if (repeatTail != 0) {
        BlockReduceSum<float>(dst_local[bodyCount], tmp_local[bodyCount * elementNum], 1, repeatTail, 1, 1, elementNum);
    }
}

__aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
    return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
}

__aicore__ inline uint32_t THE_SMALLEST_NUMBER_BETWEEN_TWO_NUMBERS(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t THE_BIGGEST_NUMBER_BETWEEN_TWO_NUMBERS(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

#endif  // QUANTIZE_ADD_LAYER_NORM_BASE_H_
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
 * \file add_layer_norm_quant_helper.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_QUANT_HELPER_H_
#define ADD_LAYER_NORM_QUANT_HELPER_H_

#include "kernel_operator.h"
#if __CCE_AICORE__ == 220
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#endif

using namespace AscendC;
static constexpr float ZERO = 0;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr int32_t ELEM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ELEM_PER_REP_FP16 = 128;
constexpr uint32_t TWO_NUMS_MAX_REP_NUM = 255;
constexpr int32_t ROW_FACTOR = 16;
constexpr float DYNAMIC_QUANT_DIVIDEND = 127.0;

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

__aicore__ inline uint32_t TWO_NUMS_MIN(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t TWO_NUMS_MAX(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

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
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    auto tensorValue = src.GetValue(elementCount - numPerBlock + i);
                    src.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dst[elementCount - numPerBlock], src, numPerBlock);
            }
        }
    }
#endif
}

/*
 * support count in (0, 255 * 64)
 * very very slow
 */
__aicore__ inline float ReduceMaxFP32(const LocalTensor<float> &src_local, int32_t count)
{
    float value = 0.0;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        AscendCUtils::SetMaskCount<float>();
        SetVectorMask<float>(0, count);
        ReduceMax(src_local, src_local, src_local, 1);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        uint64_t reg_val = get_max_min_cnt();
        value = *reinterpret_cast<float *>(&reg_val);
        SetMaskNorm();
        SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
#else
    ReduceMax(src_local, src_local, src_local, count);
    value = src_local.GetValue(0);
#endif
    return value;
}

/*
 * only support count in (128, 255 * 64)
 * about 20us faster than above in case fp16:(1024, 11264) on 910B
 */
__aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &src_local, int32_t count)
{
    uint64_t repsFp32 = count >> 6;        // 6 is cound / ELEM_PER_REP_FP32
    uint64_t offsetsFp32 = repsFp32 << 6;  // 6 is repsFp32 * ELEM_PER_REP_FP32
    uint64_t remsFp32 = count & 0x3f;      // 0x3f 63, count % ELEM_PER_REP_FP32

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(src_local, src_local[ELEM_PER_REP_FP32], src_local, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Max(src_local, src_local[offsetsFp32], src_local, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
    // 8 is rep stride
    WholeReduceMax(src_local, src_local, mask, 1, 8, 1, 8);
}

/*
 * only support count <= 255 * 64 = 16320
 * very fast
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float> &src_local, int32_t count)
{
    float value = 0.0;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        AscendCUtils::SetMaskCount<float>();
        SetVectorMask<float>(0, count);
        ReduceSum(src_local, src_local, src_local, 1);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        uint64_t acc_val = GetAccVal();
        value = *reinterpret_cast<float *>(&acc_val);
        SetMaskNorm();
        SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
#else
    ReduceSum(src_local, src_local, src_local, count);
    value = src_local.GetValue(0);
#endif
    return value;
}

/*
 * only support count in (128, 255 * 64)
 * about 6us slower than above in case fp16:(1024, 11264) on 910B
 */
__aicore__ inline void ReduceSumInplace(const LocalTensor<float> &src_local, int32_t count)
{
    uint64_t repsFp32 = count >> 6;        // 6 is cound / ELEM_PER_REP_FP32
    uint64_t offsetsFp32 = repsFp32 << 6;  // 6 is repsFp32 * ELEM_PER_REP_FP32
    uint64_t remsFp32 = count & 0x3f;      // 0x3f 63, count % ELEM_PER_REP_FP32

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Add(src_local, src_local[ELEM_PER_REP_FP32], src_local, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Add(src_local, src_local[offsetsFp32], src_local, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
    // 8 is rep stride
    WholeReduceSum(src_local, src_local, mask, 1, 8, 1, 8);
}

__aicore__ inline void DivScalarFP32(LocalTensor<float> &dstTensor, LocalTensor<float> &dividendTensor,
    LocalTensor<float> &tmpTensor, float divisorScalar, uint32_t count)
{
    uint32_t repsFp32 = count >> 6;                         // 6 is devide 64
    uint32_t offsetsFp32 = count & 0xffffffc0;              // 0xffffffc0 is floor by 64
    uint32_t remsFp32 = count & 0x3f;                       // 0x3f is mod(64)
    Duplicate(tmpTensor, divisorScalar, FLOAT_BLOCK_ELEM);  // FLOAT_BLOCK_ELEM);
    PipeBarrier<PIPE_V>();
    Div(dstTensor, dividendTensor, tmpTensor, ELEM_PER_REP_FP32, repsFp32, {1, 1, 0, 8, 8, 0});
    if ((remsFp32 > 0)) {
        Div(dstTensor[offsetsFp32], dividendTensor[offsetsFp32], tmpTensor, remsFp32, 1, {1, 1, 0, 8, 8, 0});
    }
}

__aicore__ inline void RoundFloat2Int8(LocalTensor<int8_t> &dstTensor, LocalTensor<float> &srcTensor, int32_t size)
{
    Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
}

#endif  // ADD_LAYER_NORM_QUANT_HELPER_H_

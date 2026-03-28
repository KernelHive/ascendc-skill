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
 * \file add_layer_norm_grad_utils.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_UTILS_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_UTILS_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "safe_data_copy.h"

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

using namespace AscendC;

constexpr uint32_t TAIL_BUFFER_SIZE = 32;
constexpr uint32_t REDUCE_SRC0_REPSTRIDE = 8;
constexpr uint32_t REDUCE_BUFFER_STAND_SIZE = 64;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_AlIGN = 32;
constexpr float ZERO = 0;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr uint32_t MAX_COPY_LENTH = 2000;
// define num_last_dim => reduce(gamma_axis)
// define num_first_dim => reduce(input_axis except gamma_axis)
__aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
__aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}
__aicore__ inline uint32_t ROUND_UP(uint32_t x, uint32_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}
__aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

uint32_t FLOOR_DIV(uint32_t x, uint32_t y)
{
    if (y > 0 && x + 1 - y > 0) {
        return (x + 1 - y) / y;
    }
    return 0;
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
    if (g_coreType == AIV) {
        if (likely(repeatTimes > 0)) {
            AscendCUtils::SetMask<float>(elementNumPerRep);
            ReduceSum(src_local, src_local, src_local, repeatTimes);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            uint64_t acc_val = GetAccVal();
            value = *reinterpret_cast<float *>(&acc_val);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            ReduceSum(src_local[bodyCount], src_local[bodyCount], src_local[bodyCount], 1);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            uint64_t acc_val = GetAccVal();
            value += *reinterpret_cast<float *>(&acc_val);
        }
    }
    return value;
}

__aicore__ inline float ReduceSumCustom(const LocalTensor<float> &src_local, int32_t count)
{
#if __CCE_AICORE__ == 220
    return ReduceSumFP32(src_local, count);
#else
    ReduceSum(src_local, src_local, src_local, count);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    float rstd_value = src_local.GetValue(0);
    return rstd_value;
#endif
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U &dstTensor, const R &srcTensor, const uint32_t processElem,
    const uint32_t offset, bool is_float, const uint16_t blockCount)
{
#if __CCE_AICORE__ == 220
    DataCopyParams dataCopyParamsND{blockCount, (uint16_t)(processElem * sizeof(T)), 0, 0};
    DataCopyPad(dstTensor[offset], srcTensor, dataCopyParamsND);
#else
    int32_t blockNumel = is_float ? BLOCK_AlIGN / sizeof(float) : BLOCK_AlIGN / sizeof(T);
    int32_t blockNum = processElem / blockNumel;
    int32_t tail = processElem % blockNumel;
    int32_t blkLength = blockNum * blockNumel;
    if (blockNum == 0) {
        return;
    }
    for (uint32_t idx = 0; idx < blockCount; idx++) {
        uint32_t curOffset = offset + idx * processElem;
        DataCopy(dstTensor[curOffset], srcTensor[idx * ROUND_UP(processElem, blockNumel)], blkLength);
        if (tail != 0) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (uint32_t i = 0; i < blockNumel; i++) {
                T tensorValue =
                    srcTensor.GetValue(idx * ROUND_UP(processElem, blockNumel) + processElem - blockNumel + i);
                srcTensor.SetValue(idx * ROUND_UP(processElem, blockNumel) + i, tensorValue);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(dstTensor[curOffset + processElem - blockNumel],
                srcTensor[idx * ROUND_UP(processElem, blockNumel)],
                blockNumel);
        }
    }
#endif
}

__aicore__ inline void DataCopyAutomicAdd(GlobalTensor<float> dstTensor, const LocalTensor<float> srcTensor,
    const uint32_t processElem, const uint32_t offset, const uint16_t blockCount)
{
#if __CCE_AICORE__ == 220
    DataCopyParams dataCopyParamsND{blockCount, (uint16_t)(processElem * sizeof(float)), 0, 0};
    DataCopyPad(dstTensor[offset], srcTensor, dataCopyParamsND);
#else
    SafeDataCopy<true>(dstTensor[offset], srcTensor, blockCount * processElem);
#endif
}
__aicore__ inline void InitGmData(GlobalTensor<float> outputPdGammaGm, GlobalTensor<float> outputPdBetaGm,
    const uint32_t dDimNum, LocalTensor<float> dbeta, uint32_t elemWithDInUBFp32)
{
#if __CCE_AICORE__ == 220
    if (GetBlockIdx() == 0) {
        InitOutput<float>(outputPdGammaGm, dDimNum, 0);
        InitOutput<float>(outputPdBetaGm, dDimNum, 0);
    }
#else
    if (GetBlockIdx() == 0) {
        uint32_t maxCopyLenth = elemWithDInUBFp32 - FLOAT_BLOCK_ELEM;
        maxCopyLenth = MAX(maxCopyLenth, FLOAT_BLOCK_ELEM);
        uint32_t loopNum = dDimNum / maxCopyLenth;
        uint32_t tail = dDimNum % maxCopyLenth;
        if (loopNum == 0) {
            Duplicate(dbeta, 0.0f, ROUND_UP(dDimNum, FLOAT_BLOCK_ELEM));
        } else {
            Duplicate(dbeta, 0.0f, elemWithDInUBFp32);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t idx = 0; idx < loopNum; idx++) {
            uint32_t curOffset = idx * maxCopyLenth;
            SafeDataCopy(outputPdGammaGm[curOffset], dbeta, maxCopyLenth);
            SafeDataCopy(outputPdBetaGm[curOffset], dbeta, maxCopyLenth);
        }
        if (tail != 0) {
            if (loopNum >= 1) {
                uint32_t rollbackTail = FLOAT_BLOCK_ELEM + tail;
                SafeDataCopy(outputPdGammaGm[maxCopyLenth * loopNum - FLOAT_BLOCK_ELEM], dbeta, rollbackTail);
                SafeDataCopy(outputPdBetaGm[maxCopyLenth * loopNum - FLOAT_BLOCK_ELEM], dbeta, rollbackTail);
            } else {
                SafeDataCopy(outputPdGammaGm[maxCopyLenth * loopNum], dbeta, tail);
                SafeDataCopy(outputPdBetaGm[maxCopyLenth * loopNum], dbeta, tail);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
#endif
}

#endif  // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_UTILS_H
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
 * \file add_layer_norm_base.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_BASE_H_
#define ADD_LAYER_NORM_BASE_H_

#include "kernel_operator.h"
#if __CCE_AICORE__ == 220
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#endif

using namespace AscendC;
static constexpr float ZERO = 0;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr int32_t ELEM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ELEM_PER_REP_FP16 = 128;
constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t BROADCAST_ND_DIM_NUM = 2;     // only support 1 or 2
constexpr uint32_t BROADCAST_ND_LAST_INDEX = 1;  // only support 0 or 1

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
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float> &src_local, int32_t count)
{
    int32_t elementNumPerRep = ELEM_PER_REP_FP32;
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
            ReduceSum(src_local, src_local, src_local, repeatTimes);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint64_t acc_val = GetAccVal();
            value = *reinterpret_cast<float *>(&acc_val);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            ReduceSum(src_local[bodyCount], src_local[bodyCount], src_local[bodyCount], 1);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
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
    int32_t maxRepeat = ELEM_PER_REP_FP32;
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

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(const LocalTensor<float> &dstLocal,
    const LocalTensor<float> &srcLocal, const LocalTensor<float> &tmpLocal, const uint32_t elemNum,
    const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    uint32_t elemIndex = 0;
    for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
        Add(tmpLocal,
            srcLocal[elemIndex],
            tmpLocal,
            elemNum,
            repeat,
            {1, 1, 1, FLOAT_BLOCK_ELEM, repStride, FLOAT_BLOCK_ELEM});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmpLocal,
            srcLocal[elemIndex],
            tmpLocal,
            tailCount,
            repeat,
            {1, 1, 1, FLOAT_BLOCK_ELEM, repStride, FLOAT_BLOCK_ELEM});
    }
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);  // set mask = 64
#if __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        RepeatReduceSum<float, false>(dstLocal, tmpLocal, repeat, ELEM_PER_REP_FP32, 1, 1, 1, ELEM_PER_REP_FP32);
    }
#else
    WholeReduceSum<float, false>(dstLocal, tmpLocal, MASK_PLACEHOLDER, repeat, 1, 1, ELEM_PER_REP_FP32);
#endif
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(const LocalTensor<float> &dstLocal,
    const LocalTensor<float> &srcLocal, const LocalTensor<float> &tmpLocal, const uint32_t numLastDimAligned,
    const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    uint32_t repeatTimes = repeat / MAX_REP_NUM;
    if (repeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex],
                srcLocal[repIndex * numLastDimAligned],
                tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32,
                numLastDim,
                tailCount,
                MAX_REP_NUM,
                repStride);
        }
        if (repTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex],
                srcLocal[repIndex * numLastDimAligned],
                tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32,
                numLastDim,
                tailCount,
                repTailNum,
                repStride);
        }
    }
}

__aicore__ inline void InitVAForTranspose(__ubuf__ half *transposeDstAddr, __ubuf__ half *transposeSrcAddr)
{
    uint64_t va_reg_array_1[8] = {((uint64_t)transposeDstAddr),
        ((uint64_t)(transposeDstAddr + (int64_t)128)),
        ((uint64_t)(transposeDstAddr + (int64_t)256)),
        ((uint64_t)(transposeDstAddr + (int64_t)384)),
        ((uint64_t)(transposeDstAddr + (int64_t)512)),
        ((uint64_t)(transposeDstAddr + (int64_t)640)),
        ((uint64_t)(transposeDstAddr + (int64_t)768)),
        ((uint64_t)(transposeDstAddr + (int64_t)896))};
    set_va_reg_sb(VA0, va_reg_array_1);
    uint64_t va_reg_array_2[8] = {((uint64_t)(transposeDstAddr + (int64_t)1024)),
        ((uint64_t)(transposeDstAddr + (int64_t)1152)),
        ((uint64_t)(transposeDstAddr + (int64_t)1280)),
        ((uint64_t)(transposeDstAddr + (int64_t)1408)),
        ((uint64_t)(transposeDstAddr + (int64_t)1536)),
        ((uint64_t)(transposeDstAddr + (int64_t)1664)),
        ((uint64_t)(transposeDstAddr + (int64_t)1792)),
        ((uint64_t)(transposeDstAddr + (int64_t)1920))};
    set_va_reg_sb(VA1, va_reg_array_2);
    uint64_t va_reg_array_3[8] = {((uint64_t)transposeSrcAddr),
        ((uint64_t)(transposeSrcAddr + (int64_t)16)),
        ((uint64_t)(transposeSrcAddr + (int64_t)256)),
        ((uint64_t)(transposeSrcAddr + (int64_t)272)),
        ((uint64_t)(transposeSrcAddr + (int64_t)512)),
        ((uint64_t)(transposeSrcAddr + (int64_t)528)),
        ((uint64_t)(transposeSrcAddr + (int64_t)768)),
        ((uint64_t)(transposeSrcAddr + (int64_t)784))};
    set_va_reg_sb(VA2, va_reg_array_3);
    uint64_t va_reg_array_4[8] = {((uint64_t)(transposeSrcAddr + (int64_t)1024)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1040)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1280)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1296)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1536)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1552)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1792)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1808))};
    set_va_reg_sb(VA3, va_reg_array_4);
}

/*
 * for reduce dim > 64, need run InitVAForTranspose first.
 */
__aicore__ inline void CCEBroadCastShort(__ubuf__ int16_t *dstAddr, __ubuf__ float *srcAddr,
    __ubuf__ int16_t *transposeDstAddr, __ubuf__ int16_t *transposeSrcAddr, __ubuf__ int16_t *orOffsetINT16Addr,
    const uint32_t forCount, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    SetVectorMask<float>(0x0, 0xffff);
    vector_dup(orOffsetINT16Addr, (int16_t)0, 1, 0, 0, (uint8_t)0, (uint8_t)0);  // all zero
    copy_ubuf_to_ubuf((__ubuf__ float *)transposeSrcAddr, srcAddr, 0, 1, repStride, 0, 0);
    PipeBarrier<PIPE_V>();

    vtranspose((__ubuf__ uint16_t *)transposeDstAddr, (__ubuf__ uint16_t *)transposeSrcAddr);
    AscendCUtils::SetMask<half>(ELEM_PER_REP_FP16);
    PipeBarrier<PIPE_V>();

    vor(transposeSrcAddr, transposeDstAddr, orOffsetINT16Addr, 8, 1, 1, 0, 16, 0, 0);
    vor(transposeSrcAddr + (int64_t)ELEM_PER_REP_FP16,
        transposeDstAddr + (int64_t)ELEM_PER_REP_FP16,
        orOffsetINT16Addr,
        8,
        1,
        1,
        0,
        16,
        0,
        0);
    PipeBarrier<PIPE_V>();
    scatter_vnchwconv_b16(VA0, VA2, 8, 1, 2);  // transpose
    PipeBarrier<PIPE_V>();
    for (int64_t forIndex = 0; forIndex < (int64_t)forCount; ++forIndex) {
        vor(dstAddr + (forIndex * (int64_t)ELEM_PER_REP_FP16),
            transposeDstAddr,
            orOffsetINT16Addr,
            repeat,
            1,
            0,
            0,
            repStride,
            1,
            0);
    }
    if (tailCount != 0) {
        AscendCUtils::SetMask<half>(tailCount * 2);
        vor(dstAddr + (forCount * (int64_t)ELEM_PER_REP_FP16),
            transposeDstAddr,
            orOffsetINT16Addr,
            repeat,
            1,
            0,
            0,
            repStride,
            1,
            0);
    }
}

__aicore__ inline void Level0MulFp32Short(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
    const LocalTensor<float> &src1Local, uint32_t alignElem, uint32_t repeat, uint32_t processElem)
{
    uint32_t maxElemFp32 = ELEM_PER_REP_FP32;
    uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
    uint32_t tailCount = processElem % maxElemFp32;

    uint32_t repeatTimes = repeat / MAX_REP_NUM;

    uint32_t index = 0;
    uint32_t elemIndex = 0;
    if (likely(repeatTimes == 0)) {
        elemIndex = 0;
        for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
            Mul(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                maxElemFp32,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
        if (tailCount != 0) {
            Mul(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                tailCount,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            elemIndex = 0;
            repElem = repIndex * alignElem;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
        if (repTailNum != 0) {
            elemIndex = 0;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
    }
}

__aicore__ inline void Level0AddFp32Short(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
    const LocalTensor<float> &src1Local, uint32_t alignElem, uint32_t repeat, uint32_t processElem)
{
    uint32_t maxElemFp32 = ELEM_PER_REP_FP32;
    uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
    uint32_t tailCount = processElem % maxElemFp32;

    uint32_t repeatTimes = repeat / MAX_REP_NUM;

    uint32_t index = 0;
    uint32_t elemIndex = 0;
    if (likely(repeatTimes == 0)) {
        elemIndex = 0;
        for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
            Add(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                maxElemFp32,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
        if (tailCount != 0) {
            Add(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                tailCount,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            elemIndex = 0;
            repElem = repIndex * alignElem;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
        if (repTailNum != 0) {
            elemIndex = 0;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
    }
}

#endif  // ADD_LAYER_NORM_BASE_H_

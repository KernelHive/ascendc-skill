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
 * \file mat_mul_v3_common.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_COMMON_H__
#define __OP_KERNEL_MATMUL_V3_COMMON_H__
#include "kernel_operator.h"
#include "lib/matmul_intf.h"


using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

const uint64_t NUM_TWO = 2;
const uint64_t NUM_FOUR = 4;
const uint64_t DATA_SIZE_FP16 = 2;
const uint64_t DATA_SIZE_FP32 = 4;
const uint64_t BANK_SIZE = 256;
const uint64_t ALIGN_BYTE = 256;
const uint64_t ALIGN_128_BYTE = 128;
const uint64_t MAX_BLOCK_NUM = 100;
const uint64_t DEFAULT_BLOCK_LEN = 8;
const uint64_t BLOCK_SIZE = 16;
constexpr uint64_t NUM_AIV_TO_AIC_RATIO = 2;

constexpr uint64_t ALIGNED_H = 16;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;

const uint32_t CONTROL_DB = 1;
const uint32_t ALL_L2_CACHE_ENABLE = 1;
const uint32_t C_L2_DISABLE = 16;

// common MDL config
constexpr MatmulConfig MM_CFG_MDL = GetMDLConfig();
// set isVecND2Nz
constexpr MatmulConfig MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, 0, true);
// set enUnitFlag
constexpr MatmulConfig MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, true);
// set doMTE2Preload
constexpr MatmulConfig MM_CFG_PRELOAD_MK = GetMDLConfig(false, false, 2);
constexpr MatmulConfig MM_CFG_PRELOAD_NK = GetMDLConfig(false, false, 1);

enum ND2NZ_SELECT {
    ONLY_A = 1,
    ONLY_B = 2,
    BOTH_AB = 3
};

enum FIXPIPE_OPT_SELECT {
    BASE = 0,
    BASE_ENABLE_ALIGNOUT = 1,
    VEC_NZ2ND_UNALIGNOUT = 2
};

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

__aicore__ inline uint64_t MMLcm(uint64_t m, uint64_t n) {
    if (m == 0 || n == 0) {
        return 0; // 处理输入为0的情况
    }
    uint64_t total = m * n;
    uint64_t tmp = 0;
    while (n != 0) {
        tmp = m % n;
        m = n;
        n = tmp;
    }
    return total / m;
}

__aicore__ inline void WaitFlagDevLocal(int64_t flagID)
{
    wait_flag_dev(flagID);
}

template <HardEvent event>
__aicore__ inline void TPipeSetWaitFlag() {
    auto eventID = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

template <class T>
__aicore__ inline void GetSizeC0(uint64_t &c0Size) {
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
}

template <class A_T, class B_T, class C_T, class BiasT>
__aicore__ inline void SetL2CacheEnable(const L2cacheUseInfo& l2EnableInfo,
    GlobalTensor<A_T> &aGlobal, GlobalTensor<B_T> &bGlobal,
    GlobalTensor<C_T> &cGlobal, GlobalTensor<BiasT> &biasGlobal)
{
    if ((l2EnableInfo.l2CacheFlag & ALL_L2_CACHE_ENABLE) == 0) {
        if (l2EnableInfo.l2CacheFlag & C_L2_DISABLE) {
            cGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }
}

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMV3DivCeil(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMV3CeilAlign(uint64_t a, uint64_t b)
{
    return MMV3DivCeil(a, b) * b;
}

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMV3DivFloor(uint64_t a, uint64_t b)
{
    return b == 0 ? a : a / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMV3FloorAlign(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : a / b * b;
}

template <class T>
__aicore__ inline void CopyGmToUbufAlign(__ubuf__ void* dst, __gm__ void* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP16) {
        copy_gm_to_ubuf_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP32) {
        copy_gm_to_ubuf_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        ASSERT(false);
    }
}

template <typename T>
__aicore__ inline void CopyUbufToGmAlign(__gm__ void* dst, __ubuf__ void* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP16) {
        copy_ubuf_to_gm_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP32) {
        copy_ubuf_to_gm_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        ASSERT(false);
    }
}

template <typename T>
__aicore__ inline void CopyCast(const LocalTensor<float>& ubSrc, const LocalTensor<T>& ubDst, __gm__ float* src,
    __gm__ T* dst, uint64_t offset, uint16_t nBurst, uint16_t lenBurst, uint32_t gap, uint8_t pingpongEventId)
{
    CopyGmToUbufAlign<float>((__ubuf__ float*)ubSrc.GetPhyAddr(), (__gm__ float*)src + offset, 0, nBurst,
        lenBurst * sizeof(float), 0, 0, gap * sizeof(float), 0);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingpongEventId));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingpongEventId));
    Cast(ubDst, ubSrc, RoundMode::CAST_RINT, nBurst * lenBurst);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingpongEventId));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingpongEventId));
    CopyUbufToGmAlign<T>((__gm__ T *)dst + offset, (__ubuf__ T*)ubDst.GetPhyAddr(), 0, nBurst,
        lenBurst * sizeof(T), 0, 0, 0, gap * sizeof(T));
}

#define DBCAST
#ifdef DBCAST
// v220
template <typename T>
__aicore__ inline void Cast32to16V220(__gm__ T *dst, __gm__ float *src, uint64_t size, uint32_t nCoreUse,
    uint32_t n, TBuf<TPosition::VECCALC> &tmpBuf)
{
    uint16_t dataSize = static_cast<uint16_t>(TOTAL_UB_SIZE / NUM_TWO / sizeof(float));
    uint16_t dataSize1 = static_cast<uint16_t>(TOTAL_UB_SIZE / NUM_TWO / sizeof(T));
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<T> ubDstPong = ubDstPing[dataSize1];
    LocalTensor<float> ubSrcPong = ubSrcPing[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc = ubSrcPing;

    GlobalTensor<float> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), size);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst), size);

    uint8_t pingpongEventId = 0;

    if (nCoreUse >= dataSize) { // 操作的最小一行数据量超ub时，对于每一行数据循环做cast处理
        uint64_t mRepeat = size / nCoreUse;
        uint16_t nBurst = static_cast<uint16_t>(nCoreUse / dataSize);
        uint16_t tail = static_cast<uint16_t>(nCoreUse % dataSize);

        for (uint64_t i = 0; i < mRepeat; ++i) {
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
            for (uint16_t j = 0; j < nBurst; ++j) {
                if ((j & CONTROL_DB) == 0) {
                    pingpongEventId = 0;
                    ubDst = ubDstPing;
                    ubSrc = ubSrcPing;
                } else {
                    pingpongEventId = 1;
                    ubDst = ubDstPong;
                    ubSrc = ubSrcPong;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
                CopyCast<T>(ubSrc, ubDst, src, dst, i * n + j * dataSize, 1, dataSize, 0, pingpongEventId);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
            }
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
            if (tail > 0) {
                CopyCast<T>(ubSrc, ubDst, src, dst, i * n + nBurst * dataSize, 1, tail, 0, pingpongEventId);
            }
        }
        return;
    }

    uint16_t nBurst = static_cast<uint16_t>(dataSize / nCoreUse);
    uint64_t repeat = size / (nBurst * nCoreUse);
    uint16_t tail = static_cast<uint16_t>(size % (nBurst * nCoreUse));

    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    for (uint64_t i = 0; i < repeat; ++i) {
        if ((i & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            ubDst = ubDstPing;
            ubSrc = ubSrcPing;
        } else {
            pingpongEventId = 1;
            ubDst = ubDstPong;
            ubSrc = ubSrcPong;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
        CopyCast<T>(ubSrc, ubDst, src, dst, i * nBurst * n, nBurst, static_cast<uint16_t>(nCoreUse),
            n - nCoreUse, pingpongEventId);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    if ((repeat & CONTROL_DB) == 0) {
        pingpongEventId = 0;
        ubDst = ubDstPing;
        ubSrc = ubSrcPing;
    } else {
        pingpongEventId = 1;
        ubDst = ubDstPong;
        ubSrc = ubSrcPong;
    }
    if (tail > 0) {
        uint16_t tailNBurst = static_cast<uint16_t>(tail / nCoreUse);
        CopyCast<T>(ubSrc, ubDst, src, dst, repeat * nBurst * n, tailNBurst, static_cast<uint16_t>(nCoreUse),
            n - nCoreUse, pingpongEventId);
    }
    return;
}

template <typename T>
__aicore__ inline void UnAlignedCast32to16V220(__gm__ T *dst, __gm__ float *src, uint32_t offset, uint32_t size,
    TBuf<TPosition::VECCALC> &tmpBuf)
{
    uint32_t dataSize = TOTAL_UB_SIZE / NUM_TWO / sizeof(float);
    uint32_t dataSize1 = TOTAL_UB_SIZE / NUM_TWO / sizeof(T);
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<T> ubDstPong = ubDstPing[dataSize1];
    LocalTensor<float> ubSrcPong = ubSrcPing[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc = ubSrcPing;

    GlobalTensor<float> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), size);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst), size);

    uint32_t repeat = size / dataSize;
    uint32_t tail = size % dataSize;

    uint8_t pingpongEventId = 0;

    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));

    for (uint32_t i = 0; i < repeat; ++i) {
        if ((i & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            ubDst = ubDstPing;
            ubSrc = ubSrcPing;
        } else {
            pingpongEventId = 1;
            ubDst = ubDstPong;
            ubSrc = ubSrcPong;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
        CopyCast<T>(ubSrc, ubDst, src, dst, i * dataSize, 1, dataSize, 0, pingpongEventId);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    if ((repeat & CONTROL_DB) == 0) {
        pingpongEventId = 0;
        ubDst = ubDstPing;
        ubSrc = ubSrcPing;
    } else {
        pingpongEventId = 1;
        ubDst = ubDstPong;
        ubSrc = ubSrcPong;
    }
    if (tail > 0) {
        CopyCast<T>(ubSrc, ubDst, src, dst, repeat * dataSize, 1, tail, 0, pingpongEventId);
    }
    return;
}

#endif


template <typename T1, typename T2>
__aicore__ inline void CopyRemovePad(const GlobalTensor<T2>& outputGlobal, const GlobalTensor<T1>& inputGlobal,
    const LocalTensor<T1>& srcUb, const LocalTensor<T2>& castDstUb, uint32_t nBurst, uint32_t inputWidth,
    uint32_t outputWidth)
{
    CopyGmToUbufAlign<T1>((__ubuf__ void*)srcUb.GetPhyAddr(), (__gm__ void*)inputGlobal.GetPhyAddr(), 0,
                          static_cast<uint16_t>(nBurst), inputWidth * sizeof(T1), 0, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    Cast(castDstUb, srcUb, RoundMode::CAST_RINT, nBurst * inputWidth);
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    uint32_t srcGap = (inputWidth - outputWidth) * sizeof(T2) / 32; // 32 is blocksize
    CopyUbufToGmAlign<T2>((__gm__ void*)outputGlobal.GetPhyAddr(), (__ubuf__ void*)castDstUb.GetPhyAddr(), 0,
        static_cast<uint16_t>(nBurst), outputWidth * sizeof(T2), 0, 0, srcGap, 0);
}

template <typename T1, typename T2>
__aicore__ inline void RemovePaddingImpl(GlobalTensor<T2> outputGlobal, GlobalTensor<T1> inputGlobal,
    uint32_t height, uint32_t width, uint32_t outputWidth, TBuf<TPosition::VECCALC> &tmpBuf)
{
    LocalTensor<T1> srcUb = tmpBuf.Get<T1>();
    LocalTensor<T2> castDstUb = srcUb.template ReinterpretCast<T2>();

    uint32_t nBurst = TOTAL_UB_SIZE / (width * sizeof(T1));
    if (nBurst == 0) {
        uint32_t maxWidthLen = TOTAL_UB_SIZE / sizeof(T1);
        uint32_t castTimes = width / maxWidthLen;
        uint32_t tailWidth = width - castTimes * maxWidthLen;
        uint32_t tailOutWidth = outputWidth - castTimes * maxWidthLen;
        for (uint32_t mIndex = 0; mIndex < height; ++mIndex) {
            set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
            for (uint32_t index = 0; index < castTimes; ++index) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
                UnAlignedCast32to16V220((__gm__ T2 *)(outputGlobal[index * maxWidthLen + mIndex * outputWidth].GetPhyAddr()),
                               (__gm__ float *)(inputGlobal[index * maxWidthLen + mIndex * width].GetPhyAddr()),
                               0, maxWidthLen, tmpBuf);
                set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
            }
            wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
            if (tailWidth != 0) {
                CopyRemovePad(outputGlobal[castTimes * maxWidthLen + mIndex * outputWidth],
                              inputGlobal[castTimes * maxWidthLen + mIndex * width], srcUb, castDstUb,
                              1, tailWidth, tailOutWidth);
            }
        }
        return;
    }
    uint32_t nBurstTimes = height / nBurst;
    uint32_t nBurstTail = height - nBurstTimes * nBurst;
    set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    for (uint32_t i = 0; i < nBurstTimes; ++i) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
        CopyRemovePad(outputGlobal[i * nBurst * outputWidth], inputGlobal[i * nBurst * width], srcUb, castDstUb,
                      nBurst, width, outputWidth);
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    if (nBurstTail > 0) {
        CopyRemovePad(outputGlobal[nBurstTimes * nBurst * outputWidth], inputGlobal[nBurstTimes * nBurst * width],
                      srcUb, castDstUb, nBurstTail, width, outputWidth);
    }
}

#endif // __OP_KERNEL_MATMUL_V3_H__
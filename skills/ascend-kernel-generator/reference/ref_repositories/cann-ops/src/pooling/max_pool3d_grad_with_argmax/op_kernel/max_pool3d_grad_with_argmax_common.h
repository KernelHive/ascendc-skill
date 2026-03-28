/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file max_pool3d_grad_with_argmax_common.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_COMMON
#define MAX_POOL_GRAD3D_WITH_ARGMAX_COMMON
#include "kernel_tiling/kernel_tiling.h"


namespace MaxPool3DGradWithArgmaxComm {
using namespace AscendC;

constexpr uint64_t VL_FP32 = 64;
constexpr uint64_t VL_FP16 = 128;
constexpr uint64_t MAX_LIST_NUM = 4;
constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BLOCK_NUM_16 = BLOCK_SIZE / sizeof(half);
constexpr uint64_t BLOCK_NUM_32 = BLOCK_SIZE / sizeof(float);
constexpr uint64_t LARGE_KERNEL = 0;
constexpr uint64_t LARGE_HO = 1;
constexpr uint64_t LARGE_WO = 2;
constexpr uint64_t UINT16_BITS = 16;
constexpr uint64_t UINT8_BITS = 8;
constexpr float ZERO = 0.0f;
constexpr uint64_t B32_VECTOR_MASK = 64;
constexpr uint64_t B16_VECTOR_MASK = 128;
const uint64_t NUM_TWO = 2;
const uint64_t BITSIZE = 16;

constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t UNIT_BLOCK_LEN = 32;

struct BlockParams {
    uint64_t ncCntIndex = 0;
    uint64_t doCntIndex = 0;
    uint64_t hoCntIndex = 0;
    uint64_t woCntIndex = 0;
    uint64_t ncShape = 0;
    uint64_t doShape = 0;
    uint64_t hoShape = 0;
    uint64_t woShape = 0;
    uint64_t diShape = 0;
    uint64_t hiShape = 0;
    uint64_t wiShape = 0;
    uint64_t wiShapeAlign = 0;
    uint64_t diValid = 0;
    uint64_t hiValid = 0;
    uint64_t wiValid = 0;
    uint64_t dohowoShape = 0;
    uint64_t dihiwiAlign = 0;
    uint64_t dohowoAlign8 = 0;
    uint64_t dohowoAlign16 = 0;
    uint64_t offsetX = 0;
    uint64_t offsetGrad = 0;
    uint64_t offsetArgmax = 0;
    uint64_t offsetY = 0;
    uint64_t offsetYD = 0;
    uint64_t offsetYH = 0;
    uint64_t offsetYW = 0;
    uint64_t padDTop = 0;
    uint64_t padHTop = 0;
    uint64_t padWTop = 0;
    uint64_t padDBottom = 0;
    uint64_t padHBottom = 0;
    uint64_t padWBottom = 0;
    uint64_t baseNcOffset = 0;
    uint64_t ShapeSum = 0;
};

struct TilingParams {
    uint64_t ncDim;
    uint64_t diDim;
    uint64_t hiDim;
    uint64_t wiDim;
    uint64_t doDim;
    uint64_t hoDim;
    uint64_t woDim;
    uint64_t kd;
    uint64_t kh;
    uint64_t kw;
    uint64_t sd;
    uint64_t sh;
    uint64_t sw;
    uint64_t padDTop;
    uint64_t padDBottom;
    uint64_t padHTop;
    uint64_t padHBottom;
    uint64_t padWTop;
    uint64_t padWBottom;
    uint64_t singleCoreNc;
    uint64_t singleCoreDo;
    uint64_t singleCoreHo;
    uint64_t singleCoreWo;
    uint64_t baseNc;
    uint64_t baseDo;
    uint64_t baseHo;
    uint64_t baseWo;
    uint64_t baseDoHoWo;
    uint64_t baseDoHoWoAlign8;
    uint64_t baseDoHoWoAlign16;
    uint64_t baseDi;
    uint64_t baseHi;
    uint64_t baseWi;
    uint64_t baseWiAlign;
    uint64_t baseDiHiWi;
    uint64_t baseDiHiWiAlign8;
    uint64_t baseDiHiWiAlign16;
    uint64_t baseDiHiWiAlign;
    uint64_t ncCnt;
    uint64_t doCnt;
    uint64_t hoCnt;
    uint64_t woCnt;
    uint64_t ncTail;
    uint64_t doTail;
    uint64_t hoTail;
    uint64_t woTail;
    uint64_t totalCnt;
    uint64_t needInitOutput;
    uint64_t padGmOffset;
    uint64_t ubSize;
    uint64_t usedCoreNum;
    uint64_t preCoreNum;
    uint64_t round;
    uint64_t realRound;
    uint64_t outputDataSize;
    uint64_t ncIndex;
    uint64_t ncCntRound;
    uint64_t ncRealRound;
    uint64_t diHiWiLen;
    uint64_t initLen;
    uint64_t initOffset;
};

__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y) {
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename Tp, Tp v> struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename> struct is_same : public false_type {};
template <typename Tp> struct is_same<Tp, Tp> : public true_type {};

// only support float/int32_t
// [row, col] -> [col, row]: row:align16, col:align8
template <typename T>
__aicore__ inline void TransposeBase16M8(const LocalTensor<T> &dstUb, const LocalTensor<T> &srcUb,
                                         uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = rowNum;
        }

        TransDataTo5HD<T>(dstAddrList, srcAddrList, transDataParams);
    }
}

// only support float/int32_t
// [row, col] -> [col, row]: row:align8, col:align16
template <typename T>
__aicore__ inline void TransposeBase8M16(const LocalTensor<T> &dstUb, const LocalTensor<T> &srcUb,
                                         uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];

    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
          srcAddrList[i] =
              (uint64_t)(srcUb[r * TRANS_ADDR_LEN + i % BLOCK_NUM_32 * colNum + i / BLOCK_NUM_32 * BLOCK_NUM_32]
                             .GetPhyAddr());
          dstAddrList[i] =
              (uint64_t)(dstUb[r * TRANS_ADDR_LEN * rowNum + (i % 2 * BLOCK_NUM_32 + i / 2) * rowNum].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = rowNum / BLOCK_NUM_32;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = colNum;
            transDataParams.dstRepStride = 1;
        }

        TransDataTo5HD<T>(dstAddrList, srcAddrList, transDataParams);
    }
}

// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, col:align16
template <typename T>
__aicore__ inline void TransposeBase16M16(const LocalTensor<T> &dstUb, const LocalTensor<T> &srcUb,
                                          uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_16;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = rowNum;
        }
        // TransData5HD does not support bfloat16, only use half
        TransDataTo5HD<half>(dstAddrList, srcAddrList, transDataParams);
    }
}

// only support float/int32_t
// [row, col] -> [col, row]: row:align16,max:64, col:align8
__aicore__ inline void TransposeAddrBase16M8(uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                             uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                             uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = rowNum;
    }

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        TransDataTo5HD<int32_t>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

// only support float/int32_t
// [row, col] -> [col, row]: row:align8, col:align16,max:64
__aicore__ inline void TransposeAddrBase8M16(uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                             uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                             uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = rowNum / BLOCK_NUM_32;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = colNum;
        transDataParams.dstRepStride = 1;
    }
    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        TransDataTo5HD<int32_t>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, max:64, col:align16
__aicore__ inline void TransposeAddrBase16M16(uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                              uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                              uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = colNum / BLOCK_NUM_16;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = rowNum;
    }

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        // TransData5HD does not support bfloat16, only use half
        TransDataTo5HD<half>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, col:align16, max:64
__aicore__ inline void TransposeAddrBase16M16B(uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                               uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
                                               uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = rowNum / BLOCK_NUM_16;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = colNum;
        transDataParams.dstRepStride = 1;
    }

    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        // TransData5HD does not support bfloat16, only use half
        TransDataTo5HD<half>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

} // namespace MaxPool3DGradWithArgmaxComm

#endif // MAX_POOL_GRAD3D_WITH_ARGMAX_COMMON
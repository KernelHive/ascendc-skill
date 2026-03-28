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
 * \file adaptive_max_pool3d_grad_common.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_COMMON_H
#define ADAPTIVE_MAX_POOL3D_GRAD_COMMON_H
#include "kernel_tiling/kernel_tiling.h"
 
 
namespace AdaptiveMaxPool3DGradComm {
using namespace AscendC;
 
constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t UINT8_BITS = 8;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BLOCK_NUM_16 = BLOCK_SIZE / sizeof(half);
constexpr uint64_t BLOCK_NUM_32 = BLOCK_SIZE / sizeof(float);
constexpr uint64_t UNIT_BLOCK_LEN = BLOCK_SIZE / sizeof(float);
constexpr float ZERO = 0.0f;
 
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
    float coeffD = 0.0;
    float coeffH = 0.0;
    float hwDims = 0.0;
    uint64_t maxKwAlign8 = 0;
    uint64_t maxKwAlign16 = 0;
    uint64_t maxKwAlignDtype = 0;
    uint64_t dihiwiAlign = 0;
    uint64_t diValid = 0;
    uint64_t hiValid = 0;
    uint64_t wiValid = 0;
    uint64_t dohowoShape = 0;
    uint64_t dohowoAlign8 = 0;
    uint64_t dohowoAlign16 = 0;
    uint64_t offsetX = 0;
    uint64_t offsetGrad = 0;
    uint64_t offsetArgmax = 0;
    uint64_t offsetY = 0;
    uint64_t startD = 0;
    uint64_t startH = 0;
    uint64_t startW = 0;
    uint64_t deltaD = 0;
    uint64_t deltaH = 0;
    uint64_t deltaW = 0;
    uint64_t kernelShape = 0;
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
    uint64_t singleCoreNc;
    uint64_t singleCoreDo;
    uint64_t singleCoreHo;
    uint64_t singleCoreWo;
    uint64_t baseNc;
    uint64_t baseDo;
    uint64_t baseHo;
    uint64_t baseWo;
    uint64_t ncCnt;
    uint64_t dCnt;
    uint64_t hCnt;
    uint64_t wCnt;
    uint64_t baseNcTail;
    uint64_t ncTail;
    uint64_t doTail;
    uint64_t hoTail;
    uint64_t woTail;
    uint64_t totalCnt;
    uint64_t maxKd;
    uint64_t maxKh;
    uint64_t maxKw;
    uint64_t maxKdhwLen;
    uint64_t needInitOutput;
    uint64_t usedCoreNum;
    uint64_t preCoreNum;
    uint64_t round;
    uint64_t realRound;
    uint64_t addMode;
    uint64_t ncIndex;
    uint64_t ncCntRound;
    uint64_t ncRealRound;
    uint64_t diHiWiLen;
    uint64_t ubSize;
    uint64_t initLen;
    uint64_t initOffset;
};
 
template <typename Tp, Tp v> struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename> struct is_same : public false_type {};
template <typename Tp> struct is_same<Tp, Tp> : public true_type {};
 
__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}
 
__aicore__ inline uint64_t FloorDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (uint64_t) (x / y);
}
 
// only support float/int32_t
// [row, col] -> [col, row]: row:align16, col:align8
template <typename T>
__aicore__ inline void TransposeBase16M8(LocalTensor<T> &dstUb, LocalTensor<T> &srcUb, uint64_t rowNum, uint64_t colNum)
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
 
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}
 
// only support float/int32_t
// [row, col] -> [col, row]: row:align8, col:align16
template <typename T>
__aicore__ inline void TransposeBase8M16(LocalTensor<T> &dstUb, LocalTensor<T> &srcUb, uint64_t rowNum, uint64_t colNum)
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
 
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}
 
// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, col:align16
template <typename T>
__aicore__ inline void TransposeBase16M16(LocalTensor<T> &dstUb, LocalTensor<T> &srcUb, uint64_t rowNum, uint64_t colNum)
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
        TransDataTo5HD<half>(dstAddrList, srcAddrList, transDataParams);
    }
}
 
} // namespace AdaptiveMaxPool3DGradComm
 
#endif // ADAPTIVE_MAX_POOL3D_GRAD_COMMON_H
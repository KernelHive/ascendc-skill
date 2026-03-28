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
 * \file pad_v4_grad_base.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_BASE_H_
#define _PAD_V4_GRAD_BASE_H_

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PADDING_INPUT_INDEX = 2;
constexpr int32_t Y_OUTPUT_INDEX = 0;
constexpr int32_t BUFFER_APPLY_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t COPY_LOOP = 16;
constexpr uint32_t HALF_BLOCK_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_NUM = 8;
constexpr uint32_t CAL_COUNT = 32;
constexpr uint32_t W_PAD_LOWER_LIMIT = 16;
constexpr uint32_t COPY_ROWS_AND_COLS = 16;
constexpr uint32_t MINI_SHAPE_MAX_ROWS = 128;
constexpr uint32_t TRANSDATA_BASE_H = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t SMALL_WIDTH_LIMIT = 128;
constexpr uint32_t SMALL_HEIGHT_LIMIT = 64;

using namespace AscendC;

template <typename T>
class PadV4GradBase {
   public:
    __aicore__ inline PadV4GradBase(){};
    __aicore__ inline void Init(const PadV4GradTilingData &__restrict tilingData, GM_ADDR x, GM_ADDR padding,
                                GM_ADDR y, GM_ADDR workspace);
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        if (b == (T2)0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b) {
        if (b == (T2)0) {
            return a;
        }
        return (a + b - 1) / b * b;
    };

   public:
    uint32_t batch = 0;
    uint32_t ncPerCore = 0;
    uint32_t tailNC = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    uint32_t hPad1 = 0;
    uint32_t hPad2 = 0;
    uint32_t wPad1 = 0;
    uint32_t wPad2 = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
    uint32_t wPadCopyCount = 0;
    uint64_t workspacePerCore = 0;
    int64_t batchStride = 0;
    int64_t outBatchStride = 0;
    uint32_t loopNC = 0;
    int64_t ncOffset = 0;

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T>
__aicore__ inline void PadV4GradBase<T>::Init(const PadV4GradTilingData &__restrict tilingData, GM_ADDR x,
                                              GM_ADDR padding, GM_ADDR y, GM_ADDR workspace) {
    batch = tilingData.batch;
    ncPerCore = tilingData.ncPerCore;
    tailNC = tilingData.tailNC;
    height = tilingData.height;
    width = tilingData.width;
    outHeight = tilingData.outHeight;
    outWidth = tilingData.outWidth;
    alignHeight = tilingData.alignHeight;
    alignWidth = tilingData.alignWidth;
    alignOutHeight = tilingData.alignOutHeight;
    alignOutWidth = tilingData.alignOutWidth;
    hPad1 = tilingData.hPad1;
    hPad2 = tilingData.hPad2;
    wPad1 = tilingData.wPad1;
    wPad2 = tilingData.wPad2;
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
    wPadCopyCount = tilingData.wPadCopyCount;
    workspacePerCore = tilingData.workspacePerCore / sizeof(T);
    batchStride = height * width;
    outBatchStride = outHeight * outWidth;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }
    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
    mGmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
}
#endif  // _PAD_V4_GRAD_BASE_H_
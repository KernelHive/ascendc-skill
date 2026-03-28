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
 * \file circular_pad_common.h
 * \brief
 */
#ifndef CIRCULAR_PAD_COMMON_H
#define CIRCULAR_PAD_COMMON_H
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;

struct sDataCopyExtParams
{
    DataCopyExtParams paramsIn = {0, 0, 0, 0, 0};
    DataCopyExtParams paramsOut = {0, 0, 0, 0, 0};
};
struct CopyParams
{
    __aicore__ inline CopyParams() {};
    __aicore__ inline CopyParams(int64_t offset, int64_t strideLoop, DataCopyExtParams dcParams) :
                                 offset(offset), strideLoop(strideLoop), dcParams(dcParams) {};
    int64_t offset{0};
    int64_t strideLoop{0};
    int64_t stridePage{0};
    DataCopyExtParams dcParams = {0, 0, 0, 0, 0};
};

class CircularPadCommon
{
public:
    __aicore__ inline CircularPadCommon(TPipe* pipe) : pipe_(pipe) {};
    
    __aicore__ inline void InitCommon(const CircularPadCommonTilingData& tiling_data, int64_t T1Size, int64_t T2Size)
    {
        GetTiling(tiling_data);
        if (T2Size == 0) {
            return;
        }
        Align_ = BLOCK_SIZE / T2Size;
        inputWAlign_ = GetAlign(inputW_, T1Size);
        outputWAlign_ = GetAlign(outputW_, T1Size);
        inputLen_ = inputH_ * inputW_;
        outputLen_ = outputH_ * outputW_;

        pLeft_ = GetPositive(left_);
        pRight_ = GetPositive(right_);
        pTop_ = GetPositive(top_);
        pBottom_ = GetPositive(bottom_);
        nLeft_ = GetNegtive(left_);
        nRight_ = GetNegtive(right_);
        nTop_ = GetNegtive(top_);
        nBottom_ = GetNegtive(bottom_);

        leftAlign_ = GetAlign(pLeft_, T2Size);
        rightAlign_ = GetAlign(pRight_, T2Size);
    }

    __aicore__ inline void GetTiling(const CircularPadCommonTilingData& tiling_data)
    {
        inputH_ = tiling_data.inputH;
        inputW_ = tiling_data.inputW;
        outputH_ = tiling_data.outputH;
        outputW_ = tiling_data.outputW;
        left_ = tiling_data.left;
        right_ = tiling_data.right;
        top_ = tiling_data.top;
        bottom_ = tiling_data.bottom;
        perCoreTaskNum_ = tiling_data.perCoreTaskNum;
        workspaceLen_ = tiling_data.workspaceLen;
        tailTaskNum_ = tiling_data.tailTaskNum;
        workspaceLen_ = tiling_data.workspaceLen;
    }
    
    __aicore__ inline int64_t GetAlign(int64_t len, int64_t size)
    {
        if (size == 0) {
            return 0;
        }
        return (len * size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / size;
    }

    __aicore__ inline int64_t GetPositive(int64_t len) {
        return len > 0 ? len : 0;
    }

    __aicore__ inline int64_t GetNegtive(int64_t len) {
        return len < 0 ? len : 0;
    }

    __aicore__ inline void MTE3ToMTE2Sync() {
        event_t eventId3To2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId3To2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId3To2);
    }

protected:
    TPipe* pipe_;
    int64_t inputH_{0};
    int64_t inputW_{0};
    int64_t outputH_{0};
    int64_t outputW_{0};
    int64_t left_{0};
    int64_t right_{0};
    int64_t top_{0};
    int64_t bottom_{0};
    int64_t perCoreTaskNum_{0};
    int64_t tailTaskNum_{0};
    int64_t workspaceLen_{0};

    uint8_t Align_{0};
    int64_t inputLen_{0};
    int64_t inputWAlign_{0};
    int64_t outputWAlign_{0};
    int64_t outputLen_{0};
    int64_t inOutputH_{0};
    int64_t inOutputW_{0};
    int64_t inOutputWAlign_{0};
    int64_t inOutputW32Align_{0};
    int64_t leftAlign_{0};
    int64_t rightAlign_{0};
    int64_t pLeft_{0};
    int64_t pRight_{0};
    int64_t pTop_{0};
    int64_t pBottom_{0};
    int64_t nLeft_{0};
    int64_t nRight_{0};
    int64_t nTop_{0};
    int64_t nBottom_{0};
};
#endif
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
 * \file circular_pad_2d.h
 * \brief
 */
#ifndef CIRCULAR_PAD_2D_H
#define CIRCULAR_PAD_2D_H
#include "circular_pad.h"
using namespace AscendC;

template <typename T>
class CircularPad2D : public CircularPad<T>
{
public:
    __aicore__ inline CircularPad2D(TPipe* pipe) : CircularPad<T>(pipe) {};

    __aicore__ inline void Init2D(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, 
                                  const CircularPadCommonTilingData& __restrict tiling_data)
    {
        this->Init(tiling_data);

        uint32_t blockId = static_cast<uint32_t>(GetBlockIdx());
        uint32_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < this->tailTaskNum_) {
            this->perCoreTaskNum_ += 1;
            startIdx += blockId;
        } else {
            startIdx += this->tailTaskNum_;
        }
        this->xGM_.SetGlobalBuffer((__gm__ T*)x + this->inputLen_ * startIdx,
            this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer((__gm__ T*)y + this->outputLen_ * startIdx,
            this->outputLen_ * this->perCoreTaskNum_);
        this->workspaceGM_.SetGlobalBuffer((__gm__ T*)workspace + this->workspaceLen_ * startIdx,
            this->workspaceLen_ * this->perCoreTaskNum_);
    }

    __aicore__ inline void ProcessSmallShape()
    {
        if (this->hasLR) {
            this->template PadLeftAndRightSmallShape<true>();
            this->MTE3ToMTE2Sync();
            CopyToOutSmallShape<true>();
        } else {
            this->template PadLeftAndRightSmallShape<false>();
            this->MTE3ToMTE2Sync();
            CopyToOutSmallShape<false>();
        }
    }

    __aicore__ inline void ProcessBigShape()
    {
        if (this->hasLR) {
            this->template PadLeftAndRightBigShape<true>();
        } else {
            this->template PadLeftAndRightBigShape<false>();
        }
        this->MTE3ToMTE2Sync();
        CopyToOutBigShape();
    }

private:
    /************************************小shape***************************************************/
    template <bool HASLR>
    __aicore__ inline void CopyToOutSmallShape()
    {
        CopyParams copyParamsIn;
        CopyParams copyParamsOut;
        if constexpr (HASLR) {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>((this->inOutputWAlign_ + this->leftAlign_ + 
                                                        this->rightAlign_ - this->outputW_) * this->TSize_);
            for (uint32_t i = 0; i < this->perCoreTaskNum_; i++) {
                copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                copyParamsOut.dcParams  = {blockCount, blockLen, 0, 0, 0};
                copyParamsIn.offset = i * this->workspaceLen_ + this->leftAlign_ - this->pLeft_;
                copyParamsOut.offset = i * this->outputLen_ + this->pTop_ * this->outputW_;
                this->CopyToOutSmallShapeOnePage(this->workspaceGM_, i, copyParamsIn, copyParamsOut);
            }
        } else {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>((this->inputW_ - this->outputW_) * this->TSize_);
            for (uint32_t i = 0; i < this->perCoreTaskNum_; i++) {
                copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                copyParamsOut.dcParams  = {blockCount, blockLen, 0, 0, 0};
                copyParamsIn.offset = i * this->inputLen_ - this->nTop_ * this->inputW_ - this->nLeft_;
                copyParamsOut.offset = i * this->outputLen_ + this->pTop_ * this->outputW_;
                this->CopyToOutSmallShapeOnePage(this->xGM_, i, copyParamsIn, copyParamsOut);
            }
        }
    }

    /************************************大shape***************************************************/
    __aicore__ inline void CopyToOutBigShape()
    {
        for (uint32_t i = 0; i < this->perCoreTaskNum_; i++) {
            this->CopyToOutBigShapeOnePage(i, i);
        }
    }
};
#endif
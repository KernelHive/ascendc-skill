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
 * \file circular_pad_3d.h
 * \brief
 */
#ifndef CIRCULAR_PAD_3D_H
#define CIRCULAR_PAD_3D_H
#include "circular_pad.h"
using namespace AscendC;

template <typename T>
class CircularPad3D : public CircularPad<T>
{
public:
    __aicore__ inline CircularPad3D(TPipe* pipe) : CircularPad<T>(pipe) {};

    __aicore__ inline void Init3D(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, 
                                const CircularPadCommonTilingData& __restrict tiling_data)
    {
        this->Init(tiling_data);
        
        front_ = tiling_data.front;
        back_ = tiling_data.back;
        inputL_ = tiling_data.inputL;
        outputL_ = tiling_data.outputL;

        pFront_ = this->GetPositive(front_);
        pBack_ = this->GetPositive(back_);
        nFront_ = this->GetNegtive(front_);
        nBack_ = this->GetNegtive(back_);
        inOutputL_ = inputL_ + nFront_ + nBack_;

        int32_t blockId = static_cast<uint32_t>(GetBlockIdx());
        int32_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < (this->tailTaskNum_ / inputL_)) {
            this->perCoreTaskNum_ += inputL_;
            startIdx += blockId * inputL_;
        } else {
            startIdx += this->tailTaskNum_;
        }

        this->xGM_.SetGlobalBuffer((__gm__ T*)x + this->inputLen_ * startIdx,
            this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer((__gm__ T*)y + this->outputLen_ * (startIdx * outputL_ / inputL_),
            this->outputLen_ * (this->perCoreTaskNum_ * outputL_ / inputL_));
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
        this->MTE3ToMTE2Sync();
        PadFrontAndBack();
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
        this->MTE3ToMTE2Sync();
        PadFrontAndBack();
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

            for (uint32_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
                for(int32_t pageIdx = 0; pageIdx < inOutputL_; pageIdx++){
                    copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                    copyParamsOut.dcParams  = {blockCount, blockLen, 0, 0, 0};
                    int64_t pageIdxIn = batchIdx * inputL_ + pageIdx - nFront_;
                    int64_t pageIdxOut = batchIdx * outputL_ + pageIdx + pFront_;
                    copyParamsIn.offset = pageIdxIn * this->workspaceLen_ + this->leftAlign_ - this->pLeft_;
                    copyParamsOut.offset = pageIdxOut * this->outputLen_ + this->pTop_ * this->outputW_;
                    this->CopyToOutSmallShapeOnePage(this->workspaceGM_, pageIdxOut, copyParamsIn, copyParamsOut);
                }
            }
        } else {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>((this->inputW_ - this->outputW_) * this->TSize_);
            
            for (uint32_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
                for(int32_t pageIdx = 0; pageIdx < inOutputL_; pageIdx++){
                    copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                    copyParamsOut.dcParams  = {blockCount, blockLen, 0, 0, 0};
                    int64_t pageIdxIn = batchIdx * inputL_ + pageIdx - nFront_;
                    int64_t pageIdxOut = batchIdx * outputL_ + pageIdx + pFront_;
                    copyParamsIn.offset = pageIdxIn * this->inputLen_ - this->nTop_ * this->inputW_ - this->nLeft_;
                    copyParamsOut.offset = pageIdxOut * this->outputLen_ + this->pTop_ * this->outputW_;
                    this->CopyToOutSmallShapeOnePage(this->xGM_, pageIdxOut, copyParamsIn, copyParamsOut);
                }
            }
        }
    }

    /************************************大shape***************************************************/
    __aicore__ inline void CopyToOutBigShape()
    {
        for (uint32_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
            for(int32_t pageIdx = 0; pageIdx < inOutputL_; pageIdx++){
                int64_t pageIdxIn = batchIdx * inputL_ + pageIdx - nFront_;
                int64_t pageIdxOut = batchIdx * outputL_ + pageIdx + pFront_;
                this->CopyToOutBigShapeOnePage(pageIdxIn, pageIdxOut);
            }
        }
    }

    __aicore__ inline void PadFrontAndBack()
    {
        int64_t stride = this->outputLen_ * outputL_;
        if (front_ > 0) {
            int64_t offsetIn = (outputL_ - pBack_ - front_) * this->outputLen_;
            int64_t offsetOut = 0;
            this->CopyGmToGm(front_, this->perCoreTaskNum_ / inputL_, offsetIn, offsetOut, stride);
        }
        if (back_ > 0) {
            int64_t offsetIn = pFront_ * this->outputLen_;
            int64_t offsetOut = (outputL_ - pBack_) * this->outputLen_;
            this->CopyGmToGm(back_, this->perCoreTaskNum_ / inputL_, offsetIn, offsetOut, stride);
        }
    }

private:
    int64_t inputL_{0};
    int64_t inOutputL_{0};
    int64_t outputL_{0};
    int64_t front_{0};
    int64_t back_{0};
    int64_t pFront_{0};
    int64_t pBack_{0};
    int64_t nFront_{0};
    int64_t nBack_{0};
};
#endif
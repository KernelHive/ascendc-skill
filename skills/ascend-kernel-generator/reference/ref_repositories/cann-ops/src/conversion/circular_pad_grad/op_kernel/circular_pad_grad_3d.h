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
 * \file circular_pad_grad_3d.h
 * \brief
 */
#ifndef CIRCULAR_PAD_GRAD_3D_H
#define CIRCULAR_PAD_GRAD_3D_H
#include "circular_pad_grad.h"
using namespace AscendC;

template <typename T1, typename T2, bool ISCAST = false>
class CircularPadGrad3D : public CircularPadGrad<T1, T2, ISCAST>
{
public:
    __aicore__ inline CircularPadGrad3D(TPipe* pipe) : CircularPadGrad<T1, T2, ISCAST>(pipe) {};

    __aicore__ inline void Init3D(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, 
                                const CircularPadCommonTilingData& tiling_data)
    {
        this->Init(tiling_data);
        
        front_ = tiling_data.front;
        back_ = tiling_data.back;
        inputL_ = tiling_data.inputL;
        outputL_ = tiling_data.outputL;

        pFront_ = this->GetPositive(front_);
        pBack_ = this->GetPositive(back_);

        int32_t blockId = static_cast<uint32_t>(GetBlockIdx());
        int32_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < (this->tailTaskNum_ / inputL_)) {
            this->perCoreTaskNum_ += inputL_;
            startIdx += blockId * inputL_;
        } else {
            startIdx += this->tailTaskNum_;
        }

        this->xGM_.SetGlobalBuffer((__gm__ T1*)x + this->inputLen_ * startIdx,
            this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer((__gm__ T1*)y + this->outputLen_ * (startIdx * outputL_ / inputL_),
            this->outputLen_ * (this->perCoreTaskNum_ * outputL_ / inputL_));
        this->workspaceGM_.SetGlobalBuffer((__gm__ T2*)workspace + this->workspaceLen_ * startIdx,
            this->workspaceLen_ * this->perCoreTaskNum_);
        if (this->left_ < 0 || this->right_ < 0 || this->top_ < 0 ||
            this->bottom_ < 0 || front_ < 0 || back_ < 0) {
            this->SetGMtoZero(this->outputLen_ * (this->perCoreTaskNum_ * outputL_ / inputL_));
        }
    }

    __aicore__ inline void ProcessSmallShape()
    {
        this->AddTopAndBottomSmallShape();
        this->MTE3ToMTE2Sync();
        this->AddLeftAndRightSmallShape();
        this->MTE3ToMTE2Sync();
        CopyToOutSmallShape();
    }

    __aicore__ inline void ProcessBigShape()
    {
        this->AddTopAndBottomBigShape();
        this->MTE3ToMTE2Sync();
        this->AddLeftAndRightBigShape();
        this->MTE3ToMTE2Sync();
        CopyToOutBigShapeBig();
    }

private:
    /************************************小shape***************************************************/
    __aicore__ inline void CopyToOutSmallShape()
    {
        AddFrontAndBack();
        this->MTE3ToMTE2Sync();

        sDataCopyExtParams params;
        for (int64_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
            for(int64_t pageIdx = pFront_; pageIdx < inputL_ - pBack_; pageIdx++){
                this->CalculateOutParms(params);
                int64_t i_out = this->getNewBatchOffset(pageIdx, inputL_, front_, back_) + batchIdx * outputL_;
                this->CopyToOutSmallShapeOnePage(batchIdx * inputL_ + pageIdx, i_out, params);
            }
        }
    }

    /************************************大shape重叠pad**********************************************/
    __aicore__ inline void CopyToOutBigShapeBig()
    {
        AddFrontAndBack();
        this->MTE3ToMTE2Sync();

        sDataCopyExtParams params;
        for (int64_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
            for(int64_t pageIdx = pFront_; pageIdx < inputL_ - pBack_; pageIdx++){
                this->CalculateOutParms(params);
                int64_t i_out = this->getNewBatchOffset(pageIdx, inputL_, front_, back_) + batchIdx * outputL_;
                this->CopyToOutBigShapeOnePage(batchIdx * inputL_ + pageIdx, i_out, params);
            }
        }
    }

    __aicore__ inline void AddFrontAndBack()
    {
        int64_t stride = this->workspaceLen_ * inputL_;
        if (front_ > 0) {
            int64_t offsetIn = 0;
            int64_t offsetOut = (inputL_ - pBack_ - front_) * this->workspaceLen_;
            SetAtomicAdd<T2>();
            this->CopyGmToGm(front_, this->perCoreTaskNum_ / inputL_, offsetIn, offsetOut, stride);
            SetAtomicNone();
        }
        this->MTE3ToMTE2Sync();
        if (back_ > 0) {
            int64_t offsetIn = (inputL_ - pBack_) * this->workspaceLen_;
            int64_t offsetOut = pFront_ * this->workspaceLen_;
            SetAtomicAdd<T2>();
            this->CopyGmToGm(back_, this->perCoreTaskNum_ / inputL_, offsetIn, offsetOut, stride);
            SetAtomicNone();
        }
    }

private:
    int64_t inputL_{0};
    int64_t outputL_{0};
    int64_t front_{0};
    int64_t back_{0};
    int64_t pFront_{0};
    int64_t pBack_{0};
};
#endif
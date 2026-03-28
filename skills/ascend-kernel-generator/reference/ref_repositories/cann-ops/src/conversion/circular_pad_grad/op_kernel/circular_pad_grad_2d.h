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
 * \file circular_pad_grad_2d.h
 * \brief
 */
#ifndef CIRCULAR_PAD_GRAD_2D_H
#define CIRCULAR_PAD_GRAD_2D_H
#include "circular_pad_grad.h"
using namespace AscendC;

template <typename T1, typename T2, bool ISCAST = false>
class CircularPadGrad2D : public CircularPadGrad<T1, T2, ISCAST>
{
public:
    __aicore__ inline CircularPadGrad2D(TPipe* pipe) : CircularPadGrad<T1, T2, ISCAST>(pipe) {};

    __aicore__ inline void Init2D(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, 
                                const CircularPadCommonTilingData& tiling_data)
    {
        this->Init(tiling_data);

        int64_t blockId = GetBlockIdx();
        int64_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < this->tailTaskNum_) {
            this->perCoreTaskNum_ += 1;
            startIdx += blockId;
        } else {
            startIdx += this->tailTaskNum_;
        }

        this->xGM_.SetGlobalBuffer((__gm__ T1*)x + this->inputLen_ * startIdx,
            this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer((__gm__ T1*)y + this->outputLen_ * startIdx,
            this->outputLen_ * this->perCoreTaskNum_);
        this->workspaceGM_.SetGlobalBuffer((__gm__ T2*)workspace + this->workspaceLen_ * startIdx,
            this->workspaceLen_ * this->perCoreTaskNum_);
        if (this->left_ < 0 || this->right_ < 0 || this->top_ < 0 || this->bottom_ < 0) {
            this->SetGMtoZero(this->outputLen_ * this->perCoreTaskNum_);
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
        sDataCopyExtParams params;
        for (int32_t i = 0; i < this->perCoreTaskNum_; i++) {
            this->CalculateOutParms(params);
            this->CopyToOutSmallShapeOnePage(i, i, params);
        }
    }

    /************************************大shape**********************************************/
    __aicore__ inline void CopyToOutBigShapeBig()
    {
        sDataCopyExtParams params;

        for (int64_t i = 0; i < this->perCoreTaskNum_; i++) {
            this->CalculateOutParms(params);
            this->CopyToOutBigShapeOnePage(i, i, params);
        }
    }
};
#endif
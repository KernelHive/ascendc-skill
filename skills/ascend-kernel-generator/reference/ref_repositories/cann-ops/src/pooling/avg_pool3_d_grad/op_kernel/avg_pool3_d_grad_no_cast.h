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
 * \file avg_pool3_d_grad_no_cast.h
 * \brief
 */
#ifndef AVG_POOL3D_GRAD_NO_CAST_H
#define AVG_POOL3D_GRAD_NO_CAST_H

#include "kernel_operator.h"
#include "avg_pool3_d_grad_base.h"

namespace AvgPool3DGrad {
using namespace AscendC;

template <typename T>
class KernelAvgPool3DGradNoCast : public KernelAvgPool3DGradBase<T> {
public:
    __aicore__ inline KernelAvgPool3DGradNoCast() {}
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam &tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearOutput();
    __aicore__ inline void ProcessPerLine(int64_t index, int64_t ubIndex);
    __aicore__ inline void Compute(int64_t ubIndex);
    __aicore__ inline void CopyOut(int64_t d, int64_t h, int64_t w, int64_t nc, int64_t ubIndex);
};

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::Init(GM_ADDR grads, GM_ADDR output, 
                                                          const AvgPool3dGradTilingParam &tilingData, 
                                                          GM_ADDR workspace) 
{
    // Get all required params
    this->thisCoreIdx = GetBlockIdx();
    this->coreNum = GetBlockNum();
    this->ParseTilingData(tilingData);
    this->GetEventIds();

    // Init
    this->xGm.SetGlobalBuffer((__gm__ T*)grads);
    this->yGm.SetGlobalBuffer((__gm__ T*)output);
    this->pipe.InitBuffer(this->xInBuf, this->nLine * this->ncAlign * sizeof(T));
    this->pipe.InitBuffer(this->yOutBuf, this->nLine * this->ncAlign * sizeof(T));

    ClearOutput();

    this->outputGradUb = this->xInBuf.template Get<T>();
    this->inputGradUb = this->yOutBuf.template Get<T>();
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::ClearOutput() {
    auto yGmCount = this->inD * this->inH * this->inW * this->ncTotal;
    auto formerNum = yGmCount % this->coreNum;
    auto tailNum = this->coreNum - formerNum;
    auto formerCount = (yGmCount + this->coreNum - 1) / this->coreNum;
    auto tailCount = yGmCount / this->coreNum;
    if (this->thisCoreIdx < formerNum) {
        auto offset = this->thisCoreIdx * formerCount;
        InitOutput(this->yGm[offset], formerCount, static_cast<float>(0));
    } else {
        auto offset = formerNum * formerCount + (this->thisCoreIdx - formerNum) * tailCount;
        InitOutput(this->yGm[offset], tailCount, static_cast<float>(0));
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::Process() {
    auto processStart = this->normalCoreNCNum * this->thisCoreIdx;
    auto processEnd = processStart + this->thisCoreNum;

    if (this->kernelsOverlap) {
        SetAtomicAdd<T>();
    }

    SetFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
    SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
    for (auto index = processStart; index < processEnd; index += this->nLine) {
        auto indexStart = index;
        auto indexEnd = this->min(index + this->nLine, processEnd);
        WaitFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
        WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
        for (auto i = index; i < indexEnd; ++i) {
            auto ubIndex = i - index;
            this->ProcessPerLine(i, ubIndex);
        }
        SetFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
        SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
    }
    WaitFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
    WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);

    if (this->kernelsOverlap) {
        SetAtomicNone();
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::ProcessPerLine(int64_t index, int64_t ubIndex) {
    this->CalcIndex(index);

    this->CopyIn(this->indexD, this->indexH, this->indexW, this->indexNC, ubIndex);
    SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);

    WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);
    Compute(ubIndex);
    SetFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);
    
    WaitFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);
    for (auto iD = this->dStart; iD < this->dEnd; ++iD) {
        for (auto iH = this->hStart; iH < this->hEnd; ++iH) {
            for (auto iW = this->wStart; iW < this->wEnd; ++iW) {
                CopyOut(iD, iH, iW, this->indexNC, ubIndex);
            }
        }
    }

    if (this->isDetermine) {
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::Compute(int64_t ubIndex) {
    auto offset = ubIndex * this->ncAlign;
    Muls(this->inputGradUb[offset], this->outputGradUb[offset], this->mulsFactor, this->thisIterCount);
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::CopyOut(int64_t d, int64_t h, int64_t w, int64_t nc, int64_t ubIndex) {
    auto srcUbOffset = ubIndex * this->ncAlign;
    auto dstGmOffset = d * (this->inH * this->inW * this->ncTotal) + 
                    h * (this->inW * this->ncTotal) + 
                    w * this->ncTotal + 
                    nc * this->ncCount;
    DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(this->thisIterCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->yGm[dstGmOffset], this->inputGradUb[srcUbOffset], copyExtParams);
}

}  // namespace AvgPool3DGrad
#endif  // AVG_POOL3D_GRAD_NO_CAST_H
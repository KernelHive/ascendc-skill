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
 * \file avg_pool3_d_grad_base.h
 * \brief
 */
#ifndef AVG_POOL3D_GRAD_BASE_H
#define AVG_POOL3D_GRAD_BASE_H

#include "kernel_operator.h"

namespace AvgPool3DGrad {
using namespace AscendC;

template <typename T>
class KernelAvgPool3DGradBase {
public:
    __aicore__ inline KernelAvgPool3DGradBase() {}

protected:
    __aicore__ inline int min(int a, int b);
    __aicore__ inline int max(int a, int b);
    __aicore__ inline void ParseTilingData(const AvgPool3dGradTilingParam &tilingData);
    __aicore__ inline void GetEventIds();
    __aicore__ inline void CalcIndex(int64_t index);
    __aicore__ inline void CopyIn(int64_t d, int64_t h, int64_t w, int64_t nc, int64_t ubIndex);

protected:
    TPipe pipe;
    uint64_t coreNum;
    uint64_t isDetermine;
    uint64_t thisCoreNum;
    uint64_t normalCoreNCNum, lastCoreNCNum;
    uint64_t ncCount, ncNum, ncTail, ncTotal, nLine, ncAlign;
    uint64_t thisIterCount;
    uint64_t outD, outH, outW;
    int64_t dD, dH, dW;
    int64_t kD, kH, kW, kD_, kH_, kW_;
    int64_t padD, padH, padW;
    int64_t inD, inH, inW;
    int64_t dStart, hStart, wStart;
    int64_t dEnd, hEnd, wEnd;
    int64_t indexD, indexH, indexW, indexNC;
    int64_t kernelSize;
    int64_t poolSize;
    int64_t thisCoreIdx;
    int64_t divisorOverride;
    bool kernelsOverlap;
    bool countIncludePad;
    float mulsFactor;

    TBuf<QuePosition::VECCALC> xInBuf;
    TBuf<QuePosition::VECCALC> yOutBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    LocalTensor<T> inputGradUb, outputGradUb;

    event_t eventIdMte2ToV, eventIdMte3ToV, eventIdV2Mte2, eventIdV2Mte3;
};

template <typename T>
__aicore__ inline int KernelAvgPool3DGradBase<T>::min(int a, int b) {
    return a <= b ? a : b;
}

template <typename T>
__aicore__ inline int KernelAvgPool3DGradBase<T>::max(int a, int b) {
    return a >= b ? a : b;
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBase<T>::ParseTilingData(const AvgPool3dGradTilingParam &tilingData) {
    // Parse ncParams
    this->normalCoreNCNum = tilingData.ncParam.normalCoreNCNum;
    this->lastCoreNCNum = tilingData.ncParam.lastCoreNCNum;
    this->ncTotal = tilingData.ncParam.ncTotal;
    this->ncCount = tilingData.ncParam.ncCount;
    this->ncNum = tilingData.ncParam.ncNum;
    this->ncTail = tilingData.ncParam.ncTail;
    this->nLine = tilingData.ncParam.nLine;
    this->ncAlign = tilingData.ncParam.ncAlign;

    // Parse attrParam
    this->outD = tilingData.attrParam.outD;
    this->outH = tilingData.attrParam.outH;
    this->outW = tilingData.attrParam.outW;
    this->inD = tilingData.attrParam.inD;
    this->inH = tilingData.attrParam.inH;
    this->inW = tilingData.attrParam.inW;
    this->kD = tilingData.attrParam.kD;
    this->kH = tilingData.attrParam.kH;
    this->kW = tilingData.attrParam.kW;
    this->dD = tilingData.attrParam.dD;
    this->dH = tilingData.attrParam.dH;
    this->dW = tilingData.attrParam.dW;
    this->padD = tilingData.attrParam.padD;
    this->padH = tilingData.attrParam.padH;
    this->padW = tilingData.attrParam.padW;
    this->countIncludePad = tilingData.attrParam.countIncludePad;
    this->divisorOverride = tilingData.attrParam.divisorOverride;
    this->kernelsOverlap = tilingData.attrParam.isOverLap;
    this->isDetermine = tilingData.attrParam.isDetermine;

    this->thisCoreNum = thisCoreIdx == coreNum - 1 ? lastCoreNCNum : normalCoreNCNum;
    this->nLine = min(this->nLine, this->thisCoreNum);
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBase<T>::GetEventIds() {
    eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    eventIdV2Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventIdV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBase<T>::CalcIndex(int64_t index) {
    indexD = (index / (outH * outW * ncNum)) % outD;
    indexH = (index / (outW * ncNum)) % outH;
    indexW = (index / ncNum) % outW;
    indexNC = index % ncNum;
    thisIterCount = ncCount;
    if (indexNC == ncNum - 1) {
        thisIterCount = ncTail;
    }

    dStart = indexD * dD - padD;
    hStart = indexH * dH - padH;
    wStart = indexW * dW - padW;
    dEnd = min(dStart + kD, inD + padD);
    hEnd = min(hStart + kH, inH + padH);
    wEnd = min(wStart + kW, inW + padW);
    poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
    dStart = max(dStart, 0);
    hStart = max(hStart, 0);
    wStart = max(wStart, 0);
    dEnd = min(dEnd, inD);
    hEnd = min(hEnd, inH);
    wEnd = min(wEnd, inW);
    kernelSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);

    if (divisorOverride) {
        mulsFactor = (float)1.0 / static_cast<float>(divisorOverride);
    } else if (countIncludePad) {
        mulsFactor = (float)1.0 / static_cast<float>(poolSize);
    } else {
        mulsFactor = (float)1.0 / static_cast<float>(kernelSize);
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBase<T>::CopyIn(int64_t d, int64_t h, int64_t w, int64_t nc, int64_t ubIndex) {
    auto dstUbOffset = ubIndex * this->ncAlign;
    auto srcGmOffset = d * (this->outH * this->outW * this->ncTotal) + 
                       h * (this->outW * this->ncTotal) + 
                       w * this->ncTotal + 
                       nc * this->ncCount;
    DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(this->thisIterCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> copyPadExtParams = {false, 0, 0, 0};
    DataCopyPad(this->outputGradUb[dstUbOffset], this->xGm[srcGmOffset], copyExtParams, copyPadExtParams);
}

}  // namespace AvgPool3DGrad
#endif  // AVG_POOL3D_GRAD_BASE_H

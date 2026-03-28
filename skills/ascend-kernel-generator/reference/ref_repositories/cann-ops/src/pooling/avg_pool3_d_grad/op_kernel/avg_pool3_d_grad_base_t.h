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
 * \file avg_pool3_d_grad_base_t.h
 * \brief
 */
#ifndef AVG_POOL3D_GRAD_BASE_T_H
#define AVG_POOL3D_GRAD_BASE_T_H

#include "kernel_operator.h"

namespace AvgPool3DGrad {
using namespace AscendC;

template <typename T>
class KernelAvgPool3DGradBaseT {
public:
    __aicore__ inline KernelAvgPool3DGradBaseT() {}

protected:
    __aicore__ inline int min(int a, int b);
    __aicore__ inline int max(int a, int b);
    __aicore__ inline void ParseTilingData(const AvgPool3dGradTilingParam &tilingData);
    __aicore__ inline void GetEventIds();
    __aicore__ inline void CalcIndex(int64_t index);
    __aicore__ inline void CopyIn(int64_t n, int64_t c, int64_t d, int64_t hw, int64_t ubIndex);

protected:
    TPipe pipe;
    uint64_t coreNum;
    uint64_t isDetermine;
    uint64_t thisCoreNum;
    uint64_t normalCoreHWNum, lastCoreHWNum;
    uint64_t hwCount, hwNum, hwTail, hwTotal, nLine, hwAlign;
    uint64_t thisIterCount;
    uint64_t outD, dD, kD, kD_, padD;
    uint64_t N, C, inD;
    uint64_t dStart, dEnd;
    uint64_t indexN, indexC, indexD, indexHW;
    uint64_t thisCoreIdx;
    int64_t kernelSize;
    int64_t poolSize;
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
__aicore__ inline int KernelAvgPool3DGradBaseT<T>::min(int a, int b) {
    return a <= b ? a : b;
}

template <typename T>
__aicore__ inline int KernelAvgPool3DGradBaseT<T>::max(int a, int b) {
    return a >= b ? a : b;
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBaseT<T>::ParseTilingData(const AvgPool3dGradTilingParam &tilingData) {
    // Parse hwParams
    this->normalCoreHWNum = tilingData.hwParam.normalCoreHWNum;
    this->lastCoreHWNum = tilingData.hwParam.lastCoreHWNum;
    this->hwTotal = tilingData.hwParam.hwTotal;
    this->hwCount = tilingData.hwParam.hwCount;
    this->hwNum = tilingData.hwParam.hwNum;
    this->hwTail = tilingData.hwParam.hwTail;
    this->nLine = tilingData.hwParam.nLine;
    this->hwAlign = tilingData.hwParam.hwAlign;

    // Parse attrParam
    this->N = tilingData.attrParam.N;
    this->C = tilingData.attrParam.C;
    this->outD = tilingData.attrParam.outD;
    this->inD = tilingData.attrParam.inD;
    this->kD = tilingData.attrParam.kD;
    this->dD = tilingData.attrParam.dD;
    this->padD = tilingData.attrParam.padD;
    this->countIncludePad = tilingData.attrParam.countIncludePad;
    this->divisorOverride = tilingData.attrParam.divisorOverride;
    this->kernelsOverlap = tilingData.attrParam.isOverLap;
    this->isDetermine = tilingData.attrParam.isDetermine;

    this->thisCoreNum = thisCoreIdx == coreNum - 1 ? lastCoreHWNum : normalCoreHWNum;
    this->nLine = min(this->nLine, this->thisCoreNum);
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBaseT<T>::GetEventIds() {
    eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    eventIdV2Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventIdV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBaseT<T>::CalcIndex(int64_t index) {
    indexN = (index / (C * outD * hwNum)) % N;
    indexC = (index / (outD * hwNum)) % C;
    indexD = (index / hwNum) % outD;
    indexHW = index % hwNum;
    thisIterCount = hwCount;
    if (indexHW == hwNum - 1) {
        thisIterCount = hwTail;
    }

    dStart = indexD * dD - padD;
    dEnd = min(dStart + kD, inD + padD);
    poolSize = dEnd - dStart;
    dStart = max(dStart, 0);
    dEnd = min(dEnd, inD);
    kernelSize = dEnd - dStart;

    if (divisorOverride) {
        mulsFactor = (float)1.0 / static_cast<float>(divisorOverride);
    } else if (countIncludePad) {
        mulsFactor = (float)1.0 / static_cast<float>(poolSize);
    } else {
        mulsFactor = (float)1.0 / static_cast<float>(kernelSize);
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradBaseT<T>::CopyIn(int64_t n, int64_t c, int64_t d, int64_t hw, int64_t ubIndex) {
    auto dstUbOffset = ubIndex * this->hwAlign;
    auto srcGmOffset = n * (this->C * this->outD * this->hwTotal) + 
                       c * (this->outD * this->hwTotal) + 
                       d * this->hwTotal + 
                       hw * this->hwCount;
    DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(this->thisIterCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> copyPadExtParams = {false, 0, 0, 0};
    DataCopyPad(this->outputGradUb[dstUbOffset], this->xGm[srcGmOffset], copyExtParams, copyPadExtParams);
}

}  // namespace AvgPool3DGrad
#endif  // AVG_POOL3D_GRAD_BASE_H

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
 * \file avg_pool3_d_grad_t_cast.h
 * \brief
 */
#ifndef AVG_POOL3D_GRAD_T_CAST_H
#define AVG_POOL3D_GRAD_T_CAST_H

#include "kernel_operator.h"
#include "avg_pool3_d_grad_base_t.h"

namespace AvgPool3DGrad {
using namespace AscendC;

template <typename T>
class KernelAvgPool3DGradTCast : public KernelAvgPool3DGradBaseT<T> {
public:
    __aicore__ inline KernelAvgPool3DGradTCast() {}
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam &tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearOutput();
    __aicore__ inline void ProcessAndCopyout();
    __aicore__ inline void ProcessPerLine(int64_t index, int64_t ubIndex);
    __aicore__ inline void ParseTilingDataForCast(const AvgPool3dGradTilingParam &tilingData);
    __aicore__ inline void ProcessWorkspace();
    __aicore__ inline void Compute(int64_t ubIndex);
    __aicore__ inline void CopyOut(int64_t n, int64_t c, int64_t d, int64_t hw, int64_t ubIndex);
    __aicore__ inline void CopyFromWorkspaceToGm(int64_t offset, int64_t count);

private:
    TPipe pipeCopyCast;
    uint64_t thisCoreCopyNum;
    uint64_t maxDataNumInUb, normalCoreNum, tailCoreNum, normalCoreDataNum, tailCoreDataNum;
    uint64_t tileFormerLen, tileFormerLoop, tileTailLen, tileTailLoop, offset;

    TBuf<QuePosition::VECCALC> calcBuffer;
    GlobalTensor<float> workspaceGm;
    LocalTensor<float> calcUb;
};

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::Init(GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam &tilingData, GM_ADDR workspace) {
    // Get all required params
    this->thisCoreIdx = GetBlockIdx();
    this->coreNum = GetBlockNum();
    this->ParseTilingData(tilingData);
    this->ParseTilingDataForCast(tilingData);
    this->GetEventIds();

    // Init
    this->xGm.SetGlobalBuffer((__gm__ T*)grads);
    this->yGm.SetGlobalBuffer((__gm__ T*)output);
    this->workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);
    this->pipe.InitBuffer(this->xInBuf, this->nLine * this->hwAlign * sizeof(T));
    this->pipe.InitBuffer(this->calcBuffer, this->nLine * this->hwAlign * sizeof(float));

    this->outputGradUb = this->xInBuf.template Get<T>();
    this->calcUb = this->calcBuffer.template Get<float>();

    // If no overlap, init yOutBuf and alloc inputGradUb
    if (!this->kernelsOverlap) {
        this->pipe.InitBuffer(this->yOutBuf, this->nLine * this->hwAlign * sizeof(T));
        this->inputGradUb = this->yOutBuf.template Get<T>();
    }

    ClearOutput();
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::ClearOutput() {
    auto yGmCount = this->N * this->C * this->inD * this->hwTotal;
    auto formerNum = yGmCount % this->coreNum;
    auto tailNum = this->coreNum - formerNum;
    auto formerCount = (yGmCount + this->coreNum - 1) / this->coreNum;
    auto tailCount = yGmCount / this->coreNum;
    if (this->thisCoreIdx < formerNum) {
        auto offset = this->thisCoreIdx * formerCount;
        if (this->kernelsOverlap) {
            InitOutput(workspaceGm[offset], formerCount, static_cast<float>(0));
        } else {
            InitOutput(this->yGm[offset], formerCount, static_cast<T>(0));
        }
    } else {
        auto offset = formerNum * formerCount + (this->thisCoreIdx - formerNum) * tailCount;
        if (this->kernelsOverlap) {
            InitOutput(workspaceGm[offset], tailCount, static_cast<float>(0));
        } else {
            InitOutput(this->yGm[offset], tailCount, static_cast<T>(0));
        }
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::Process() {
    this->ProcessAndCopyout();
    if (this->kernelsOverlap) {
        SyncAll();
        this->pipe.Destroy();
        this->ProcessWorkspace();
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::ProcessAndCopyout() {
    auto processStart = this->normalCoreHWNum * this->thisCoreIdx;
    auto processEnd = processStart + this->thisCoreNum;

    if (this->kernelsOverlap) {
        SetAtomicAdd<float>();
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
__aicore__ inline void KernelAvgPool3DGradTCast<T>::ProcessPerLine(int64_t index, int64_t ubIndex) {
    this->CalcIndex(index);
    
    this->CopyIn(this->indexN, this->indexC, this->indexD, this->indexHW, ubIndex);
    SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);

    WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);
    this->Compute(ubIndex);
    SetFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);

    WaitFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);
    for (auto iD = this->dStart; iD < this->dEnd; ++iD) {
        this->CopyOut(this->indexN, this->indexC, iD, this->indexHW, ubIndex);
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::ParseTilingDataForCast(const AvgPool3dGradTilingParam &tilingData) {
    // Parse castCopyParam
    this->maxDataNumInUb = tilingData.castCopyParam.maxDataNumInUb;
    this->normalCoreNum = tilingData.castCopyParam.normalCoreNum;
    this->tailCoreNum = tilingData.castCopyParam.tailCoreNum;
    this->normalCoreDataNum = tilingData.castCopyParam.normalCoreDataNum;
    this->tailCoreDataNum = tilingData.castCopyParam.tailCoreDataNum;
    if (this->thisCoreIdx < normalCoreNum) {
        tileFormerLen = tilingData.castCopyParam.normalCoreFormerDataNum;
        tileFormerLoop = tilingData.castCopyParam.normalCoreFormerCopyTime;
        tileTailLen = tilingData.castCopyParam.normalCoreTailDataNum;
        tileTailLoop = tilingData.castCopyParam.normalCoreTailCopyTime;
        offset = this->thisCoreIdx * normalCoreDataNum;
    } else {
        tileFormerLen = tilingData.castCopyParam.tailCoreFormerDataNum;
        tileFormerLoop = tilingData.castCopyParam.tailCoreFormerCopyTime;
        tileTailLen = tilingData.castCopyParam.tailCoreTailDataNum;
        tileTailLoop = tilingData.castCopyParam.tailCoreTailCopyTime;
        offset = normalCoreNum * normalCoreDataNum 
                + (this->thisCoreIdx - normalCoreNum) * tailCoreDataNum;
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::ProcessWorkspace() {
    this->pipeCopyCast.InitBuffer(this->yOutBuf, this->maxDataNumInUb * sizeof(T));
    this->pipeCopyCast.InitBuffer(this->calcBuffer, this->maxDataNumInUb * sizeof(float));

    this->inputGradUb = this->yOutBuf.template Get<T>();
    this->calcUb = this->calcBuffer.template Get<float>();

    SetFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
    SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
    for (auto process = 0; process < tileFormerLoop + tileTailLoop; process++) {
        auto thisIterNum = process < tileFormerLoop ? tileFormerLen : tileTailLen;
        auto workspaceOffset = process < tileFormerLoop ?
                                offset + process * tileFormerLen :
                                offset + tileFormerLoop * tileFormerLen + (process - tileFormerLoop) * tileTailLen;
        WaitFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
        WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
        CopyFromWorkspaceToGm(workspaceOffset, thisIterNum);
        SetFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
        SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
    }
    WaitFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
    WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::Compute(int64_t ubIndex) {
    auto offset = ubIndex * this->hwAlign;
    Cast(calcUb[offset], this->outputGradUb[offset], RoundMode::CAST_NONE, this->thisIterCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(calcUb[offset], calcUb[offset], this->mulsFactor, this->thisIterCount);
    if (!this->kernelsOverlap) {
        Cast(this->inputGradUb[offset], calcUb[offset], RoundMode::CAST_RINT, this->thisIterCount);
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::CopyOut(int64_t n, int64_t c, int64_t d, int64_t hw, int64_t ubIndex) {
    auto srcUbOffset = ubIndex * this->hwAlign;
    auto dstGmOffset = n * (this->C * this->inD * this->hwTotal) + 
                       c * (this->inD * this->hwTotal) + 
                       d * this->hwTotal + 
                       hw * this->hwCount;
    if (!this->kernelsOverlap) {
        DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(this->thisIterCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(this->yGm[dstGmOffset], this->inputGradUb[srcUbOffset], copyExtParams);
    } else {
        DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(this->thisIterCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(workspaceGm[dstGmOffset], calcUb[srcUbOffset], copyExtParams);
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradTCast<T>::CopyFromWorkspaceToGm(int64_t offset, int64_t count) {
    // CopyIn
    DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> copyPadExtParams = {false, 0, 0, 0};
    DataCopyPad(calcUb, workspaceGm[offset], copyExtParams, copyPadExtParams);
    SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);

    // Cast
    WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);
    Cast(this->inputGradUb, calcUb, RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);

    // CopyOut
    WaitFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);
    copyExtParams = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->yGm[offset], this->inputGradUb, copyExtParams);
}

}  // namespace AvgPool3DGrad

#endif  // AVG_POOL3D_GRAD_T_CAST_H

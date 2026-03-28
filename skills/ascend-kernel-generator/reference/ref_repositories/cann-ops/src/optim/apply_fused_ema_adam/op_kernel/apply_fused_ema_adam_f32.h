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
 * \file apply_fused_ema_adam_f32.h
 * \brief
 */

#ifndef APPLY_FUSED_EMA_ADAM_F32_H
#define APPLY_FUSED_EMA_ADAM_F32_H

#include "apply_fused_ema_adam_base.h"

namespace FusedEmaAdam {
using namespace AscendC;

template <typename T>
class FusedEmaAdamF32 : public FusedEmaAdamBase<T> {
public:
    __aicore__ inline FusedEmaAdamF32(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v,
        GM_ADDR s, GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
        GM_ADDR s_ref,const ApplyFusedEmaAdamTilingData& tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void Compute(const uint64_t index, const uint64_t dataCount);
    
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

    TBuf<QuePosition::VECCALC> powTBuf1, powTBuf2;

    GlobalTensor<T> gmGrad, gmVar, gmM, gmV, gmS;
    GlobalTensor<int64_t> gmStep;
    GlobalTensor<T> gmVarRef, gmMRef, gmVRef, gmSRef;

    float step_ = 0;
    int32_t INPUT_NUM = 5;
    int32_t OUTPUT_NUM = 4;
    uint32_t blockIdx;
    uint64_t blockOffset;
    int64_t mOffset;
    int64_t vOffset;
    int64_t varOffset;
    int64_t sOffset;
    int64_t gradOffset;
};

template <typename T>
__aicore__ inline void FusedEmaAdamF32<T>::Init(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v,
                        GM_ADDR s, GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
                        GM_ADDR s_ref, const ApplyFusedEmaAdamTilingData& tiling, TPipe *pipe) {
    this->InitData(tiling);
    blockIdx = GetBlockIdx();

    if (blockIdx < this->frontCoreNum) {
        blockOffset = this->coreCalcNum * blockIdx;
    } else if (this->coreCalcNum - 1 != 0) {
        blockOffset = this->coreCalcNum * this->frontCoreNum + (blockIdx - this->frontCoreNum) * (this->coreCalcNum - 1);
    }

    gmGrad.SetGlobalBuffer((__gm__ T*)grad + blockOffset);
    gmVar.SetGlobalBuffer((__gm__ T*)var + blockOffset);
    gmM.SetGlobalBuffer((__gm__ T*)m + blockOffset);
    gmV.SetGlobalBuffer((__gm__ T*)v + blockOffset);
    gmS.SetGlobalBuffer((__gm__ T*)s + blockOffset);

    gmVarRef.SetGlobalBuffer((__gm__ T*)var_ref + blockOffset);
    gmMRef.SetGlobalBuffer((__gm__ T*)m_ref + blockOffset);
    gmVRef.SetGlobalBuffer((__gm__ T*)v_ref + blockOffset);
    gmSRef.SetGlobalBuffer((__gm__ T*)s_ref + blockOffset);

    if (this->mode == 1) {
        INPUT_NUM -= 1;
    }
    pipe->InitBuffer(inQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * INPUT_NUM);
    pipe->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * OUTPUT_NUM);

    varOffset = this->coreCalcMax * INDEX_VAR;
    mOffset = this->coreCalcMax * INDEX_M;
    vOffset = this->coreCalcMax * INDEX_V;
    sOffset = this->coreCalcMax * INDEX_S;
    gradOffset = this->mode == 0 ? this->coreCalcMax * INDEX_GRAD : varOffset;

    pipe->InitBuffer(powTBuf1, BYTE_ONE_BLOCK);
    pipe->InitBuffer(powTBuf2, BYTE_ONE_BLOCK);

    gmStep.SetGlobalBuffer((__gm__ int64_t*)step, 1);
    step_ = static_cast<float>(gmStep.GetValue(0));
    if (this->biasCorrection == 1) {
        this->beta1Correction = 1.0f - this->ScalarPow(powTBuf1, powTBuf2, inQue, this->beta1, step_);
        this->beta2Correction = 1.0f - this->ScalarPow(powTBuf1, powTBuf2, inQue, this->beta2, step_);
    }
}

template <typename T>
__aicore__ inline void FusedEmaAdamF32<T>::Compute(const uint64_t index, const uint64_t dataCount) {
    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    
    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();
    
    // grad = grad [+ weight_decay*var if mode == 0]
    if (this->mode == 0) {
        DataCopyPad(inLocal[varOffset], gmVar[offset], copyParams, padParams);
        this->PipeM2V();
        Muls(outLocal[mOffset], inLocal[varOffset], this->weightDecay, dataCount);
    }
    DataCopyPad(inLocal[gradOffset], gmGrad[offset], copyParams, padParams);
    this->PipeM2V();
    if (this->mode == 0) {
        Add(inLocal[gradOffset], outLocal[mOffset], inLocal[gradOffset], dataCount);
    }

    // m = beta1*m + (1-beta1)*grad, next_m = m/beta1_correction
    DataCopyPad(inLocal[mOffset], gmM[offset], copyParams, padParams);
    this->PipeM2V();
    Muls(inLocal[mOffset], inLocal[mOffset], this->beta1, dataCount);
    Muls(outLocal[mOffset], inLocal[gradOffset], 1 - this->beta1, dataCount);
    Add(outLocal[mOffset], outLocal[mOffset], inLocal[mOffset], dataCount);
    this->PipeVM3();
    DataCopyPad(gmMRef[offset], outLocal[mOffset], copyParams);
    Muls(inLocal[mOffset], outLocal[mOffset], 1 / this->beta1Correction, dataCount);

    // v = beta2*v + (1-beta2)*grad*grad, next_v = v/beta2_correction
    DataCopyPad(inLocal[vOffset], gmV[offset], copyParams, padParams);
    this->PipeM2V();
    Muls(inLocal[vOffset], inLocal[vOffset], this->beta2, dataCount);
    Mul(outLocal[vOffset], inLocal[gradOffset], inLocal[gradOffset], dataCount);
    if (this->mode == 1) {
        this->PipeVM2();
    }
    Muls(outLocal[vOffset], outLocal[vOffset], 1 - this->beta2, dataCount);
    Add(outLocal[vOffset], outLocal[vOffset], inLocal[vOffset], dataCount);
    this->PipeVM3();
    DataCopyPad(gmVRef[offset], outLocal[vOffset], copyParams);
    Muls(inLocal[vOffset], outLocal[vOffset], 1 / this->beta2Correction, dataCount);

    // denom = sqrt(next_v) + eps, update = next_m/denom [+ weight_decay*var if mode == 1]
    Sqrt(inLocal[vOffset], inLocal[vOffset], dataCount);
    Adds(inLocal[vOffset], inLocal[vOffset], this->eps, dataCount);
    Div(inLocal[mOffset], inLocal[mOffset], inLocal[vOffset], dataCount);
    if (this->mode == 1) {
        DataCopyPad(inLocal[varOffset], gmVar[offset], copyParams, padParams);
        this->PipeM2V();
        Muls(inLocal[vOffset], inLocal[varOffset], this->weightDecay, dataCount);
        Add(inLocal[mOffset], inLocal[mOffset], inLocal[vOffset], dataCount);
    }

    // var = var - lr*update
    Muls(inLocal[mOffset], inLocal[mOffset], this->lr, dataCount);
    Sub(outLocal[varOffset], inLocal[varOffset], inLocal[mOffset], dataCount);
    this->PipeVM3();
    DataCopyPad(gmVarRef[offset], outLocal[varOffset], copyParams);
    Muls(inLocal[varOffset], outLocal[varOffset], 1 - this->emaDecay, dataCount);

    // s = ema_decay*s + (1-ema_decay)*var
    DataCopyPad(inLocal[sOffset], gmS[offset], copyParams, padParams);
    this->PipeM2V();
    Muls(inLocal[sOffset], inLocal[sOffset], this->emaDecay, dataCount);
    Add(outLocal[sOffset], inLocal[sOffset], inLocal[varOffset], dataCount);
    this->PipeVM3();
    DataCopyPad(gmSRef[offset], outLocal[sOffset], copyParams);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void FusedEmaAdamF32<T>::Process() {
    for (uint64_t n =0; n < this->loopNum - 1; n++) {
        Compute(n, this->coreCalcMax);
    }
    if (blockIdx < this->frontCoreNum) {
        Compute(this->loopNum - 1, this->frontCalcExtra);
    } else if (this->tailCalcExtra != 0) {
        Compute(this->loopNum - 1, this->tailCalcExtra);
    }
}

} // namespace FusedEmaAdam

#endif // APPLY_FUSED_EMA_ADAM_F32_H
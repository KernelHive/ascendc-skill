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
 * \file apply_fused_ema_adam_f_bf16.h
 * \brief
 */

#ifndef APPLY_FUSED_EMA_ADAM_F_BF_16_H
#define APPLY_FUSED_EMA_ADAM_F_BF_16_H

#include "apply_fused_ema_adam_base.h"

namespace FusedEmaAdam {
using namespace AscendC;

template <typename T>
class FusedEmaAdamF16 : public FusedEmaAdamBase<T> {
public:
    __aicore__ inline FusedEmaAdamF16(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v,
        GM_ADDR s, GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
        GM_ADDR s_ref,const ApplyFusedEmaAdamTilingData& tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void Compute(const uint64_t index, const uint64_t dataCount);
    
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

    TBuf<QuePosition::VECCALC> inCastBuf, outCastBuf, powTBuf1, powTBuf2;

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
__aicore__ inline void FusedEmaAdamF16<T>::Init(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v,
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

    pipe->InitBuffer(inCastBuf, this->coreCalcMax * sizeof(float) * INPUT_NUM);
    pipe->InitBuffer(outCastBuf, this->coreCalcMax * sizeof(float) * OUTPUT_NUM);

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
__aicore__ inline void FusedEmaAdamF16<T>::Compute(const uint64_t index, const uint64_t dataCount) {
    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    
    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();
    LocalTensor<float> inLocalC = inCastBuf.Get<float>();
    LocalTensor<float> outLocalC = outCastBuf.Get<float>();
    
    // grad = grad [+ weight_decay*var if mode == 0]
    if (this->mode == 0) {
        DataCopyPad(inLocal[varOffset], gmVar[offset], copyParams, padParams);
        this->PipeM2V();
        Cast(inLocalC[varOffset], inLocal[varOffset], RoundMode::CAST_NONE, dataCount);
        Muls(outLocalC[mOffset], inLocalC[varOffset], this->weightDecay, dataCount);
    }
    DataCopyPad(inLocal[gradOffset], gmGrad[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[gradOffset], inLocal[gradOffset], RoundMode::CAST_NONE, dataCount);
    if (this->mode == 0) {
        Add(inLocalC[gradOffset], outLocalC[mOffset], inLocalC[gradOffset], dataCount);
    }

    // m = beta1*m + (1-beta1)*grad, next_m = m/beta1_correction
    DataCopyPad(inLocal[mOffset], gmM[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[mOffset], inLocal[mOffset], RoundMode::CAST_NONE, dataCount);
    Muls(inLocalC[mOffset], inLocalC[mOffset], this->beta1, dataCount);
    Muls(outLocalC[mOffset], inLocalC[gradOffset], 1 - this->beta1, dataCount);
    Add(outLocalC[mOffset], outLocalC[mOffset], inLocalC[mOffset], dataCount);
    Cast(outLocal[mOffset], outLocalC[mOffset], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmMRef[offset], outLocal[mOffset], copyParams);
    Muls(inLocalC[mOffset], outLocalC[mOffset], 1 / this->beta1Correction, dataCount);

    // v = beta2*v + (1-beta2)*grad*grad, next_v = v/beta2_correction
    DataCopyPad(inLocal[vOffset], gmV[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[vOffset], inLocal[vOffset], RoundMode::CAST_NONE, dataCount);
    Muls(inLocalC[vOffset], inLocalC[vOffset], this->beta2, dataCount);
    Mul(outLocalC[vOffset], inLocalC[gradOffset], inLocalC[gradOffset], dataCount);
    if (this->mode == 1) {
        this->PipeVM2();
    }
    Muls(outLocalC[vOffset], outLocalC[vOffset], 1 - this->beta2, dataCount);
    Add(outLocalC[vOffset], outLocalC[vOffset], inLocalC[vOffset], dataCount);
    Cast(outLocal[vOffset], outLocalC[vOffset], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmVRef[offset], outLocal[vOffset], copyParams);
    Muls(inLocalC[vOffset], outLocalC[vOffset], 1 / this->beta2Correction, dataCount);

    // denom = sqrt(next_v) + eps, update = next_m/denom [+ weight_decay*var if mode == 1]
    Sqrt(inLocalC[vOffset], inLocalC[vOffset], dataCount);
    Adds(inLocalC[vOffset], inLocalC[vOffset], this->eps, dataCount);
    Div(inLocalC[mOffset], inLocalC[mOffset], inLocalC[vOffset], dataCount);
    if (this->mode == 1) {
        DataCopyPad(inLocal[varOffset], gmVar[offset], copyParams, padParams);
        this->PipeM2V();
        Cast(inLocalC[varOffset], inLocal[varOffset], RoundMode::CAST_NONE, dataCount);
        Muls(inLocalC[vOffset], inLocalC[varOffset], this->weightDecay, dataCount);
        Add(inLocalC[mOffset], inLocalC[mOffset], inLocalC[vOffset], dataCount);
    }

    // var = var - lr*update
    Muls(inLocalC[mOffset], inLocalC[mOffset], this->lr, dataCount);
    Sub(outLocalC[varOffset], inLocalC[varOffset], inLocalC[mOffset], dataCount);
    Cast(outLocal[varOffset], outLocalC[varOffset], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmVarRef[offset], outLocal[varOffset], copyParams);
    Muls(inLocalC[varOffset], outLocalC[varOffset], 1 - this->emaDecay, dataCount);

    // s = ema_decay*s + (1-ema_decay)*var
    DataCopyPad(inLocal[sOffset], gmS[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[sOffset], inLocal[sOffset], RoundMode::CAST_NONE, dataCount);
    Muls(inLocalC[sOffset], inLocalC[sOffset], this->emaDecay, dataCount);
    Add(outLocalC[sOffset], inLocalC[sOffset], inLocalC[varOffset], dataCount);
    Cast(outLocal[sOffset], outLocalC[sOffset], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmSRef[offset], outLocal[sOffset], copyParams);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void FusedEmaAdamF16<T>::Process() {
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

#endif // APPLY_FUSED_EMA_ADAM_F_BF_16_H
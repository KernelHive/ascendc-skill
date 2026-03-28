/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file apply_adam_wv2_fp.h
 */
#ifndef APPLYADAM_W_V2_FP_H
#define APPLYADAM_W_V2_FP_H

#include "apply_adam_wv2_base.h"

namespace ApplyAdamWV2 {
using namespace AscendC;

template <typename T, typename U>
class ApplyAdamWV2Fp {
public:
    __aicore__ inline ApplyAdamWV2Fp(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad, GM_ADDR step,
                                GM_ADDR maxGradNorm,  GM_ADDR workspace, const ApplyAdamWV2TilingData* tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ParseTilingData(const ApplyAdamWV2TilingData* tilingData);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline void Compute(int32_t dataCount);
    __aicore__ inline void CopyOut(int64_t index, int64_t dataCount);
    __aicore__ inline float ScalarPow(float x, float y);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    TBuf<QuePosition::VECCALC> powTempBuf1_;
    TBuf<QuePosition::VECCALC> powTempBuf2_;

    GlobalTensor<T> gmVar_;
    GlobalTensor<T> gmExpAvg_;
    GlobalTensor<T> gmExpAvgSq_;
    GlobalTensor<T> gmMaxGradNorm_;
    GlobalTensor<T> gmGrad_;
    GlobalTensor<U> gmStep_;

    float step_ = 0;

    int64_t numPerLoop_ = 0;
    int64_t loopNumPerCore_ = 0;
    int64_t numLastLoop_ = 0;
    int64_t handleExtraLoopCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    bool amsgrad_ = false;
    float beta1_ = 0;
    float beta2_ = 0;
    float lr_ = 0;
    float weightDecay_ = 0;
    float eps_ = 0;
    bool maximize_ = false;

    T realWeightDecay_ = 0;
    T stepSize_ = 0;
    T biasCorrection2Sqrt_ = 0;
    T oneSubBeta1_ = 0;
    T oneSubBeta2_ = 0;
    T realBeta2_ = 0;
    T negOne_ = -1;
    T realEps_ = 0;

    int64_t varOffset_ = 0;
    int64_t expAvgOffset_ = 0;
    int64_t expAvgSqOffset_ = 0;
    int64_t gradOffset_ = 0;
    int64_t maxGradNormOffset_ = 0;
    int64_t maxGradOutOffset_ = 0;
    int64_t blockIdx_ = GetBlockIdx();
};

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2Fp<T, U>::Init(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad,
    GM_ADDR step, GM_ADDR maxGradNorm,  GM_ADDR workspace, const ApplyAdamWV2TilingData* tilingData) {

    this->ParseTilingData(tilingData);
    gmStep_.SetGlobalBuffer((__gm__ U*)step, 1);
    step_ = static_cast<float>(gmStep_.GetValue(0));
    int64_t gmOffset = blockIdx_  * numPerLoop_;

    gmVar_.SetGlobalBuffer((__gm__ T*)var + gmOffset);
    gmExpAvg_.SetGlobalBuffer((__gm__ T*)expAvg + gmOffset);
    gmExpAvgSq_.SetGlobalBuffer((__gm__ T*)expAvgSq + gmOffset);
    gmGrad_.SetGlobalBuffer((__gm__ T*)grad + gmOffset);

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, IN_BUFFER_NUM * numPerLoop_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, OUT_BUFFER_NUM * numPerLoop_ * sizeof(T));

    pipe_.InitBuffer(powTempBuf1_, BYTE_ONE_BLOCK);
    pipe_.InitBuffer(powTempBuf2_, BYTE_ONE_BLOCK);

    if(amsgrad_){
        gmMaxGradNorm_.SetGlobalBuffer((__gm__ T*)maxGradNorm + gmOffset);
   }

  step_ += 1;
  float biasCorrection1 = 1.0f - ScalarPow(beta1_, step_);
  float biasCorrection2 = 1.0f - ScalarPow(beta2_, step_);

  stepSize_ = static_cast<T>(lr_ / biasCorrection1);
  biasCorrection2Sqrt_ = static_cast<T>(1.0f / sqrt(biasCorrection2));

  realWeightDecay_ = static_cast<T>(1.0f - lr_ * (weightDecay_));
  oneSubBeta1_ = static_cast<T>(1.0f - beta1_);
  oneSubBeta2_ = static_cast<T>(1.0f - beta2_);
  realBeta2_ = static_cast<T>(beta2_);
  realEps_ = static_cast<T>(eps_);
  varOffset_ = VAR_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
  expAvgOffset_ = EXP_AVG_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
  expAvgSqOffset_ = EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
  gradOffset_ = GRAD_NORM_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
  maxGradNormOffset_ = MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
  maxGradOutOffset_ = MAX_GRAD_NORM_ORDER_IN_OUT_LOCAL_TENSOR * numPerLoop_;
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2Fp<T, U>::ParseTilingData(
    const ApplyAdamWV2TilingData* tilingData) {
    numPerLoop_ = tilingData->numPerLoop;
    loopNumPerCore_ = tilingData->loopNumPerCore;
    numLastLoop_ = tilingData->numLastLoop;
    usedCoreNum_ = tilingData->usedCoreNum;
    handleExtraLoopCoreNum_ = tilingData->handleExtraLoopCoreNum;
    beta1_ = tilingData->beta1;
    beta2_ = tilingData->beta2;
    lr_ = tilingData->lr;
    weightDecay_ = tilingData->weightDecay;
    eps_ = tilingData->eps;

    if (tilingData->amsgrad != 0){
        amsgrad_ = true;
    }

    if (tilingData->maximize != 0){
        maximize_ = true;
    }
}

template <typename T, typename U>
__aicore__ inline float ApplyAdamWV2Fp<T, U>::ScalarPow(float x, float y){
    LocalTensor<float> baseLocal = powTempBuf1_.Get<float>();
    LocalTensor<float> outLocal = powTempBuf2_.Get<float>();
    pipe_barrier(PIPE_V);
    Duplicate(baseLocal, x, BLOCK_SIZE_FOR_FLOAT32);
    pipe_barrier(PIPE_V);
    Power<float, false>(outLocal, baseLocal, y, BLOCK_SIZE_FOR_FLOAT32);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float result = outLocal.GetValue(0);
    pipe_barrier(PIPE_ALL);
    return result;
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2Fp<T, U>::CopyIn(int64_t index, int64_t dataCount) {

    int64_t offset = usedCoreNum_ * index * numPerLoop_;
    LocalTensor<T> dataLocal = inQueue_.AllocTensor<T>();

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};

    DataCopyPad(dataLocal[varOffset_], gmVar_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(dataLocal[expAvgOffset_], gmExpAvg_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(dataLocal[expAvgSqOffset_], gmExpAvgSq_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(dataLocal[gradOffset_], gmGrad_[offset], dataCopyParams, dataCopyPadParams);

    if (amsgrad_){
        DataCopyPad(dataLocal[maxGradNormOffset_], gmMaxGradNorm_[offset], dataCopyParams, dataCopyPadParams);
    }
    inQueue_.EnQue(dataLocal);
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2Fp<T, U>::Compute(int32_t dataCount) {
    LocalTensor<T> dataLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dataOutLocal = outQueue_.AllocTensor<T>();
    pipe_barrier(PIPE_V);
    if (maximize_){
        // grad = -grad
        Muls(dataLocal[gradOffset_], dataLocal[gradOffset_], negOne_, dataCount);
    }
    pipe_barrier(PIPE_V);
    // param.mul_(1 - lr * weight_decay)
    Muls(dataOutLocal[varOffset_], dataLocal[varOffset_], realWeightDecay_, dataCount);

    // exp_avg.lerp_(grad, 1 - beta1)
    pipe_barrier(PIPE_V);
    Sub(dataOutLocal[maxGradOutOffset_], dataLocal[gradOffset_], dataLocal[expAvgOffset_], dataCount);
    pipe_barrier(PIPE_V);
    Muls(dataLocal[varOffset_], dataOutLocal[maxGradOutOffset_], oneSubBeta1_, dataCount);
    pipe_barrier(PIPE_V);
    Add(dataOutLocal[expAvgOffset_], dataLocal[varOffset_], dataLocal[expAvgOffset_], dataCount);

    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    pipe_barrier(PIPE_V);
    Muls(dataLocal[varOffset_], dataLocal[expAvgSqOffset_], realBeta2_, dataCount);
    pipe_barrier(PIPE_V);
    Mul(dataOutLocal[maxGradOutOffset_], dataLocal[gradOffset_], dataLocal[gradOffset_], dataCount);
    pipe_barrier(PIPE_V);
    Muls(dataOutLocal[maxGradOutOffset_], dataOutLocal[maxGradOutOffset_], oneSubBeta2_, dataCount);
    pipe_barrier(PIPE_V);
    Add(dataOutLocal[expAvgSqOffset_], dataLocal[varOffset_], dataOutLocal[maxGradOutOffset_], dataCount);
    pipe_barrier(PIPE_V);

    if (amsgrad_){
        // torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
        pipe_barrier(PIPE_V);
        Max(dataOutLocal[maxGradOutOffset_], dataLocal[maxGradNormOffset_], dataOutLocal[expAvgSqOffset_], dataCount);
        pipe_barrier(PIPE_V);
        Sqrt(dataLocal[varOffset_], dataOutLocal[maxGradOutOffset_], dataCount);
        pipe_barrier(PIPE_V);
        Muls(dataLocal[expAvgOffset_], dataLocal[varOffset_], biasCorrection2Sqrt_, dataCount);
        pipe_barrier(PIPE_V);
        Adds(dataLocal[varOffset_], dataLocal[expAvgOffset_], realEps_, dataCount);
    } else {
        // denom = (exp_avg_sq.sqrt() / bias_corrections_sqrt) + eps
        pipe_barrier(PIPE_V);
        Sqrt(dataLocal[varOffset_], dataOutLocal[expAvgSqOffset_], dataCount);
        pipe_barrier(PIPE_V);
        Muls(dataLocal[expAvgOffset_], dataLocal[varOffset_], biasCorrection2Sqrt_, dataCount);
        pipe_barrier(PIPE_V);
        Adds(dataLocal[varOffset_], dataLocal[expAvgOffset_], realEps_, dataCount);
    }

    // param.addcdiv_(exp_avg, denom, value=-step_size)
    pipe_barrier(PIPE_V);
    Div(dataLocal[expAvgOffset_], dataOutLocal[expAvgOffset_], dataLocal[varOffset_], dataCount);
    pipe_barrier(PIPE_V);
    Muls(dataLocal[varOffset_], dataLocal[expAvgOffset_], stepSize_, dataCount);
    pipe_barrier(PIPE_V);
    Sub(dataOutLocal[varOffset_], dataOutLocal[varOffset_], dataLocal[varOffset_], dataCount);

    inQueue_.FreeTensor(dataLocal);
    outQueue_.EnQue(dataOutLocal);
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2Fp<T, U>::CopyOut(int64_t index, int64_t dataCount) {

    int64_t offset = usedCoreNum_ * index * numPerLoop_;
    LocalTensor<T> dataOutLocal = outQueue_.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};

    DataCopyPad(gmVar_[offset], dataOutLocal[varOffset_], copyParams);
    DataCopyPad(gmExpAvg_[offset], dataOutLocal[expAvgOffset_], copyParams);
    DataCopyPad(gmExpAvgSq_[offset], dataOutLocal[expAvgSqOffset_], copyParams);

    if (amsgrad_){
        DataCopyPad(gmMaxGradNorm_[offset], dataOutLocal[maxGradOutOffset_], copyParams);
    }
    outQueue_.FreeTensor(dataOutLocal);
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2Fp<T, U>::Process() {
    if (blockIdx_  < usedCoreNum_) {
        int64_t curLoopCount = loopNumPerCore_;
        if (blockIdx_  < handleExtraLoopCoreNum_ - 1){
            curLoopCount += 1;
        }

        for(int64_t n = 0; n < curLoopCount; n++) {
            CopyIn(n, numPerLoop_);
            Compute(numPerLoop_);
            CopyOut(n, numPerLoop_);
        }

        // 尾loop
        if (blockIdx_  == handleExtraLoopCoreNum_ - 1){
            CopyIn(loopNumPerCore_, numLastLoop_);
            Compute(numLastLoop_);
            CopyOut(loopNumPerCore_, numLastLoop_);
        }
    }
}

}  // namespace ApplyAdamWV2

#endif  // APPLYADAM_W_V2_FP_H
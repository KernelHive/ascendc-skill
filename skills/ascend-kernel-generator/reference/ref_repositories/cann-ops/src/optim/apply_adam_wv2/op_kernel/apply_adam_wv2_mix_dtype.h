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
 * @file apply_adam_wv2_mix_dtype.h
 */
#ifndef APPLYADAM_W_V2_MIX_DTYPE_H
#define APPLYADAM_W_V2_MIX_DTYPE_H

#include "apply_adam_wv2_base.h"

namespace ApplyAdamWV2 {
using namespace AscendC;

constexpr int32_t IN_BUFFER_NUM_TYPE_T = 3;
constexpr int32_t IN_BUFFER_NUM_TYPE_U = 2;
constexpr int32_t OUT_BUFFER_NUM_TYPE_T = 3;
constexpr int32_t OUT_BUFFER_NUM_TYPE_U = 1;

constexpr int32_t VAR_ORDER_IN_LOCAL_TENSOR_T = 0;
constexpr int32_t EXP_AVG_ORDER_IN_LOCAL_TENSOR_T = 1;
constexpr int32_t EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR_T = 2;
constexpr int32_t GRAD_NORM_ORDER_IN_LOCAL_TENSOR_U = 0;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR_U = 1;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_OUT_LOCAL_TENSOR_U = 0;

template <typename T, typename U, typename Z>
class ApplyAdamWV2MixType {
public:
    __aicore__ inline ApplyAdamWV2MixType(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad, GM_ADDR step,
                                GM_ADDR maxGradNorm, GM_ADDR workspace, const ApplyAdamWV2TilingData* tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ParseTilingData(const ApplyAdamWV2TilingData* tilingData);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline void Compute(int32_t realProcCount);
    __aicore__ inline void CopyOut(int64_t index, int64_t dataCount);
    __aicore__ inline float ScalarPow(float x, float y);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTypeT_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTypeU_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueTypeT_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueTypeU_;

    TBuf<QuePosition::VECCALC> resultTempBuf1_;
    TBuf<QuePosition::VECCALC> resultTempBuf2_;
    TBuf<QuePosition::VECCALC> powTempBuf1_;
    TBuf<QuePosition::VECCALC> powTempBuf2_;

    GlobalTensor<T> gmVar_;
    GlobalTensor<T> gmExpAvg_;
    GlobalTensor<T> gmExpAvgSq_;
    GlobalTensor<U> gmMaxGradNorm_;
    GlobalTensor<U> gmGrad_;
    GlobalTensor<Z> gmStep_;

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
    bool isBfloat16_ = false;

    T realWeightDecay_ = 0;
    T stepSize_ = 0;
    T biasCorrection2Sqrt_ = 0;
    T oneSubBeta1_ = 0;
    T oneSubBeta2_ = 0;
    T realBeta2_ = 0;
    T negOne_ = -1;

    int64_t varOffset_ = 0;
    int64_t expAvgOffset_ = 0;
    int64_t expAvgSqOffset_ = 0;
    int64_t gradOffset_ = 0;
    int64_t maxGradNormOffset_ = 0;
    int64_t maxGradOutOffset_ = 0;
    int64_t blockIdx_ = GetBlockIdx();
};

template <typename T, typename U, typename Z>
__aicore__ inline float ApplyAdamWV2MixType<T, U, Z>::ScalarPow(float x, float y){
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

template <typename T, typename U, typename Z>
__aicore__ inline void ApplyAdamWV2MixType<T, U, Z>::Init(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad,
    GM_ADDR step, GM_ADDR maxGradNorm, GM_ADDR workspace, const ApplyAdamWV2TilingData* tilingData) {

    this->ParseTilingData(tilingData);

    gmStep_.SetGlobalBuffer((__gm__ Z*)step, 1);
    step_ = static_cast<float>(gmStep_.GetValue(0));

    int64_t gmOffset = blockIdx_  * numPerLoop_;

    gmVar_.SetGlobalBuffer((__gm__ T*)var + gmOffset);
    gmExpAvg_.SetGlobalBuffer((__gm__ T*)expAvg + gmOffset);
    gmExpAvgSq_.SetGlobalBuffer((__gm__ T*)expAvgSq + gmOffset);
    gmGrad_.SetGlobalBuffer((__gm__ U*)grad + gmOffset);

    pipe_.InitBuffer(inQueueTypeT_, BUFFER_NUM, IN_BUFFER_NUM_TYPE_T * numPerLoop_ * sizeof(T));
    pipe_.InitBuffer(inQueueTypeU_, BUFFER_NUM, IN_BUFFER_NUM_TYPE_U * numPerLoop_ * sizeof(U));
    pipe_.InitBuffer(outQueueTypeT_, BUFFER_NUM, OUT_BUFFER_NUM_TYPE_T * numPerLoop_ * sizeof(T));
    pipe_.InitBuffer(outQueueTypeU_, BUFFER_NUM, OUT_BUFFER_NUM_TYPE_U * numPerLoop_ * sizeof(U));

    if(amsgrad_){
        gmMaxGradNorm_.SetGlobalBuffer((__gm__ U*)maxGradNorm + gmOffset);
    }
    pipe_.InitBuffer(resultTempBuf1_, BUFFER_NUM * numPerLoop_ * sizeof(T));
    pipe_.InitBuffer(resultTempBuf2_, BUFFER_NUM * numPerLoop_ * sizeof(T));
    pipe_.InitBuffer(powTempBuf1_, BYTE_ONE_BLOCK);
    pipe_.InitBuffer(powTempBuf2_, BYTE_ONE_BLOCK);

  step_ += 1;
  float biasCorrection1 = 1.0f - ScalarPow(beta1_, step_);
  float biasCorrection2 = 1.0f - ScalarPow(beta2_, step_);

  stepSize_ = static_cast<T>(lr_ / biasCorrection1);
  biasCorrection2Sqrt_ = static_cast<T>(1.0f / sqrt(biasCorrection2));

  realWeightDecay_ = static_cast<T>(1.0f - lr_ * weightDecay_);
  oneSubBeta1_ = static_cast<T>(1.0f - beta1_);
  oneSubBeta2_ = static_cast<T>(1.0f - beta2_);
  realBeta2_ = static_cast<T>(beta2_);

  varOffset_ = VAR_ORDER_IN_LOCAL_TENSOR_T * numPerLoop_;
  expAvgOffset_ = EXP_AVG_ORDER_IN_LOCAL_TENSOR_T * numPerLoop_;
  expAvgSqOffset_ = EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR_T * numPerLoop_;
  gradOffset_ = GRAD_NORM_ORDER_IN_LOCAL_TENSOR_U * numPerLoop_;
  maxGradNormOffset_ = MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR_U * numPerLoop_;
  maxGradOutOffset_ = MAX_GRAD_NORM_ORDER_IN_OUT_LOCAL_TENSOR_U * numPerLoop_;
}

template <typename T, typename U, typename Z>
__aicore__ inline void ApplyAdamWV2MixType<T, U, Z>::ParseTilingData(const ApplyAdamWV2TilingData* tilingData) {
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

    if (tilingData->isBfloat16 != 0){
        isBfloat16_ = true;
    }
}

template <typename T, typename U, typename Z>
__aicore__ inline void ApplyAdamWV2MixType<T, U, Z>::CopyIn(int64_t index, int64_t dataCount) {

    int64_t offset = usedCoreNum_ * index * numPerLoop_;
    LocalTensor<T> dataLocalT = inQueueTypeT_.AllocTensor<T>();
    LocalTensor<U> dataLocalU = inQueueTypeU_.AllocTensor<U>();

    DataCopyExtParams dataCopyParamsT{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParamsT{false, 0, 0, 0};
    DataCopyExtParams dataCopyParamsU{1, static_cast<uint32_t>(dataCount * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> dataCopyPadParamsU{false, 0, 0, 0};
    DataCopyPad(dataLocalT[varOffset_], gmVar_[offset], dataCopyParamsT, dataCopyPadParamsT);
    DataCopyPad(dataLocalT[expAvgOffset_], gmExpAvg_[offset], dataCopyParamsT, dataCopyPadParamsT);
    DataCopyPad(dataLocalT[expAvgSqOffset_], gmExpAvgSq_[offset], dataCopyParamsT, dataCopyPadParamsT);
    DataCopyPad(dataLocalU[gradOffset_], gmGrad_[offset], dataCopyParamsU, dataCopyPadParamsU);
    inQueueTypeT_.EnQue(dataLocalT);
    if (amsgrad_){
        DataCopyPad(dataLocalU[maxGradNormOffset_], gmMaxGradNorm_[offset], dataCopyParamsU, dataCopyPadParamsU);
    }
    inQueueTypeU_.EnQue(dataLocalU);
}

template <typename T, typename U, typename Z>
__aicore__ inline void ApplyAdamWV2MixType<T, U, Z>::Compute(int32_t realProcCount) {
    LocalTensor<T> dataLocalT = inQueueTypeT_.DeQue<T>();
    LocalTensor<U> dataLocalU = inQueueTypeU_.DeQue<U>();

    LocalTensor<T> dataOutLocalT = outQueueTypeT_.AllocTensor<T>();
    LocalTensor<T> resultTempLocal1 = resultTempBuf1_.Get<T>();
    LocalTensor<T> resultTempLocal2 = resultTempBuf2_.Get<T>();
    pipe_barrier(PIPE_V);
    Cast(resultTempLocal1, dataLocalU[gradOffset_], RoundMode::CAST_NONE, realProcCount);
    pipe_barrier(PIPE_V);
    if (maximize_){
        // grad = -grad
        Muls(resultTempLocal1, resultTempLocal1, negOne_, realProcCount);
    }
    // param.mul_(1 - lr * weight_decay)
    pipe_barrier(PIPE_V);
    Muls(dataOutLocalT[varOffset_], dataLocalT[varOffset_], realWeightDecay_, realProcCount);

    // exp_avg.lerp_(grad, 1 - beta1)
    pipe_barrier(PIPE_V);
    Sub(dataOutLocalT[expAvgOffset_], resultTempLocal1, dataLocalT[expAvgOffset_], realProcCount);
    pipe_barrier(PIPE_V);
    Muls(dataOutLocalT[expAvgOffset_], dataOutLocalT[expAvgOffset_], oneSubBeta1_, realProcCount);
    pipe_barrier(PIPE_V);
    Add(dataOutLocalT[expAvgOffset_], dataOutLocalT[expAvgOffset_], dataLocalT[expAvgOffset_], realProcCount);

    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    pipe_barrier(PIPE_V);
    Muls(dataLocalT[expAvgSqOffset_], dataLocalT[expAvgSqOffset_], realBeta2_, realProcCount);
    pipe_barrier(PIPE_V);
    Mul(resultTempLocal1, resultTempLocal1, resultTempLocal1, realProcCount);
    pipe_barrier(PIPE_V);
    Muls(resultTempLocal1, resultTempLocal1, oneSubBeta2_, realProcCount);
    pipe_barrier(PIPE_V);
    Add(dataOutLocalT[expAvgSqOffset_], dataLocalT[expAvgSqOffset_], resultTempLocal1, realProcCount);
    pipe_barrier(PIPE_V);

    if (amsgrad_){
        LocalTensor<U> dataOutLocalU = outQueueTypeU_.AllocTensor<U>();
        // torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
        pipe_barrier(PIPE_V);
        Cast(resultTempLocal2, dataLocalU[maxGradNormOffset_], RoundMode::CAST_NONE, realProcCount);
        pipe_barrier(PIPE_V);
        Max(resultTempLocal2, resultTempLocal2, dataOutLocalT[expAvgSqOffset_], realProcCount);
        pipe_barrier(PIPE_V);
        if(isBfloat16_){
           Cast(dataOutLocalU[maxGradOutOffset_], resultTempLocal2, RoundMode::CAST_ROUND, realProcCount);
        } else {
           Cast(dataOutLocalU[maxGradOutOffset_], resultTempLocal2, RoundMode::CAST_RINT, realProcCount);
        }
        pipe_barrier(PIPE_V);
        outQueueTypeU_.EnQue(dataOutLocalU);

        Sqrt(dataLocalT[varOffset_], resultTempLocal2, realProcCount);
        pipe_barrier(PIPE_V);
        Muls(dataLocalT[varOffset_], dataLocalT[varOffset_], biasCorrection2Sqrt_, realProcCount);
        pipe_barrier(PIPE_V);
        Adds(dataLocalT[varOffset_], dataLocalT[varOffset_], eps_, realProcCount);
    } else {
        // denom = (exp_avg_sq.sqrt() / bias_corrections_sqrt) + eps
        pipe_barrier(PIPE_V);
        Sqrt(dataLocalT[varOffset_], dataOutLocalT[expAvgSqOffset_], realProcCount);
        pipe_barrier(PIPE_V);
        Muls(dataLocalT[varOffset_], dataLocalT[varOffset_], biasCorrection2Sqrt_, realProcCount);
        pipe_barrier(PIPE_V);
        Adds(dataLocalT[varOffset_], dataLocalT[varOffset_], eps_, realProcCount);
    }

    // param.addcdiv_(exp_avg, denom, value=-step_size)
    pipe_barrier(PIPE_V);
    Div(dataLocalT[varOffset_], dataOutLocalT[expAvgOffset_], dataLocalT[varOffset_], realProcCount);
    pipe_barrier(PIPE_V);
    Muls(dataLocalT[varOffset_], dataLocalT[varOffset_], stepSize_, realProcCount);
    pipe_barrier(PIPE_V);
    Sub(dataOutLocalT[varOffset_], dataOutLocalT[varOffset_], dataLocalT[varOffset_], realProcCount);

    inQueueTypeT_.FreeTensor(dataLocalT);
    inQueueTypeU_.FreeTensor(dataLocalU);
    outQueueTypeT_.EnQue(dataOutLocalT);
}

template <typename T, typename U, typename Z>
__aicore__ inline void ApplyAdamWV2MixType<T, U, Z>::CopyOut(int64_t index, int64_t dataCount) {

    int64_t offset = usedCoreNum_ * index * numPerLoop_;
    LocalTensor<T> dataOutLocalT = outQueueTypeT_.DeQue<T>();

    DataCopyExtParams copyParamsT{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(gmVar_[offset], dataOutLocalT[varOffset_], copyParamsT);
    DataCopyPad(gmExpAvg_[offset], dataOutLocalT[expAvgOffset_], copyParamsT);
    DataCopyPad(gmExpAvgSq_[offset], dataOutLocalT[expAvgSqOffset_], copyParamsT);
    outQueueTypeT_.FreeTensor(dataOutLocalT);

    if (amsgrad_){
        DataCopyExtParams copyParamsU{1, static_cast<uint32_t>(dataCount * sizeof(U)), 0, 0, 0};
        LocalTensor<U> dataOutLocalU = outQueueTypeU_.DeQue<U>();
        DataCopyPad(gmMaxGradNorm_[offset], dataOutLocalU[maxGradOutOffset_], copyParamsU);
        outQueueTypeU_.FreeTensor(dataOutLocalU);
    }
}

template <typename T, typename U, typename Z>
__aicore__ inline void ApplyAdamWV2MixType<T, U, Z>::Process() {
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

#endif  // APPLYADAM_W_V2_MIX_DTYPE_H
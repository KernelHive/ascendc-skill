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
 * \file cross_entropy_loss_grad_base.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_GRAD_BASE_H
#define CROSS_ENTROPY_LOSS_GRAD_BASE_H
#include "kernel_operator.h"

namespace CrossEntropyLossGrad {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t MASK = 64;
constexpr int32_t REDUCTION_NONE = 0;
constexpr int32_t REDUCTION_MEAN = 1;
constexpr int32_t REDUCTION_SUM = 2;

template <typename T>
class CrossEntropyLossGradBase {
public:
    __aicore__ inline CrossEntropyLossGradBase(){};
    __aicore__ inline void Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss,
                                GM_ADDR lse_for_zloss, GM_ADDR x_grad, GM_ADDR workspace,
                                const CrossEntropyLossGradTilingData& tilingData);
    __aicore__ inline void InitData(const CrossEntropyLossGradTilingData& tilingData);
    __aicore__ inline void InitUB();
    __aicore__ inline void CopyInLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen);
    __aicore__ inline void CopyOutLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen);
    __aicore__ inline void ComputeIgnoreMask(uint64_t targetNum);
    __aicore__ inline void GradReductionNone(uint64_t targetNum);
    __aicore__ inline void GradReductionMeanSum();
    __aicore__ inline void PipeM2V();

protected:
  TPipe pipe;
  GlobalTensor<T> gradLossGm;
  GlobalTensor<T> logProbGm;
  GlobalTensor<float> weightGm;
  GlobalTensor<int64_t> targetGm;
  GlobalTensor<T> xGradGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueGradLoss;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueLogProb;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueTarget;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueXGrad;

  TBuf<TPosition::VECCALC> gradLossFp32Buf;
  TBuf<TPosition::VECCALC> maskBuf;
  TBuf<TPosition::VECCALC> ignoreSelectBuf;
  TBuf<TPosition::VECCALC> targetCastBuf;

  LocalTensor<float> gradLossFp32Local;
  LocalTensor<uint8_t> mask1;
  LocalTensor<float> ignoreSelect;
  LocalTensor<float> targetCast;

  // tilingdata
  uint64_t reduction;
  int64_t ignoreIndex;
  float labelSmoothing;
  uint64_t rowVal;
  uint64_t colVal;
  uint64_t frontCoreNum;
  uint64_t tailCoreNum;
  uint64_t usedCoreNum;
  uint64_t frontRowNum;
  uint64_t tailRowNum;
  uint64_t alignColLoopNum;
  uint64_t colLoop;
  uint64_t colLoopNumTail;
  uint64_t targetSize;
  uint64_t targetCastSize;
  uint64_t gradLossSize;
  uint64_t gradLossFp32Size;
  uint64_t ignoreSize;
  uint64_t maskSize;
  uint64_t targetWeightSize;
  uint64_t tBuf2Size;
  uint64_t tBuf3Size;

  // init tmp data
  uint32_t coreIndex;
  uint64_t logOffset;
  uint64_t targetOffset;
  uint64_t nLoopNum;
  float meanSumOutGrad;
};

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::InitData(const CrossEntropyLossGradTilingData& tiling) {
  reduction = tiling.reduction;
  ignoreIndex = tiling.ignoreIndex;
  labelSmoothing = tiling.labelSmoothing;
  rowVal = tiling.rowVal;
  colVal = tiling.colVal;
  frontCoreNum = tiling.frontCoreNum;
  tailCoreNum = tiling.tailCoreNum;
  usedCoreNum = tiling.usedCoreNum;
  frontRowNum = tiling.frontRowNum;
  tailRowNum = tiling.tailRowNum;
  alignColLoopNum = tiling.alignColLoopNum;
  colLoop = tiling.colLoop;
  colLoopNumTail = tiling.colLoopNumTail;
  targetSize = tiling.targetSize;
  targetCastSize = tiling.targetCastSize;
  gradLossSize = tiling.gradLossSize;
  gradLossFp32Size = tiling.gradLossFp32Size;
  ignoreSize = tiling.ignoreSize;
  maskSize = tiling.maskSize;
  targetWeightSize = tiling.targetWeightSize;
  tBuf2Size = tiling.tBuf2Size;
  tBuf3Size = tiling.tBuf3Size;
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                         GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                         GM_ADDR x_grad, GM_ADDR workspace,
                                                         const CrossEntropyLossGradTilingData& tilingData) {
  InitData(tilingData);
  coreIndex = GetBlockIdx();
  InitUB();

  if (coreIndex < frontCoreNum) {
    logOffset = coreIndex * colVal * frontRowNum;
    targetOffset = coreIndex * frontRowNum;
    nLoopNum = frontRowNum;
  } else {
    logOffset = frontCoreNum * colVal * frontRowNum + (coreIndex - frontCoreNum) * colVal * tailRowNum;
    targetOffset = frontCoreNum * frontRowNum + (coreIndex - frontCoreNum) * tailRowNum;
    nLoopNum = tailRowNum;   // 确定该核需要处理多少行，target一次处理多少个数
  }

  gradLossGm.SetGlobalBuffer((__gm__ T*)grad_loss);
  logProbGm.SetGlobalBuffer((__gm__ T*)log_prob + logOffset);
  targetGm.SetGlobalBuffer((__gm__ int64_t*)target);
  weightGm.SetGlobalBuffer((__gm__ float*)weight);
  xGradGm.SetGlobalBuffer((__gm__ T*)x_grad + logOffset);
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::InitUB() {
  uint32_t inOutQueInputSize = MASK;
  inOutQueInputSize = this->alignColLoopNum > this->colLoopNumTail ?
                      this->alignColLoopNum : AlignUp(this->colLoopNumTail, MASK);
  this->pipe.InitBuffer(inQueLogProb, BUFFER_NUM, inOutQueInputSize * sizeof(T));
  this->pipe.InitBuffer(outQueXGrad, BUFFER_NUM, inOutQueInputSize * sizeof(T));
  this->pipe.InitBuffer(inQueTarget, BUFFER_NUM, this->targetSize);
  if (this->reduction == REDUCTION_NONE) {
    this->pipe.InitBuffer(inQueGradLoss, BUFFER_NUM, this->gradLossSize);
    this->pipe.InitBuffer(gradLossFp32Buf, this->gradLossFp32Size);
    gradLossFp32Local = gradLossFp32Buf.Get<float>();
  }
  this->pipe.InitBuffer(maskBuf, this->maskSize);
  this->pipe.InitBuffer(ignoreSelectBuf, this->ignoreSize);
  this->pipe.InitBuffer(targetCastBuf, this->targetCastSize);

  mask1 = maskBuf.Get<uint8_t>();
  ignoreSelect = ignoreSelectBuf.Get<float>();
  targetCast = targetCastBuf.Get<float>();
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::CopyInLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen) {
  LocalTensor<T> logProbLocal = inQueLogProb.AllocTensor<T>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLen * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  DataCopyPad(logProbLocal, this->logProbGm[nLoopIdx * this->colVal + cLoopIdx * this->alignColLoopNum],
              copyParams, padParams);
  inQueLogProb.EnQue(logProbLocal);
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::CopyOutLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen) {
  LocalTensor<T> xGradLocal = outQueXGrad.DeQue<T>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLen * sizeof(T)), 0, 0, 0};
  DataCopyPad(this->xGradGm[nLoopIdx * this->colVal + cLoopIdx * this->alignColLoopNum], xGradLocal, copyParams);
  outQueXGrad.FreeTensor(xGradLocal);
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::ComputeIgnoreMask(uint64_t targetNum) {
  LocalTensor<int64_t> targetLocal = inQueTarget.AllocTensor<int64_t>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(targetNum * sizeof(int64_t)), 0, 0, 0};
  DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
  DataCopyPad(targetLocal, this->targetGm[this->targetOffset], copyParams, padParams);
  inQueTarget.EnQue(targetLocal);
  targetLocal = inQueTarget.DeQue<int64_t>();

  uint64_t repeat = (targetNum + MASK - 1) / MASK;
  BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
  Cast(targetCast, targetLocal, RoundMode::CAST_RINT, targetNum);  // int64 -> fp32
  inQueTarget.FreeTensor(targetLocal);
  Duplicate(ignoreSelect, static_cast<float>(this->ignoreIndex), targetNum);
  if (targetNum >= MASK) {
    Compare(mask1, targetCast, ignoreSelect, CMPMODE::EQ, MASK, repeat, repeatParams);
  } else {
    Compare(mask1, targetCast, ignoreSelect, CMPMODE::EQ, targetNum, 1, repeatParams);
  }
  Duplicate(ignoreSelect, static_cast<float>(0), targetNum);
  AscendC::Select(ignoreSelect, mask1, ignoreSelect, static_cast<float>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, targetNum);
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::GradReductionNone(uint64_t targetNum) {
  LocalTensor<T> gradLossLocal = inQueGradLoss.AllocTensor<T>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(targetNum * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  DataCopyPad(gradLossLocal, this->gradLossGm[this->targetOffset], copyParams, padParams);
  inQueGradLoss.EnQue(gradLossLocal);
  gradLossLocal = inQueGradLoss.DeQue<T>();
  if constexpr (!IsSameType<T, float>::value) {
    Cast(gradLossFp32Local, gradLossLocal, RoundMode::CAST_NONE, targetNum);   // 16->32
    inQueGradLoss.FreeTensor(gradLossLocal);
    Muls(gradLossFp32Local, gradLossFp32Local, 1 - this->labelSmoothing, targetNum);
    Mul(targetCast, ignoreSelect, gradLossFp32Local, targetNum);    // loss_out_grad
  } else {
    Muls(gradLossLocal, gradLossLocal, 1 - this->labelSmoothing, targetNum);
    Mul(targetCast, ignoreSelect, gradLossLocal, targetNum);    // loss_out_grad
    inQueGradLoss.FreeTensor(gradLossLocal);
  }
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::GradReductionMeanSum() {
  if constexpr (std::is_same<T, bfloat16_t>::value) {
    meanSumOutGrad = ToFloat(this->gradLossGm.GetValue(0)) * (1 - labelSmoothing);
  } else if constexpr (std::is_same<T, half>::value) {
    meanSumOutGrad = static_cast<float>(this->gradLossGm.GetValue(0)) * (1 - labelSmoothing);
  } else if constexpr (std::is_same<T, float>::value) {
    meanSumOutGrad = this->gradLossGm.GetValue(0) * (1 - labelSmoothing);
  }
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradBase<T>::PipeM2V() {
    event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
}

} // namespace CrossEntropyLossGrad
#endif  // CROSS_ENTROPY_LOSS_GRAD_BASE_H
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
 * \file cross_entropy_loss_grad_weight_none.h
 * \brief
 */
#ifndef CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NONE_H
#define CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NONE_H

#include "cross_entropy_loss_grad_base.h"

namespace CrossEntropyLossGrad {
using namespace AscendC;

template <typename T>
class CrossEntropyLossGradWeightNone: protected CrossEntropyLossGradBase<T> {
public:
  __aicore__ inline CrossEntropyLossGradWeightNone(){};
  __aicore__ inline void Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss,
                              GM_ADDR lse_for_zloss, GM_ADDR x_grad, GM_ADDR workspace,
                              const CrossEntropyLossGradTilingData& tilingData);
  __aicore__ inline void Process();

protected:
  __aicore__ inline void ComputerEachBatch(uint64_t nLoopIdx, uint64_t nLoopNum);
  __aicore__ inline void ComputeLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen);
  __aicore__ inline void LogProbGradLoss(uint64_t nLoopIdx, uint64_t calcLen, uint64_t cloopOffset,
                                         uint64_t targetValue, uint64_t posIdx);
  __aicore__ inline void WeightAfterMaskSum(uint64_t targetNum);

private:
  GlobalTensor<float> workspaceGm;

  TBuf<TPosition::VECCALC> fp32Buf4;
  TBuf<TPosition::VECCALC> blockBuf2;
  TBuf<TPosition::VECCALC> weightMask;

  LocalTensor<T> gradLossLocal;
  LocalTensor<T> xGradLocal;
  LocalTensor<float> fp32Buf4Local;
  LocalTensor<float> blockBuf2Local;
  LocalTensor<float> weightMaskLocal;
};

template <typename T>
__aicore__ inline void CrossEntropyLossGradWeightNone<T>::Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                           GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                           GM_ADDR x_grad, GM_ADDR workspace,
                                                           const CrossEntropyLossGradTilingData& tilingData) {
  CrossEntropyLossGradBase<T>::Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
                                    x_grad, workspace, tilingData);
  workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);

  this->pipe.InitBuffer(fp32Buf4, this->alignColLoopNum * 4);
  fp32Buf4Local = fp32Buf4.Get<float>();

  if (this->reduction == REDUCTION_MEAN) {
    this->pipe.InitBuffer(blockBuf2, this->tBuf2Size);
    this->pipe.InitBuffer(weightMask, this->tBuf3Size);
    blockBuf2Local = blockBuf2.Get<float>();
    weightMaskLocal = weightMask.Get<float>();
  }
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradWeightNone<T>::WeightAfterMaskSum(uint64_t targetNum) {
  if (targetNum != 0) {
    ReduceSum(blockBuf2Local, this->ignoreSelect, this->ignoreSelect, targetNum);
  }
  DataCopyPad(workspaceGm[this->coreIndex], blockBuf2Local, {1, (uint32_t)(1 * sizeof(float)), 0, 0, 0});
  #ifndef __CCE_KT_TEST__
    SyncAll();
  #endif
  DataCopyPad(weightMaskLocal, workspaceGm, {1, (uint32_t)(this->usedCoreNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
  this->PipeM2V();
  ReduceSum(blockBuf2Local, weightMaskLocal, weightMaskLocal, this->usedCoreNum);  // blockBuf2Local为48个数的和
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradWeightNone<T>::LogProbGradLoss(uint64_t nLoopIdx, uint64_t calcLen,
                          uint64_t cloopOffset, uint64_t targetValue, uint64_t posIdx) {
  float nllLossGradScalar = this->targetCast.GetValue(nLoopIdx);
  Muls(fp32Buf4Local, fp32Buf4Local, nllLossGradScalar, calcLen);
  if (cloopOffset <= targetValue && targetValue <= cloopOffset + calcLen) {
    fp32Buf4Local.SetValue(posIdx, fp32Buf4Local.GetValue(posIdx) - nllLossGradScalar);
  }
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradWeightNone<T>::ComputeLog(uint64_t nLoopIdx, uint64_t cLoopIdx,
                                                                     uint64_t calcLen) {
  LocalTensor<T> logProbLocal = this->inQueLogProb.template DeQue<T>();
  xGradLocal = this->outQueXGrad.template AllocTensor<T>();
  uint64_t cloopOffset = cLoopIdx * this->alignColLoopNum;    // 一个核内，一行的偏移量
  uint64_t targetValue = this->targetGm.GetValue(this->targetOffset + nLoopIdx);
  uint64_t posIdx = targetValue - cLoopIdx * this->alignColLoopNum;

  if constexpr (!IsSameType<T, float>::value) {
    Cast(fp32Buf4Local, logProbLocal, RoundMode::CAST_NONE, calcLen);
    this->inQueLogProb.template FreeTensor(logProbLocal);
    Exp(fp32Buf4Local, fp32Buf4Local, calcLen);
    LogProbGradLoss(nLoopIdx, calcLen, cloopOffset, targetValue, posIdx);
    if (this->labelSmoothing == 0) {
      Cast(xGradLocal, fp32Buf4Local, RoundMode::CAST_RINT, calcLen);
      this->outQueXGrad.template EnQue<T>(xGradLocal);
    }
  } else {
    Exp(fp32Buf4Local, logProbLocal, calcLen);
    this->inQueLogProb.template FreeTensor(logProbLocal);
    LogProbGradLoss(nLoopIdx, calcLen, cloopOffset, targetValue, posIdx);
    if (this->labelSmoothing == 0) {
      Copy(xGradLocal, fp32Buf4Local, MASK, (calcLen + MASK -1) / MASK, {1, 1, 8, 8});
      this->outQueXGrad.template EnQue<T>(xGradLocal);
    }
  }
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradWeightNone<T>::ComputerEachBatch(uint64_t nLoopIdx, uint64_t nLoopNum) {
  for (uint64_t cLoopIdx = 0; cLoopIdx < this->colLoop; cLoopIdx++) {
    this->CopyInLog(nLoopIdx, cLoopIdx, this->alignColLoopNum);
    ComputeLog(nLoopIdx, cLoopIdx, this->alignColLoopNum);
    this->CopyOutLog(nLoopIdx, cLoopIdx, this->alignColLoopNum);
  }
  if (this->colLoopNumTail != 0) {
    this->CopyInLog(nLoopIdx, this->colLoop, this->colLoopNumTail);
    pipe_barrier(PIPE_ALL);
    ComputeLog(nLoopIdx, this->colLoop, this->colLoopNumTail);
    this->CopyOutLog(nLoopIdx, this->colLoop, this->colLoopNumTail);
    pipe_barrier(PIPE_ALL);
  }
}

template <typename T>
__aicore__ inline void CrossEntropyLossGradWeightNone<T>::Process() {
  this->ComputeIgnoreMask(this->nLoopNum);  // ignore_mask
  if (this->reduction == REDUCTION_NONE) {
    pipe_barrier(PIPE_ALL);
    this->GradReductionNone(this->nLoopNum);
    for (uint64_t nLoopIdx = 0; nLoopIdx < this->nLoopNum; nLoopIdx++) {
      ComputerEachBatch(nLoopIdx, this->nLoopNum);
    }
  }
  else if (this->reduction == REDUCTION_MEAN) {
    WeightAfterMaskSum(this->nLoopNum);
    this->GradReductionMeanSum();
    float loss_out_grad = this->meanSumOutGrad / blockBuf2Local.GetValue(0);
    Muls(this->targetCast, this->ignoreSelect, loss_out_grad, this->nLoopNum);   // loss_out_grad = nll_loss_grad
    for (uint64_t nLoopIdx = 0; nLoopIdx < this->nLoopNum; nLoopIdx++) {
      ComputerEachBatch(nLoopIdx, this->nLoopNum);
    }
  }
  else {
    this->GradReductionMeanSum();
    pipe_barrier(PIPE_ALL);
    Muls(this->targetCast, this->ignoreSelect, this->meanSumOutGrad, this->nLoopNum);   // loss_out_grad = nll_loss_grad
    for (uint64_t nLoopIdx = 0; nLoopIdx < this->nLoopNum; nLoopIdx++) {
      ComputerEachBatch(nLoopIdx, this->nLoopNum);
    }
  }
}
} // namespace CrossEntropyLossGrad
#endif  // CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NONE_H
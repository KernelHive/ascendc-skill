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
 * \file avg_pool3d_ncdhw_reduce_d.h
 * \brief
 */

#ifndef AVG_POOL3D_NCDHW_REDUCE_D_H_
#define AVG_POOL3D_NCDHW_REDUCE_D_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dReduceD {
public:
  __aicore__ inline KernelAvgPool3dReduceD() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe);
  __aicore__ inline void Process();

private:
  __aicore__ inline void InitTiling(const AvgPool3DTilingData* tiling);
  __aicore__ inline void CopyIn(int64_t offset, int64_t len);
  __aicore__ inline void CopyOut(int64_t offset, int64_t len);
  __aicore__ inline void ReduceMeanDWindow(int64_t dIdx);
  __aicore__ inline void ReduceSumDWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t startOffset, int64_t len);
  
  TPipe* pipe;
  TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
  TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

  TBuf<TPosition::VECCALC> sumBuf;
  LocalTensor<float> sumBufLocal;

  GlobalTensor<T> inputGlobal;
  GlobalTensor<T> outputGlobal;

  int64_t hwLength;
  int64_t tileHW;
  int64_t ncdBlockLength;
  int64_t ncdOffset;

  ShapeAvgPool3d inputShape;
  ShapeAvgPool3d outputShape;

  int64_t indexBufLen;
  IndexBuffer indexBuf;
  PoolParameter poolParam;

  uint32_t numPerBlock;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* tiling) {
  inputShape = ShapeAvgPool3d(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
  outputShape = ShapeAvgPool3d(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

  poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                            tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);
  
  numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);

  hwLength = tiling->inH * tiling->inW;
  tileHW = tiling->tileHW;

  ncdBlockLength = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
  ncdOffset = GetBlockIdx() < tiling->formerNum
    ? tiling->formerLength * GetBlockIdx()
    : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::CopyIn(int64_t offset, int64_t len) {
  LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
  if constexpr (std::is_same_v<T, float>) {
    DataCopy(inputLocal, inputGlobal[offset], len);
  } else {
    DataCopy(inputLocal[tileHW], inputGlobal[offset], len);
  }
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  if constexpr (std::is_same_v<T, float>) {
    DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
  } else {
    DataCopyPad(inputLocal[tileHW], inputGlobal[offset], copyParams, padParams);
  }
#endif
  inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
  LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
  DataCopy(outputGlobal[offset], outputLocal, len);
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPad(outputGlobal[offset], outputLocal, copyParams);
#endif
  outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::ReduceSumDWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t startOffset, int64_t len) {
  int64_t dstart = index.D.start;
  int64_t dend = index.D.end;

  for (int64_t id = dstart; id < dend; ++id) {
    int64_t dOffset = id * inputShape.strideD;

    CopyIn(startOffset + dOffset, len);

    LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
    if constexpr (std::is_same_v<T, float>) {
      Add(sumBufLocal, sumBufLocal, inputLocal, len);
    } else {
      Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[tileHW], RoundMode::CAST_NONE, len);
      Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>(), len);
    }
    inputQueue.FreeTensor(inputLocal);
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::ReduceMeanDWindow(int64_t dIdx) {
  Index index;

  uint64_t ncIdx = dIdx / outputShape.D;
  uint64_t outputDIdx = dIdx % outputShape.D;
  index.D.Compute(outputDIdx, inputShape.D, poolParam.kernelD, poolParam.strideD, poolParam.padD,
                  poolParam.countIncludePad);

  int64_t poolSize = poolParam.divisorOverride ? poolParam.divisorOverride : index.D.poolSize;
  float factor = 1.0f / static_cast<float>(poolSize);

  SToVSync();

  int64_t hwLoop = (hwLength + tileHW - 1) / tileHW;
  int64_t hwOffset = 0;
  for (int64_t i = 0; i < hwLoop; ++i) {
    int64_t count = i < hwLoop - 1 ? tileHW : hwLength - (hwLoop - 1) * tileHW;

    Duplicate(sumBufLocal, 0.0f, count);

    int64_t startOffset = ncIdx * inputShape.strideC + hwOffset;

    ReduceSumDWindow(index, sumBufLocal, startOffset, count);
    Muls(sumBufLocal, sumBufLocal, factor, count);

    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
      DataCopy(outputLocal, sumBufLocal, AlignUp(count, numPerBlock));
    } else {
      Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, count);
    }
    outputQueue.EnQue(outputLocal);

    CopyOut(ncIdx * outputShape.strideC + outputDIdx * hwLength + hwOffset, count);

    hwOffset += count;
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe) {
  InitTiling(tiling);

  inputGlobal.SetGlobalBuffer((__gm__ T*)x);
  outputGlobal.SetGlobalBuffer((__gm__ T*)y);

  pipe->InitBuffer(inputQueue, QUEUE_DEPTH, tileHW * sizeof(float));
  pipe->InitBuffer(outputQueue, QUEUE_DEPTH, tileHW * sizeof(T));

  pipe->InitBuffer(sumBuf, tileHW * sizeof(float));
  sumBufLocal = sumBuf.Get<float>();
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::Process() {
  for (int64_t dIdx = ncdOffset; dIdx < ncdOffset + ncdBlockLength; ++dIdx) {
    ReduceMeanDWindow(dIdx);
  }
}

#endif // AVG_POOL3D_NCDHW_REDUCE_D_H_

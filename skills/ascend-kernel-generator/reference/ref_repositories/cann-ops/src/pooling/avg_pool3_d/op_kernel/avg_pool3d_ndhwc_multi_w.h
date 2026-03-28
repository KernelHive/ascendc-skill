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
 * \file avg_pool3d_ndhwc_multi_w.h
 * \brief
 */

#ifndef AVG_POOL3D_NDHWC_MULTI_W_H_
#define AVG_POOL3D_NDHWC_MULTI_W_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dMultiW {
public:
  __aicore__ inline KernelAvgPool3dMultiW() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe);
  __aicore__ inline void Process();

private:
  __aicore__ inline void InitTiling(const AvgPool3DTilingData* tiling);
  __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding);
  __aicore__ inline void CopyOut(int64_t offset, uint16_t blockCount, uint32_t blockLen);
  __aicore__ inline void ReduceMeanMultiWindow(int64_t outputPointIdx, int64_t windowNum);
  __aicore__ inline void ReduceSumMultiWindow(
    const Index& startIndex, const Index& endIndex, LocalTensor<float>& sumBufLocal,
    int64_t outputPointIdx, int64_t nOffset, int64_t windowNum);
  __aicore__ inline void ReduceSumRow(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal,
    int64_t outputPointIdx, int64_t windowNum);
  __aicore__ inline void ReduceSumRowRepeat(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal, int64_t windowNum);
  
  TPipe* pipe;
  TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
  TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

  TBuf<TPosition::VECCALC> sumBuf;
  TBuf<TPosition::VECCALC> castBuf;
  LocalTensor<float> sumBufLocal;
  LocalTensor<float> castBufLocal;

  GlobalTensor<T> inputGlobal;
  GlobalTensor<T> outputGlobal;

  int64_t inC;
  int64_t alignC;
  int64_t outputPointNum;
  int64_t outputPointOffset;
  int64_t windowWNum;

  ShapeAvgPool3d inputShape;
  ShapeAvgPool3d outputShape;

  int64_t indexBufLen;
  IndexBuffer indexBuf;
  PoolParameter poolParam;

  uint32_t numPerBlock;
  uint32_t inputBufLen;

  bool isSumWithRepeat;
  bool isSamePoolSize;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* tiling) {
  inputShape = ShapeAvgPool3d(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
  outputShape = ShapeAvgPool3d(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

  poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                            tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

  indexBuf.SetComputeParameter(outputShape, inputShape, poolParam);
  
  numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
  inC = tiling->inC;
  alignC = AlignUp(inC, numPerBlock);
  windowWNum = tiling->windowWNum;

  outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
  outputPointOffset = GetBlockIdx() < tiling->formerNum
    ? tiling->formerLength * GetBlockIdx()
    : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);

  indexBufLen = tiling->indexBufLen;

  uint32_t floatNumPerBlock = GetDataBlockSizeInBytes() / sizeof(float);
  uint32_t src1RepStride = alignC / floatNumPerBlock * poolParam.strideW;

  isSumWithRepeat = (poolParam.padW == 0 && !tiling->ceilMode) && src1RepStride <= UINT8_MAX;
  isSamePoolSize = 
    poolParam.divisorOverride || ((poolParam.countIncludePad || poolParam.padW == 0) && !tiling->ceilMode);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
  LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
  if constexpr (std::is_same_v<T, float>) {
    DataCopy(inputLocal, inputGlobal[offset], blockCount * blockLen);
  } else {
    DataCopy(inputLocal[inputBufLen], inputGlobal[offset], blockCount * blockLen);
  }
#else
  DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};
  if constexpr (std::is_same_v<T, float>) {
    DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
  } else {
    DataCopyPad(inputLocal[inputBufLen], inputGlobal[offset], copyParams, padParams);
  }
#endif
  inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyOut(
    int64_t offset, uint16_t blockCount, uint32_t blockLen) {
  LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
  DataCopy(outputGlobal[offset], outputLocal, blockCount * blockLen);
#else
  DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
  DataCopyPad(outputGlobal[offset], outputLocal, copyParams);
#endif
  outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumRow(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal,
    int64_t outputPointIdx, int64_t windowNum) {
  for (int64_t in = outputPointIdx, offset = 0; in < outputPointIdx + windowNum; ++in, offset += alignC) {
    Index index;
    indexBuf.GetWIndex(in, index);

    SToVSync();

    for (int64_t iw = index.W.start - startIndex.W.start; iw < index.W.end - startIndex.W.start; ++iw) {
      if constexpr (std::is_same_v<T, float>) {
        Add(sumBufLocal[offset], sumBufLocal[offset], inputLocal[iw * alignC], alignC);
      } else {
        Add(sumBufLocal[offset], sumBufLocal[offset],
            inputLocal.template ReinterpretCast<float>()[iw * alignC], alignC);
      }
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumRowRepeat(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal, int64_t windowNum) {
  int64_t poolSize = startIndex.W.end - startIndex.W.start;

  uint32_t floatNumPerBlock = GetDataBlockSizeInBytes() / sizeof(float);
  int64_t loop = (alignC + floatNumPerBlock * 8 - 1) / (floatNumPerBlock * 8);

  uint8_t repStride = alignC / floatNumPerBlock;
  uint8_t src1RepStride = alignC / floatNumPerBlock * poolParam.strideW;

  for (int64_t i = 0; i < poolSize; ++i) {
    for (int64_t j = 0; j < loop; ++j) {
      int64_t mask = j < loop - 1 ? floatNumPerBlock * 8 : alignC - (loop - 1) * floatNumPerBlock * 8;

      BinaryRepeatParams repeatParams;
      repeatParams.dstBlkStride = 1;
      repeatParams.src0BlkStride = 1;
      repeatParams.src1BlkStride = 1;
      repeatParams.dstRepStride = repStride;
      repeatParams.src0RepStride = repStride;
      repeatParams.src1RepStride = src1RepStride;

      int64_t offset = j * floatNumPerBlock * 8;
      int64_t src1Offset = i * alignC + offset;

      if constexpr (std::is_same_v<T, float>) {
        Add(sumBufLocal[offset], sumBufLocal[offset], inputLocal[src1Offset], mask, windowNum, repeatParams);
      } else {
        Add(sumBufLocal[offset], sumBufLocal[offset],
            inputLocal.template ReinterpretCast<float>()[src1Offset], mask, windowNum, repeatParams);
      }
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumMultiWindow(
    const Index& startIndex, const Index& endIndex, LocalTensor<float>& sumBufLocal,
    int64_t outputPointIdx, int64_t nOffset, int64_t windowNum) {
  int64_t dstart = startIndex.D.start;
  int64_t dend = startIndex.D.end;
  int64_t hstart = startIndex.H.start;
  int64_t hend = startIndex.H.end;
  int64_t wStartOffset = startIndex.W.start * inC;

  uint16_t blockCount = static_cast<uint16_t>(endIndex.W.end - startIndex.W.start);
  uint8_t rightPadding = static_cast<uint8_t>(alignC - inC);

  for (int64_t id = dstart; id < dend; ++id) {
    int64_t dOffset = id * inputShape.strideD * inC;
    for (int64_t ih = hstart; ih < hend; ++ih) {
      int64_t hOffset = ih * inputShape.strideH * inC;

      CopyIn(nOffset * inputShape.strideN + dOffset + hOffset + wStartOffset, blockCount, inC, rightPadding);
      LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();

      if constexpr (!std::is_same_v<T, float>) {
        Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[inputBufLen], RoundMode::CAST_NONE, inputBufLen);
      }

      if (isSumWithRepeat) [[likely]] {
        ReduceSumRowRepeat(startIndex, sumBufLocal, inputLocal, windowNum);
      } else {
        ReduceSumRow(startIndex, sumBufLocal, inputLocal, outputPointIdx, windowNum);
      }

      inputQueue.FreeTensor(inputLocal);
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceMeanMultiWindow(
    int64_t outputPointIdx, int64_t windowNum) {
  Index startIndex;
  indexBuf.GetIndex(outputPointIdx, startIndex);
  Index endIndex;
  indexBuf.GetIndex(outputPointIdx + windowNum - 1, endIndex);

  int64_t len = windowNum * alignC;

  SToVSync();

  Duplicate(sumBufLocal, 0.0f, len);

  ReduceSumMultiWindow(startIndex, endIndex, sumBufLocal, outputPointIdx,
                       outputPointIdx / outputShape.strideC, windowNum);

  if (isSamePoolSize) [[likely]] {
    int64_t poolSize = poolParam.divisorOverride
                        ? poolParam.divisorOverride
                        : startIndex.D.poolSize * startIndex.H.poolSize * startIndex.W.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);

    Muls(sumBufLocal, sumBufLocal, factor, windowWNum * alignC);
  } else {
    for (int64_t i = outputPointIdx, offset = 0; i < outputPointIdx + windowNum; ++i, offset += alignC) {
      Index index;
      indexBuf.GetWIndex(i, index);
      int64_t poolSize = startIndex.D.poolSize * startIndex.H.poolSize * index.W.poolSize;
      float factor = 1.0f / static_cast<float>(poolSize);

      SToVSync();

      Muls(sumBufLocal[offset], sumBufLocal[offset], factor, alignC);
    }
  }

  LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
  if constexpr (std::is_same_v<T, float>) {
    DataCopy(outputLocal, sumBufLocal, len);
  } else {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
  }
  outputQueue.EnQue(outputLocal);

  CopyOut(outputPointIdx * inC, static_cast<uint16_t>(windowNum), static_cast<uint32_t>(inC));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe) {
  InitTiling(tiling);

  inputGlobal.SetGlobalBuffer((__gm__ T*)x);
  outputGlobal.SetGlobalBuffer((__gm__ T*)y);

  inputBufLen = (windowWNum * poolParam.strideW + poolParam.kernelW) * alignC;

  pipe->InitBuffer(inputQueue, QUEUE_DEPTH, inputBufLen * sizeof(float));
  pipe->InitBuffer(outputQueue, QUEUE_DEPTH, windowWNum * alignC * sizeof(T));

  pipe->InitBuffer(sumBuf, windowWNum * alignC * sizeof(float));
  sumBufLocal = sumBuf.Get<float>();

  indexBuf.Init(pipe, indexBufLen);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::Process() {
  int64_t curWindowWNum = windowWNum;
  for (int64_t outputPointIdx = outputPointOffset, count = 0;
       outputPointIdx < outputPointOffset + outputPointNum; outputPointIdx += curWindowWNum, count += curWindowWNum) {
    curWindowWNum = (count + windowWNum) < outputPointNum ? windowWNum : outputPointNum - count;

    int64_t newRowWindowWNum = (outputPointIdx + curWindowWNum) % outputShape.W;
    curWindowWNum = newRowWindowWNum != 0 && newRowWindowWNum < curWindowWNum
                      ? curWindowWNum - newRowWindowWNum : curWindowWNum;

    ReduceMeanMultiWindow(outputPointIdx, curWindowWNum);
  }
}

#endif // AVG_POOL3D_NDHWC_MULTI_W_H_

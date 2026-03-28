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
 * \file adaptive_avg_pool3d_split_w.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_SPLIT_W_H_
#define ADAPTIVE_AVG_POOL3D_SPLIT_W_H_

#include "kernel_operator.h"
#include "adaptive_avg_pool3d_common.h"

template <typename T, int32_t QUEUE_DEPTH>
class KernelAdaptiveAvgPool3dSplitW {
public:
  __aicore__ inline KernelAdaptiveAvgPool3dSplitW() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                              const AdaptiveAvgPool3dTilingData* tiling, TPipe* pipe);
  __aicore__ inline void Process();

private:
  __aicore__ inline void InitTiling(const AdaptiveAvgPool3dTilingData* tiling);
  __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding);
  __aicore__ inline void CopyOut(int64_t outputPointIdx, int64_t len);
  __aicore__ inline void DataCopyOutNonPad(LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint32_t validDataLen);
  __aicore__ inline void ReduceMeanWindow(int64_t outputPointIdx, int64_t bufIdx);
  __aicore__ inline void ReduceSumWindow(const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset);

  TPipe* pipe;
  TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
  TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;
  TQue<QuePosition::VECOUT, QUEUE_DEPTH> zeroQueue;
  TBuf<QuePosition::VECCALC> tmpPattern;

  TBuf<TPosition::VECCALC> sumBuf;
  TBuf<TPosition::VECCALC> castBuf;

  GlobalTensor<T> inputGlobal;
  GlobalTensor<T> outputGlobal;

  int64_t cLength;
  int64_t cLengthAligned;
  int64_t outputPointNum;
  int64_t outputPointOffset;
  int64_t lastPointOffset;
  int64_t inputPointTileNum;
  int64_t atomicAddNum;

  AdaAvgPool3DShape inputShape;
  AdaAvgPool3DShape outputShape;

  int64_t indexBufLen;
  IndexBuffer indexBuf;

  uint32_t numPerBlock;
  int32_t validTailLen;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::InitTiling(
    const AdaptiveAvgPool3dTilingData* tiling) {
  inputShape = AdaAvgPool3DShape(tiling->inD, tiling->inH, tiling->inW);
  outputShape = AdaAvgPool3DShape(tiling->outD, tiling->outH, tiling->outW);

  numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
  cLength = tiling->dimC;
  cLengthAligned = AlignUp(cLength, numPerBlock);
  inputPointTileNum = tiling->inputTileNum;
  indexBufLen = tiling->indexBufLen;

  outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
  outputPointOffset = GetBlockIdx() < tiling->formerNum 
    ? tiling->formerLength * GetBlockIdx()
    : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
  lastPointOffset = outputPointNum + outputPointOffset - 1;
  atomicAddNum = outputPointNum < tiling->atomicAddNum ? outputPointNum : tiling->atomicAddNum;
  validTailLen = cLength % numPerBlock;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::CopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
  LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
  if (blockLen == cLengthAligned) {
    DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
    DataCopy(inputLocal, inputGlobal[offset], copyParams);
  } else {
    for (int i = 0; i < blockCount; i++) {
      DataCopy(inputLocal[i * cLengthAligned], inputGlobal[offset + i * blockLen], cLengthAligned);
    }
  }
#else
  DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};
  DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
#endif
  inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::DataCopyOutNonPad(
  LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint32_t validDataLen) {
  if ((validDataLen < numPerBlock) && (outputPointIdx >= lastPointOffset - atomicAddNum)) {
    uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
    uint64_t mask[2] = {mask0, 0};
    Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
    if (outputPointIdx > lastPointOffset - atomicAddNum) {
      SetAtomicAdd<T>();
      DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, cLengthAligned);
      SetAtomicNone();
    } else {
      DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, cLengthAligned);
    }
  } else if (outputPointIdx == lastPointOffset) {
    DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, cLengthAligned - numPerBlock);
    int32_t lastLeftShift = validTailLen;
    uint32_t mask = numPerBlock * 2;
    uint64_t rsvdCnt = 0;
    uint64_t gatherOffset = cLengthAligned - mask;
    MTE3ToVSync();
    if constexpr (std::is_same_v<T, float>) {
      LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
      int32_t preLeftShift = numPerBlock + lastLeftShift;

      bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
      GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
    } else {
      LocalTensor<uint16_t> bufPattern = tmpPattern.Get<uint16_t>();
      int32_t preLeftShift = numPerBlock - lastLeftShift;

      bufPattern.SetValue(0, ((1u << preLeftShift) - 1u) << lastLeftShift);
      bufPattern.SetValue(1, (1u << lastLeftShift) - 1u);
      GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
    }
    DataCopy(outputGlobal[(outputPointIdx + 1) * validDataLen - numPerBlock], outputLocal[gatherOffset], numPerBlock);
  } else {
    DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, cLengthAligned);
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::CopyOut(int64_t outputPointIdx, int64_t len) {
  LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
  if (len == cLengthAligned) {
    DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
    DataCopy(outputGlobal[outputPointIdx * len], outputLocal, copyParams);
  } else {
    DataCopyOutNonPad(outputLocal, outputPointIdx, len);
  }
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPad(outputGlobal[outputPointIdx * len], outputLocal, copyParams);
#endif
  outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::ReduceSumWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset) {
  int64_t dstart = index.dstart;
  int64_t dend = index.dend;
  int64_t hstart = index.hstart;
  int64_t hend = index.hend;
  int64_t wstart = index.wstart;
  int64_t wend = index.wend;

  int64_t wLoop = (wend - wstart + inputPointTileNum - 1) / inputPointTileNum;
  uint8_t rightPadding = static_cast<uint8_t>(cLengthAligned - cLength);
  int64_t kW = wend - wstart;

  for (int64_t id = dstart; id < dend; ++id) {
    int64_t dOffset = id * inputShape.dstride * cLength;
    for (int64_t ih = hstart; ih < hend; ++ih) {
      int64_t hOffset = ih * inputShape.hstride * cLength;
      for (int64_t i = 0, iw = wstart; i < wLoop; ++i) {
        int64_t tileNum = i < wLoop - 1 ? inputPointTileNum : kW - (wLoop - 1) * inputPointTileNum;

        CopyIn(nOffset + dOffset + hOffset + iw * cLength, static_cast<uint16_t>(tileNum),
               static_cast<uint32_t>(cLength), rightPadding);
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();

        for (int64_t i = 0; i < tileNum; ++i) {
          if constexpr (std::is_same_v<T, float>) {
            Add(sumBufLocal, sumBufLocal, inputLocal[i * cLengthAligned], cLengthAligned);
          } else {
            LocalTensor<float> castBufLocal = castBuf.Get<float>();
            Cast(castBufLocal, inputLocal[i * cLengthAligned], RoundMode::CAST_NONE, cLengthAligned);
            Add(sumBufLocal, sumBufLocal, castBufLocal, cLengthAligned);
          }
        }
        
        iw += tileNum;

        inputQueue.FreeTensor(inputLocal);
      }
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::ReduceMeanWindow(
    int64_t outputPointIdx, int64_t bufIdx) {
  Index index;
  GetIndexFromBuffer(indexBuf, bufIdx, bufIdx, index);

  SToVSync();

  LocalTensor<float> sumBufLocal = sumBuf.Get<float>();
  Duplicate(sumBufLocal, 0.0f, cLengthAligned);
  int64_t nOffset = outputPointIdx / outputShape.nstride * inputShape.nstride * cLength;

  ReduceSumWindow(index, sumBufLocal, nOffset);

  float factor = 1.0f / (
    static_cast<float>(index.dend - index.dstart) * (index.hend - index.hstart) * (index.wend - index.wstart));
  Muls(sumBufLocal, sumBufLocal, factor, cLengthAligned);

  LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
  if constexpr (std::is_same_v<T, float>) {
    DataCopy(outputLocal, sumBufLocal, cLengthAligned);
  } else if constexpr (std::is_same_v<T, half>) {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, cLengthAligned);
  } else {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, cLengthAligned);
  }
  outputQueue.EnQue(outputLocal);

  CopyOut(outputPointIdx, cLength);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AdaptiveAvgPool3dTilingData* tiling, TPipe* pipe) {
  InitTiling(tiling);

  inputGlobal.SetGlobalBuffer((__gm__ T *)x);
  outputGlobal.SetGlobalBuffer((__gm__ T *)y);

  pipe->InitBuffer(inputQueue, QUEUE_DEPTH, inputPointTileNum * cLengthAligned * sizeof(T));
  pipe->InitBuffer(outputQueue, QUEUE_DEPTH, cLengthAligned * sizeof(T));
#if __CCE_AICORE__ < 220
  if (atomicAddNum) {
    pipe->InitBuffer(zeroQueue, QUEUE_DEPTH, cLengthAligned * sizeof(T));
  } else if (validTailLen != 0) {
    pipe->InitBuffer(tmpPattern, numPerBlock * sizeof(T));
  }
#endif

  if constexpr (!std::is_same_v<T, float>) {
    pipe->InitBuffer(castBuf, cLengthAligned * sizeof(float));
  }
  pipe->InitBuffer(sumBuf, cLengthAligned * sizeof(float));

  pipe->InitBuffer(indexBuf.startDIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endDIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.startHIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endHIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.startWIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endWIndexBuf, indexBufLen * sizeof(int64_t));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitW<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
  if (atomicAddNum) {
    LocalTensor<T> zeroTensor = zeroQueue.template AllocTensor<T>();
    Duplicate<T>(zeroTensor, 0, numPerBlock);
    int64_t curOutputPointIdx = lastPointOffset;
    for (int i = 0; i < atomicAddNum; i++, curOutputPointIdx--) {
      DataCopy<T>(outputGlobal[curOutputPointIdx * cLength], zeroTensor, numPerBlock);
    }
  }
#endif
  for (int64_t outputPointIdx = outputPointOffset, count = 0;
       outputPointIdx < outputPointOffset + outputPointNum; ++outputPointIdx, ++count) {
    int64_t bufIdx = count % indexBufLen;

    if (bufIdx == 0) [[unlikely]] {
      int64_t len = (count + indexBufLen) < outputPointNum ? indexBufLen : outputPointNum - count;
      CalculateIndex(indexBuf, inputShape, outputShape, outputPointIdx, len);
    }

    ReduceMeanWindow(outputPointIdx, bufIdx);
  }
}

#endif // ADAPTIVE_AVG_POOL3D_SPLIT_W_H_

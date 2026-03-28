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
 * \file adaptive_avg_pool3d_multi_w.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_MULTI_W_H_
#define ADAPTIVE_AVG_POOL3D_MULTI_W_H_

#include "kernel_operator.h"
#include "adaptive_avg_pool3d_common.h"

template <typename T, int32_t QUEUE_DEPTH>
class KernelAdaptiveAvgPool3dMultiW {
public:
  __aicore__ inline KernelAdaptiveAvgPool3dMultiW() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                              const AdaptiveAvgPool3dTilingData* tiling, TPipe* pipe);
  __aicore__ inline void Process();

private:
  __aicore__ inline void InitTiling(const AdaptiveAvgPool3dTilingData* tiling);
  __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding);
  __aicore__ inline void CopyOut(int64_t outputPointIdx, uint16_t blockCount, uint32_t blockLen);
  __aicore__ inline void DataCopyOutNonPad(LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint16_t blockCount, uint32_t validDataLen);
  __aicore__ inline void ReduceMeanMultiWindow(int64_t outputPointIdx, int64_t bufIdx, int64_t windowNum);
  __aicore__ inline void ReduceSumMultiWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t bufIdx, int64_t windowNum, int64_t nOffset);

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
  int64_t windowWNum;
  int64_t maxWindowWLength;
  int64_t atomicAddNum;

  AdaAvgPool3DShape inputShape;
  AdaAvgPool3DShape outputShape;

  int64_t indexBufLen;
  IndexBuffer indexBuf;

  uint32_t numPerBlock;
  int32_t validTailLen;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::InitTiling(
    const AdaptiveAvgPool3dTilingData* tiling) {
  inputShape = AdaAvgPool3DShape(tiling->inD, tiling->inH, tiling->inW);
  outputShape = AdaAvgPool3DShape(tiling->outD, tiling->outH, tiling->outW);

  numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
  cLength = tiling->dimC;
  cLengthAligned = AlignUp(cLength, numPerBlock);
  windowWNum = tiling->windowWNum;
  maxWindowWLength = tiling->maxWindowWLength;
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
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyIn(
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
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::DataCopyOutNonPad(
  LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint16_t blockCount, uint32_t validDataLen) {
  int64_t curPointIdx = outputPointIdx;
  for (int i = 0; i < blockCount; i++, curPointIdx++) {
    PipeBarrier<PIPE_MTE3>();
    if ((validDataLen < numPerBlock) && (curPointIdx >= lastPointOffset - atomicAddNum)) {
      uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
      uint64_t mask[2] = {mask0, 0};
      Duplicate<T>(outputLocal[i * cLengthAligned], 0, mask, 1, 1, 1);
      if (curPointIdx > lastPointOffset - atomicAddNum) {
        SetAtomicAdd<T>();
        DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * cLengthAligned], cLengthAligned);
        SetAtomicNone();
      } else {
        DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * cLengthAligned], cLengthAligned);
      }
    } else if (curPointIdx == lastPointOffset) {
      DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * cLengthAligned], cLengthAligned - numPerBlock);
      int32_t lastLeftShift = validTailLen;
      uint32_t mask = numPerBlock * 2;
      uint64_t rsvdCnt = 0;
      uint64_t gatherOffset = blockCount * cLengthAligned - mask;
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
      DataCopy(outputGlobal[(curPointIdx + 1) * validDataLen - numPerBlock], outputLocal[gatherOffset], numPerBlock);
    } else {
      DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * cLengthAligned], cLengthAligned);
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyOut(
    int64_t outputPointIdx, uint16_t blockCount, uint32_t blockLen) {
  LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
  if (blockLen == cLengthAligned) {
    DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
    DataCopy(outputGlobal[outputPointIdx * blockLen], outputLocal, copyParams);
  } else {
    DataCopyOutNonPad(outputLocal, outputPointIdx, blockCount, blockLen);
  }
#else
  DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
  DataCopyPad(outputGlobal[outputPointIdx * blockLen], outputLocal, copyParams);
#endif
  outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumMultiWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t bufIdx, int64_t windowNum, int64_t nOffset) {
  LocalTensor<int64_t> startWIndexLocal = indexBuf.startWIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endWIndexLocal = indexBuf.endWIndexBuf.Get<int64_t>();
  
  int64_t dstart = index.dstart;
  int64_t dend = index.dend;
  int64_t hstart = index.hstart;
  int64_t hend = index.hend;
  int64_t wStartOffset = index.wstart * cLength;

  uint16_t blockCount = static_cast<uint16_t>(index.wend - index.wstart);
  uint8_t rightPadding = static_cast<uint8_t>(cLengthAligned - cLength);

  for (int64_t id = dstart; id < dend; ++id) {
    int64_t dOffset = id * inputShape.dstride * cLength;
    for (int64_t ih = hstart; ih < hend; ++ih) {
      int64_t hOffset = ih * inputShape.hstride * cLength;

      CopyIn(nOffset + dOffset + hOffset + wStartOffset, blockCount, cLength, rightPadding);
      LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();

      for (int64_t in = bufIdx, offset = 0; in < bufIdx + windowNum; ++in, offset += cLengthAligned) {
        int64_t wstart = startWIndexLocal.GetValue(in);
        int64_t wend = endWIndexLocal.GetValue(in);

        SToVSync();

        for (int64_t iw = wstart - index.wstart; iw < wend - index.wstart; ++iw) {
          if constexpr (std::is_same_v<T, float>) {
            Add(sumBufLocal[offset], sumBufLocal[offset], inputLocal[iw * cLengthAligned], cLengthAligned);
          } else {
            LocalTensor<float> castBufLocal = castBuf.Get<float>();
            Cast(castBufLocal, inputLocal[iw * cLengthAligned], RoundMode::CAST_NONE, cLengthAligned);
            Add(sumBufLocal[offset], sumBufLocal[offset], castBufLocal, cLengthAligned);
          }
        }
      }

      inputQueue.FreeTensor(inputLocal);
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceMeanMultiWindow(
    int64_t outputPointIdx, int64_t bufIdx, int64_t windowNum) {
  Index index;
  int64_t endBufIdx = bufIdx + windowNum;
  GetIndexFromBuffer(indexBuf, bufIdx, endBufIdx - 1, index);

  int64_t len = windowNum * cLengthAligned;
  int64_t nOffset = outputPointIdx / outputShape.nstride * inputShape.nstride * cLength;

  SToVSync();

  LocalTensor<float> sumBufLocal = sumBuf.Get<float>();
  Duplicate(sumBufLocal, 0.0f, len);

  ReduceSumMultiWindow(index, sumBufLocal, bufIdx, windowNum, nOffset);

  LocalTensor<int64_t> startWIndexLocal = indexBuf.startWIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endWIndexLocal = indexBuf.endWIndexBuf.Get<int64_t>();

  for (int64_t i = bufIdx, offset = 0; i < endBufIdx; ++i, offset += cLengthAligned) {
    int64_t wstart = startWIndexLocal.GetValue(i);
    int64_t wend = endWIndexLocal.GetValue(i);

    float factor = 1.0f / (
        static_cast<float>(index.dend - index.dstart) * (index.hend - index.hstart) * (wend - wstart));

    SToVSync();

    Muls(sumBufLocal[offset], sumBufLocal[offset], factor, cLengthAligned);
  }

  LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
  if constexpr (std::is_same_v<T, float>) {
    DataCopy(outputLocal, sumBufLocal, len);
  } else if constexpr (std::is_same_v<T, half>) {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, len);
  } else {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
  }
  outputQueue.EnQue(outputLocal);

  CopyOut(outputPointIdx, static_cast<uint16_t>(windowNum), static_cast<uint32_t>(cLength));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AdaptiveAvgPool3dTilingData* tiling, TPipe* pipe) {
  InitTiling(tiling);

  inputGlobal.SetGlobalBuffer((__gm__ T *)x);
  outputGlobal.SetGlobalBuffer((__gm__ T *)y);

  pipe->InitBuffer(inputQueue, QUEUE_DEPTH, windowWNum * maxWindowWLength * cLengthAligned * sizeof(T));
  pipe->InitBuffer(outputQueue, QUEUE_DEPTH, windowWNum * cLengthAligned * sizeof(T));

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
  pipe->InitBuffer(sumBuf, windowWNum * cLengthAligned * sizeof(float));

  pipe->InitBuffer(indexBuf.startDIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endDIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.startHIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endHIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.startWIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endWIndexBuf, indexBufLen * sizeof(int64_t));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dMultiW<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
  if (atomicAddNum) {
    LocalTensor<T> zeroTensor = zeroQueue.template AllocTensor<T>();
    Duplicate<T>(zeroTensor, 0, numPerBlock);
    int64_t curOutputPointIdx = lastPointOffset;
    for (int i = 0; i < atomicAddNum; i++, curOutputPointIdx--) {
      DataCopy(outputGlobal[curOutputPointIdx * cLength], zeroTensor, numPerBlock);
    }
  }
#endif
  int64_t curWindowWNum = windowWNum;
  for (int64_t outputPointIdx = outputPointOffset, count = 0;
       outputPointIdx < outputPointOffset + outputPointNum; outputPointIdx += curWindowWNum, count += curWindowWNum) {
    int64_t bufIdx = count % indexBufLen;

    if (bufIdx == 0) [[unlikely]] {
      int64_t len = (count + indexBufLen) < outputPointNum ? indexBufLen : outputPointNum - count;
      CalculateIndex(indexBuf, inputShape, outputShape, outputPointIdx, len);
    }

    curWindowWNum = (count + windowWNum) < outputPointNum ? windowWNum : outputPointNum - count;
    curWindowWNum = (bufIdx + curWindowWNum) < indexBufLen ? curWindowWNum : indexBufLen - bufIdx;

    int64_t newRowWindowWNum = (outputPointIdx + curWindowWNum) % outputShape.w;
    curWindowWNum = newRowWindowWNum != 0 && newRowWindowWNum < curWindowWNum
                      ? curWindowWNum - newRowWindowWNum : curWindowWNum;

    ReduceMeanMultiWindow(outputPointIdx, bufIdx, curWindowWNum);
  }
}

#endif // ADAPTIVE_AVG_POOL3D_MULTI_W_H_

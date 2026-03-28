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
 * \file adaptive_avg_pool3d_split_c.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_SPLIT_C_H_
#define ADAPTIVE_AVG_POOL3D_SPLIT_C_H_

#include "kernel_operator.h"
#include "adaptive_avg_pool3d_common.h"

template <typename T, int32_t QUEUE_DEPTH>
class KernelAdaptiveAvgPool3dSplitC {
public:
  __aicore__ inline KernelAdaptiveAvgPool3dSplitC() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                              const AdaptiveAvgPool3dTilingData* tiling, TPipe* pipe);
  __aicore__ inline void Process();

private:
  __aicore__ inline void InitTiling(const AdaptiveAvgPool3dTilingData* tiling);
  __aicore__ inline void CopyIn(int64_t offset, int64_t len);
  __aicore__ inline void CopyOut(int64_t offset, int64_t len);
  __aicore__ inline void DataCopyOutNonPad(LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen);
  __aicore__ inline void ReduceMean(int64_t outputPointIdx, int64_t bufIdx);
  __aicore__ inline void ReduceSum(const Index& index, LocalTensor<float>& sumBufLocal, int64_t cOffset, int64_t len, int64_t nOffset);

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
  int64_t cTileLength;
  int64_t outputPointNum;
  int64_t outputPointOffset;
  int64_t nextCoreAddrOffset;
  int64_t atomicAddNum;
  int64_t cTailLength;
  int64_t cTailAlign;

  AdaAvgPool3DShape inputShape;
  AdaAvgPool3DShape outputShape;

  int64_t indexBufLen;
  IndexBuffer indexBuf;

  uint32_t numPerBlock;
  int32_t validTailLen;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::InitTiling(
    const AdaptiveAvgPool3dTilingData* tiling) {
  inputShape = AdaAvgPool3DShape(tiling->inD, tiling->inH, tiling->inW);
  outputShape = AdaAvgPool3DShape(tiling->outD, tiling->outH, tiling->outW);

  numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);

  cLength = tiling->dimC;
  cTileLength = tiling->cTileLength;
  indexBufLen = tiling->indexBufLen;

  outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
  outputPointOffset = GetBlockIdx() < tiling->formerNum 
    ? tiling->formerLength * GetBlockIdx()
    : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
  nextCoreAddrOffset = (outputPointOffset + outputPointNum) * cLength;
  atomicAddNum = tiling->atomicAddNum;
  cTailLength = cLength % cTileLength;
  cTailAlign = AlignUp(cTailLength, numPerBlock);
  validTailLen = cTailLength % numPerBlock;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::CopyIn(int64_t offset, int64_t len) {
  LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
  if (len == cTileLength) {
    DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
    DataCopy(inputLocal, inputGlobal[offset], copyParams);
  } else {
    DataCopy(inputLocal, inputGlobal[offset], cTailAlign);
  }
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
#endif
  inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::DataCopyOutNonPad(
  LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen) {
  if ((validDataLen < numPerBlock) && (offset + validDataLen * atomicAddNum >= nextCoreAddrOffset)) {
    uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
    uint64_t mask[2] = {mask0, 0};
    Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
    SetAtomicAdd<T>();
    DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
    SetAtomicNone();
  } else if ((validTailLen != 0) && (offset + validDataLen == nextCoreAddrOffset)) {
    DataCopy(outputGlobal[offset], outputLocal, cTailAlign - numPerBlock);
    int32_t lastLeftShift = validTailLen;
    uint32_t mask = numPerBlock * 2;
    uint64_t rsvdCnt = 0;
    uint64_t gatherOffset = cTailAlign - mask;
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
    DataCopy(outputGlobal[nextCoreAddrOffset - numPerBlock], outputLocal[gatherOffset], numPerBlock);
  } else {
    DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
  LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
  if (len == cTileLength) {
    DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
    DataCopy(outputGlobal[offset], outputLocal, copyParams);
  } else {
    DataCopyOutNonPad(outputLocal, offset, len);
  }
#else
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
  DataCopyPad(outputGlobal[offset], outputLocal, copyParams);
#endif
  outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::ReduceSum(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t cOffset, int64_t len, int64_t nOffset) {
  int64_t dstart = index.dstart;
  int64_t dend = index.dend;
  int64_t hstart = index.hstart;
  int64_t hend = index.hend;
  int64_t wstart = index.wstart;
  int64_t wend = index.wend;
  
  int startOffset = cOffset;
  for (int64_t id = dstart; id < dend; ++id) {
    int64_t dOffset = id * inputShape.dstride;
    for (int64_t ih = hstart; ih < hend; ++ih) {
      int64_t hOffset = ih * inputShape.hstride;
      for (int64_t iw = wstart; iw < wend; ++iw) {
        CopyIn(startOffset + (nOffset + dOffset + hOffset + iw * inputShape.wstride) * cLength, len);

        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
          Add(sumBufLocal, sumBufLocal, inputLocal, len);
        } else {
          LocalTensor<float> castBufLocal = castBuf.Get<float>();
          Cast(castBufLocal, inputLocal, RoundMode::CAST_NONE, len);
          Add(sumBufLocal, sumBufLocal, castBufLocal, len);
        }
        inputQueue.FreeTensor(inputLocal);
      }
    }
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::ReduceMean(
    int64_t outputPointIdx, int64_t bufIdx) {
  Index index;
  GetIndexFromBuffer(indexBuf, bufIdx, bufIdx, index);

  SToVSync();

  float factor = 1.0f / (
    static_cast<float>(index.dend - index.dstart) * (index.hend - index.hstart) * (index.wend - index.wstart));

  int64_t cLoop = (cLength + cTileLength - 1) / cTileLength;
  int64_t cOffset = 0;
  int64_t nOffset = outputPointIdx / outputShape.nstride * inputShape.nstride;
  for (int64_t i = 0; i < cLoop; ++i) {
    int64_t count = i < cLoop - 1 ? cTileLength : cLength - (cLoop - 1) * cTileLength;

    LocalTensor<float> sumBufLocal = sumBuf.Get<float>();
    Duplicate(sumBufLocal, 0.0f, count);
  
    ReduceSum(index, sumBufLocal, cOffset, count, nOffset);
    Muls(sumBufLocal, sumBufLocal, factor, count);

    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
      DataCopy(outputLocal, sumBufLocal, AlignUp(count, numPerBlock));
    } else if constexpr (std::is_same_v<T, half>) {
      Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, count);
    } else {
      Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, count);
    }
    outputQueue.EnQue(outputLocal);

    CopyOut(outputPointIdx * cLength + cOffset, count);

    cOffset += count;
  }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AdaptiveAvgPool3dTilingData* tiling, TPipe* pipe) {
  InitTiling(tiling);

  inputGlobal.SetGlobalBuffer((__gm__ T *)x);
  outputGlobal.SetGlobalBuffer((__gm__ T *)y);

  pipe->InitBuffer(inputQueue, QUEUE_DEPTH, cTileLength * sizeof(T));
  pipe->InitBuffer(outputQueue, QUEUE_DEPTH, cTileLength * sizeof(T));

#if __CCE_AICORE__ < 220
  if (atomicAddNum) {
    pipe->InitBuffer(zeroQueue, QUEUE_DEPTH, cTailAlign * sizeof(T));
  } else if (validTailLen != 0) {
    pipe->InitBuffer(tmpPattern, numPerBlock * sizeof(T));
  }
#endif

  if constexpr (!std::is_same_v<T, float>) {
    pipe->InitBuffer(castBuf, cTileLength * sizeof(float));
  }
  pipe->InitBuffer(sumBuf, cTileLength * sizeof(float));

  pipe->InitBuffer(indexBuf.startDIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endDIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.startHIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endHIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.startWIndexBuf, indexBufLen * sizeof(int64_t));
  pipe->InitBuffer(indexBuf.endWIndexBuf, indexBufLen * sizeof(int64_t));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAdaptiveAvgPool3dSplitC<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
  if (atomicAddNum) {
    LocalTensor<T> zeroTensor = zeroQueue.template AllocTensor<T>();
    Duplicate<T>(zeroTensor, 0, numPerBlock);
    int64_t curOutputPointIdx = outputPointNum + outputPointOffset - 1;
    for (int i = 0; i < atomicAddNum; i++, curOutputPointIdx--) {
      DataCopy<T>(outputGlobal[curOutputPointIdx * cTailLength], zeroTensor, numPerBlock);
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

    ReduceMean(outputPointIdx, bufIdx);
  }
}

#endif // ADAPTIVE_AVG_POOL3D_SPLIT_C_H_
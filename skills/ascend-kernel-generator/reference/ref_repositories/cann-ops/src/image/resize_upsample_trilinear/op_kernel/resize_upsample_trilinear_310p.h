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
 * \file resize_upsample_trilinear_310p.h
 * \brief
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_310P_H
#define RESIZE_UPSAMPLE_TRILINEAR_310P_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleTrilinearNs {
using namespace AscendC;

constexpr int64_t DEFAULT_CLEAR_UB_SIZE = 10 * 1024;

template <typename T>
class KernelUpsampleTrilinear310p {
 public:
  TPipe pipe;

  __aicore__ inline KernelUpsampleTrilinear310p(){}; 
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, UpsampleTrilinearTilingData* tilingData);
  __aicore__ inline void Process();

 private:
  __aicore__ inline bool FloatEqual(float a, float b) {
    float closeTo0 = float(1e-6);
    if (a > b) {
      return a - b < closeTo0;
    } else {
      return b - a < closeTo0;
    }
  };

  template <typename T1, typename T2>
  __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
    if (b == 0) {
      return a;
    }
    return (a + b - 1) / b;
  };

  template <typename T1>
  __aicore__ inline T1 Min(T1 a, T1 b) {
    return a < b ? a : b;
  };

  template <typename T1>
  __aicore__ inline T1 Max(T1 a, T1 b) {
    return a > b ? a : b;
  };

  __aicore__ inline void ParseTilingData(UpsampleTrilinearTilingData* tilingData);
  __aicore__ inline void ClearGM();
  __aicore__ inline void ProcessSingleSlide(int64_t slideIndex);
  __aicore__ inline void ComputeSmallBatch();
  __aicore__ inline void ComputeW();
  __aicore__ inline void ComputeLargeBatch(int64_t batchOffset, int64_t batchCount);
  __aicore__ inline float AreaPixelComputeSourceIndex(float scale, int64_t dst_index);
  __aicore__ inline void CopyIn(int64_t indexInput, int64_t calCount);
  __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calCount);

 private:
  TQue<QuePosition::VECIN, 1> inputQueue;
  TQue<QuePosition::VECOUT, 1> outputQueue;
  TQue<QuePosition::VECIN, 1> syncWorkQueue;
  TBuf<TPosition::VECCALC> realIndexQueue;
  TBuf<TPosition::VECCALC> cacheTensorBuff1;
  TBuf<TPosition::VECCALC> cacheTensorBuff2;
  TBuf<TPosition::VECCALC> castInputBuff;
  TBuf<TPosition::VECCALC> castOutputBuff;
  TBuf<TPosition::VECCALC> clearTensorBuff;

  GlobalTensor<T> inTensorsGM;
  GlobalTensor<T> outTensorsGM;
  GlobalTensor<int32_t> syncTensorsGM;
  LocalTensor<float> realIndexTensor;
  LocalTensor<float> cacheTensor1;
  LocalTensor<float> cacheTensor2;
  LocalTensor<float> castInputTensor;
  LocalTensor<float> castOutputTensor;

  float scale_w, scale_h, scale_d;
  int64_t blockIdx, blockSize;
  uint16_t total_core_num;
  uint32_t real_core_num;
  bool widthZoom, heightZoom, depthZoom;
  int64_t align_corners;
  int64_t output_w, output_h, output_d;
  int64_t batches, batch_size;
  int64_t input_w, input_h, input_d;
  int64_t slide_size, tensor_size;

  int64_t slideIndexStart = -1;
  int64_t slideIndexEnd = -1;
  int64_t tailSlideIndex = -1;
  int64_t startW, dataLength, lastStartW;
  int64_t srcStartW, srcDataLength;
  int64_t indexD, indexH;
  int64_t srcIndexD0, srcIndexH0, srcIndexW0;
  int64_t dSize, hSize, wSize;
  float lambdaD1, lambdaH1, lambdaW1;
};

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                            UpsampleTrilinearTilingData* tilingData) {
  // parse tilingdata
  blockIdx = GetBlockIdx();
  blockSize = 32 / sizeof(T);
  ParseTilingData(tilingData);

  inTensorsGM.SetGlobalBuffer((__gm__ T*)input);
  outTensorsGM.SetGlobalBuffer((__gm__ T*)output);
  syncTensorsGM.SetGlobalBuffer((__gm__ int32_t *)workspace, total_core_num * 8 * 32);

  pipe.InitBuffer(inputQueue, 1, CeilA2B(tensor_size * batch_size * sizeof(T), 32) * 32);
  pipe.InitBuffer(outputQueue, 1, CeilA2B(slide_size * batch_size * sizeof(T), 32) * 32);
  pipe.InitBuffer(syncWorkQueue, 1, 8 * 32 * sizeof(int32_t));
  pipe.InitBuffer(realIndexQueue, CeilA2B(slide_size * sizeof(float), 32) * 32);
  pipe.InitBuffer(cacheTensorBuff1, CeilA2B(tensor_size * batch_size * sizeof(float), 32) * 32);
  pipe.InitBuffer(cacheTensorBuff2, CeilA2B(batch_size * sizeof(float), 32) * 32);
  pipe.InitBuffer(castInputBuff, CeilA2B(tensor_size * batch_size * sizeof(float), 32) * 32);
  pipe.InitBuffer(castOutputBuff, CeilA2B(slide_size * batch_size * sizeof(float), 32) * 32);
  pipe.InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::Process() {
  if ((batches % blockSize) != 0) {
    ClearGM();
    LocalTensor<int32_t> syncLocalTensor = syncWorkQueue.AllocTensor<int32_t>();
    SyncAll(syncTensorsGM, syncLocalTensor, int32_t(total_core_num));
    syncWorkQueue.FreeTensor(syncLocalTensor);
  }

  if (blockIdx >= real_core_num) {
    return;                                                                                                 
  }
  realIndexTensor = realIndexQueue.Get<float>();
  cacheTensor1 = cacheTensorBuff1.Get<float>();
  cacheTensor2 = cacheTensorBuff2.Get<float>();
  castInputTensor = castInputBuff.Get<float>();
  castOutputTensor = castOutputBuff.Get<float>();
  lastStartW = -1;
  if(slideIndexStart < slideIndexEnd) {
    for(int64_t slideIndex = slideIndexStart; slideIndex < slideIndexEnd; slideIndex ++) {
      ProcessSingleSlide(slideIndex);
    }
  }
  if(tailSlideIndex >= 0) {
    ProcessSingleSlide(tailSlideIndex);
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::ClearGM() {
  LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
  int64_t totalNum = output_w * output_h * output_d * batches;
  int64_t totalBlockNum = (totalNum + blockSize - 1) / blockSize;
  int64_t preCoreBlockCnt = totalBlockNum / total_core_num;
  int64_t tailBlockCnt = totalBlockNum % total_core_num;
  int64_t realNeedCore = 1;

  if (preCoreBlockCnt > 0) {
    realNeedCore = total_core_num;
  }
  if (blockIdx >= realNeedCore) {
    return;
  }
  
  int64_t preCoreDataCnt = preCoreBlockCnt * blockSize;
  int64_t loopCnt = preCoreDataCnt  / DEFAULT_CLEAR_UB_SIZE;
  int64_t tailCnt = preCoreDataCnt % DEFAULT_CLEAR_UB_SIZE;
  int64_t offset = blockIdx * preCoreDataCnt;

  Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

  for(int i = 0; i < loopCnt; i++) {
    DataCopy(outTensorsGM[offset], clearUb, DEFAULT_CLEAR_UB_SIZE);
    offset += DEFAULT_CLEAR_UB_SIZE;
  }

  if(tailCnt > 0){
    tailCnt = (tailCnt + blockSize - 1) / blockSize * blockSize;
    DataCopy(outTensorsGM[offset], clearUb, tailCnt);
  }

  if ((tailBlockCnt > 0) && (blockIdx==0)) {
    tailCnt = tailBlockCnt * blockSize;
    offset = preCoreDataCnt * realNeedCore;
    DataCopy(outTensorsGM[offset], clearUb, tailCnt);
  }

  LocalTensor<int32_t> clearWorkspaceUb = clearTensorBuff.Get<int32_t>();
  DataCopy(syncTensorsGM, clearWorkspaceUb, total_core_num * 8 * 32);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::ProcessSingleSlide(int64_t slideIndex) {
  indexD = (slideIndex % (output_d * output_h)) / output_h;
  float realIndexD = AreaPixelComputeSourceIndex(scale_d, indexD);
  srcIndexD0 = Min(static_cast<int64_t>(realIndexD), input_d - 1);
  if (depthZoom || srcIndexD0 == input_d - 1) {
    dSize = 1;
    lambdaD1 = static_cast<float>(0);
  } else {
    dSize = 2;
    lambdaD1 = Min(Max(realIndexD - (float)srcIndexD0, static_cast<float>(0)), static_cast<float>(1));
  }

  indexH = slideIndex % output_h;
  float realIndexH = AreaPixelComputeSourceIndex(scale_h, indexH);
  srcIndexH0 = Min(static_cast<int64_t>(realIndexH), input_h - 1);
  if (heightZoom || srcIndexH0 == input_h - 1) {
    hSize = 1;
    lambdaH1 = static_cast<float>(0);
  } else {
    hSize = 2;
    lambdaH1 = Min(Max(realIndexH - (float)srcIndexH0, static_cast<float>(0)), static_cast<float>(1));
  }

  startW = (slideIndex / (output_h * output_d)) * slide_size;
  int64_t loop = CeilA2B(batches, batch_size);
  if (loop == 1) {
    if (startW != lastStartW) {
      dataLength = Min(startW + slide_size, output_w) - startW;
      for(int64_t i = 0;i < dataLength;i ++) {
        float realIndex = AreaPixelComputeSourceIndex(scale_w, startW + i);
        realIndexTensor.SetValue(i, realIndex);
      }
      srcStartW = Min(static_cast<int64_t>(realIndexTensor.GetValue(0)), input_w - 1);
      srcDataLength = Min(static_cast<int64_t>(realIndexTensor.GetValue(dataLength - 1)) + 1, input_w - 1) - srcStartW + 1;
      lastStartW = startW;
    }
    ComputeSmallBatch();
  } else {
    float realIndexW = AreaPixelComputeSourceIndex(scale_w, startW);
    srcIndexW0 = Min(static_cast<int64_t>(realIndexW), input_w - 1);
    if (widthZoom || srcIndexW0 == input_w - 1) {
      wSize = 1;
      lambdaW1 = static_cast<float>(0);
    } else {
      wSize = 2;
      lambdaW1 = Min(Max(realIndexW - (float)srcIndexW0, static_cast<float>(0)), static_cast<float>(1));
    }
    for (int64_t i = 0;i < loop;i ++) {
      ComputeLargeBatch(i * batch_size, (i == loop-1) ? batches - i * batch_size : batch_size);
    }
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::ComputeSmallBatch() {
  int64_t calCount = srcDataLength * batches;
  Duplicate(cacheTensor1, static_cast<float>(0), CeilA2B(calCount, blockSize) * blockSize);
  for(int64_t d = 0;d < dSize;d ++) {
    float weightD = d == 0 ? static_cast<float>(1) - lambdaD1 : lambdaD1;
    for(int64_t h = 0;h < hSize;h ++) {
      float weightH = h == 0 ? static_cast<float>(1) - lambdaH1 : lambdaH1;
      int64_t indexInput = ((srcIndexD0 + d) * input_h * input_w + (srcIndexH0 + h) * input_w + srcStartW) * batches;
      CopyIn(indexInput, CeilA2B(calCount, blockSize) * blockSize);
      if (std::is_same_v<T, float>) {
        LocalTensor<float> srcDataLocal = inputQueue.DeQue<float>();
        Muls(castInputTensor, srcDataLocal, weightD * weightH, calCount);
        Add(cacheTensor1, cacheTensor1, castInputTensor, calCount);
        inputQueue.FreeTensor(srcDataLocal);
      } else {
        LocalTensor<T> srcDataLocal = inputQueue.DeQue<T>();
        Cast(castInputTensor, srcDataLocal, RoundMode::CAST_NONE, calCount);
        Muls(castInputTensor, castInputTensor, weightD * weightH, calCount);
        Add(cacheTensor1, cacheTensor1, castInputTensor, calCount);
        inputQueue.FreeTensor(srcDataLocal);
      }
    }
  }

  ComputeW();
  int64_t indexOutput = (indexD * output_h * output_w + indexH * output_w + startW) * batches;
  CopyOut(indexOutput, dataLength * batches);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::ComputeW() {
  if (std::is_same_v<T, float>) {
    LocalTensor<float> dstDataLocal = outputQueue.AllocTensor<float>();
    Duplicate(dstDataLocal, static_cast<float>(0), CeilA2B(dataLength * batches, blockSize) * blockSize);
    for (int64_t w = 0;w < dataLength;w ++) {
      int32_t srcIndexW0 = Min(static_cast<int64_t>(realIndexTensor.GetValue(w)), input_w - 1);
      if (widthZoom || srcIndexW0 == input_w - 1) {
        Muls(dstDataLocal[w*batches], cacheTensor1[(srcIndexW0-srcStartW)*batches], static_cast<float>(1), batches);
      } else {
        float weight1 = Min(Max(realIndexTensor.GetValue(w) - static_cast<float>(srcIndexW0), static_cast<float>(0)), static_cast<float>(1));
        Muls(cacheTensor2, cacheTensor1[(srcIndexW0-srcStartW)*batches], static_cast<float>(1) - weight1, batches);
        Muls(dstDataLocal[w*batches], cacheTensor1[(srcIndexW0+1-srcStartW)*batches], weight1, batches);
        Add(dstDataLocal[w*batches], dstDataLocal[w*batches], cacheTensor2, batches);
      }
    }
    outputQueue.EnQue(dstDataLocal);
  } else {
    LocalTensor<T> dstDataLocal = outputQueue.AllocTensor<T>();
    Duplicate(dstDataLocal, static_cast<T>(0), CeilA2B(dataLength * batches, blockSize) * blockSize);
    for (int64_t w = 0;w < dataLength;w ++) {
      int32_t srcIndexW0 = Min(static_cast<int64_t>(realIndexTensor.GetValue(w)), input_w - 1);
      if (widthZoom || srcIndexW0 == input_w - 1) {
        Muls(castOutputTensor[w*batches], cacheTensor1[(srcIndexW0-srcStartW)*batches], static_cast<float>(1), batches);
      } else {
        float weight1 = Min(Max(realIndexTensor.GetValue(w) - static_cast<float>(srcIndexW0), static_cast<float>(0)), static_cast<float>(1));
        Muls(cacheTensor2, cacheTensor1[(srcIndexW0-srcStartW)*batches], static_cast<float>(1) - weight1, batches);
        Muls(castOutputTensor[w*batches], cacheTensor1[(srcIndexW0+1-srcStartW)*batches], weight1, batches);
        Add(castOutputTensor[w*batches], castOutputTensor[w*batches], cacheTensor2, batches);
      }
    }
    Cast(dstDataLocal, castOutputTensor, RoundMode::CAST_RINT, dataLength * batches);
    outputQueue.EnQue(dstDataLocal);
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::ComputeLargeBatch(int64_t batchOffset, int64_t batchCount) {
  if (std::is_same_v<T, float>) {
    LocalTensor<float> dstDataLocal = outputQueue.AllocTensor<float>();
    Duplicate(dstDataLocal, static_cast<float>(0), CeilA2B(batchCount, blockSize) * blockSize);
    for(int64_t d = 0;d < dSize;d ++) {
      float weightD = d == 0 ? static_cast<float>(1) - lambdaD1 : lambdaD1;
      for(int64_t h = 0;h < hSize;h ++) {
        float weightH = h == 0 ? static_cast<float>(1) - lambdaH1 : lambdaH1;
        for(int64_t w = 0;w < wSize;w ++) {
          float weightW = w == 0 ? static_cast<float>(1) - lambdaW1 : lambdaW1;
          int64_t indexInput = ((srcIndexD0 + d) * input_h * input_w + (srcIndexH0 + h) * input_w + srcIndexW0 + w) * batches + batchOffset;
          CopyIn(indexInput, CeilA2B(batchCount, blockSize) * blockSize);
          LocalTensor<float> srcDataLocal = inputQueue.DeQue<float>();
          Muls(castInputTensor, srcDataLocal, weightD * weightH * weightW, batchCount);
          Add(dstDataLocal, dstDataLocal, castInputTensor, batchCount);
          inputQueue.FreeTensor(srcDataLocal);
        }
      }
    }
    outputQueue.EnQue(dstDataLocal);
  } else {
    LocalTensor<T> dstDataLocal = outputQueue.AllocTensor<T>();
    Duplicate(dstDataLocal, static_cast<T>(0), CeilA2B(batchCount, blockSize) * blockSize);
    Duplicate(castOutputTensor, static_cast<float>(0), CeilA2B(batchCount, blockSize) * blockSize);
    for(int64_t d = 0;d < dSize;d ++) {
      float weightD = d == 0 ? static_cast<float>(1) - lambdaD1 : lambdaD1;
      for(int64_t h = 0;h < hSize;h ++) {
        float weightH = h == 0 ? static_cast<float>(1) - lambdaH1 : lambdaH1;
        for(int64_t w = 0;w < wSize;w ++) {
          float weightW = w == 0 ? static_cast<float>(1) - lambdaW1 : lambdaW1;
          int64_t indexInput = ((srcIndexD0 + d) * input_h * input_w + (srcIndexH0 + h) * input_w + srcIndexW0 + w) * batches + batchOffset;
          CopyIn(indexInput, CeilA2B(batchCount, blockSize) * blockSize);
          LocalTensor<T> srcDataLocal = inputQueue.DeQue<T>();
          Cast(castInputTensor, srcDataLocal, RoundMode::CAST_NONE, batchCount);
          Muls(castInputTensor, castInputTensor, weightD * weightH * weightW, batchCount);
          Add(castOutputTensor, castOutputTensor, castInputTensor, batchCount);
          inputQueue.FreeTensor(srcDataLocal);
        }
      }
    }
    Cast(dstDataLocal, castOutputTensor, RoundMode::CAST_RINT, batchCount);
    outputQueue.EnQue(dstDataLocal);
  }
  int64_t indexOutput = (indexD * output_h * output_w + indexH * output_w + startW) * batches + batchOffset;
  CopyOut(indexOutput, batchCount);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::CopyIn(int64_t indexInput, int64_t calCount) {
  LocalTensor<T> srcDataLocal = inputQueue.AllocTensor<T>();
  DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount);
  inputQueue.EnQue(srcDataLocal);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::CopyOut(int64_t indexOutput, int64_t calCount) {
  LocalTensor<T> dstDataLocal = outputQueue.DeQue<T>();
  if ((calCount % blockSize) == 0) {
    DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
  } else {
    int64_t blockCalCount = (calCount + blockSize - 1) / blockSize * blockSize;
    SetAtomicAdd<T>();
    DataCopy(outTensorsGM[indexOutput], dstDataLocal, blockCalCount);
    SetAtomicNone();
  }
  
  outputQueue.FreeTensor(dstDataLocal);
}

template <typename T>
__aicore__ inline float KernelUpsampleTrilinear310p<T>::AreaPixelComputeSourceIndex(float scale, int64_t dst_index) {
  // calc coordinate range with group
  float result;
  if (align_corners == 1) {
    result = scale * (float)dst_index;
  } else {
    auto zero = static_cast<float>(0.);
    float src_idx = static_cast<float>(scale * ((float)dst_index + (float)0.5) - (float)0.5);
    result = (src_idx < zero) ? float(0.) : src_idx;
  }
  return result;
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear310p<T>::ParseTilingData(UpsampleTrilinearTilingData* tilingData) {
  scale_w = tilingData->scale_w;
  scale_h = tilingData->scale_h;
  scale_d = tilingData->scale_d;
  output_w = tilingData->output_w;
  output_h = tilingData->output_h;
  output_d = tilingData->output_d;
  input_w = tilingData->input_w;
  input_h = tilingData->input_h;
  input_d = tilingData->input_d;
  batches = tilingData->batches;
  batch_size = tilingData->batch_size;

  align_corners = tilingData->align_corners;
  total_core_num = tilingData->total_core_num;
  real_core_num = tilingData->real_core_num;

  slide_size = tilingData->slide_size;
  tensor_size = tilingData->tensor_size;

  widthZoom = FloatEqual(scale_w, 1.0);
  heightZoom = FloatEqual(scale_h, 1.0);
  depthZoom = FloatEqual(scale_d, 1.0);

  int64_t eachCoreSlideNum = tilingData->each_core_slide_num;
  int64_t remainder = tilingData->remainder;
  int64_t tailStartSlideNum = tilingData->tail_start_slide_num;
  if (eachCoreSlideNum > 0) {
    slideIndexStart = eachCoreSlideNum * blockIdx;
    slideIndexEnd = slideIndexStart + eachCoreSlideNum;
  }
  if (remainder > 0 && blockIdx < remainder) {
    tailSlideIndex = tailStartSlideNum + blockIdx;
  }
}
}

#endif  // RESIZE_UPSAMPLE_TRILINEAR_310P_H
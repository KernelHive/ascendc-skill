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
 * \file upsample_bicubic2d_aa.h
 * \brief
 */
#ifndef UPSAMPLE_BICUBIC2D_AA
#define UPSAMPLE_BICUBIC2D_AA

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2dAA {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t EACH_SLICE_HANDLE_NUM = 16;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleBicubic2dAAND {
 public:
  TPipe pipe;
  matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>, MDL_CFG>
      matmulW;

  matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>, MDL_CFG>
      matmulH;

  __aicore__ inline UpsampleBicubic2dAAND(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                              UpsampleBicubic2dAATilingData* tilingData);
  __aicore__ inline void Process();

 private:
  template <typename T1, typename T2>
  __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
    if (b == 0) {
      return a;
    }
    return (a + b - 1) / b * b;
  };
  template <typename T1>
  __aicore__ inline T1 WeightCalculate(T1 x) {
    if (x < 0) {
      x = -1 * x;
    }
    if (x < (T1)1.0) {
      return ((T1)1.5 * x - (T1)2.5) * x * x + (T1)1.0;
    }
    if (x < (T1)2.0) {
      return (((T1)-0.5 * x + (T1)2.5) * x - 4) * x + (T1)2.0;
    }
    return 0.0;
  };
  template <typename T1, typename T2>
  __aicore__ inline T1 MinFun(T1 a, T2 b) {
    return a < b ? a : b;
  };
  template <typename T1, typename T2>
  __aicore__ inline T1 MaxFun(T1 a, T2 b) {
    return a > b ? a : b;
  };

  __aicore__ inline bool FloatEqual(float a, float b) {
    float epsilon = float(1e-8);
    if (a > b) {
      return (a - b) < epsilon;
    } else {
      return (b - a) < epsilon;
    }
  };

  __aicore__ inline void ParseTilingData(UpsampleBicubic2dAATilingData* tilingData);
  __aicore__ inline void CalculateIndexTensor(int32_t index, int32_t length, uint8_t direction, int32_t inputSize);
  __aicore__ inline void CalculateRadioTensor(int32_t index, int32_t length, int32_t tensorLength, float invscale);
  __aicore__ inline void CalculateWidthExtension(int32_t tensorCIndex, int32_t rowStart, int32_t rowEnd);
  __aicore__ inline void CalculateHeightExtension(int32_t tensorCIndex, int32_t batchStart, int32_t batchEnd);
  __aicore__ inline void CopyRadioTensorToGm(int32_t length);
  __aicore__ inline void ProcessWidthDirection();
  __aicore__ inline void ProcessHeightDirection();
  __aicore__ inline int32_t GetWidthTensorSize();
  __aicore__ inline int32_t GetHeightTensorSize();

 private:
  // 系数矩阵下标队列
  TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioQueue;
  TBuf<> centerTensorBuff;
  TBuf<> xMinTensorBuff;
  TBuf<> xSizeTensorBuff;
  TBuf<> weightTensorBuff;

  const TCubeTiling* __restrict matmulTilingW;
  const TCubeTiling* __restrict matmulTilingH;

  GlobalTensor<T> inTensorsGM;
  GlobalTensor<T> outTensorsGM;
  GlobalTensor<T> intermediateTensorGm;

  LocalTensor<float> centerTensor;
  LocalTensor<float> xMinTensor;
  LocalTensor<float> xSizeTensor;
  LocalTensor<float> weightTensor;

  int64_t blockIdx = 0;
  int32_t sliceSize = 16;
  float scaleW;
  float scaleH;
  float invscaleW;
  float invscaleH;
  float supportW;
  float supportH;
  int32_t maxInterpSizeW;
  int32_t maxInterpSizeH;
  int32_t maxInterpSize;
  uint32_t needCoreNumW;
  uint32_t needCoreNumH;

  uint64_t intermediateMatrixSize;
  int64_t workSpaceRadioOffset;
  uint32_t radioMatrixSize;
  uint32_t radioWLength;
  uint32_t radioHLength;

  int32_t sliceStartW;
  int32_t sliceEndW;
  int32_t tailSliceStartW;
  int32_t tailSliceEndW;
  int32_t tailRowStartW;
  int32_t tailRowEndW;

  int32_t sliceStartH;
  int32_t sliceEndH;
  int32_t tailSliceStartH;
  int32_t tailSliceEndH;
  int32_t tailBatchStartH;
  int32_t tailBatchEndH;

  int32_t inputShapes[4] = {0, 0, 0, 0};
  int32_t outputShapes[4] = {0, 0, 0, 0};

  int64_t singleCoreK = 0;
  int64_t xMin = 0;
  bool widthFixed;
  bool heightFixed;
  bool useWidthDirectCopy;
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                     UpsampleBicubic2dAATilingData* tilingData) {
  blockIdx = GetBlockIdx() / 2;

  ParseTilingData(tilingData);

  int32_t tensorWidthSize = GetWidthTensorSize();
  int32_t tensorHeightSize = GetHeightTensorSize();
  int32_t cacheBufferSize = MaxFun(tensorWidthSize, tensorHeightSize);

  pipe.InitBuffer(radioQueue, NO_BUFFER_NUM, radioMatrixSize * sizeof(float) / sizeof(T));
  pipe.InitBuffer(centerTensorBuff, cacheBufferSize);
  pipe.InitBuffer(xMinTensorBuff, cacheBufferSize);
  pipe.InitBuffer(xSizeTensorBuff, cacheBufferSize);
  
  pipe.InitBuffer(weightTensorBuff, CeilA2B(maxInterpSize * sizeof(float), 32));

  intermediateTensorGm.SetGlobalBuffer((__gm__ T*)workspace);
  inTensorsGM.SetGlobalBuffer((__gm__ T*)x);
  outTensorsGM.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::Process() {
  // 先横向扩展
  ProcessWidthDirection();
  SyncAll();
  // 如果shape不变，直接copy输出结束
  if (widthFixed && heightFixed && useWidthDirectCopy) {
    return;
  }
  // 再纵向扩展
  ProcessHeightDirection();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::ProcessWidthDirection() {
  if (GetSubBlockIdx() == 1) {
    return;
  }

  if (blockIdx >= needCoreNumW) {
    return;
  }

  if ((!widthFixed) || (widthFixed && heightFixed && useWidthDirectCopy)) {
    if (sliceStartW < sliceEndW) {
      for (int32_t index = sliceStartW; index < sliceEndW; index += sliceSize) {
        int32_t length = MinFun(sliceSize, sliceEndW - index);
        CalculateIndexTensor(index, length, 0, inputShapes[3]);
        CalculateRadioTensor(0, length, radioWLength, invscaleW);
        CopyRadioTensorToGm(radioWLength);
        CalculateWidthExtension(index, 0, 0);
      }
    }

    if (tailSliceStartW < tailSliceEndW) {
      for (int32_t index = tailSliceStartW; index < tailSliceEndW; index += sliceSize) {
        int32_t length = MinFun(sliceSize, tailSliceEndW - index);
        CalculateIndexTensor(index, length, 0, inputShapes[3]);
        CalculateRadioTensor(0, length, radioWLength, invscaleW);
        CopyRadioTensorToGm(radioWLength);
        CalculateWidthExtension(index, tailRowStartW, tailRowEndW);
      }
    }
  }
}


template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::ProcessHeightDirection() {
  if (GetSubBlockIdx() == 1) {
    return;
  }

  if (blockIdx >= needCoreNumH) {
    return;
  }
  if ((!heightFixed) || (widthFixed && heightFixed && !useWidthDirectCopy)) {
    if (sliceStartH < sliceEndH) {
      for (int32_t index = sliceStartH; index < sliceEndH; index += sliceSize) {
        int32_t length = MinFun(sliceSize, sliceEndH - index);
        CalculateIndexTensor(index, length, 1, inputShapes[2]);
        CalculateRadioTensor(0, length, radioHLength, invscaleH);
        CopyRadioTensorToGm(radioHLength);
        CalculateHeightExtension(index, 0, 0);
      }
    }

    if (tailSliceStartH < tailSliceEndH) {
      for (int32_t index = tailSliceStartH; index < tailSliceEndH; index += sliceSize) {
        int32_t length = MinFun(sliceSize, tailSliceEndH - index);
        CalculateIndexTensor(index, length, 1, inputShapes[2]);
        CalculateRadioTensor(0, length, radioHLength, invscaleH);
        CopyRadioTensorToGm(radioHLength);
        CalculateHeightExtension(index, tailBatchStartH, tailBatchEndH);
      }
    }
  }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::CalculateIndexTensor(int32_t index, int32_t length, uint8_t direction, int32_t inputSize) {
  int32_t realDataCount = MaxFun(length, EACH_SLICE_HANDLE_NUM);

  centerTensor = centerTensorBuff.Get<float>();
  xMinTensor = xMinTensorBuff.Get<float>();
  xSizeTensor = xSizeTensorBuff.Get<float>();
  float scale = scaleW;
  float support = supportW;
  int32_t interpSize = maxInterpSizeW;
  if (direction == 1) {
    scale = scaleH;
    support = supportH;
    interpSize = maxInterpSizeH;
  }

  ArithProgression(centerTensor, static_cast<float>(index), (float)1.0, realDataCount);

  Adds(centerTensor, centerTensor, (float)0.5, realDataCount);
  Muls(centerTensor, centerTensor, scale, realDataCount);

  Adds(xMinTensor, centerTensor, (float)0.5 - support, realDataCount);
  Floor(xMinTensor, xMinTensor, realDataCount);
  Maxs(xMinTensor, xMinTensor, (float)0.0, realDataCount);

  Adds(xSizeTensor, centerTensor, (float)0.5 + support, realDataCount);
  Floor(xSizeTensor, xSizeTensor, realDataCount);
  Mins(xSizeTensor, xSizeTensor, static_cast<float>(inputSize), realDataCount);
  Sub(xSizeTensor, xSizeTensor, xMinTensor, realDataCount);
  Mins(xSizeTensor, xSizeTensor, static_cast<float>(interpSize), realDataCount);
  Maxs(xSizeTensor, xSizeTensor, (float)0.0, realDataCount);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::CalculateRadioTensor(int32_t index, int32_t length, int32_t tensorLength, float invscale) {
  
  LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
  centerTensor = centerTensorBuff.Get<float>();
  xMinTensor = xMinTensorBuff.Get<float>();
  xSizeTensor = xSizeTensorBuff.Get<float>();
  weightTensor = weightTensorBuff.Get<float>();

  xMin = static_cast<int64_t>(xMinTensor.GetValue(index));
  Duplicate(radioTensor, (float)0.0, tensorLength);
  singleCoreK = 0;
  for (int32_t i = index; i < index+length; i++) {
    float totalW = 0.0;
    float distanceOffset = xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5;
    for (int32_t j = 0; j < static_cast<int32_t>(xSizeTensor.GetValue(i)); j++) {
      float w = WeightCalculate((j + distanceOffset) * invscale);
      weightTensor.SetValue(j, w);
      totalW += w;
    }

    if (totalW > (float)0.0) {
      int32_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
      int32_t indexOffset = i - index;
      for (int32_t j = 0; j < static_cast<int32_t>(xSizeTensor.GetValue(i)); j++) {
        float weight = weightTensor.GetValue(j) / totalW;
        int32_t yIndexValue = j + yIndexOffset;
        singleCoreK = singleCoreK < yIndexValue + 1 ? yIndexValue + 1 : singleCoreK;
        int64_t index = yIndexValue * sliceSize + indexOffset;
        radioTensor.SetValue(index, weight);
      }
    }
  }
  if (sizeof(T) == 2) {
    Cast(radioTensor.ReinterpretCast<T>(), radioTensor, RoundMode::CAST_RINT, tensorLength);
  }
  radioQueue.EnQue(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::CopyRadioTensorToGm(int32_t length) {
  LocalTensor<T> radioTensor = radioQueue.DeQue<T>();

  workSpaceRadioOffset = (intermediateMatrixSize + radioMatrixSize * blockIdx) / sizeof(T);

  DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, length);
  radioQueue.FreeTensor(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::CalculateWidthExtension(int32_t tensorCIndex, int32_t rowStart,
                                                                        int32_t rowEnd) {
  int64_t singleCoreM = matmulTilingW->singleCoreM;
  int64_t singleCoreN = matmulTilingW->singleCoreN;
  if (rowEnd != 0) {
    singleCoreM = rowEnd - rowStart;
  }
  matmulW.SetOrgShape(singleCoreM, singleCoreN, inputShapes[3], singleCoreK, outputShapes[3]);
  matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
  

  if (tensorCIndex + sliceSize > outputShapes[3]) {
    matmulW.SetTail(singleCoreM, outputShapes[3] - tensorCIndex, singleCoreK);
  }
  int64_t xIndex = xMin + rowStart * inputShapes[3];
  int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * outputShapes[3];

  matmulW.SetTensorA(inTensorsGM[xIndex], false);
  matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
  if (heightFixed) {
    matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
  } else {
    matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
  }
  matmulW.End();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::CalculateHeightExtension(int32_t tensorCIndex, int32_t batchStart,
                                                                        int32_t batchEnd) {
  int64_t singleCoreM = matmulTilingH->singleCoreM;
  int64_t singleCoreN = matmulTilingH->singleCoreN;

  matmulH.SetOrgShape(singleCoreM, outputShapes[3], singleCoreK);
  matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

  if (tensorCIndex + sliceSize > outputShapes[2]) {
    matmulH.SetTail(outputShapes[2] - tensorCIndex, singleCoreN, singleCoreK);
  }

  if (batchEnd == 0) {
    batchEnd = inputShapes[0] * inputShapes[1];
  }

  int64_t weightOffsetSize = inputShapes[2] * outputShapes[3];
  int64_t outputOffsetSize = outputShapes[2] * outputShapes[3];
  int64_t xMinOffset = xMin * outputShapes[3];
  int64_t tensorCIndexOffset = tensorCIndex * outputShapes[3];
  matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], true);
  for (int64_t j = batchStart; j < batchEnd; j++) {
    int64_t xIndex = xMinOffset + j * weightOffsetSize;
    int64_t tensorCIndexWithOffset = tensorCIndexOffset + j * outputOffsetSize;
    if (widthFixed) {
      matmulH.SetTensorB(inTensorsGM[xIndex], false);
    } else {
      matmulH.SetTensorB(intermediateTensorGm[xIndex], false);
    }
    matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    matmulH.End();
  }
}

template <typename T>
__aicore__ inline int32_t UpsampleBicubic2dAAND<T>::GetWidthTensorSize() {
  int32_t size = sliceSize;
  size = CeilA2B(size * sizeof(float), 32);
  return size;
}

template <typename T>
__aicore__ inline int32_t UpsampleBicubic2dAAND<T>::GetHeightTensorSize() {
  int32_t size = sliceSize;
  size = CeilA2B(size * sizeof(float), 32);
  return size;
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dAAND<T>::ParseTilingData(UpsampleBicubic2dAATilingData* tilingData) {
  scaleW = tilingData->scaleW;
  scaleH = tilingData->scaleH;
  invscaleW = tilingData->invscaleW;
  invscaleH = tilingData->invscaleH;
  supportW = tilingData->supportW;
  supportH = tilingData->supportH;
  maxInterpSizeW = tilingData->maxInterpSizeW;
  maxInterpSizeH = tilingData->maxInterpSizeH;
  maxInterpSize = MaxFun(maxInterpSizeW, maxInterpSizeH);
  needCoreNumW = tilingData->needCoreNumW;
  needCoreNumH = tilingData->needCoreNumH;
  sliceSize = tilingData->sliceSize;
  for (int8_t i = 0; i < 4; i++) {
    outputShapes[i] = tilingData->outputShapes[i];
  }
  for (int8_t i = 0; i < 4; i++) {
    inputShapes[i] = tilingData->inputShapes[i];
  }

  intermediateMatrixSize = tilingData->intermediateMatrixSize;
  uint32_t radioMatrixWSize = CeilA2B(tilingData->radioMatrixWSize, 32);
  uint32_t radioMatrixHSize = CeilA2B(tilingData->radioMatrixHSize, 32);
  radioWLength = radioMatrixWSize / sizeof(T);
  radioHLength = radioMatrixHSize / sizeof(T);
  radioMatrixSize = MaxFun(radioMatrixWSize, radioMatrixHSize);
  radioMatrixSize = (radioMatrixSize + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

  sliceStartW = tilingData->sliceStartListW[blockIdx];
  sliceEndW = tilingData->sliceEndListW[blockIdx];
  tailSliceStartW = tilingData->tailSliceStartListW[blockIdx];
  tailSliceEndW = tilingData->tailSliceEndListW[blockIdx];
  tailRowStartW = tilingData->tailRowStartListW[blockIdx];
  tailRowEndW = tilingData->tailRowEndListW[blockIdx];

  sliceStartH = tilingData->sliceStartListH[blockIdx];
  sliceEndH = tilingData->sliceEndListH[blockIdx];
  tailSliceStartH = tilingData->tailSliceStartListH[blockIdx];
  tailSliceEndH = tilingData->tailSliceEndListH[blockIdx];
  tailBatchStartH = tilingData->tailBatchStartListH[blockIdx];
  tailBatchEndH = tilingData->tailBatchEndListH[blockIdx];
  widthFixed = FloatEqual(scaleW, 1.0);
  heightFixed = FloatEqual(scaleH, 1.0);
  useWidthDirectCopy = needCoreNumW > needCoreNumH ? true : false;

  matmulTilingW = &tilingData->matmulTilingW;
  matmulTilingH = &tilingData->matmulTilingH;
}
}  // namespace UpsampleBicubic2dAA

#endif  // UPSAMPLE_BICUBIC2D_AA
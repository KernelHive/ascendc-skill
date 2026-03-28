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
 * \file dynamic_quant_update_scatter_comm.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_COMM_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_COMM_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t EIGHT = 8;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t INDICES_RANK2 = 2;
constexpr uint32_t MASK = 64;
constexpr float QUANT_CONST_VALUE = 127.0;
constexpr float MIN_FLOAT_VALUE = -3.4028235e+38;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterBase {
 public:
  __aicore__ inline DynamicQuantUpdateScatterBase() {
  }

  __aicore__ inline void InitBase(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR smoothScales,
                                  const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    blockIdx = GetBlockIdx();

    uint64_t indexAlignElements =
        (tilingPtr->indexElements * sizeof(T2) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO / sizeof(T2);

    if (smoothScales != nullptr) {
      hasSmoothScales = true;
      smoothScalesGm.SetGlobalBuffer((__gm__ T3*)smoothScales, tilingPtr->varOrigLastDimSize);
    }

    if (blockIdx < tilingPtr->coreNum - 1) {
      varScaleOutElemens = tilingPtr->eachCoreBsNum * tilingPtr->quantReptNum * tilingPtr->updateAxisShape;
      coreBatchNum = tilingPtr->eachCoreBsNum;
    } else {
      varScaleOutElemens = tilingPtr->lastCoreBsNum * tilingPtr->quantReptNum * tilingPtr->updateAxisShape;
      coreBatchNum = tilingPtr->lastCoreBsNum;
    }
    coreBsNumElements = coreBatchNum * tilingPtr->srcBsStride;

    varGm.SetGlobalBuffer((__gm__ T1*)var, tilingPtr->varElements);
    varScalesGm.SetGlobalBuffer((__gm__ float*)varScale, tilingPtr->varScalesElements);
    indicesGm.SetGlobalBuffer((__gm__ T2*)indices, tilingPtr->indexElements);

    pipe.InitBuffer(indicesInQueue, BUFFER_NUM, indexAlignElements * sizeof(T2));
  }

  __aicore__ inline void InitvarScaleOutQueue(int64_t ubSize) {
    pipe.InitBuffer(varScaleOutQueue, BUFFER_NUM, ubSize);
  }

  __aicore__ inline void InitUpdateGM(GM_ADDR updates, int64_t Offset, int64_t elements) {
    updatesGm.SetGlobalBuffer((__gm__ T3*)updates + Offset, elements);
  }

  __aicore__ inline void InitUpdatesInQueue(int64_t ubSize) {
    pipe.InitBuffer(updatesInQueue, BUFFER_NUM, ubSize);
  }

  __aicore__ inline void InitSmoothScalesInQueue(int64_t ubSize, int64_t ubF32Size) {
    if (hasSmoothScales) {
      pipe.InitBuffer(smoothScalesInQueue, BUFFER_NUM, ubSize);
#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
      if (hasSmoothScales) {
        pipe.InitBuffer(smoothScaleF32, ubF32Size);
      }
#endif
    }
  }

  __aicore__ inline void InitVarOutQueue(int64_t ubSize) {
    pipe.InitBuffer(varOutQueue, BUFFER_NUM, ubSize);
  }

  __aicore__ inline void InitTempF32(int64_t ubSize) {
    pipe.InitBuffer(tempF32, ubSize);
  }

  __aicore__ inline void InitTempI32(int64_t ubSize) {
    pipe.InitBuffer(tempI32, ubSize);
  }

  __aicore__ inline void InitUpdateF32(int64_t ubSize) {
    pipe.InitBuffer(updateF32, ubSize);
  }

  __aicore__ inline void CopyIndicesIn(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    LocalTensor<T2> indicesLocal = indicesInQueue.AllocTensor<T2>();

    int64_t eleOneBlock = THIRTY_TWO / sizeof(T2);
    DataCopy(indicesLocal, indicesGm[0], (tilingPtr->indexElements + eleOneBlock - 1) / eleOneBlock * eleOneBlock);

    indicesInQueue.EnQue(indicesLocal);
  }

  __aicore__ inline void CopySmoothScalesIn(int64_t offset, int64_t elements) {
    if (hasSmoothScales) {
      LocalTensor<T3> smoothScalesLocal = smoothScalesInQueue.AllocTensor<T3>();
      DataCopy(smoothScalesLocal, smoothScalesGm[offset], elements);
      smoothScalesInQueue.EnQue(smoothScalesLocal);
    }
  }

  __aicore__ inline void ComputeForLittleQuant(const DynamicQuantUpdateScatterTilingData* tilingPtr,
                                               float& scalesQuant) {
    LocalTensor<T3> updatesLocal = updatesInQueue.template DeQue<T3>();
    LocalTensor<T3> smoothScalesLocal;
    LocalTensor<float> tempUpdatesCast;
    LocalTensor<float> tempsmoothScalesCast;

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    tempUpdatesCast = updateF32.template Get<float>();
    Cast(tempUpdatesCast, updatesLocal, RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
    if (hasSmoothScales) {
      smoothScalesLocal = smoothScalesInQueue.template DeQue<T3>();
      tempsmoothScalesCast = smoothScaleF32.template Get<float>();
      Cast(tempsmoothScalesCast, smoothScalesLocal[0], RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
    }
    ComputeQuantByOne(tempUpdatesCast, tempsmoothScalesCast, tilingPtr, scalesQuant);
#else
    if (hasSmoothScales) {
      smoothScalesLocal = smoothScalesInQueue.template DeQue<T3>();
    }
    ComputeQuantByOne(updatesLocal, smoothScalesLocal, tilingPtr, scalesQuant);
#endif

    updatesInQueue.FreeTensor(updatesLocal);
    if (hasSmoothScales) {
      smoothScalesInQueue.EnQue(smoothScalesLocal);
    }

    return;
  }

  __aicore__ inline int64_t GetDetOffsetNeg2LargeEle(int64_t coreBatchIndex, int64_t updateDim0Index,
                                                     int64_t updateDim1Index, LocalTensor<T2>& indicesLocal,
                                                     const DynamicQuantUpdateScatterTilingData* tiling) {
    int64_t dstOffset = 0;
    int64_t actualBatchIdx = 0;
    if (tiling->indicesShapeRank == INDICES_RANK2) {
      int64_t indexIdx = updateDim0Index;
      int64_t bsIdx = indicesLocal.GetValue(indexIdx * INDICES_RANK2);
      int64_t validIdx = indicesLocal.GetValue(indexIdx * INDICES_RANK2 + 1);
      actualBatchIdx = bsIdx * tiling->dstFirSecBsStride + tiling->dstBsStride * updateDim1Index;
      dstOffset = actualBatchIdx + (validIdx + tiling->eachCoreBsNum * blockIdx + coreBatchIndex) * tiling->sizePerHead;
    } else {
      int64_t indexIdx = updateDim0Index;
      int64_t bsIdx = indicesLocal.GetValue(indexIdx);
      actualBatchIdx = indexIdx * tiling->dstFirSecBsStride + tiling->dstBsStride * updateDim1Index;
      dstOffset = actualBatchIdx + (bsIdx + tiling->eachCoreBsNum * blockIdx + coreBatchIndex) * tiling->sizePerHead;
    }

    return dstOffset;
  }

  __aicore__ inline void CopyOutScales(int64_t dstOffset, int64_t elements) {
    LocalTensor<float> varScaleLocal = varScaleOutQueue.DeQue<float>();

    DataCopyParams copyParams{1, (uint16_t)(elements * sizeof(float)), 0, 0};
    DataCopyPad(varScalesGm[dstOffset], varScaleLocal[0], copyParams);

    varScaleOutQueue.FreeTensor(varScaleLocal);
  }

  __aicore__ inline void CopyUpdatesAndScalesByEle(int64_t offset, int64_t loopIndex, int64_t Elements,
                                                   const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    LocalTensor<T3> updatesLocal = updatesInQueue.AllocTensor<T3>();
    LocalTensor<T3> smoothScalesLocal;

    DataCopy(updatesLocal, updatesGm[offset], Elements);
    updatesInQueue.EnQue(updatesLocal);

    if (hasSmoothScales) {
      smoothScalesLocal = smoothScalesInQueue.AllocTensor<T3>();
      DataCopy(smoothScalesLocal, smoothScalesGm[loopIndex * tilingPtr->innerLoopEle], Elements);
      smoothScalesInQueue.EnQue(smoothScalesLocal);
    }
  }

  __aicore__ inline void ComputeGetMaxUpdates(int64_t elements, float& maxUpdatesValue) {
    LocalTensor<T3> updatesLocal = updatesInQueue.DeQue<T3>();
    LocalTensor<T3> smoothScalesLocal;
    LocalTensor<float> tempUpdatesCast;
    LocalTensor<float> temp = tempF32.Get<float>();
    LocalTensor<float> tempsmoothScalesCast;

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    tempUpdatesCast = updateF32.Get<float>();
    Cast(tempUpdatesCast, updatesLocal, RoundMode::CAST_NONE, elements);
    if (hasSmoothScales) {
      smoothScalesLocal = smoothScalesInQueue.DeQue<T3>();
      tempsmoothScalesCast = smoothScaleF32.Get<float>();
      Cast(tempsmoothScalesCast, smoothScalesLocal[0], RoundMode::CAST_NONE, elements);
      Mul(tempUpdatesCast, tempUpdatesCast, tempsmoothScalesCast, elements);
      smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }
    Abs(tempUpdatesCast, tempUpdatesCast, elements);
    ReduceMax(temp, tempUpdatesCast, tempUpdatesCast, elements, false);
    float max_value = temp.GetValue(0);
    maxUpdatesValue = (maxUpdatesValue > max_value) ? maxUpdatesValue : max_value;
#else
    if (hasSmoothScales) {
      smoothScalesLocal = smoothScalesInQueue.DeQue<T3>();
      Mul(updatesLocal, updatesLocal, smoothScalesLocal, elements);
      smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }
    Abs(updatesLocal, updatesLocal, elements);
    ReduceMax(temp, updatesLocal, updatesLocal, elements, false);
    float max_value = temp.GetValue(0);
    maxUpdatesValue = (maxUpdatesValue > max_value) ? maxUpdatesValue : max_value;
#endif
    updatesInQueue.FreeTensor(updatesLocal);

    return;
  }

  __aicore__ inline void ComputeQuantByEle(LocalTensor<T1>& varOutLocal, LocalTensor<float>& Updates, float scale,
                                           int64_t elements) {
    Muls(Updates, Updates, scale, elements);
#if defined(ORIG_DTYPE_VAR) && (ORIG_DTYPE_VAR == DT_INT32)
    Cast(varOutLocal, Updates, RoundMode::CAST_NONE, elements);
#else
    LocalTensor<float> temp = tempF32.Get<float>();
    LocalTensor<int32_t> tempInt32 = tempF32.Get<int32_t>();
    Cast(tempInt32, Updates, RoundMode::CAST_RINT, elements);
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    SetDeqScale(static_cast<half>(1.0));
    Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, elements);
    Cast(varOutLocal, tempHalfCast, RoundMode::CAST_TRUNC, elements);
#endif

    return;
  }

  __aicore__ inline void ComputeByEle(float scale, int64_t elements) {
    LocalTensor<T3> updatesLocal = updatesInQueue.DeQue<T3>();
    LocalTensor<float> tempUpdatesCast;
    LocalTensor<T1> varOutLocal = varOutQueue.AllocTensor<T1>();

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    tempUpdatesCast = updateF32.Get<float>();
    Cast(tempUpdatesCast, updatesLocal, RoundMode::CAST_NONE, elements);
    if (hasSmoothScales) {
      LocalTensor<T3> smoothScalesLocal = smoothScalesInQueue.DeQue<T3>();
      LocalTensor<float> tempsmoothScalesCast = smoothScaleF32.Get<float>();
      Cast(tempsmoothScalesCast, smoothScalesLocal, RoundMode::CAST_NONE, elements);
      Mul(tempUpdatesCast, tempUpdatesCast, tempsmoothScalesCast, elements);
      smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }
    ComputeQuantByEle(varOutLocal, tempUpdatesCast, scale, elements);
#else
    if (hasSmoothScales) {
      LocalTensor<T3> smoothScalesLocal = smoothScalesInQueue.DeQue<T3>();
      Mul(updatesLocal, updatesLocal, smoothScalesLocal, elements);
      smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }
    ComputeQuantByEle(varOutLocal, updatesLocal, scale, elements);
#endif
    varOutQueue.EnQue(varOutLocal);
    updatesInQueue.FreeTensor(updatesLocal);

    return;
  }

  __aicore__ inline void CopyUpdatesIn(int64_t offset, int64_t elements) {
    LocalTensor<T3> updatesLocal = updatesInQueue.AllocTensor<T3>();
    DataCopy(updatesLocal, updatesGm[offset], elements);
    updatesInQueue.EnQue(updatesLocal);
  }

  __aicore__ inline void CopyOutByBatchs(int64_t coreBsNum, const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    LocalTensor<T2> indicesLocal = indicesInQueue.DeQue<T2>();
    LocalTensor<T1> varOutLocal = varOutQueue.DeQue<T1>();
    LocalTensor<float> varScaleLocal = varScaleOutQueue.DeQue<float>();
    LocalTensor<float> temp = tempF32.Get<float>();
    int64_t quantNumOneCore = tilingPtr->quantReptNum * tilingPtr->updateAxisShape;

    for (int64_t i = 0; i < coreBsNum; i++) {
      int64_t dstOffset = GetDetOffsetNeg2(i, indicesLocal, tilingPtr);
      DataCopy(varGm[dstOffset], varOutLocal[i * tilingPtr->srcBsStride], tilingPtr->srcBsStride);

      for (int64_t index = 0; index < quantNumOneCore; index++) {
        temp.SetValue(index, varScaleLocal[i * quantNumOneCore].GetValue(index));
      }
      DataCopyParams copyParams{1, (uint16_t)(quantNumOneCore * sizeof(float)), 0, 0};
      DataCopyPad(varScalesGm[dstOffset / tilingPtr->varOrigLastDimSize], temp, copyParams);
    }

    indicesInQueue.FreeTensor(indicesLocal);
    varOutQueue.FreeTensor(varOutLocal);
    varScaleOutQueue.FreeTensor(varScaleLocal);
  }

  __aicore__ inline void FreeIndices() {
    LocalTensor<T2> indicesLocal = indicesInQueue.DeQue<T2>();
    indicesInQueue.FreeTensor(indicesLocal);
  }

  __aicore__ inline void FreeSmoothScales() {
    if (hasSmoothScales) {
      LocalTensor<T3> smoothScalesLocal = smoothScalesInQueue.DeQue<T3>();
      smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }
  }

  __aicore__ inline void CopyVarOut(int64_t dstOffset, int64_t elements) {
    LocalTensor<T1> varOutLocal = varOutQueue.DeQue<T1>();
    DataCopy(varGm[dstOffset], varOutLocal, elements);
    varOutQueue.FreeTensor(varOutLocal);
  }

  __aicore__ inline void CopyOutByOneBatch(int64_t coreBatchIndex, int scalesNum,
                                           const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    LocalTensor<T2> indicesLocal = indicesInQueue.DeQue<T2>();
    LocalTensor<T1> varOutLocal = varOutQueue.DeQue<T1>();
    LocalTensor<float> varScaleLocal = varScaleOutQueue.DeQue<float>();

    int64_t dstOffset = GetDetOffsetNeg2(coreBatchIndex, indicesLocal, tilingPtr);
    DataCopy(varGm[dstOffset], varOutLocal[0], tilingPtr->srcBsStride);

    DataCopyParams copyParams{1, (uint16_t)(scalesNum * sizeof(float)), 0, 0};
    DataCopyPad(varScalesGm[dstOffset / tilingPtr->varOrigLastDimSize], varScaleLocal[0], copyParams);

    indicesInQueue.EnQue(indicesLocal);
    varOutQueue.FreeTensor(varOutLocal);
    varScaleOutQueue.FreeTensor(varScaleLocal);
  }

  __aicore__ inline void ComputeQuantByOne(LocalTensor<float>& Updates, LocalTensor<float>& smoothScales,
                                           const DynamicQuantUpdateScatterTilingData* tilingPtr, float& scalesQuant) {
    LocalTensor<T1> varOutLocal = varOutQueue.AllocTensor<T1>();
    LocalTensor<float> temp = tempF32.Get<float>();
    LocalTensor<int32_t> tempInt32 = tempI32.Get<int32_t>();
    AscendC::LocalTensor<float> temp127 = tempInt32.ReinterpretCast<float>();

    Duplicate<float>(temp127, QUANT_CONST_VALUE, tilingPtr->varOrigLastDimSize);
    if (hasSmoothScales) {
      Mul(Updates, Updates, smoothScales, tilingPtr->varOrigLastDimSize);
    }
    Abs(temp, Updates, tilingPtr->varOrigLastDimSize);
    Div(temp, temp127, temp, tilingPtr->varOrigLastDimSize);
    ReduceMin(temp, temp, temp, tilingPtr->varOrigLastDimSize, false);
    scalesQuant = 1 / temp.GetValue(0);
    Muls(Updates, Updates, temp.GetValue(0), tilingPtr->varOrigLastDimSize);
#if defined(ORIG_DTYPE_VAR) && (ORIG_DTYPE_VAR == DT_INT32)
    Cast(varOutLocal, Updates, RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
#else
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    Cast(tempInt32, Updates, RoundMode::CAST_RINT, tilingPtr->varOrigLastDimSize);
    SetDeqScale(static_cast<half>(1.0));
    Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingPtr->varOrigLastDimSize);
    Cast(varOutLocal, tempHalfCast, RoundMode::CAST_TRUNC, tilingPtr->varOrigLastDimSize);
#endif
    varOutQueue.EnQue(varOutLocal);

    return;
  }

  __aicore__ inline void ComputeQuant(LocalTensor<float>& Updates, LocalTensor<float>& smoothScales,
                                      const DynamicQuantUpdateScatterTilingData* tilingPtr, int64_t rptNum) {
    LocalTensor<T1> varOutLocal = varOutQueue.AllocTensor<T1>();
    LocalTensor<float> varScaleOutLocal = varScaleOutQueue.AllocTensor<float>();

    LocalTensor<float> temp = tempF32.Get<float>();
    LocalTensor<int32_t> tempInt32 = tempI32.Get<int32_t>();
    AscendC::LocalTensor<float> temp127 = tempInt32.ReinterpretCast<float>();

    for (int64_t i = 0; i < rptNum; i++) {
      Duplicate<float>(temp127, QUANT_CONST_VALUE, tilingPtr->varOrigLastDimSize);
      int64_t offset = i * tilingPtr->varOrigLastDimSize;
      if (hasSmoothScales) {
        Mul(Updates[offset], Updates[offset], smoothScales, tilingPtr->varOrigLastDimSize);
      }
      Abs(temp, Updates[offset], tilingPtr->varOrigLastDimSize);
      Div(temp, temp127, temp, tilingPtr->varOrigLastDimSize);
      ReduceMin(temp, temp, temp, tilingPtr->varOrigLastDimSize, false);

      varScaleOutLocal.SetValue(i, 1 / temp.GetValue(0));
      Muls(Updates[offset], Updates[offset], temp.GetValue(0), tilingPtr->varOrigLastDimSize);
#if defined(ORIG_DTYPE_VAR) && (ORIG_DTYPE_VAR == DT_INT32)
      Cast(varOutLocal[offset], Updates[offset], RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
#else
      Cast(tempInt32, Updates[offset], RoundMode::CAST_RINT, tilingPtr->varOrigLastDimSize);
      AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
      SetDeqScale(static_cast<half>(1.0));
      Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingPtr->varOrigLastDimSize);
      Cast(varOutLocal[offset], tempHalfCast, RoundMode::CAST_TRUNC, tilingPtr->varOrigLastDimSize);
#endif
    }
    varOutQueue.EnQue(varOutLocal);
    varScaleOutQueue.EnQue(varScaleOutLocal);

    return;
  }

  __aicore__ inline int64_t GetDetOffsetNeg2(int64_t coreBatchIndex, LocalTensor<T2>& indicesLocal,
                                             const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    int64_t dstOffset = 0;
    int64_t actualBatchIdx = 0;
    if (tilingPtr->indicesShapeRank == INDICES_RANK2) {
      int64_t indexIdx = (blockIdx * tilingPtr->eachCoreBsNum + coreBatchIndex) / tilingPtr->numHead;
      int64_t bsIdx = indicesLocal.GetValue(indexIdx * INDICES_RANK2);
      int64_t validIdx = indicesLocal.GetValue(indexIdx * INDICES_RANK2 + 1);
      actualBatchIdx =
          bsIdx * tilingPtr->numHead + (blockIdx * tilingPtr->eachCoreBsNum + coreBatchIndex) % tilingPtr->numHead;
      dstOffset = actualBatchIdx * tilingPtr->dstBsStride + validIdx * tilingPtr->sizePerHead;
    } else {
      int64_t indexIdx = blockIdx * tilingPtr->eachCoreBsNum + coreBatchIndex;
      int64_t validIdx = indicesLocal.GetValue(indexIdx / tilingPtr->numHead);
      dstOffset = (indexIdx * tilingPtr->dstBsStride + validIdx * tilingPtr->sizePerHead);
    }

    return dstOffset;
  }

 protected:
  /* ascendc variable */
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> indicesInQueue, updatesInQueue, smoothScalesInQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> varOutQueue, varScaleOutQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> updateF32, smoothScaleF32, tempF32, tempI32;

  /* global memory address */
  GlobalTensor<T1> varGm;
  GlobalTensor<float> varScalesGm;
  GlobalTensor<T2> indicesGm;
  GlobalTensor<T3> updatesGm;
  GlobalTensor<T3> smoothScalesGm;

  /* variable */
  int64_t blockIdx;
  int64_t coreBatchNum;
  int64_t coreBsNumElements;
  int64_t varScaleOutElemens;
  uint64_t varScaleOutAlignElemens;
  bool hasSmoothScales = false;
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT

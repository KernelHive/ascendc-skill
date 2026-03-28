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
 * \file dynamic_quant_update_scatter_each_core_large_batch.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_BATCH_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_BATCH_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_comm.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterEachCoreLargeBatch : DynamicQuantUpdateScatterBase<T1, T2, T3> {
 public:
  __aicore__ inline DynamicQuantUpdateScatterEachCoreLargeBatch() {
  }

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                              const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    int64_t eachCoreBsNumElements = tilingPtr->srcBsStride * tilingPtr->eachCoreBsNum;
    quantNumOneCore = tilingPtr->quantReptNum * tilingPtr->updateAxisShape;

    this->InitBase(var, varScale, indices, smoothScales, tilingPtr);
    int64_t varScaleOutAlignElemens =
        (quantNumOneCore * sizeof(float) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO / sizeof(float);
    this->InitUpdateGM(updates, this->blockIdx * eachCoreBsNumElements, this->coreBsNumElements);
    this->InitUpdatesInQueue(tilingPtr->srcBsStride * sizeof(T3));
    this->InitSmoothScalesInQueue(tilingPtr->varOrigLastDimSize * sizeof(T3),
                                  tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitVarOutQueue(tilingPtr->srcBsStride * sizeof(T1));
    this->InitvarScaleOutQueue(varScaleOutAlignElemens * sizeof(float));
    this->InitTempF32(tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitTempI32(tilingPtr->varOrigLastDimSize * sizeof(int32_t));

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    this->InitUpdateF32(tilingPtr->srcBsStride * sizeof(float));
#endif
  }

  __aicore__ inline void Process(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    if (this->blockIdx < tilingPtr->coreNum) {
      LoopProcessIn(tilingPtr);
    }
  }

 private:
  __aicore__ inline void LoopProcessIn(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    this->CopyIndicesIn(tilingPtr);
    this->CopySmoothScalesIn(0, tilingPtr->varOrigLastDimSize);

    for (int64_t coreBatchIndex = 0; coreBatchIndex < this->coreBatchNum; coreBatchIndex++) {
      this->CopyUpdatesIn(coreBatchIndex * tilingPtr->srcBsStride, tilingPtr->srcBsStride);
      Compute(tilingPtr);
      this->CopyOutByOneBatch(coreBatchIndex, quantNumOneCore, tilingPtr);
    }

    this->FreeIndices();
    this->FreeSmoothScales();
  }

  __aicore__ inline void Compute(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    LocalTensor<T3> updatesLocal = this->updatesInQueue.template DeQue<T3>();
    LocalTensor<T3> smoothScalesLocal;
    LocalTensor<float> tempUpdatesCast;
    LocalTensor<float> tempsmoothScalesCast;

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    tempUpdatesCast = this->updateF32.template Get<float>();
    Cast(tempUpdatesCast, updatesLocal, RoundMode::CAST_NONE, tilingPtr->srcBsStride);
    if (this->hasSmoothScales) {
      smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
      tempsmoothScalesCast = this->smoothScaleF32.template Get<float>();
      Cast(tempsmoothScalesCast, smoothScalesLocal[0], RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
    }
    this->ComputeQuant(tempUpdatesCast, tempsmoothScalesCast, tilingPtr, quantNumOneCore);
#else
    if (this->hasSmoothScales) {
      smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
    }
    this->ComputeQuant(updatesLocal, smoothScalesLocal, tilingPtr, quantNumOneCore);
#endif
    this->updatesInQueue.FreeTensor(updatesLocal);
    if (this->hasSmoothScales) {
      this->smoothScalesInQueue.EnQue(smoothScalesLocal);
    }

    return;
  }

 private:
  /* ascendc variable */
  int64_t quantNumOneCore;
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT

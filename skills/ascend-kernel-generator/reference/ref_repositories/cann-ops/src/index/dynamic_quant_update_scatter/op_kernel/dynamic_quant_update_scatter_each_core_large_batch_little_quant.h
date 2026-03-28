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
 * \file dynamic_quant_update_scatter_each_core_large_batch_little_quant.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_BATCH_LITTLE_QUANT_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_BATCH_LITTLE_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_comm.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterEachCoreLargeBatchLittleQuant : DynamicQuantUpdateScatterBase<T1, T2, T3> {
 public:
  __aicore__ inline DynamicQuantUpdateScatterEachCoreLargeBatchLittleQuant() {
  }

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                              const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    int64_t eachCoreBsNumElements = tilingPtr->srcBsStride * tilingPtr->eachCoreBsNum;
    this->InitBase(var, varScale, indices, smoothScales, tilingPtr);
    int64_t varScaleOutAlignElemens =
        (tilingPtr->eachCoreBsNum * tilingPtr->quantReptNum * sizeof(float) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO /
        sizeof(float);
    this->InitUpdateGM(updates, this->blockIdx * eachCoreBsNumElements, this->coreBsNumElements);
    this->InitUpdatesInQueue(tilingPtr->varOrigLastDimSize * sizeof(T3));
    this->InitSmoothScalesInQueue(tilingPtr->varOrigLastDimSize * sizeof(T3),
                                  tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitVarOutQueue(tilingPtr->varOrigLastDimSize * sizeof(T1));
    this->InitvarScaleOutQueue(varScaleOutAlignElemens * sizeof(float));
    this->InitTempF32(tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitTempI32(tilingPtr->varOrigLastDimSize * sizeof(int32_t));

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    this->InitUpdateF32(tilingPtr->varOrigLastDimSize * sizeof(float));
#endif
  }

  __aicore__ inline void Process(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    if (this->blockIdx < tilingPtr->coreNum) {
      LoopProcessSub(tilingPtr);
    }
  }

 private:
  __aicore__ inline void LoopProcessSub(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    this->CopyIndicesIn(tilingPtr);
    this->CopySmoothScalesIn(0, tilingPtr->varOrigLastDimSize);
    int64_t dstBaseOffsetOrg = 0;
    int64_t srcBaseOffsetOrg = 0;
    int64_t dstBaseOffset = 0;
    int64_t srcBaseOffset = 0;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    float scalesQuant = 0;

    LocalTensor<T2> indicesLocal = this->indicesInQueue.template DeQue<T2>();
    for (int64_t coreBatchIndex = 0; coreBatchIndex < this->coreBatchNum; coreBatchIndex++) {
      dstBaseOffsetOrg = this->GetDetOffsetNeg2(coreBatchIndex, indicesLocal, tilingPtr);
      srcBaseOffsetOrg = coreBatchIndex * tilingPtr->srcBsStride;
      for (int64_t updateAxisIndex = 0; updateAxisIndex < tilingPtr->updateAxisShape; updateAxisIndex++) {
        dstBaseOffset = dstBaseOffsetOrg + updateAxisIndex * tilingPtr->sizePerHead;
        srcBaseOffset = srcBaseOffsetOrg + updateAxisIndex * tilingPtr->sizeSrcPerHead;
        LocalTensor<float> varScaleOutLocal = this->varScaleOutQueue.template AllocTensor<float>();
        for (int64_t quantIndex = 0; quantIndex < tilingPtr->quantReptNum; quantIndex++) {
          srcOffset = srcBaseOffset + quantIndex * tilingPtr->varOrigLastDimSize;
          dstOffset = dstBaseOffset + quantIndex * tilingPtr->varOrigLastDimSize;
          this->CopyUpdatesIn(srcOffset, tilingPtr->varOrigLastDimSize);
          this->ComputeForLittleQuant(tilingPtr, scalesQuant);
          varScaleOutLocal.SetValue(quantIndex, scalesQuant);
          this->CopyVarOut(dstOffset, tilingPtr->varOrigLastDimSize);
        }
        this->varScaleOutQueue.EnQue(varScaleOutLocal);
        this->CopyOutScales(dstBaseOffset / tilingPtr->varOrigLastDimSize, tilingPtr->quantReptNum);
      }
    }

    this->indicesInQueue.FreeTensor(indicesLocal);
    this->FreeSmoothScales();
  }

 private:
  /* ascendc variable */
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT

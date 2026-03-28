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
 * \file dynamic_quant_update_scatter_each_core_large_batch_large_quant.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_BATCH_LARGE_QUANT_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_BATCH_LARGE_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_comm.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterEachCoreLargeBatchLargeQuant : DynamicQuantUpdateScatterBase<T1, T2, T3> {
 public:
  __aicore__ inline DynamicQuantUpdateScatterEachCoreLargeBatchLargeQuant() {
  }

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                              const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    int64_t eachCoreBsNumElements = tilingPtr->srcBsStride * tilingPtr->eachCoreBsNum;
    this->InitBase(var, varScale, indices, smoothScales, tilingPtr);
    int64_t varScaleOutAlignElemens =
        (tilingPtr->eachCoreBsNum * tilingPtr->quantReptNum * sizeof(float) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO /
        sizeof(float);
    this->InitUpdateGM(updates, this->blockIdx * eachCoreBsNumElements, this->coreBsNumElements);

    this->InitvarScaleOutQueue(varScaleOutAlignElemens * sizeof(float));
    this->InitUpdatesInQueue(tilingPtr->innerLoopEle * sizeof(T3));
    this->InitSmoothScalesInQueue(tilingPtr->innerLoopEle * sizeof(T3), tilingPtr->innerLoopEle * sizeof(float));
    this->InitVarOutQueue(tilingPtr->innerLoopEle * sizeof(T1));
    this->InitTempF32(tilingPtr->innerLoopEle * sizeof(float));
    this->InitTempI32(tilingPtr->innerLoopEle * sizeof(int32_t));

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    this->InitUpdateF32(tilingPtr->innerLoopEle * sizeof(float));
#endif
  }

  __aicore__ inline void Process(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    if (this->blockIdx < tilingPtr->coreNum) {
      LoopProcessInner(tilingPtr);
    }
  }

 private:
  __aicore__ inline void CalcEachQuant(int64_t dstBaseOffsetOrg, int64_t srcBaseOffsetOrg,
                                       const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    int64_t dstBaseOffset = 0;
    int64_t srcBaseOffset = 0;
    LocalTensor<float> temp = this->tempF32.template Get<float>();

    for (int64_t updateAxisIndex = 0; updateAxisIndex < tilingPtr->updateAxisShape; updateAxisIndex++) {
      dstBaseOffset = dstBaseOffsetOrg + updateAxisIndex * tilingPtr->sizePerHead;
      srcBaseOffset = srcBaseOffsetOrg + updateAxisIndex * tilingPtr->sizeSrcPerHead;
      LocalTensor<float> varScaleOutLocal = this->varScaleOutQueue.template AllocTensor<float>();
      for (int64_t quantIndex = 0; quantIndex < tilingPtr->quantReptNum; quantIndex++) {
        srcOffset = srcBaseOffset + quantIndex * tilingPtr->varOrigLastDimSize;
        dstOffset = dstBaseOffset + quantIndex * tilingPtr->varOrigLastDimSize;
        maxUpdatesValue = MIN_FLOAT_VALUE;
        for (int64_t loopIndex = 0; loopIndex < tilingPtr->innerLoopTimes; loopIndex++) {
          srcOffset = srcOffset + loopIndex * tilingPtr->innerLoopEle;
          this->CopyUpdatesAndScalesByEle(srcOffset, loopIndex, tilingPtr->innerLoopEle, tilingPtr);
          this->ComputeGetMaxUpdates(tilingPtr->innerLoopEle, maxUpdatesValue);
        }
        if (tilingPtr->innerLoopTail != 0) {
          srcOffset = srcBaseOffset + tilingPtr->innerLoopTimes * tilingPtr->innerLoopEle;
          this->CopyUpdatesAndScalesByEle(srcOffset, tilingPtr->innerLoopTimes, tilingPtr->innerLoopTail, tilingPtr);
          this->ComputeGetMaxUpdates(tilingPtr->innerLoopTail, maxUpdatesValue);
        }

        temp.SetValue(0, QUANT_CONST_VALUE);
        temp.SetValue(EIGHT, maxUpdatesValue);
        Div(temp[0], temp[0], temp[EIGHT], 1);
        scale = temp.GetValue(0);
        varScaleOutLocal.SetValue(quantIndex, 1 / temp.GetValue(0));

        for (int64_t loopIndex = 0; loopIndex < tilingPtr->innerLoopTimes; loopIndex++) {
          srcOffset = srcBaseOffset + loopIndex * tilingPtr->innerLoopEle;
          dstOffset = dstBaseOffset + loopIndex * tilingPtr->innerLoopEle;
          this->CopyUpdatesAndScalesByEle(srcOffset, loopIndex, tilingPtr->innerLoopEle, tilingPtr);
          this->ComputeByEle(scale, tilingPtr->innerLoopEle);
          this->CopyVarOut(dstOffset, tilingPtr->innerLoopEle);
        }
        if (tilingPtr->innerLoopTail != 0) {
          srcOffset = srcBaseOffset + tilingPtr->innerLoopTimes * tilingPtr->innerLoopEle;
          dstOffset = dstBaseOffset + tilingPtr->innerLoopTimes * tilingPtr->innerLoopEle;
          this->CopyUpdatesAndScalesByEle(srcOffset, tilingPtr->innerLoopTimes, tilingPtr->innerLoopTail, tilingPtr);
          this->ComputeByEle(scale, tilingPtr->innerLoopTail);
          this->CopyVarOut(dstOffset, tilingPtr->innerLoopTail);
        }
      }
      this->varScaleOutQueue.EnQue(varScaleOutLocal);
      this->CopyOutScales(dstBaseOffset / tilingPtr->varOrigLastDimSize, tilingPtr->quantReptNum);
    }
  }

  __aicore__ inline void LoopProcessInner(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    this->CopyIndicesIn(tilingPtr);
    int64_t srcBaseOffset = 0;
    int64_t dstBaseOffset = 0;

    LocalTensor<T2> indicesLocal = this->indicesInQueue.template DeQue<T2>();
    for (int64_t coreBatchIndex = 0; coreBatchIndex < this->coreBatchNum; coreBatchIndex++) {
      dstBaseOffset = this->GetDetOffsetNeg2(coreBatchIndex, indicesLocal, tilingPtr);
      srcBaseOffset = coreBatchIndex * tilingPtr->srcBsStride;
      CalcEachQuant(dstBaseOffset, srcBaseOffset, tilingPtr);
    }

    this->indicesInQueue.FreeTensor(indicesLocal);
  }

 private:
  float maxUpdatesValue;
  float scale;
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT

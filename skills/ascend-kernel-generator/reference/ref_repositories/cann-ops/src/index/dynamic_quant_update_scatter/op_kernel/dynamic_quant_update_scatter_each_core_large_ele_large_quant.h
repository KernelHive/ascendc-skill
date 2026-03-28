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
 * \file dynamic_quant_update_scatter_each_core_large_ele_large_quant.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_ELE_LARGE_QUANT_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_ELE_LARGE_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_comm.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterEachCoreLargeEleLargeQuant : DynamicQuantUpdateScatterBase<T1, T2, T3> {
 public:
  __aicore__ inline DynamicQuantUpdateScatterEachCoreLargeEleLargeQuant() {
  }

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                              const DynamicQuantUpdateScatterTilingData* tiling) {
    this->InitBase(var, varScale, indices, smoothScales, tiling);
    int64_t eachCoreQuantNums = tiling->eachCoreBsNum * tiling->quantReptNum;
    int64_t varScaleOutAlignElemens =
        (eachCoreQuantNums * sizeof(float) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO / sizeof(float);
    this->InitUpdateGM(updates, 0, tiling->updatesElements);
    this->InitUpdatesInQueue(tiling->innerLoopEle * sizeof(T3));
    this->InitSmoothScalesInQueue(tiling->innerLoopEle * sizeof(T3), tiling->innerLoopEle * sizeof(float));
    this->InitVarOutQueue(tiling->innerLoopEle * sizeof(T1));
    this->InitvarScaleOutQueue(varScaleOutAlignElemens * sizeof(float));
    this->InitTempF32(tiling->innerLoopEle * sizeof(float));
    this->InitTempI32(tiling->innerLoopEle * sizeof(int32_t));

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    this->InitUpdateF32(tiling->innerLoopEle * sizeof(float));
#endif
  }

  __aicore__ inline void Process(const DynamicQuantUpdateScatterTilingData* tiling) {
    if (this->blockIdx < tiling->coreNum) {
      LoopProcess(tiling);
    }
  }

 private:
  __aicore__ inline void LoopProcessCalc(int64_t dstOffsetBaseOrg, int64_t srcOffsetBaseOrg,
                                         const DynamicQuantUpdateScatterTilingData* tiling) {
    int64_t dstOffsetBase = 0;
    int64_t srcOffsetBase = 0;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    LocalTensor<float> temp = this->tempF32.template Get<float>();
    LocalTensor<float> varScaleOutLocal = this->varScaleOutQueue.template AllocTensor<float>();
    for (int64_t quantRptIndex = 0; quantRptIndex < tiling->quantReptNum; quantRptIndex++) {
      maxUpdatesValue = MIN_FLOAT_VALUE;
      dstOffsetBase = dstOffsetBaseOrg + quantRptIndex * tiling->varOrigLastDimSize;
      srcOffsetBase = srcOffsetBaseOrg + quantRptIndex * tiling->varOrigLastDimSize;

      for (int64_t loopIndex = 0; loopIndex < tiling->innerLoopTimes; loopIndex++) {
        srcOffset = srcOffsetBase + loopIndex * tiling->innerLoopEle;
        this->CopyUpdatesAndScalesByEle(srcOffset, loopIndex, tiling->innerLoopEle, tiling);
        this->ComputeGetMaxUpdates(tiling->innerLoopEle, maxUpdatesValue);
      }
      if (tiling->innerLoopTail != 0) {
        srcOffset = srcOffsetBase + tiling->innerLoopTimes * tiling->innerLoopEle;
        this->CopyUpdatesAndScalesByEle(srcOffset, tiling->innerLoopTimes, tiling->innerLoopTail, tiling);
        this->ComputeGetMaxUpdates(tiling->innerLoopTail, maxUpdatesValue);
      }
      temp.SetValue(0, QUANT_CONST_VALUE);
      temp.SetValue(EIGHT, maxUpdatesValue);
      Div(temp[0], temp[0], temp[EIGHT], 1);
      scale = temp.GetValue(0);
      varScaleOutLocal.SetValue(quantRptIndex, 1 / temp.GetValue(0));

      for (int64_t loopIndex = 0; loopIndex < tiling->innerLoopTimes; loopIndex++) {
        srcOffset = srcOffsetBase + loopIndex * tiling->innerLoopEle;
        dstOffset = dstOffsetBase + loopIndex * tiling->innerLoopEle;
        this->CopyUpdatesAndScalesByEle(srcOffset, loopIndex, tiling->innerLoopEle, tiling);
        this->ComputeByEle(scale, tiling->innerLoopEle);
        this->CopyVarOut(dstOffset, tiling->innerLoopEle);
      }
      if (tiling->innerLoopTail != 0) {
        srcOffset = srcOffsetBase + tiling->innerLoopTimes * tiling->innerLoopEle;
        dstOffset = dstOffsetBase + tiling->innerLoopTimes * tiling->innerLoopEle;
        this->CopyUpdatesAndScalesByEle(srcOffset, tiling->innerLoopTimes, tiling->innerLoopTail, tiling);
        this->ComputeByEle(scale, tiling->innerLoopTail);
        this->CopyVarOut(dstOffset, tiling->innerLoopTail);
      }
    }
    this->varScaleOutQueue.EnQue(varScaleOutLocal);
  }

  __aicore__ inline void LoopProcess(const DynamicQuantUpdateScatterTilingData* tiling) {
    LocalTensor<T2> indicesLocal;
    this->CopyIndicesIn(tiling);
    int64_t dstOffsetBaseOrg = 0;
    int64_t srcOffsetBaseOrg = 0;
    int64_t eachBsOffset = tiling->eachCoreBsNum * this->blockIdx;

    indicesLocal = this->indicesInQueue.template DeQue<T2>();
    for (int64_t dim0Index = 0; dim0Index < tiling->updateDim0; dim0Index++) {
      for (int64_t dim1Index = 0; dim1Index < tiling->updateDim1; dim1Index++) {
        for (int64_t coreBatchIndex = 0; coreBatchIndex < this->coreBatchNum; coreBatchIndex++) {
          dstOffsetBaseOrg = this->GetDetOffsetNeg2LargeEle(coreBatchIndex, dim0Index, dim1Index, indicesLocal, tiling);
          srcOffsetBaseOrg = dim0Index * tiling->srcFirBsStride + dim1Index * tiling->srcBsStride +
                             (coreBatchIndex + eachBsOffset) * tiling->sizeSrcPerHead;
          LoopProcessCalc(dstOffsetBaseOrg, srcOffsetBaseOrg, tiling);
          this->CopyOutScales(dstOffsetBaseOrg / tiling->varOrigLastDimSize, tiling->quantReptNum);
        }
      }
    }

    this->indicesInQueue.FreeTensor(indicesLocal);
  }

 private:
  float maxUpdatesValue;
  float scale;
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT

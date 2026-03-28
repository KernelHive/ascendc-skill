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
 * \file dynamic_quant_update_scatter_each_core_large_ele_little_quant_mod.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_ELE_LITTLT_QUANT_MOD_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_LARGE_ELE_LITTLT_QUANT_MOD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_comm.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterEachCoreLargeEleLittleQuantMod : DynamicQuantUpdateScatterBase<T1, T2, T3> {
 public:
  __aicore__ inline DynamicQuantUpdateScatterEachCoreLargeEleLittleQuantMod() {
  }

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                              const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    this->InitBase(var, varScale, indices, smoothScales, tilingPtr);
    int64_t echoCoreQuantRpetNum = 0;
    if (this->blockIdx < tilingPtr->coreNum - 1) {
      echoCoreQuantRpetNum = tilingPtr->eachCoreBsNum * tilingPtr->quantReptNum;
      loopNum = tilingPtr->innerLoopTimes;
      tailNum = tilingPtr->innerLoopTail;
    } else {
      echoCoreQuantRpetNum = tilingPtr->lastCoreBsNum * tilingPtr->quantReptNum;
      loopNum = tilingPtr->innerLoopTimesLastCore;
      tailNum = tilingPtr->innerLoopTailLastCore;
    }
    innerLoopFullRpt = tilingPtr->innerLoopFullRpt;

    int64_t varScaleOutAlignElemens =
        (echoCoreQuantRpetNum * sizeof(float) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO / sizeof(float);
    this->InitUpdateGM(updates, 0, tilingPtr->updatesElements);

    this->InitSmoothScalesInQueue(tilingPtr->varOrigLastDimSize * sizeof(T3),
                                  tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitUpdatesInQueue(tilingPtr->innerLoopEle * sizeof(T3));
    this->InitVarOutQueue(tilingPtr->innerLoopEle * sizeof(T1));
    this->InitvarScaleOutQueue(varScaleOutAlignElemens * sizeof(float));
    this->InitTempI32(tilingPtr->innerLoopEle * sizeof(int32_t));
    this->InitTempF32(tilingPtr->innerLoopEle * sizeof(float));

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    this->InitUpdateF32(tilingPtr->innerLoopEle * sizeof(float));
#endif
  }

  __aicore__ inline void Process(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    if (this->blockIdx < tilingPtr->coreNum) {
      LoopProcess(tilingPtr);
    }
  }

 private:
  __aicore__ inline void LoopProcess(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    this->CopyIndicesIn(tilingPtr);
    this->CopySmoothScalesIn(0, tilingPtr->varOrigLastDimSize);

    int64_t dstBaseOffset = 0;
    int64_t srcBaseOffset = 0;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    int64_t dstScalesBaseOffset = 0;
    int64_t eachBsOffset = tilingPtr->eachCoreBsNum * this->blockIdx;
    float scalesQuant = 0;

    LocalTensor<T2> indicesLocal = this->indicesInQueue.template DeQue<T2>();
    for (int64_t dim0Index = 0; dim0Index < tilingPtr->updateDim0; dim0Index++) {
      for (int64_t dim1Index = 0; dim1Index < tilingPtr->updateDim1; dim1Index++) {
        for (int64_t fullIdx = 0; fullIdx < loopNum; fullIdx++) {
          srcBaseOffset = dim0Index * tilingPtr->srcFirBsStride + dim1Index * tilingPtr->srcBsStride +
                          (fullIdx * innerLoopFullRpt + eachBsOffset) * tilingPtr->sizeSrcPerHead;
          this->CopyUpdatesIn(srcBaseOffset, tilingPtr->innerLoopEle);
          Compute(tilingPtr, tilingPtr->innerLoopEle);

          dstBaseOffset =
              this->GetDetOffsetNeg2LargeEle(fullIdx * innerLoopFullRpt, dim0Index, dim1Index, indicesLocal, tilingPtr);
          this->CopyVarOut(dstBaseOffset, tilingPtr->innerLoopEle);
          this->CopyOutScales(dstBaseOffset / tilingPtr->sizeSrcPerHead, tilingPtr->quantReptNum * innerLoopFullRpt);
        }

        if (tailNum != 0) {
          uint64_t tailElements = tailNum * tilingPtr->varOrigLastDimSize * tilingPtr->quantReptNum;
          srcBaseOffset = dim0Index * tilingPtr->srcFirBsStride + dim1Index * tilingPtr->srcBsStride +
                          (loopNum * innerLoopFullRpt + eachBsOffset) * tilingPtr->sizeSrcPerHead;
          this->CopyUpdatesIn(srcBaseOffset, tailElements);
          Compute(tilingPtr, tailElements);

          dstBaseOffset =
              this->GetDetOffsetNeg2LargeEle(loopNum * innerLoopFullRpt, dim0Index, dim1Index, indicesLocal, tilingPtr);
          this->CopyVarOut(dstBaseOffset, tailElements);
          this->CopyOutScales(dstBaseOffset / tilingPtr->sizeSrcPerHead, tilingPtr->quantReptNum * tailNum);
        }
      }
    }

    this->indicesInQueue.FreeTensor(indicesLocal);
    if (this->hasSmoothScales) {
      LocalTensor<T3> smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
      this->smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }
  }

  __aicore__ inline void ComputeQuantMod(LocalTensor<float>& Updates, LocalTensor<float>& smoothScales,
                                         const DynamicQuantUpdateScatterTilingData* tilingPtr, int64_t elements) {
    LocalTensor<T1> varOutLocal = this->varOutQueue.template AllocTensor<T1>();
    LocalTensor<float> varScaleOutLocal = this->varScaleOutQueue.template AllocTensor<float>();
    LocalTensor<float> temp = this->tempF32.template Get<float>();
    LocalTensor<int32_t> tempInt32 = this->tempI32.template Get<int32_t>();

    AscendC::LocalTensor<float> temp127 = tempInt32.ReinterpretCast<float>();
    Duplicate<float>(temp127, QUANT_CONST_VALUE, elements);
    int64_t rpt = elements / tilingPtr->varOrigLastDimSize;
    if (this->hasSmoothScales) {
      int64_t timesOfmask = tilingPtr->varOrigLastDimSize / MASK;
      uint8_t stride = (uint8_t)EIGHT * timesOfmask;
      for (int64_t timesIdx = 0; timesIdx < timesOfmask; timesIdx++) {
        int64_t maskOffset = timesIdx * MASK;
        Mul(Updates[maskOffset], Updates[maskOffset], smoothScales[maskOffset], MASK, rpt,
            {1, 1, 1, stride, stride, 0});
      }
    }
    Abs(temp, Updates, elements);
    Div(temp, temp127, temp, elements);
    for (int idx = 0; idx < rpt; idx++) {
      int64_t offset = idx * tilingPtr->varOrigLastDimSize;
      ReduceMin(temp, temp[offset], temp, tilingPtr->varOrigLastDimSize, false);
      varScaleOutLocal.SetValue(idx, 1 / temp.GetValue(0));
      Muls(Updates[offset], Updates[offset], temp.GetValue(0), tilingPtr->varOrigLastDimSize);
    }
#if defined(ORIG_DTYPE_VAR) && (ORIG_DTYPE_VAR == DT_INT32)
    Cast(varOutLocal, Updates, RoundMode::CAST_NONE, elements);
#else
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    Cast(tempInt32, Updates, RoundMode::CAST_RINT, elements);
    SetDeqScale(static_cast<half>(1.0));
    Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, elements);
    Cast(varOutLocal, tempHalfCast, RoundMode::CAST_TRUNC, elements);
#endif

    this->varOutQueue.EnQue(varOutLocal);
    this->varScaleOutQueue.EnQue(varScaleOutLocal);
    return;
  }

  __aicore__ inline void Compute(const DynamicQuantUpdateScatterTilingData* tilingPtr, uint64_t elements) {
    LocalTensor<T3> updatesLocal = this->updatesInQueue.template DeQue<T3>();
    LocalTensor<T3> smoothScalesLocal;
    LocalTensor<float> tempUpdatesCast;
    LocalTensor<float> tempsmoothScalesCast;

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    tempUpdatesCast = this->updateF32.template Get<float>();
    Cast(tempUpdatesCast, updatesLocal, RoundMode::CAST_NONE, elements);
    if (this->hasSmoothScales) {
      smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
      tempsmoothScalesCast = this->smoothScaleF32.template Get<float>();
      Cast(tempsmoothScalesCast, smoothScalesLocal[0], RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
    }
    ComputeQuantMod(tempUpdatesCast, tempsmoothScalesCast, tilingPtr, elements);
#else
    if (this->hasSmoothScales) {
      smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
    }
    ComputeQuantMod(updatesLocal, smoothScalesLocal, tilingPtr, elements);
#endif

    this->updatesInQueue.FreeTensor(updatesLocal);
    if (this->hasSmoothScales) {
      this->smoothScalesInQueue.EnQue(smoothScalesLocal);
    }

    return;
  }

 private:
  int64_t innerLoopFullRpt;
  int64_t loopNum;
  int64_t tailNum;
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT

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
 * \file weight_quant_batch_matmul_v2_weight_nz_performance.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_H

#include "weight_quant_batch_matmul_v2_weight_nz_performance_base.h"

namespace WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          QuantType antiQuantType, bool hasAntiQuantOffset>
class WeightQuantBatchMatmulV2WeightNzPerformanceKernel
    : public WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                        hasAntiQuantOffset> {
public:
    __aicore__ inline WeightQuantBatchMatmulV2WeightNzPerformanceKernel(){};
    __aicore__ inline void Process();
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void WeightQuantBatchMatmulV2WeightNzPerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                              antiQuantType, hasAntiQuantOffset>::Process()
{
    LocalTensor<xType> bL1Local = this->InBufBL1_.template Get<xType>();
    LocalTensor<xType> aL1Local = this->InBufAL1_.template Get<xType>();

    LocalTensor<float> biasFp32Local = this->apiTmpBuf_.template GetWithOffset<float>(this->elemsBiasFp32_, this->offsetBiasFp32_);
    LocalTensor<xType> resCNz = this->apiTmpBuf_.template GetWithOffset<xType>(this->resCNzElem_, this->resCNzOffset_);
    LocalTensor<yType> resCNd = this->apiTmpBuf_.template GetWithOffset<yType>(this->resCNdElem_, this->resCNdOffset_);

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte3ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    for (int32_t mCoreFactorIdx = 0; mCoreFactorIdx < this->mCoreLoopNum_; mCoreFactorIdx++) {
      int64_t mBlockOffset = this->mBlockOffset_ + mCoreFactorIdx * this->mAL1Size_;
      for (int32_t nCoreFactorIdx = 0; nCoreFactorIdx < this->nCoreLoopNum_; nCoreFactorIdx++) {
        int64_t nBlockOffset = this->nBlockOffset_ + nCoreFactorIdx * this->nBL1Size_;

        int32_t realNLen = this->min(this->nBL1Size_, this->tiling_->nSize - nBlockOffset);
        this->CopyInAddMul(nBlockOffset, realNLen);

        int64_t mAL1Offset = mBlockOffset;
        int64_t nBL1Offset = nBlockOffset;
        int32_t mL0Len = this->min(this->tiling_->mSize - mAL1Offset, this->mL0Size_);
        int32_t nL0Len = this->min(this->tiling_->nSize - nBL1Offset, this->nL0Size_);
        for (int32_t kFactorIdx = 0; kFactorIdx < this->kSingleCore_; kFactorIdx++) {
          int64_t kBlockOffset = kFactorIdx * this->kL1Size_;
          int32_t kL1Len = this->min(this->tiling_->kSize - kBlockOffset, this->kL1Size_);
          this->BL1Process(bL1Local, nBL1Offset, kBlockOffset, kL1Len, nL0Len);
          this->AL1Process(aL1Local, mAL1Offset, kBlockOffset, kL1Len, mL0Len);

          if (this->biasFlag_) {
            this->BiasProcess(biasFp32Local, nBlockOffset);
          }

          SetFlag<HardEvent::MTE3_MTE1>(eventIdMte3ToMte1);
          WaitFlag<HardEvent::MTE3_MTE1>(eventIdMte3ToMte1);

          this->CubeProcess(aL1Local, bL1Local, biasFp32Local, mL0Len, nL0Len, kL1Len, kFactorIdx);

          SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
          WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        this->mmObj_.GetTensorC(resCNz);

        AscendC::PipeBarrier<PIPE_V>();

        this->PostProcess(mL0Len, nL0Len, mAL1Offset, nBL1Offset);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        int64_t yGmOffset = mAL1Offset * this->tiling_->nSize + nBL1Offset;
        this->CopyVec2Out(yGmOffset, mL0Len, nL0Len, resCNd);

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
      }
    }
}

}  // namespace WeightQuantBatchMatmulV2

#endif  // WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_H
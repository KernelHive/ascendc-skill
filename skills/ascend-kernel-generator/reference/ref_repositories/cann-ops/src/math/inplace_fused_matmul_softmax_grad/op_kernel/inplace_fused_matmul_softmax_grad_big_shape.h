/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BIG_SHAPE_H
#define INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BIG_SHAPE_H

#include "inplace_fused_matmul_softmax_grad_base.h"

namespace InplaceFusedMatmulSoftmaxGradOpt {

template <typename mmType, typename dataType, bool isAligned, bool isCast>
class InplaceFusedMatmulSoftmaxGradBigShape : public InplaceFusedMatmulSoftmaxGradBase<mmType, dataType, isAligned, isCast> {
public:
    __aicore__ inline InplaceFusedMatmulSoftmaxGradBigShape(mmType &mm) : InplaceFusedMatmulSoftmaxGradBase<mmType, dataType, isAligned, isCast>(mm)
    {}

    __aicore__ inline void Init(GM_ADDR softmaxOutput, GM_ADDR gradOutput, GM_ADDR values,
        GM_ADDR workspace, const InplaceFusedMatmulSoftmaxGradTilingData *__restrict tilingData, AscendC::TPipe *pipe)
    {
        this->baseTilingData_ = tilingData->baseTilingData;
        this->cubeTilingData_ = tilingData->cubeTilingData;
        this->headSoftMaxGradTilingData_ = tilingData->headSoftMaxGradTilingData;
        this->blockIdx_ = AscendC::GetBlockIdx();

        this->pPipe_ = pipe;

        this->rowLen_ = (this->blockIdx_ < this->baseTilingData_.headCoreNum)
                            ? this->baseTilingData_.rowLenPerHeadCore
                            : this->baseTilingData_.rowLenPerTailCore;

        // gm数据
        this->softmaxOutputGm_.SetGlobalBuffer((__gm__ dataType *)softmaxOutput,
            this->baseTilingData_.b * this->baseTilingData_.m * this->baseTilingData_.n);
        this->gradOutputGm_.SetGlobalBuffer(
            (__gm__ dataType *)gradOutput, this->baseTilingData_.b * this->baseTilingData_.m * this->baseTilingData_.k);
        this->valuesGm_.SetGlobalBuffer(
            (__gm__ dataType *)values, this->baseTilingData_.b * this->baseTilingData_.n * this->baseTilingData_.k);
        this->gradXGm_.SetGlobalBuffer(
            (__gm__ dataType *)softmaxOutput, this->baseTilingData_.b * this->baseTilingData_.m * this->baseTilingData_.n);
        this->gradSoftmaxGm_.SetGlobalBuffer(
            (__gm__ float *)workspace, this->baseTilingData_.b * this->baseTilingData_.m * this->baseTilingData_.n);

        this->pPipe_->InitBuffer(this->inQueueSoftmaxOutput_,
            BUFFER_NUM,
            this->baseTilingData_.basicRowLenHeadCore * this->baseTilingData_.innerLoopHeadColLen * sizeof(dataType));
        this->pPipe_->InitBuffer(this->inQueueGradSoftmax_,
            BUFFER_NUM,
            this->baseTilingData_.basicRowLenHeadCore * this->baseTilingData_.innerLoopHeadColLen * sizeof(float));
        this->pPipe_->InitBuffer(this->outQueueGradX_,
            BUFFER_NUM,
            this->baseTilingData_.basicRowLenHeadCore * this->baseTilingData_.innerLoopHeadColLen * sizeof(dataType));
        if constexpr (isCast) {
            this->pPipe_->InitBuffer(this->SoftmaxOutput32Temp_, this->baseTilingData_.innerLoopHeadColLen * sizeof(float));
            this->softmaxOutputUB32Temp = this->SoftmaxOutput32Temp_.template Get<float>();
        }
        this->pPipe_->InitBuffer(this->sharedTempBuf1_, 
            this->baseTilingData_.basicRowLenHeadCore * this->baseTilingData_.innerLoopHeadColLen * sizeof(float));
        this->pPipe_->InitBuffer(this->sharedTempBuf2_, 
            this->baseTilingData_.basicRowLenHeadCore * this->baseTilingData_.innerLoopHeadColLen * sizeof(float));

        this->softmaxTmpUb1_ = this->sharedTempBuf1_.template Get<float>();
        this->softmaxTmpUb2_ = this->sharedTempBuf2_.template Get<float>();
    }

    __aicore__ inline void Process()
    {
        uint32_t rowNumOffset = 0;
        uint32_t nBatchOffset = 0;
        uint32_t softmaxOutputOffsetGm = 0;
        uint32_t gradOutputOffsetGm = 0;
        uint32_t valuesOffsetGm = 0;
        uint32_t gradSoftmaxOffsetGm = 0;
        uint32_t gradXOffsetGm = 0;
        ComputeGmOffset(rowNumOffset, nBatchOffset, softmaxOutputOffsetGm, gradOutputOffsetGm, valuesOffsetGm, gradSoftmaxOffsetGm, gradXOffsetGm);
        CubeCompute(rowNumOffset, nBatchOffset, gradOutputOffsetGm, valuesOffsetGm, gradSoftmaxOffsetGm);
        for (uint32_t roundIdx = 0; roundIdx < this->rowLen_; ++roundIdx) {
            float tmpreduce = 0.0;
            uint32_t colIdx = 0;
            for (; colIdx < this->baseTilingData_.innerLoopTimes ; ++colIdx){
                uint32_t offset = softmaxOutputOffsetGm + colIdx * this->baseTilingData_.innerLoopHeadColLen + roundIdx * this->baseTilingData_.colLen;
                CopyIn(offset, this->baseTilingData_.innerLoopHeadColLen);
                MulSum(this->baseTilingData_.innerLoopHeadColLen);
                tmpreduce += static_cast<float>(this->softmaxTmpUb2_.GetValue(0));
            }
            if (this->baseTilingData_.innerLoopTailColLen > 0) {
                uint32_t offset = softmaxOutputOffsetGm + colIdx * this->baseTilingData_.innerLoopHeadColLen + roundIdx * this->baseTilingData_.colLen;
                CopyIn(offset, this->baseTilingData_.innerLoopTailColLen);
                MulSum(this->baseTilingData_.innerLoopTailColLen);
                tmpreduce += static_cast<float>(this->softmaxTmpUb2_.GetValue(0));
            }
            AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventID);
            AscendC::PipeBarrier<PIPE_V>();
            this->softmaxTmpUb1_.SetValue(0, tmpreduce);
            const uint32_t srcShape_[1] = {1};
            const uint32_t dstShape_[1] = {this->baseTilingData_.innerLoopHeadColLen};
            AscendC::Broadcast<float, 1, 0>(this->softmaxTmpUb2_, this->softmaxTmpUb1_, dstShape_, srcShape_);
            AscendC::PipeBarrier<PIPE_V>();
            for (colIdx = 0; colIdx < this->baseTilingData_.innerLoopTimes; ++colIdx){
                uint32_t offset = softmaxOutputOffsetGm + colIdx * this->baseTilingData_.innerLoopHeadColLen + roundIdx * this->baseTilingData_.colLen;
                CopyIn(offset, this->baseTilingData_.innerLoopHeadColLen);
                SubMul(this->baseTilingData_.innerLoopHeadColLen);
                CopyOut(offset, this->baseTilingData_.innerLoopHeadColLen);
            }
            if (this->baseTilingData_.innerLoopTailColLen > 0) {
                uint32_t offset = softmaxOutputOffsetGm + colIdx * this->baseTilingData_.innerLoopHeadColLen + roundIdx * this->baseTilingData_.colLen;
                CopyIn(offset, this->baseTilingData_.innerLoopTailColLen);
                SubMul(this->baseTilingData_.innerLoopTailColLen);
                CopyOut(offset, this->baseTilingData_.innerLoopTailColLen);
            }    
        }
    }

    __aicore__ inline void ComputeGmOffset(uint32_t &rowNumOffset, uint32_t &nBatchOffset,
        uint32_t &softmaxOutputOffsetGm, uint32_t &gradOutputOffsetGm, uint32_t &valuesOffsetGm, uint32_t &gradSoftmaxOffsetGm,
        uint32_t &gradXOffsetGm)
    {
        if (this->blockIdx_ < this->baseTilingData_.headCoreNum) {
            rowNumOffset = this->blockIdx_ * this->baseTilingData_.rowLenPerHeadCore;
        } else {
            rowNumOffset =
                this->baseTilingData_.headCoreNum * this->baseTilingData_.rowLenPerHeadCore +
                (this->blockIdx_ - this->baseTilingData_.headCoreNum) * this->baseTilingData_.rowLenPerTailCore;
        }
        nBatchOffset = rowNumOffset / this->baseTilingData_.m;
        softmaxOutputOffsetGm = rowNumOffset * this->baseTilingData_.n;
        gradOutputOffsetGm = rowNumOffset * this->baseTilingData_.k;
        valuesOffsetGm = nBatchOffset * this->baseTilingData_.n * this->baseTilingData_.k;
        gradSoftmaxOffsetGm = softmaxOutputOffsetGm;
        gradXOffsetGm = softmaxOutputOffsetGm;
    }

    __aicore__ inline void CubeCompute(uint32_t &rowNumOffset, uint32_t &nBatchOffset, uint32_t &gradOutputOffsetGm,
        uint32_t &valuesOffsetGm, uint32_t &gradSoftmaxOffsetGm)
    {
        uint32_t lastTailMLen = rowNumOffset - nBatchOffset * this->baseTilingData_.m;
        uint32_t curMPos = (this->baseTilingData_.m - lastTailMLen) > this->rowLen_ ? this->rowLen_ : (this->baseTilingData_.m - lastTailMLen);
        uint32_t endMPos = this->rowLen_;   

        uint32_t singleM = curMPos; 
        uint32_t singleN = this->baseTilingData_.n; 
        uint32_t singleK = this->baseTilingData_.k; 
        uint32_t offsetA = gradOutputOffsetGm;  
        uint32_t offsetB = valuesOffsetGm;
        uint32_t offsetC = gradSoftmaxOffsetGm;
        
        while (true) {
            this->mm1_.SetTensorA(this->gradOutputGm_[offsetA]);
            this->mm1_.SetTensorB(this->valuesGm_[offsetB], true);
            this->mm1_.SetSingleShape(singleM, singleN, singleK);
            this->mm1_.IterateAll(this->gradSoftmaxGm_[offsetC]);

            if (endMPos - curMPos >= this->baseTilingData_.m) {
                singleM = this->baseTilingData_.m;
                singleN = this->baseTilingData_.n;
                singleK = this->baseTilingData_.k;
                offsetA = gradOutputOffsetGm + curMPos * this->baseTilingData_.k;
                offsetB += this->baseTilingData_.n * this->baseTilingData_.k;
                offsetC = gradSoftmaxOffsetGm + curMPos * this->baseTilingData_.n;
                curMPos += this->baseTilingData_.m; 
            } else {
                break;
            }
        }
        if (endMPos - curMPos > 0) {
            singleM = endMPos - curMPos;
            singleN = this->baseTilingData_.n;
            singleK = this->baseTilingData_.k;
            offsetA = gradOutputOffsetGm + curMPos * this->baseTilingData_.k;
            offsetB += this->baseTilingData_.n * this->baseTilingData_.k;
            offsetC = gradSoftmaxOffsetGm + curMPos * this->baseTilingData_.n;
            this->mm1_.SetTensorA(this->gradOutputGm_[offsetA]);
            this->mm1_.SetTensorB(this->valuesGm_[offsetB], true);
            this->mm1_.SetSingleShape(singleM, singleN, singleK);
            this->mm1_.IterateAll(this->gradSoftmaxGm_[offsetC]);
        }
        this->mm1_.End();
    }

        __aicore__ inline void CopyIn(uint32_t offset, uint32_t curColLen)
    {
        AscendC::LocalTensor<dataType> softmaxOutputUB = this->inQueueSoftmaxOutput_.template AllocTensor<dataType>();
        AscendC::LocalTensor<float> gradSoftmaxUB = this->inQueueGradSoftmax_.template AllocTensor<float>();

        if constexpr (isAligned) {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(curColLen * sizeof(dataType) / BLOCK_SIZE), 0, 0};
            AscendC::DataCopy(softmaxOutputUB, this->softmaxOutputGm_[offset], copyParams);
            
            DataCopyParams copyParams1 = {1, static_cast<uint16_t>(curColLen * sizeof(float) / BLOCK_SIZE), 0, 0};
            AscendC::DataCopy(gradSoftmaxUB, this->gradSoftmaxGm_[offset], copyParams1);
        } else {
            if (curColLen == this->baseTilingData_.innerLoopHeadColLen) {
                DataCopyParams copyParams = {1, static_cast<uint16_t>(curColLen * sizeof(dataType) / BLOCK_SIZE), 0, 0};
                AscendC::DataCopy(softmaxOutputUB, this->softmaxOutputGm_[offset], copyParams);
                
                DataCopyParams copyParams1 = {1, static_cast<uint16_t>(curColLen * sizeof(float) / BLOCK_SIZE), 0, 0};
                AscendC::DataCopy(gradSoftmaxUB, this->gradSoftmaxGm_[offset], copyParams1);
            } else {
                DataCopyParams copyParams= {1, static_cast<uint16_t>(curColLen * sizeof(dataType)), 0, 0};
                DataCopyPadParams padParams = {true,
                0,
                static_cast<uint8_t>(this-> template AlignUp<uint32_t>(curColLen, this->baseTilingData_.blockNum) - curColLen),
                0};
                AscendC::DataCopyPad(softmaxOutputUB, this->softmaxOutputGm_[offset], copyParams, padParams);

                DataCopyParams copyParams1= {1, static_cast<uint16_t>(curColLen * sizeof(float)), 0, 0};
                DataCopyPadParams padParams1 = {true,
                0,
                static_cast<uint8_t>(this-> template AlignUp<uint32_t>(curColLen, 8) - curColLen),
                0};
                AscendC::DataCopyPad(gradSoftmaxUB, this->gradSoftmaxGm_[offset], copyParams1, padParams1);
            }
        }
        event_t eventIdMTE2ToV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV2);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV2);
        this->inQueueSoftmaxOutput_.EnQue(softmaxOutputUB);
        this->inQueueGradSoftmax_.EnQue(gradSoftmaxUB);
    }

    __aicore__ inline void MulSum(uint32_t curColLen)
    {
        LocalTensor<dataType> softmaxOutputUb = this->inQueueSoftmaxOutput_.template DeQue<dataType>();
        LocalTensor<float> gradSoftmaxUb = this->inQueueGradSoftmax_.template DeQue<float>();
        if constexpr (isCast) {
            AscendC::Cast(this->softmaxOutputUB32Temp, softmaxOutputUb, AscendC::RoundMode::CAST_NONE, curColLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(this->softmaxTmpUb1_, gradSoftmaxUb, this->softmaxOutputUB32Temp, curColLen);
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(this->softmaxTmpUb1_, gradSoftmaxUb, softmaxOutputUb, curColLen); 
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventID);

        uint32_t inner = (curColLen * sizeof(float) + BLOCK_SIZE -1) / BLOCK_SIZE * BLOCK_SIZE / sizeof(float);
        AscendC::SumParams params = {1, inner, curColLen};
        AscendC::Sum(this->softmaxTmpUb2_, this->softmaxTmpUb1_, params);
        
        this->inQueueSoftmaxOutput_.template FreeTensor<dataType>(softmaxOutputUb);
        this->inQueueGradSoftmax_.template FreeTensor<float>(gradSoftmaxUb);
    }

    __aicore__ inline void SubMul(uint32_t curColLen)
    {
        LocalTensor<dataType> softmaxOutputUb = this->inQueueSoftmaxOutput_.template DeQue<dataType>();
        LocalTensor<float> gradSoftmaxUb = this->inQueueGradSoftmax_.template DeQue<float>();
        LocalTensor<dataType> gradXUb = this->outQueueGradX_.template AllocTensor<dataType>();
        if constexpr (isCast) {
            AscendC::Sub(this->softmaxTmpUb1_, gradSoftmaxUb, this->softmaxTmpUb2_, curColLen);
            AscendC::Cast(this->softmaxOutputUB32Temp, softmaxOutputUb, AscendC::RoundMode::CAST_NONE, curColLen);
            AscendC::TEventID eventID0 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventID0);
            
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(this->softmaxTmpUb1_, this->softmaxTmpUb1_, this->softmaxOutputUB32Temp, curColLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::TEventID eventID1 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventID1);
            AscendC::Cast(gradXUb, this->softmaxTmpUb1_, AscendC::RoundMode::CAST_RINT, curColLen);
        } else {
            AscendC::Sub(this->softmaxTmpUb1_, gradSoftmaxUb, this->softmaxTmpUb2_, curColLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::TEventID eventID1 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventID1);
            AscendC::Mul(gradXUb, this->softmaxTmpUb1_, softmaxOutputUb, curColLen);
            AscendC::TEventID eventID0 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventID0);
        }  
        this->outQueueGradX_.template EnQue<dataType>(gradXUb);
        this->inQueueGradSoftmax_.template FreeTensor<float>(gradSoftmaxUb);
        this->inQueueSoftmaxOutput_.template FreeTensor<dataType>(softmaxOutputUb);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curColLen)
    {
        LocalTensor<dataType> gradXUb = this->outQueueGradX_.template DeQue<dataType>();

        if constexpr (isAligned) {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(curColLen * sizeof(dataType) / BLOCK_SIZE), 0, 0};
            AscendC::DataCopy(this->gradXGm_[offset], gradXUb, copyParams);
        } else {
            if (curColLen == this->baseTilingData_.innerLoopHeadColLen) {
                DataCopyParams copyParams = {1, static_cast<uint16_t>(curColLen * sizeof(dataType) / BLOCK_SIZE), 0, 0};
                AscendC::DataCopy(this->gradXGm_[offset], gradXUb, copyParams);
            } else {
                DataCopyParams copyParams = {1, static_cast<uint16_t>(curColLen * sizeof(dataType)), 0, 0};
                AscendC::DataCopyPad(this->gradXGm_[offset], gradXUb, copyParams);
            }
        }
        this->outQueueGradX_.FreeTensor(gradXUb);
    }
};
}  // namespace InplaceFusedMatmulSoftmaxGradOpt
#endif  // INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BIG_SHAPE_H
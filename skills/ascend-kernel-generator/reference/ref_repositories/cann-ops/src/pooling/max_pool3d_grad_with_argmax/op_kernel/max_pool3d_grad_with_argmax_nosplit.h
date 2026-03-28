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
 * \file max_pool3d_grad_with_argmax_nosplit.h
 * \brief
 */
 
#ifndef OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_NOSPLIT_H
#define OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_NOSPLIT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_grad_with_argmax_base.h"

using namespace AscendC;

template <typename T, typename S>
class KernelMaxPool3DGradWithArgmaxNoSplit : public KernelMaxPool3DGradWithArgmaxBase<T,S> {
    public:
        __aicore__ KernelMaxPool3DGradWithArgmaxNoSplit(const MaxPool3DGradWithArgmaxNoSplitTilingData* __restrict tilingData_) :
                                                        KernelMaxPool3DGradWithArgmaxBase<T,S>(tilingData_->inputShapes[D_DIM], tilingData_->inputShapes[H_DIM], tilingData_->inputShapes[W_DIM],
                                                                                               tilingData_->outShapes[D_DIM], tilingData_->outShapes[H_DIM], tilingData_->outShapes[W_DIM],
                                                                                               tilingData_->kD, tilingData_->kH, tilingData_->kW,
                                                                                               tilingData_->sD, tilingData_->sH, tilingData_->sW,
                                                                                               tilingData_->pD, tilingData_->pH, tilingData_->pW,
                                                                                               tilingData_->dD, tilingData_->dH, tilingData_->dW,
                                                                                               tilingData_->partD, tilingData_->partH, tilingData_->partW,
                                                                                               tilingData_->partOutD, tilingData_->partOutH, tilingData_->partOutW,
                                                                                               tilingData_->batchesPerCore, tilingData_->leftOverBatches,
                                                                                               tilingData_->ceilD, tilingData_->ceilH, tilingData_->ceilW,
                                                                                               tilingData_->sizeUb1, tilingData_->sizeUb2, tilingData_->sizeValues){}
        
        __aicore__ void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR indices, GM_ADDR gradInput, GM_ADDR workspace) {  
            this->GMInit(gradOutput, self, indices, gradInput, workspace);
        
            PrepareScalars();
            auto event0 = this->pipe.FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(event0);
            WaitFlag<HardEvent::S_V>(event0);

            this->UbInit();
        }

        __aicore__ void Process() {
            if ((g_coreType != AIV) || (this->batchesCurCore == 0)) {
                return;
            }
            Compute();
            return;
        }
    private:
        __aicore__ void PrepareScalars() {
            this->PrepareBaseScalars();
        } 
    
        template <typename V, bool indices = false>
        __aicore__ void CopyIn(TQue<TPosition::VECIN, 1>& inQue, GlobalTensor<V> srcGm, const uint64_t& partNC) {
            constexpr uint64_t curBlockLen = BLOCK_SIZE / sizeof(V);
            uint64_t partInputSize = this->partRoundDhwInp * partNC;
            LocalTensor<V> dstUb = inQue.template AllocTensor<V>();
            
            if constexpr (indices && (sizeof(T) == sizeof(S)) && (sizeof(T) == sizeof(half))) {
                if (this->partRoundDhwInp % this->blockLength == 0) {
                    DataCopy(dstUb, srcGm, partInputSize);
                } else {
                    DataCopyExtParams copyPadParamsInput{static_cast<uint16_t>(partNC), static_cast<uint32_t>((this->partRoundDhwInp) * sizeof(V)), 0, (this->partRoundDhwInp % this->blockLength) <= curBlockLen, 0};
                    DataCopyPadExtParams<V> paddingParamsInput{false, 0, 0, 0};
                    DataCopyPad(dstUb, srcGm, copyPadParamsInput, paddingParamsInput);
                }
            } else {
                if (this->partRoundDhwInp % curBlockLen == 0) {
                    DataCopy(dstUb, srcGm, partInputSize);
                } else {
                    DataCopyExtParams copyPadParamsInput{static_cast<uint16_t>(partNC), static_cast<uint32_t>((this->partRoundDhwInp) * sizeof(V)), 0, 0, 0};
                    DataCopyPadExtParams<V> paddingParamsInput{false, 0, 0, 0};
                    DataCopyPad(dstUb, srcGm, copyPadParamsInput, paddingParamsInput);
                }
            } 
            inQue.template EnQue<V>(dstUb);
        }
        
        __aicore__ void Im2ColCall(const LocalTensor<T>& Im2ColTensor, bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8);
        __aicore__ void Compute();
        __aicore__ void ComputeIteration(const uint64_t& partNC,
                                         const uint64_t& inputOffset, const uint64_t& outputOffset,
                                         const uint64_t& padDL, const uint64_t& padDR,
                                         const uint64_t& padHL, const uint64_t& padHR,
                                         const uint64_t& padWL, const uint64_t& padWR,
                                         bool& addMulti, bool& addTail, bool& onlyTail);
        __aicore__ inline void CreateGeneralOffsets();
        __aicore__ void CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC);
};

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DGradWithArgmaxNoSplit<T,S>::CreateGeneralOffsets() {
    uint64_t i = 0;
    LocalTensor<float> hTmp = this->generalOffsets[this->partAlignRoundDhwInp];
    LocalTensor<float> wTmp = hTmp[this->partAlignRoundDhwInp];
    
    int64_t offD = -this->pD;
    int64_t offH = -this->pH;
    int64_t offW = -this->pW;

    for (int64_t curD = offD; curD < offD + (int)this->partD * (int)this->sD; curD += (int)this->sD) {
        for (int64_t curH = offH; curH < offH + (int)this->partH * (int)this->sH; curH += (int)this->sH) {
            for (int64_t curW = offW; curW < offW + (int)this->partW * (int)this->sW; curW += (int)this->sW) {
                this->generalOffsets.SetValue(i, 1.f * curD);
                hTmp.SetValue(i, 1.f * curH);
                wTmp.SetValue(i, 1.f * curW);
                ++i;
            }
        }
    }
    auto event0 = this->pipe.FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxNoSplit<T, S>::CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC) {
    LocalTensor<S> Im2ColTensor = this->queOutVals.template DeQue<S>();
    
    if (this->transAlignRoundPartOutSize == this->outSize) {
        DataCopy(dstGm, Im2ColTensor, partNC * this->transAlignRoundPartOutSize);
    } else {
        DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>(this->outSize * sizeof(S)), 
                              static_cast<uint16_t>((this->transAlignRoundPartOutSize - this->outSize) / this->blockLengthS), 0};
        DataCopyPad(dstGm, Im2ColTensor, params);
    }
    
    this->queOutVals.template FreeTensor<S>(Im2ColTensor);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxNoSplit<T,S>::Im2ColCall(const LocalTensor<T>& Im2ColTensor, bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8) {
    uint32_t sliceByWidthOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t curD = 0;
    uint64_t curHeightOut = 0;
    
    if (!this->dAdd && !this->hAdd && !this->wAdd) {
        for (; curD < this->partD; dstOffset += this->gmDOff, ++curD) {
            for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            }
            curHeightOut = 0;
        }
    } else {
        if (this->dAdd && !this->hAdd && !this->wAdd) {
            for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) { // there is a chance that sH >= kH and sW >= kW which means that the first slicing window may be coppied, not added 
                this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            }
            PipeBarrier<PIPE_V>();
            curHeightOut = 0;

            ++curD;
            dstOffset += this->gmDOff;
        } else if (this->hAdd && !this->wAdd) {
            this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            PipeBarrier<PIPE_V>();
            
            ++curHeightOut;
            dstOffset += this->gmHOff;
        }
        
        for (; curD < this->partD; dstOffset += this->gmDOff, ++curD) {
            for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            }
            curHeightOut = 0;
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxNoSplit<T,S>::ComputeIteration(const uint64_t& partNC, 
                                                                            const uint64_t& inputOffset, const uint64_t& outputOffset,
                                                                            const uint64_t& padDL, const uint64_t& padDR,
                                                                            const uint64_t& padHL, const uint64_t& padHR,
                                                                            const uint64_t& padWL, const uint64_t& padWR,
                                                                            bool& addMulti, bool& addTail, bool& onlyTail) {
    CopyIn<int32_t, true>(this->queInInds, this->indicesGm[inputOffset], partNC);

    LocalTensor<float> dTensor_ = this->queInInds.template DeQue<float>();
    this->CastInds(dTensor_, partNC * this->partAlignRoundDhwInp);
    PipeBarrier<PIPE_V>();
    
    LocalTensor<float> Im2ColTensor_ = this->queOutVals.template AllocTensor<float>();
    LocalTensor<float> tmpBufFp32 = this->tmpBuf.template ReinterpretCast<float>();
    LocalTensor<float> dTmp = this->generalOffsets;
    LocalTensor<float> hTmp = dTmp[this->partAlignRoundDhwInp];
    LocalTensor<float> wTmp = hTmp[this->partAlignRoundDhwInp];
    
    this->IndexRecalcFirst(dTensor_, tmpBufFp32, Im2ColTensor_, partNC * this->partAlignRoundDhwInp);
    PipeBarrier<PIPE_V>();
    
    for (uint64_t i = 0; i < this->blockAlPartDhwInp; i += this->partAlignRoundDhwInp) {
        Sub(dTensor_[i], dTensor_[i], dTmp, this->partAlignRoundDhwInp);
        Sub(tmpBufFp32[i], tmpBufFp32[i], hTmp, this->partAlignRoundDhwInp);
        Sub(Im2ColTensor_[i], Im2ColTensor_[i], wTmp, this->partAlignRoundDhwInp);
    }
    PipeBarrier<PIPE_V>();
    
    this->IndexRecalcSecond(dTensor_, tmpBufFp32, Im2ColTensor_, this->blockAlPartDhwInp);   
    
    LocalTensor<T> dTensor;
    if constexpr (sizeof(T) == sizeof(float)) {
        dTensor = dTensor_;
    } else {
        this->CastIndsToHalf(dTensor_, partNC * this->partAlignRoundDhwInp);
        dTensor = dTensor_.template ReinterpretCast<T>();
        PipeBarrier<PIPE_V>();
    }
    
    LocalTensor<T> transDataTensorIndicesUb = dTensor[this->sizeUb1];
    this->template PrepareInput<true>(dTensor, transDataTensorIndicesUb, this->partRoundDhwInp);
    
    CopyIn<S>(this->queInVals, this->gradOutputGm[inputOffset], partNC);
    LocalTensor<S> curInVals = this->queInVals.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2;
    if constexpr(sizeof(T) == sizeof(S)) {
        transDataTensorUb2 = curInVals;
    } else {
        transDataTensorUb2 = this->tmpBuf;
        Cast(transDataTensorUb2, curInVals, RoundMode::CAST_NONE, partNC * this->RoundUpBlock(this->partRoundDhwInp, this->blockLengthS));
        PipeBarrier<PIPE_V>();
    }
    
    this->PrepareInput(transDataTensorUb2, dTensor, this->partRoundDhwInp); 
    this->queInVals.template FreeTensor<S>(curInVals);
    
    LocalTensor<T> Im2ColTensor = Im2ColTensor_.template ReinterpretCast<T>();
    
    if (!this->kernelIsBlock){
        this->CompareSelect(dTensor, Im2ColTensor, this->inputSize);
    } else {
        this->CompareSelectBlockKernel(dTensor, Im2ColTensor, this->inputSize);
    }
    this->queInInds.template FreeTensor<float>(dTensor_);

    Duplicate(this->tmpBuf, static_cast<T>(0.f), this->sizeUb2);
    PipeBarrier<PIPE_V>();
    
    Im2ColCall(Im2ColTensor, addMulti, addTail, onlyTail, this->addParamsOverUint8);
    PipeBarrier<PIPE_V>();

    uint64_t tmpBufOffset = padDL * this->partHwOut * this->blockLength;
    if (padHL || padWL || padHR || padWR) {
        tmpBufOffset = 0;
        bool smallSingleMask = ((this->stripPadRepeatTimes == 0) || (this->stripPadRepeatTimes == 1 && this->stripPadTail == 0)) ? true : false;
        bool doTail = ((this->stripPadRepeatTimes == 1 && this->stripPadTail == 0) || (this->stripPadTail == 0)) ? false : true;

        this->StripPad(padDL, padHL, padWL, this->depthOut, this->heightOut, this->widthOut, this->partOutD, this->partOutH, this->partOutW, smallSingleMask, doTail);
        PipeBarrier<PIPE_V>();
    }
    this->template TransposeBack<S>(Im2ColTensor, this->tmpBuf[tmpBufOffset], partNC, this->transAlignRoundPartOutSize);
    PipeBarrier<PIPE_V>();

    CopyOut(this->gradInputGm[outputOffset], partNC);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxNoSplit<T,S>::Compute() {
    this->CreateKernelIndexes();
    CreateGeneralOffsets();
    
    bool singleAddOrCopy = this->im2colAddRepeatTimes == 1;
    bool multipleAddOrCopy = this->im2colAddRepeatTimes > 1;
    bool doTail = this->im2colAddTail != 0;
    bool onlyTail = this->im2colAddRepeatTimes == 0 && doTail;

    if (this->im2colSingleWOut) {
        singleAddOrCopy = false;
        multipleAddOrCopy = this->im2colCopyFullIters > 0;
        doTail = this->im2colCopyRepeatTimesTail > 0;
        onlyTail = !multipleAddOrCopy && doTail;
    }

    for (int64_t b = 0; b < this->roundBatches; b += this->blockLength) {
        ComputeIteration(this->blockLength, b * this->inputSize, b * this->outSize,
                         this->pD, this->ceilD, this->pH, this->ceilH, this->pW, this->ceilW,
                         multipleAddOrCopy, doTail, onlyTail);
    }
    
    if (this->partNC != 0) {
        ComputeIteration(this->partNC, this->roundBatches * this->inputSize, this->roundBatches * this->outSize,
                         this->pD, this->ceilD, this->pH, this->ceilH, this->pW, this->ceilW, multipleAddOrCopy, doTail, onlyTail);
    }
}

#endif
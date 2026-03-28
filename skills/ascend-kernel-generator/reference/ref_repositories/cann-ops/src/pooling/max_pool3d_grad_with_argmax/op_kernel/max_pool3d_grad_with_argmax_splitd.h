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
 * \file max_pool3d_grad_with_argmax_splitd.h
 * \brief
 */
 
#ifndef OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_SPLITD_H
#define OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_SPLITD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_grad_with_argmax_base.h"
 
using namespace AscendC;
 
template <typename T, typename S>
class KernelMaxPool3DGradWithArgmaxSplitD : public KernelMaxPool3DGradWithArgmaxBase<T,S> {
    private:
        uint64_t tailD;
        uint64_t tailOutD;
        const uint64_t overlapPart = (this->kD + ((this->kD - 1) * (this->dD - 1)) - this->sD) * this->partOutH * this->partOutW * this->blockLength;
        const uint64_t workspaceSizePerCore = this->roundPartOutSize * this->blockLength; 
        TEventID eventMTE3_MTE2_workspace = this->pipe.template AllocEventID<HardEvent::MTE3_MTE2>();
     
    public:
        __aicore__ KernelMaxPool3DGradWithArgmaxSplitD(const MaxPool3DGradWithArgmaxSplitDTilingData *__restrict tilingData_) :
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
                                                                                              tilingData_->sizeUb1, tilingData_->sizeUb2, tilingData_->sizeValues) {}
         
        __aicore__ void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR indices, GM_ADDR gradInput, GM_ADDR workspace) {
            this->GMInit(gradOutput, self, indices, gradInput, workspace);
            if (sizeof(T) == sizeof(S)) {
                this->pipe.template ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
            } else {
                if (this->dAdd) {
                    this->workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + this->coreIdx * workspaceSizePerCore);
                } else {
                    this->pipe.template ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
                }
            }
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
            auto itD = this->RoundDownBlock(this->depthInput, this->partD);
            this->roundDepthOut = this->sD * (itD - 1) + 1;
            tailD = this->depthInput - itD;
            tailOutD = this->dD * (this->kD - 1) + this->sD * (tailD - 1) + 1;
            
            this->PrepareBaseScalars();
        } 
        
        __aicore__ void CreateGeneralOffsets();
        __aicore__ void CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC, const uint64_t& realPartFull);
        __aicore__ void ProcessBatches(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD);
        __aicore__ void Im2ColCall(const LocalTensor<T>& Im2ColTensor, const uint64_t& part, bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8);
        __aicore__ void Compute();
        __aicore__ void ComputeIteration(const int64_t& dOff, const uint64_t& partNC, 
                                         const uint64_t& partOut, const uint64_t& part,
                                         const uint64_t& inputOffset, const uint64_t& outputOffset,
                                         const uint64_t& padDL, const uint64_t& padDR,
                                         const uint64_t& padHL, const uint64_t& padHR,
                                         const uint64_t& padWL, const uint64_t& padWR,
                                         bool& addMulti, bool& addTail, bool& onlyTail,
                                         bool firstIter, bool computeTail);
        
        __aicore__ void ComputeAll(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD, 
                                   bool& addMulti, bool& addTail, bool& onlyTail);
};

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::CreateGeneralOffsets() {
    uint64_t i = 0;
    int64_t offD = -this->pD;
    int64_t offH = -this->pH;
    int64_t offW = -this->pW;
    LocalTensor<float> dTmp = this->generalOffsets;
    LocalTensor<float> hTmp = dTmp[this->partAlignRoundDhwInp];
    LocalTensor<float> wTmp = hTmp[this->partAlignRoundDhwInp];
    
    if ((this->partD == 1) && (this->partW % this->blockLength == 0) && (sizeof(T) == BLOCK_SIZE / this->blockLength)) {
        Duplicate<float>(dTmp, 1.f * offD, this->partAlignRoundDhwInp);
        ArithProgression<float>(wTmp, 1.f * offW, 1.f * this->sW, this->partW);
        for (int64_t curH = offH; curH < offH + (int)this->partH * (int)this->sH; curH += (int)this->sH) {
            Duplicate<float>(hTmp[i], 1.f * curH, this->partW);
            DataCopy(wTmp[i], wTmp, this->partW);
            i += this->partW;
        }
    } else {
        for (int64_t curD = offD; curD < offD + (int)this->partD * (int)this->sD; curD += (int)this->sD) {
            for (int64_t curH = offH; curH < offH + (int)this->partH * (int)this->sH; curH += (int)this->sH) {
                for (int64_t curW = offW; curW < offW + (int)this->partW * (int)this->sW; curW += (int)this->sW) {
                    dTmp.SetValue(i, 1.f * curD);
                    hTmp.SetValue(i, 1.f * curH);
                    wTmp.SetValue(i, 1.f * curW);
                    ++i;
                }
            }
        }
    }
    auto event0 = this->pipe.FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC, const uint64_t& realPartFull) {
    auto downSize = this->RoundDownBlock(realPartFull, this->blockLengthS);
    auto roundSize = this->RoundUpBlock(realPartFull, MIN_TRANSPOSE_ROWS);
    LocalTensor<S> Im2ColTensor = this->queOutVals.template DeQue<S>();
    
    if (this->outSize % this->blockLengthS == 0 && realPartFull == downSize) {
        if (((this->outSize - realPartFull) / this->blockLengthS) < MAX_UINT16) {
            DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>(realPartFull / this->blockLengthS),
                                  static_cast<uint16_t>((roundSize - realPartFull) / this->blockLengthS),
                                  static_cast<uint16_t>((this->outSize - realPartFull) / this->blockLengthS)};
            DataCopy(dstGm, Im2ColTensor, params);
        } else {
            DataCopyParams params {static_cast<uint16_t>(1), static_cast<uint16_t>(realPartFull / this->blockLengthS), 0, 0};
            for (uint64_t nc = 0; nc < partNC; nc++) {
                DataCopy(dstGm[this->outSize * nc], Im2ColTensor[nc * roundSize], params);
            }
        }
    } else {
        DataCopyExtParams params{static_cast<uint16_t>(partNC), static_cast<uint32_t>(realPartFull * sizeof(S)), 
                                 static_cast<uint32_t>((roundSize - realPartFull) / this->blockLengthS), 
                                 static_cast<uint32_t>((this->outSize - realPartFull) * sizeof(S)), 0};
        DataCopyPad(dstGm, Im2ColTensor, params);
    }

    this->queOutVals.template FreeTensor<S>(Im2ColTensor);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::Im2ColCall(const LocalTensor<T>& Im2ColTensor, const uint64_t& part,
                                                                     bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8) {
    uint32_t sliceByWidthOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t curD = 0;
    uint64_t curHeightOut = 0;
    
    if constexpr (sizeof(T) == sizeof(S)) { // fp32 | fp16 case
        if (!this->dAdd && !this->hAdd && !this->wAdd) {
            for (; curD < part; dstOffset += this->gmDOff, ++curD) {
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
            
            for (; curD < part; dstOffset += this->gmDOff, ++curD) {
                for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                    this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                }
                curHeightOut = 0;
            }
        }
    } else {  // bfloat16 case
        if (!this->dAdd) {
            if (!this->hAdd && !this->wAdd) {
                for (; curD < part; dstOffset += this->gmDOff, ++curD) {
                    for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                        this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                    }
                    curHeightOut = 0;
                }
            } else if (this->hAdd && !this->wAdd) { // overlap only by H, which means that every first H slice may be coppied, not added
                for (; curD < part; dstOffset += this->gmDOff, ++curD) {
                    this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                    PipeBarrier<PIPE_V>();
                    
                    ++curHeightOut;
                    dstOffset += this->gmHOff;
                    
                    for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                        this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                    }
                    curHeightOut = 0;
                }
            } else {
                for (; curD < part; dstOffset += this->gmDOff, ++curD) {
                    for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                        this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                    }
                    curHeightOut = 0;
                }
            }
        } else { // SetAtomicAdd is replaced by copying workspace values
                 // The DataCopy needs to be called for Im2Col tensor prior to this function
            for (; curD < part; dstOffset += this->gmDOff, ++curD) {
                for (; curHeightOut < this->partH; dstOffset += this->gmHOff, ++curHeightOut) {
                    this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, this->curDepthOff, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                }
                curHeightOut = 0;
            }
        }
    }
}

template <typename T, typename S> 
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::ComputeIteration(const int64_t& dOff, const uint64_t& partNC,
                                                                           const uint64_t& partOut, const uint64_t& part,
                                                                           const uint64_t& inputOffset, const uint64_t& outputOffset,
                                                                           const uint64_t& padDL, const uint64_t& padDR,
                                                                           const uint64_t& padHL, const uint64_t& padHR,
                                                                           const uint64_t& padWL, const uint64_t& padWR,
                                                                           bool& addMulti, bool& addTail, bool& onlyTail,
                                                                           bool firstIter, bool computeTail) {
    const uint64_t realTransRoundPartOutSize = (partOut - padDL - padDR) * this->hwOutputSize;
    const uint64_t realTransAlignRoundPartOutSize = this->RoundUpBlock(realTransRoundPartOutSize, MIN_TRANSPOSE_ROWS);
    const uint64_t partFull = part * this->partHwInp;
    const uint64_t partRound = this->RoundUpBlock(partFull);
    
    this->template CopyIn<int32_t, true>(this->queInInds, this->indicesGm[inputOffset], partNC, partFull);
    LocalTensor<float> dTensor_ = this->queInInds.template DeQue<float>();
    this->CastInds(dTensor_, partNC * partRound);
    PipeBarrier<PIPE_V>();
        
    LocalTensor<float> Im2ColTensor_ = this->queOutVals.template AllocTensor<float>();
    LocalTensor<float> tmpBufFp32 = this->tmpBuf.template ReinterpretCast<float>();
    LocalTensor<float> dTmp = this->generalOffsets;
    LocalTensor<float> hTmp = dTmp[this->partAlignRoundDhwInp];
    LocalTensor<float> wTmp = hTmp[this->partAlignRoundDhwInp];

    this->IndexRecalcFirst(dTensor_, tmpBufFp32, Im2ColTensor_, partNC * partRound);
    
    float generalOffsetD = -1.f * (dOff + this->pD);
    Adds(dTensor_, dTensor_, generalOffsetD, partNC * partRound);
    PipeBarrier<PIPE_V>();
    
    for (uint64_t i = 0; i < partNC * partRound; i += partRound) {
        Sub(dTensor_[i], dTensor_[i], dTmp, partRound);
        Sub(tmpBufFp32[i], tmpBufFp32[i], hTmp, partRound);
        Sub(Im2ColTensor_[i], Im2ColTensor_[i], wTmp, partRound);
    }
    PipeBarrier<PIPE_V>();
    
    this->transDataParamsReverse.repeatTimes = realTransAlignRoundPartOutSize / NCHW_CONV_ADDR_LIST_SIZE;
    this->transDataParamsReverse.dstRepStride = (this->transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    this->transDataParamsReverse.srcRepStride = (this->transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;

    if (computeTail) {
        this->transDataParams.repeatTimes = this->RoundUpBlock(partFull, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    }

    this->IndexRecalcSecond(dTensor_, tmpBufFp32, Im2ColTensor_, partNC * partRound);
    
    LocalTensor<T> Im2ColTensor = Im2ColTensor_.template ReinterpretCast<T>();
    
    LocalTensor<T> dTensor;
    if constexpr (sizeof(T) == sizeof(float)) {
        dTensor = dTensor_;
    } else {
        this->CastIndsToHalf(dTensor_, partNC * partRound);
        dTensor = dTensor_.template ReinterpretCast<T>();
        PipeBarrier<PIPE_V>();
    }
    
    LocalTensor<T> transDataTensorIndicesUb = dTensor[this->sizeUb1];
    this->template PrepareInput<true>(dTensor, transDataTensorIndicesUb, partFull);
    
    this->template CopyIn<S>(this->queInVals, this->gradOutputGm[inputOffset], partNC, partFull);
    LocalTensor<S> curInVals = this->queInVals.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2;
    
    if constexpr(sizeof(T) == sizeof(S)) {
        transDataTensorUb2 = curInVals;
    } else {
        transDataTensorUb2 = this->tmpBuf;
        Cast(transDataTensorUb2, curInVals, RoundMode::CAST_NONE, partNC * this->RoundUpBlock(partFull, this->blockLengthS));
        PipeBarrier<PIPE_V>();
    }
    
    this->PrepareInput(transDataTensorUb2, dTensor, partFull);
    this->queInVals.template FreeTensor<S>(curInVals);
    
    if (!this->kernelIsBlock) {
        this->CompareSelect(dTensor, Im2ColTensor, partFull);
    } else {
        this->CompareSelectBlockKernel(dTensor, Im2ColTensor, partFull);
    }
    this->queInInds.template FreeTensor<float>(dTensor_);
    
    auto eventMTE2_V = this->pipe.template AllocEventID<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != sizeof(S)) { // bfloat16 case
        if (this->dAdd) {
            auto event0 = this->pipe.FetchEventID(HardEvent::V_MTE2);
            
            SetFlag<HardEvent::V_MTE2>(event0);
            WaitFlag<HardEvent::V_MTE2>(event0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
            if (!firstIter) {
                DataCopy(this->tmpBuf, this->workspaceGm, overlapPart);
            }
            SetFlag<HardEvent::MTE2_V>(eventMTE2_V);
        }
    }
    
    if constexpr (sizeof(T) == sizeof(S)) { // fp32 | fp16 case 
        Duplicate(this->tmpBuf, static_cast<T>(0.f), this->sizeUb2);
        PipeBarrier<PIPE_V>();
    } else {
        if (!this->dAdd) { 
            Duplicate(this->tmpBuf, static_cast<T>(0.f), this->sizeUb2);
            PipeBarrier<PIPE_V>();
        } else {
            if (firstIter) {
                Duplicate(this->tmpBuf, 0.f, this->sizeUb2);
            } else {
                Duplicate(this->tmpBuf[overlapPart], 0.f, this->sizeUb2 - overlapPart);
            }
            WaitFlag<HardEvent::MTE2_V>(eventMTE2_V);
            PipeBarrier<PIPE_V>();
        }
    }

    Im2ColCall(Im2ColTensor, part, addMulti, addTail, onlyTail, this->addParamsOverUint8);
    this->pipe.template ReleaseEventID<HardEvent::MTE2_V>(eventMTE2_V);
    PipeBarrier<PIPE_V>();
    
    if constexpr(sizeof(T) != sizeof(S)) { // bfloat16 case
        if (this->dAdd) {
            auto event0 = this->pipe.FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(event0);
            WaitFlag<HardEvent::V_MTE3>(event0);

            DataCopy(this->workspaceGm, this->tmpBuf[partOut * this->partHwOut * this->blockLength - overlapPart], overlapPart);
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);

            event0 = this->pipe.FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(event0);
            WaitFlag<HardEvent::MTE3_V>(event0);
        } 
    }

    uint64_t tmpBufOffset = padDL * this->partHwOut * this->blockLength;
    if (padHL || padWL || padHR || padWR) {
        tmpBufOffset = 0;
        bool smallSingleMask = ((this->stripPadRepeatTimes == 0) || (this->stripPadRepeatTimes == 1 && this->stripPadTail == 0)) ? true : false;
        bool doTail = ((this->stripPadRepeatTimes == 1 && this->stripPadTail == 0) || (this->stripPadTail == 0)) ? false : true;

        this->StripPad(padDL, padHL, padWL, partOut - padDL - padDR, this->heightOut, this->widthOut, partOut, this->partOutH, this->partOutW, smallSingleMask, doTail);
        PipeBarrier<PIPE_V>();
    }

    this->template TransposeBack<S>(Im2ColTensor, this->tmpBuf[tmpBufOffset], partNC, realTransAlignRoundPartOutSize);
    PipeBarrier<PIPE_V>();
    
    if (computeTail) {
        this->transDataParams.repeatTimes = this->RoundUpBlock(this->partRoundDhwInp, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    }
    
    if constexpr (sizeof(T) == sizeof(S)) { // fp32/fp16 case 
        if (this->dAdd) {
            SetAtomicAdd<S>();
            CopyOut(this->gradInputGm[outputOffset], partNC, realTransRoundPartOutSize);
            SetAtomicNone();
        } else {
            CopyOut(this->gradInputGm[outputOffset], partNC, realTransRoundPartOutSize);
        }
    } else {
        CopyOut(this->gradInputGm[outputOffset], partNC, realTransRoundPartOutSize);
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::ComputeAll(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD,
                                                                     bool& addMulti, bool& addTail, bool& onlyTail) {
    uint64_t d = 0;
    uint64_t dOut = 0;
    uint64_t padDL = ((int(this->pD - dOut) > 0) && (this->pD != 0)) ? this->pD - dOut : 0;
    uint64_t padDR = (((dOut + this->partOutD) > (this->depthOut + this->pD)) && (this->ceilD != 0)) ? (dOut + this->partOutD) - (this->depthOut + this->pD) : 0;
    uint64_t inputOffset = b * this->inputSize + d * this->hwInputSize;
    uint64_t outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize;
     
    ComputeIteration(d * this->sD - this->pD, batchesPerIter, this->partOutD, this->partD, inputOffset, outputOffset, padDL, padDR, this->pH, this->ceilH, this->pW, this->ceilW, addMulti, addTail, onlyTail, true, false);
    dOut += offD;
    d += this->partD;

    for (; dOut < this->roundDepthOut; dOut += offD, d += this->partD) {
        padDL = ((int(this->pD - dOut) > 0) && (this->pD != 0)) ? this->pD - dOut : 0;
        padDR = (((dOut + this->partOutD) > (this->depthOut + this->pD)) && (this->ceilD != 0)) ? (dOut + this->partOutD) - (this->depthOut + this->pD) : 0;
        inputOffset = b * this->inputSize + d * this->hwInputSize;
        outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize;

        ComputeIteration(d * this->sD - this->pD, batchesPerIter, this->partOutD, this->partD, inputOffset, outputOffset, padDL, padDR, this->pH, this->ceilH, this->pW, this->ceilW, addMulti, addTail, onlyTail, false, false);
    } 
    if (tailD >= 1) {
        padDL = ((int(this->pD - dOut) > 0) && (this->pD != 0)) ? this->pD - dOut : 0; 
        padDR = (((dOut + this->tailOutD) > (this->depthOut + this->pD)) && (this->ceilD != 0)) ? (dOut + this->tailOutD) - (this->depthOut + this->pD) : 0;
        inputOffset = b * this->inputSize + d * this->hwInputSize;
        outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize;
        
        ComputeIteration(d * this->sD - this->pD, batchesPerIter, tailOutD, tailD, inputOffset, outputOffset, padDL, padDR, this->pH, this->ceilH, this->pW, this->ceilW, addMulti, addTail, onlyTail, false, true);
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::ProcessBatches(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD) { 
    bool singleAddOrCopy = this->im2colAddRepeatTimes == 1;
    bool multipleAddOrCopy = this->im2colAddRepeatTimes > 1;
    bool doTail = this->im2colAddTail != 0;
    bool onlyTail = this->im2colAddRepeatTimes == 0 && doTail;

    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd) { 
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
        }
    }

    if (this->im2colSingleWOut) {
        singleAddOrCopy = false;
        multipleAddOrCopy = this->im2colCopyFullIters > 0;
        doTail = this->im2colCopyRepeatTimesTail > 0;
        onlyTail = !multipleAddOrCopy && doTail;
    } 

    ComputeAll(b, batchesPerIter, offD, multipleAddOrCopy, doTail, onlyTail);
    
    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
        }
    }
}

template <typename T, typename S> 
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitD<T,S>::Compute() {
    this->CreateKernelIndexes();
    CreateGeneralOffsets();
    
    uint64_t offD = this->partOutD - (this->kD - 1) * this->dD + this->sD - 1;
    
    for (int64_t b = 0; b < this->roundBatches; b += this->blockLength) {
        ProcessBatches(b, this->blockLength, offD);
    }
    
    if (this->partNC != 0) {
        ProcessBatches(this->roundBatches, this->partNC, offD);
    }
}
 
#endif
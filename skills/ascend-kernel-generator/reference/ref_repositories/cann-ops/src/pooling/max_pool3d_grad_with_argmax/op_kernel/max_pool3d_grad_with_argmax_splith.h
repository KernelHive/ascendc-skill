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
 * \file max_pool3d_grad_with_argmax_splith.h
 * \brief
 */
 
#ifndef OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_SPLITH_H
#define OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_SPLITH_H
 
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_grad_with_argmax_base.h"
 
using namespace AscendC;
 
template <typename T, typename S>
class KernelMaxPool3DGradWithArgmaxSplitH : public KernelMaxPool3DGradWithArgmaxBase<T,S> {
    private:
        uint64_t curWorkspaceOffset = 0;
        uint64_t tailH;
        uint64_t tailOutH;
        const uint64_t workspaceSizePerCore = (this->depthOut + this->pD + this->ceilD) * (this->heightOut + this->pH + this->ceilH) * this->partOutW * this->blockLength; 
        
        TEventID eventMTE3_MTE2_workspace = this->pipe.template AllocEventID<HardEvent::MTE3_MTE2>();

    public:
        __aicore__ KernelMaxPool3DGradWithArgmaxSplitH(const MaxPool3DGradWithArgmaxSplitHTilingData *__restrict tilingData_) :
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
                if (this->dAdd || this->hAdd) {
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
            this->roundDepthOut = this->sD * (this->depthInput - 1) + 1;
            
            auto itH = this->RoundDownBlock(this->heightInput, this->partH);
            this->roundHeightOut = this->sH * (itH - 1) + 1;
            tailH = this->heightInput - itH;
            tailOutH = this->dH * (this->kH - 1) + this->sH * (tailH - 1) + 1;
            
            this->PrepareBaseScalars();
        } 
        
        __aicore__ inline void CreateGeneralOffsets();
        template <typename V>
        __aicore__ inline void TransposeBack(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint64_t& partNC, 
                                             const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& dataLen);
        template <typename V>
        __aicore__ inline void TransposeBack(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor, const uint64_t& partNC, 
                                             const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& dataLen);
        __aicore__ void CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC, const uint64_t& realPartOutD, const uint64_t& realPartOutH);
        __aicore__ void ProcessBatches(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD, const uint64_t& offH);
        __aicore__ void Im2ColCall(const LocalTensor<T>& Im2ColTensor, const uint64_t& part, const uint64_t& curDepthOff_,
                                   bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8);
        __aicore__ void Compute();
        __aicore__ void ComputeIteration(const int64_t& dOff, const int64_t& hOff,
                                         const uint64_t& partNC, const uint64_t& partOut, const uint64_t& part,
                                         const uint64_t& inputOffset, const uint64_t& outputOffset,
                                         const uint64_t& padDL, const uint64_t& padDR,
                                         const uint64_t& padHL, const uint64_t& padHR,
                                         const uint64_t& padWL, const uint64_t& padWR,
                                         bool& addMulti, bool& addTail, bool& onlyTail, bool computeTail);
                                         
        __aicore__ void ComputeAll(const uint64_t& b, const uint64_t& batchesPerIter,
                                   const uint64_t& offD, const uint64_t& offH,
                                   bool& addMulti, bool& addTail, bool& onlyTail);
};

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::CreateGeneralOffsets() {
    uint64_t i = 0;
    int64_t offD = -this->pD;
    int64_t offH = -this->pH;
    int64_t offW = -this->pW;
    LocalTensor<float> dTmp = this->generalOffsets;
    LocalTensor<float> hTmp = dTmp[this->partAlignRoundDhwInp];
    LocalTensor<float> wTmp = hTmp[this->partAlignRoundDhwInp];
    
    Duplicate<float>(dTmp, 1.f * offD, this->partAlignRoundDhwInp);
    if ((this->partW % this->blockLength == 0) && (sizeof(T) == BLOCK_SIZE / this->blockLength)) {
        ArithProgression<float>(wTmp, 1.f * offW, 1.f * this->sW, this->partW);
        Duplicate<float>(hTmp, 1.f * offH, this->partW);
        PipeBarrier<PIPE_V>();
        i += this->partW;
        for (int64_t curH = offH + (int)this->sH; curH < offH + (int)this->partH * (int)this->sH; curH += (int)this->sH) {
            Duplicate<float>(hTmp[i], 1.f * curH, this->partW);
            DataCopy(wTmp[i], wTmp, this->partW);
            i += this->partW;
        }
    } else {
        for (int64_t curH = offH; curH < offH + (int)this->partH * (int)this->sH; curH += (int)this->sH) {
            for (int64_t curW = offW; curW < offW + (int)this->partW * (int)this->sW; curW += (int)this->sW) {
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
template <typename V>
__aicore__ inline void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::TransposeBack(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint64_t& partNC, 
                                                                               const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& dataLen) {
    auto curPartHwOut = realPartOutH * this->widthOut;
    auto alPartHwOut = this->RoundUpBlock(curPartHwOut, this->blockLengthS);

    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    LocalTensor<float> tmp = srcTensor[this->blockLength];
    LocalTensor<float> tmp2 = dstTensor[this->blockLength];
    for (uint64_t j = 0; j < realPartOutD; ++j) {
        for (uint64_t i = 0; i < this->blockLength; ++i) {
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i] = (uint64_t)srcTensor[NCHW_CONV_ADDR_LIST_SIZE * i + j * curPartHwOut * this->blockLength].GetPhyAddr();
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i + 1] = (uint64_t)tmp[NCHW_CONV_ADDR_LIST_SIZE * i + j * curPartHwOut * this->blockLength].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i] = (uint64_t)dstTensor[i * realPartOutD * this->RoundUpBlock(alPartHwOut, MIN_TRANSPOSE_ROWS) + j * alPartHwOut].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i + 1] = (uint64_t)tmp2[i * realPartOutD * this->RoundUpBlock(alPartHwOut, MIN_TRANSPOSE_ROWS) + j * alPartHwOut].GetPhyAddr();
        }
        TransDataTo5HD<float>(dstLocalList, srcLocalList, this->transDataParamsReverse);
    }
    PipeBarrier<PIPE_V>();
    
    this->CastBF16Back(dstTensor, partNC, realPartOutD * dataLen, sizeof(V) == sizeof(bfloat16_t));
}

template <typename T, typename S>
template <typename V>
__aicore__ inline void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::TransposeBack(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor, const uint64_t& partNC, 
                                                                               const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& dataLen) {
    auto curPartHwOut = realPartOutH * this->widthOut;
    auto alPartHwOut = this->RoundUpBlock(curPartHwOut, this->blockLengthS);

    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    for (uint64_t j = 0; j < realPartOutD; ++j) {
        for (uint64_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; ++i) {
            srcLocalList[i] = (uint64_t)srcTensor[NCHW_CONV_ADDR_LIST_SIZE * i + j * curPartHwOut * this->blockLength].GetPhyAddr();
            dstLocalList[i] = (uint64_t)dstTensor[i * realPartOutD * this->RoundUpBlock(alPartHwOut, MIN_TRANSPOSE_ROWS) + j * alPartHwOut].GetPhyAddr();
        }
        TransDataTo5HD<half>(dstLocalList, srcLocalList, this->transDataParamsReverse);
    }
    PipeBarrier<PIPE_V>();
    
    this->CastBF16Back(dstTensor, partNC);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC, const uint64_t& realPartOutD, const uint64_t& realPartOutH) {
    uint64_t hwOut = realPartOutH * this->widthOut;
    uint64_t downSize = this->RoundDownBlock(hwOut, this->blockLengthS);
    uint64_t roundSize = this->RoundUpBlock(hwOut, this->blockLengthS);
    LocalTensor<S> Im2ColTensor = this->queOutVals.template DeQue<S>();
    
    auto transAlPartDhwOut = realPartOutD * this->RoundUpBlock(roundSize, MIN_TRANSPOSE_ROWS);

    if (this->hwOutputSize % this->blockLengthS == 0 && hwOut == downSize) {
        if ((this->hwOutputSize - hwOut) / this->blockLengthS < MAX_UINT16) {
            DataCopyParams copyParams{static_cast<uint16_t>(realPartOutD), static_cast<uint16_t>(hwOut / this->blockLengthS),
                                      static_cast<uint16_t>((roundSize - hwOut) / this->blockLengthS), static_cast<uint16_t>((this->hwOutputSize - hwOut) / this->blockLengthS)};
            for (uint64_t nc = 0; nc < partNC; ++nc) {
                DataCopy(dstGm[nc * this->outSize], Im2ColTensor[nc * transAlPartDhwOut], copyParams);
            }
        } else {
            DataCopyParams copyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(hwOut / this->blockLengthS), 0, 0};
            for (uint64_t nc = 0; nc < partNC; ++nc) {
                auto tmpSrc = Im2ColTensor[nc * transAlPartDhwOut];
                auto tmpDst = dstGm[nc * this->outSize];
                for (uint64_t dOut = 0; dOut < realPartOutD; ++dOut) {
                    DataCopy(tmpDst[dOut * this->hwOutputSize], tmpSrc[dOut * hwOut], copyParams);
                }
            }
        }
    } else {
        DataCopyExtParams copyPadParams {static_cast<uint16_t>(realPartOutD), static_cast<uint32_t>(hwOut * sizeof(S)),
                                         static_cast<uint32_t>((roundSize - hwOut) / this->blockLengthS),
                                         static_cast<uint32_t>((this->hwOutputSize - hwOut) * sizeof(S)), 0}; 
        for (uint64_t nc = 0; nc < partNC; ++nc) {
            DataCopyPad(dstGm[nc * this->outSize], Im2ColTensor[nc * transAlPartDhwOut], copyPadParams);
        }
    }
    
    this->queOutVals.template FreeTensor<S>(Im2ColTensor);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::Im2ColCall(const LocalTensor<T>& Im2ColTensor, const uint64_t& part, const uint64_t& curDepthOff_,
                                                                     bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8) {
    uint32_t sliceByWidthOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t curHeightOut = 0;
    
    if constexpr (sizeof(T) == sizeof(S)) { // fp32 | fp16 case
        if (!this->hAdd && !this->wAdd) {
            for (; curHeightOut < part; dstOffset += this->gmHOff, ++curHeightOut) {
                this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            }
        } else {
            if (this->hAdd && !this->wAdd) {
                this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                PipeBarrier<PIPE_V>();
                
                ++curHeightOut;
                dstOffset += this->gmHOff;
                
                for (; curHeightOut < part; dstOffset += this->gmHOff, ++curHeightOut) {
                    this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                }
            } else {
                for (; curHeightOut < part; dstOffset += this->gmHOff, ++curHeightOut) {
                    this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
                }
            }
        }
    } else {  // bfloat16 case
        if (!this->dAdd && !this->hAdd && !this->wAdd) {
            for (; curHeightOut < part; dstOffset += this->gmHOff, ++curHeightOut) {
                this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            }
        } else {
            for (; curHeightOut < part; dstOffset += this->gmHOff, ++curHeightOut) {
                this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, this->partOutW, this->partW, addMulti, addTail, onlyTail, overUint8);
            }
        }
    }
}

template <typename T, typename S> 
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::ComputeIteration(const int64_t& dOff, const int64_t& hOff, const uint64_t& partNC,
                                                                           const uint64_t& partOut, const uint64_t& part,
                                                                           const uint64_t& inputOffset, const uint64_t& outputOffset,
                                                                           const uint64_t& padDL, const uint64_t& padDR,
                                                                           const uint64_t& padHL, const uint64_t& padHR,
                                                                           const uint64_t& padWL, const uint64_t& padWR,
                                                                           bool& addMulti, bool& addTail, bool& onlyTail, bool computeTail) {
    const uint64_t realTransRoundPartOutSize = (partOut - padHL - padHR) * this->widthOut;
    const uint64_t realTransAlignRoundPartOutSize = this->RoundUpBlock(realTransRoundPartOutSize, MIN_TRANSPOSE_ROWS);
    const uint64_t partFull = this->partD * part * this->partW;
    const uint64_t partRound = this->RoundUpBlock(partFull);
    uint64_t curDepthOff_ = this->curDepthOff;
    
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
    float generalOffsetH = -1.f * (hOff + this->pH);
    Adds(dTensor_, dTensor_, generalOffsetD, partNC * partRound);
    Adds(tmpBufFp32, tmpBufFp32, generalOffsetH, partNC * partRound);
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
        this->gmDOff = (this->sD * partOut * this->partOutW - part * this->sH * this->partOutW) * this->blockLength;
        this->transDataParams.repeatTimes = this->RoundUpBlock(partFull, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;

        curDepthOff_ = this->dD * partOut * this->partOutW * this->blockLength;
    }
    
    this->IndexRecalcSecond(dTensor_, tmpBufFp32, Im2ColTensor_, partNC * partRound);

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
    
    LocalTensor<T> Im2ColTensor = Im2ColTensor_.template ReinterpretCast<T>();
    
    if (!this->kernelIsBlock) {
        this->CompareSelect(dTensor, Im2ColTensor, partFull);
    } else {
        this->CompareSelectBlockKernel(dTensor, Im2ColTensor, partFull);
    }
    
    this->queInInds.template FreeTensor<float>(dTensor_);
    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd) {
            auto event0 = this->pipe.FetchEventID(HardEvent::V_MTE2);
            
            SetFlag<HardEvent::V_MTE2>(event0);
            WaitFlag<HardEvent::V_MTE2>(event0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
            for (uint64_t curD = 0; curD < this->partOutD; ++curD) {
                DataCopy(this->tmpBuf[curD * partOut * this->partOutW * this->blockLength], this->workspaceGm[curWorkspaceOffset + curD * (this->heightOut + this->pH + this->ceilH) * this->partOutW * this->blockLength], partOut * this->partOutW * this->blockLength); 
            }  

            event0 = this->pipe.FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(event0);
            WaitFlag<HardEvent::MTE2_V>(event0);
        } else {
            Duplicate(this->tmpBuf, static_cast<T>(0.f), this->sizeUb2);
        }
    } else { 
        Duplicate(this->tmpBuf, static_cast<T>(0.f), this->sizeUb2);
    }
    
    Im2ColCall(Im2ColTensor, part, curDepthOff_, addMulti, addTail, onlyTail, this->addParamsOverUint8);
    PipeBarrier<PIPE_V>();
    
    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd) {
            auto event0 = this->pipe.FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(event0);
            WaitFlag<HardEvent::V_MTE3>(event0);
            for (uint64_t curD = 0; curD < this->partOutD; ++curD) {
                DataCopy(this->workspaceGm[curWorkspaceOffset + curD * (this->heightOut + this->pH + this->ceilH) * this->partOutW * this->blockLength], this->tmpBuf[curD * partOut * this->partOutW * this->blockLength],  partOut * this->partOutW * this->blockLength); 
            }  
            
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);

            event0 = this->pipe.FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(event0);
            WaitFlag<HardEvent::MTE3_V>(event0);
        } 
    }

    uint64_t tmpBufOffset = padDL * partOut * this->partOutW * this->blockLength;
    if (padHL || padWL || padHR || padWR) {
        tmpBufOffset = 0;
        bool smallSingleMask = ((this->stripPadRepeatTimes == 0) || (this->stripPadRepeatTimes == 1 && this->stripPadTail == 0)) ? true : false;
        bool doTail = ((this->stripPadRepeatTimes == 1 && this->stripPadTail == 0) || (this->stripPadTail == 0)) ? false : true;

        this->StripPad(padDL, padHL, padWL, this->partOutD - padDL - padDR, partOut - padHL - padHR, this->widthOut, this->partOutD, partOut, this->partOutW, smallSingleMask, doTail);
        PipeBarrier<PIPE_V>();
    }

    TransposeBack<S>(Im2ColTensor, this->tmpBuf[tmpBufOffset], partNC, this->partOutD - padDL - padDR, partOut - padHL - padHR, realTransAlignRoundPartOutSize);
    PipeBarrier<PIPE_V>();
    
    if (computeTail) {
        this->gmDOff = (this->sD * this->partOutH * this->partOutW - this->partH * this->sH * this->partOutW) * this->blockLength;
        
        this->transDataParams.repeatTimes = this->RoundUpBlock(this->partRoundDhwInp, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    }
   
    if constexpr (sizeof(T) == sizeof(S)) { // fp32/fp16 case 
        if (this->dAdd || this->hAdd) {
            SetAtomicAdd<S>();
            CopyOut(this->gradInputGm[outputOffset], partNC, this->partOutD - padDL - padDR, partOut - padHL - padHR);
            SetAtomicNone();
        } else {
            CopyOut(this->gradInputGm[outputOffset], partNC, this->partOutD - padDL - padDR, partOut - padHL - padHR);
        }
    } else {
        CopyOut(this->gradInputGm[outputOffset], partNC, this->partOutD - padDL - padDR, partOut - padHL - padHR);
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::ComputeAll(const uint64_t& b, const uint64_t& batchesPerIter,
                                                                     const uint64_t& offD, const uint64_t& offH,
                                                                     bool& addMulti, bool& addTail, bool& onlyTail) {
    uint64_t d = 0, h = 0;
    uint64_t dOut = 0, hOut = 0;
    for (dOut = 0; dOut < this->roundDepthOut; dOut += offD, d += this->partD) {
        uint64_t padDL = ((int(this->pD - dOut) > 0) && (this->pD != 0)) ? this->pD - dOut : 0;
        uint64_t padDR = (((dOut + this->partOutD) > (this->depthOut + this->pD)) && (this->ceilD != 0)) ? (dOut + this->partOutD) - (this->depthOut + this->pD) : 0;
        uint64_t prevWorkspaceOffset = curWorkspaceOffset;
        h = 0;
        for (hOut = 0; hOut < this->roundHeightOut; hOut += offH, h += this->partH) {
            uint64_t padHL = ((int(this->pH - hOut) > 0) && (this->pH != 0)) ? this->pH - hOut : 0;
            uint64_t padHR = (((hOut + this->partOutH) > (this->heightOut + this->pH)) && (this->ceilH != 0)) ? (hOut + this->partOutH) - (this->heightOut + this->pH) : 0;
            uint64_t inputOffset = b * this->inputSize + d * this->hwInputSize + h * this->widthInput;
            uint64_t outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize + (hOut >= this->pH ? hOut - this->pH : 0) * this->widthOut;
            
            ComputeIteration(d * this->sD - this->pD, h * this->sH - this->pH, batchesPerIter, this->partOutH, this->partH, inputOffset, outputOffset, padDL, padDR, padHL, padHR, this->pW, this->ceilW, addMulti, addTail, onlyTail, false);
            curWorkspaceOffset += this->sH * this->partH * this->partOutW * this->blockLength;
        }
        if (tailH >= 1) {
            uint64_t padHL = ((int(this->pH - hOut) > 0) && (this->pH != 0)) ? this->pH - hOut : 0;
            uint64_t padHR = (((hOut + this->tailOutH) > (this->heightOut + this->pH)) && (this->ceilH != 0)) ? (hOut + this->tailOutH) - (this->heightOut + this->pH) : 0;
            uint64_t inputOffset = b * this->inputSize + d * this->hwInputSize + h * this->widthInput;
            uint64_t outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize + (hOut >= this->pH ? hOut - this->pH : 0) * this->widthOut;
            
            ComputeIteration(d * this->sD - this->pD, h * this->sH - this->pH, batchesPerIter, tailOutH, tailH, inputOffset, outputOffset, padDL, padDR, padHL, padHR, this->pW, this->ceilW, addMulti, addTail, onlyTail, true);
        }
        curWorkspaceOffset = prevWorkspaceOffset + this->sD * (this->heightOut + this->pH + this->ceilH) * this->partOutW * this->blockLength;
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::ProcessBatches(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD, const uint64_t& offH) {
    bool singleAddOrCopy = this->im2colAddRepeatTimes == 1;
    bool multipleAddOrCopy = this->im2colAddRepeatTimes > 1;
    bool doTail = this->im2colAddTail != 0;
    bool onlyTail = this->im2colAddRepeatTimes == 0 && doTail;

    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd) { 
            InitGlobalMemory(this->workspaceGm, workspaceSizePerCore, 0.f);
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
        }
    }
    
    if (this->im2colSingleWOut) {
        singleAddOrCopy = false;
        multipleAddOrCopy = this->im2colCopyFullIters > 0;
        doTail = this->im2colCopyRepeatTimesTail > 0;
        onlyTail = !multipleAddOrCopy && doTail;
    }

    ComputeAll(b, batchesPerIter, offD, offH, multipleAddOrCopy, doTail, onlyTail);
    curWorkspaceOffset = 0;
    
    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd) { 
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
        }
    }
}

template <typename T, typename S> 
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitH<T,S>::Compute() {
    this->CreateKernelIndexes();
    CreateGeneralOffsets();
    
    uint64_t offD = this->partOutD - (this->kD - 1) * this->dD + this->sD - 1;
    uint64_t offH = this->partOutH - (this->kH - 1) * this->dH + this->sH - 1;
    
    for (int64_t b = 0; b < this->roundBatches; b += this->blockLength) {
        ProcessBatches(b, this->blockLength, offD, offH);
    }
    
    if (this->partNC != 0) {
        ProcessBatches(this->roundBatches, this->partNC, offD, offH);
    }
}

#endif
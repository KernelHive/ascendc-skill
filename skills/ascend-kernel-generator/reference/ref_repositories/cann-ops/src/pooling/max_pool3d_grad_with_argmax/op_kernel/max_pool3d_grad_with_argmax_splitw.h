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
 * \file max_pool3d_grad_with_argmax_splitw.h
 * \brief
 */
 
#ifndef OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_SPLITW_H
#define OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_SPLITW_H
 
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_grad_with_argmax_base.h"
 
using namespace AscendC;
 
template<typename T, typename S>
class KernelMaxPool3DGradWithArgmaxSplitW : public KernelMaxPool3DGradWithArgmaxBase<T,S> {
    private:
        uint64_t curWorkspaceOffset = 0;
        uint64_t tailW;
        uint64_t tailOutW;
        const uint64_t workspaceSizePerCore = (this->depthOut + this->pD + this->ceilD) * (this->heightOut + this->pH + this->ceilH) * (this->widthOut + this->pW + this->ceilW) * this->blockLength; 
        
        TEventID eventMTE3_MTE2_workspace = this->pipe.template AllocEventID<HardEvent::MTE3_MTE2>();

    public:
        __aicore__ KernelMaxPool3DGradWithArgmaxSplitW(const MaxPool3DGradWithArgmaxSplitWTilingData *__restrict tilingData_) :
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
                if (this->dAdd || this->hAdd || this->wAdd) {
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
            this->roundHeightOut = this->sH * (this->heightInput - 1) + 1;
            auto itW = this->RoundDownBlock(this->widthInput, this->partW);
            this->roundWidthOut = this->sW * (itW - 1) + 1;
            
            tailW = this->widthInput - itW;
            tailOutW = this->dW * (this->kW - 1) + this->sW * (tailW - 1) + 1;
            
            this->PrepareBaseScalars();
        } 
        
        __aicore__ inline void CreateGeneralOffsets();
        template <typename V>
        __aicore__ inline void TransposeBack(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint64_t& partNC,
                                             const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& realPartOutW, const uint64_t& dataLen);
        template <typename V>
        __aicore__ inline void TransposeBack(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor, const uint64_t& partNC,
                                             const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& realPartOutW, const uint64_t& dataLen);
        __aicore__ void CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC, const uint64_t& realPartOutD,
                                const uint64_t& realPartOutH, const uint64_t& realPartOutW);
        __aicore__ void ProcessBatches(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD, const uint64_t& offH, const uint64_t& offW);
        __aicore__ void Compute();
        __aicore__ void ComputeIteration(const int64_t& dOff, const int64_t& hOff, const int64_t& wOff,
                                         const uint64_t& partNC, const uint64_t& partOut, const uint64_t& part,
                                         const uint64_t& inputOffset, const uint64_t& outputOffset,
                                         const uint64_t& padDL, const uint64_t& padDR,
                                         const uint64_t& padHL, const uint64_t& padHR,
                                         const uint64_t& padWL, const uint64_t& padWR,
                                         bool& addMulti, bool& addTail, bool& onlyTail, bool computeTail);
                                        
        __aicore__ void ComputeAll(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD,
                                   const uint64_t& offH, const uint64_t& offW,
                                   bool& addMulti, bool& addTail, bool& onlyTail);
};

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::CreateGeneralOffsets() {
    LocalTensor<float> wTmp = this->generalOffsets[(TMP_TENSORS_NUM - 1) * this->partAlignRoundDhwInp];
    ArithProgression(wTmp, 0.f, 1.f * this->sW, this->RoundUpBlock(this->partW));
    auto event0 = this->pipe.FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
}

template <typename T, typename S>
template <typename V>
__aicore__ inline void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::TransposeBack(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint64_t& partNC,
                                                                               const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& realPartOutW, const uint64_t& dataLen) {
    auto hwPartOut = realPartOutW * realPartOutH;
    auto alWOut = this->RoundUpBlock(realPartOutW, this->blockLengthS);

    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    LocalTensor<float> tmp = srcTensor[this->blockLength];
    LocalTensor<float> tmp2 = dstTensor[this->blockLength];
    
    for (uint64_t k = 0; k < realPartOutD; ++k) {
        for (uint64_t j = 0; j < realPartOutH; ++j) {
            for (uint64_t i = 0; i < this->blockLength; ++i) {
                srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i] = (uint64_t)srcTensor[NCHW_CONV_ADDR_LIST_SIZE * i + j * realPartOutW * this->blockLength + k * hwPartOut * this->blockLength].GetPhyAddr();
                srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i + 1] = (uint64_t)tmp[NCHW_CONV_ADDR_LIST_SIZE * i + j * realPartOutW * this->blockLength + k * hwPartOut * this->blockLength].GetPhyAddr();
                dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i] = (uint64_t)dstTensor[i * realPartOutD * realPartOutH * this->RoundUpBlock(alWOut, MIN_TRANSPOSE_ROWS) + j * alWOut + k * alWOut * realPartOutH].GetPhyAddr();
                dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / this->blockLength * i + 1] = (uint64_t)tmp2[i * realPartOutD *realPartOutH * this->RoundUpBlock(alWOut, MIN_TRANSPOSE_ROWS) + j * alWOut + k * alWOut * realPartOutH].GetPhyAddr();
            }
            TransDataTo5HD<float>(dstLocalList, srcLocalList, this->transDataParamsReverse);
        }
    }
    PipeBarrier<PIPE_V>();
    
    this->CastBF16Back(dstTensor, partNC, realPartOutD * realPartOutH * dataLen, sizeof(V) == sizeof(bfloat16_t));
}


template <typename T, typename S>
template <typename V>
__aicore__ inline void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::TransposeBack(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor, const uint64_t& partNC,
                                                                               const uint64_t& realPartOutD, const uint64_t& realPartOutH, const uint64_t& realPartOutW, const uint64_t& dataLen) {
    auto hwPartOut = realPartOutW * realPartOutH;
    auto alWOut = this->RoundUpBlock(realPartOutW, this->blockLengthS);

    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

    for (uint64_t k = 0; k < realPartOutD; ++k) {
        for (uint64_t j = 0; j < realPartOutH; ++j) {
            for (uint64_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; ++i) {
                srcLocalList[i] = (uint64_t)srcTensor[NCHW_CONV_ADDR_LIST_SIZE * i + j * realPartOutW * this->blockLength + k * hwPartOut * this->blockLength].GetPhyAddr();
                dstLocalList[i] = (uint64_t)dstTensor[i * realPartOutD * realPartOutH * this->RoundUpBlock(alWOut, MIN_TRANSPOSE_ROWS) + j * alWOut + k * alWOut * realPartOutH].GetPhyAddr();
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, this->transDataParamsReverse);
        }
    }
    PipeBarrier<PIPE_V>();
    
    this->CastBF16Back(dstTensor, partNC);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::CopyOut(GlobalTensor<S> dstGm, const uint64_t& partNC, const uint64_t& realPartOutD,
                                                                  const uint64_t& realPartOutH, const uint64_t& realPartOutW) {
    uint64_t downPartOutW = this->RoundDownBlock(realPartOutW, this->blockLengthS);
    uint64_t alWOut = this->RoundUpBlock(realPartOutW, this->blockLengthS);
    uint64_t transAlPartDhwOut = realPartOutD * realPartOutH * this->RoundUpBlock(alWOut, MIN_TRANSPOSE_ROWS);
    
    LocalTensor<S> Im2ColTensor = this->queOutVals.template DeQue<S>();
    
    if (this->widthOut % this->blockLengthS == 0 && downPartOutW == realPartOutW) {
        if ((this->widthOut - realPartOutW) / this->blockLengthS < MAX_UINT16) {
            DataCopyParams copyParams {static_cast<uint16_t>(realPartOutH), static_cast<uint16_t>(realPartOutW / this->blockLengthS),
                                       static_cast<uint16_t>((alWOut - realPartOutW) / this->blockLengthS), 
                                       static_cast<uint16_t>((this->widthOut - realPartOutW) / this->blockLengthS)};
            for (uint64_t nc = 0; nc < partNC; ++nc) {
                auto tmpSrc = Im2ColTensor[nc * transAlPartDhwOut];
                auto tmpDst = dstGm[nc * this->outSize];
                for (uint64_t dOut = 0; dOut < realPartOutD; ++dOut) {
                    DataCopy(tmpDst[dOut * this->hwOutputSize], tmpSrc[dOut * realPartOutH * realPartOutW], copyParams);
                }
            }
        } else {
            DataCopyParams copyParams {static_cast<uint16_t>(1), static_cast<uint16_t>(realPartOutW / this->blockLengthS), 0, 0};
            for (uint64_t nc = 0; nc < partNC; ++nc) {
                auto tmpSrc = Im2ColTensor[nc * transAlPartDhwOut];
                auto tmpDst = dstGm[nc * this->outSize];
                for (uint64_t dOut = 0; dOut < realPartOutD; ++dOut) {
                    auto srcUbTensor = tmpSrc[dOut * realPartOutH * realPartOutW];
                    auto dstGmTensor = tmpDst[dOut * this->hwOutputSize];
                    for (uint64_t hOut = 0; hOut < realPartOutH; ++hOut) {
                        DataCopy(dstGmTensor[hOut * this->widthOut], srcUbTensor[hOut * realPartOutW], copyParams);
                    }
                }
            }
        }
    } else {
        DataCopyExtParams copyPadParams {static_cast<uint16_t>(realPartOutH), static_cast<uint32_t>(realPartOutW * sizeof(S)),
                                         static_cast<uint32_t>((alWOut - realPartOutW) / this->blockLengthS), 
                                         static_cast<uint32_t>((this->widthOut - realPartOutW) * sizeof(S)), 0};
        for (uint64_t nc = 0; nc < partNC; nc++) {
            auto tmpSrc = Im2ColTensor[nc * transAlPartDhwOut];
            auto tmpDst = dstGm[nc * this->outSize];
            for (uint64_t dOut = 0; dOut < realPartOutD; ++dOut) {
                DataCopyPad(tmpDst[dOut * this->hwOutputSize], tmpSrc[dOut * realPartOutH * alWOut], copyPadParams);
            }
        }
    }
    
    this->queOutVals.template FreeTensor<S>(Im2ColTensor);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::ComputeIteration(const int64_t& dOff, const int64_t& hOff, const int64_t& wOff,
                                                                           const uint64_t& partNC, const uint64_t& partOut, const uint64_t& part,
                                                                           const uint64_t& inputOffset, const uint64_t& outputOffset,
                                                                           const uint64_t& padDL, const uint64_t& padDR,
                                                                           const uint64_t& padHL, const uint64_t& padHR,
                                                                           const uint64_t& padWL, const uint64_t& padWR,
                                                                           bool& addMulti, bool& addTail, bool& onlyTail, bool computeTail) {                                                                   
    const uint64_t realTransAlignRoundPartOutSize = this->RoundUpBlock(partOut - padWL - padWR, MIN_TRANSPOSE_ROWS);    
    const uint64_t partRound = this->RoundUpBlock(part);
    uint64_t curDepthOff_ = this->curDepthOff;

    this->template CopyIn<int32_t, true>(this->queInInds, this->indicesGm[inputOffset], partNC, part);
    LocalTensor<float> dTensor_ = this->queInInds.template DeQue<float>();
    this->CastInds(dTensor_, partNC * partRound);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> Im2ColTensor_ = this->queOutVals.template AllocTensor<float>();
    LocalTensor<float> tmpBufFp32 = this->tmpBuf.template ReinterpretCast<float>();
    LocalTensor<float> wTmp = this->generalOffsets[(TMP_TENSORS_NUM - 1) * this->partAlignRoundDhwInp];
    
    this->template IndexRecalcFirst<true>(dTensor_, tmpBufFp32, Im2ColTensor_, partNC * partRound, dOff, hOff);
    PipeBarrier<PIPE_V>();
    
    float generalOffsetW = -1.f * wOff;
    Adds(Im2ColTensor_, Im2ColTensor_, generalOffsetW, partNC * partRound);
    PipeBarrier<PIPE_V>();
    
    for (uint64_t i = 0; i < partNC * partRound; i += partRound) {
        Sub(Im2ColTensor_[i], Im2ColTensor_[i], wTmp, partRound);
    }
    PipeBarrier<PIPE_V>();
    
    this->transDataParamsReverse.repeatTimes = realTransAlignRoundPartOutSize / NCHW_CONV_ADDR_LIST_SIZE;
    this->transDataParamsReverse.dstRepStride = (this->transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    this->transDataParamsReverse.srcRepStride = (this->transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;

    if (computeTail) {
        this->gmHOff = this->sH * partOut * this->blockLength;
        this->gmDOff = (this->sD * this->partOutH - this->partH * this->sH) * partOut * this->blockLength;
        this->addParamsOverUint8 = (this->dH * partOut) > UINT8_MAX;

        if (!this->im2colSingleWOut) {
            this->Im2colNewParamsTail.dstRepeatSize  = static_cast<uint16_t>(this->dH * partOut);
            this->im2ColAddParamsTail.dstRepStride = static_cast<uint8_t>(this->dH * partOut);
            this->im2ColAddParamsTail.src0RepStride = static_cast<uint8_t>(this->dH * partOut);
            
            if (this->im2colAddRepeatTimes == 1) {
                this->Im2colNewParams.dstRepeatSize = this->dH * partOut;
                
                if (!this->addParamsOverUint8) { 
                    this->im2ColAddParams.dstRepStride = this->dH * partOut;
                    this->im2ColAddParams.src0RepStride = this->dH * partOut;
                    this->im2ColAddParams.src1RepStride = this->kW;
                }
            }
        } else {
            this->im2colCopyFullIters = part / UINT8_MAX;
            this->processedIm2colCopyPartW = UINT8_MAX * this->im2colCopyFullIters;
            this->im2colCopyRepeatTimesTail = part - this->processedIm2colCopyPartW;
        }
    
        this->transDataParams.repeatTimes = this->RoundUpBlock(part, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;

        curDepthOff_ = this->dD * this->partOutH * partOut * this->blockLength;
    }
    
    this->IndexRecalcSecond(dTensor_, tmpBufFp32, Im2ColTensor_, partNC * partRound);

    LocalTensor<T> dTensor;
    if constexpr (sizeof(float) == sizeof(T)) {
        dTensor = dTensor_;
    } else {
        this->CastIndsToHalf(dTensor_, partNC * partRound);
        dTensor = dTensor_.template ReinterpretCast<T>();
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<T> transDataTensorIndicesUb = dTensor[this->sizeUb1];
    this->template PrepareInput<true>(dTensor, transDataTensorIndicesUb, part);
    
    LocalTensor<T> Im2ColTensor = Im2ColTensor_.template ReinterpretCast<T>();
    
    this->template CopyIn<S>(this->queInVals, this->gradOutputGm[inputOffset], partNC, part); 
    LocalTensor<S> curInVals = this->queInVals.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2;
    
    if constexpr(sizeof(T) == sizeof(S)) {
        transDataTensorUb2 = curInVals;
    } else {
        transDataTensorUb2 = this->tmpBuf;
        Cast(transDataTensorUb2, curInVals, RoundMode::CAST_NONE, partNC * this->RoundUpBlock(part, this->blockLengthS));
        PipeBarrier<PIPE_V>();
    }
    
    this->PrepareInput(transDataTensorUb2, dTensor, part);
    this->queInVals.template FreeTensor<S>(curInVals);
    
    uint64_t rowLen = part;
    uint32_t sliceByWidthOffset = 0;
    uint64_t dstOffset = 0;
    
    if (!this->kernelIsBlock) {
        this->CompareSelect(dTensor, Im2ColTensor, rowLen);
    } else {
        this->CompareSelectBlockKernel(dTensor, Im2ColTensor, rowLen);
    }
    this->queInInds.template FreeTensor<float>(dTensor_);
    
    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd || this->wAdd) {
            auto event0 = this->pipe.FetchEventID(HardEvent::V_MTE2);
            
            SetFlag<HardEvent::V_MTE2>(event0);
            WaitFlag<HardEvent::V_MTE2>(event0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
            for (uint64_t curD = 0; curD < this->partOutD; ++curD) {
                for (uint64_t curH = 0; curH < this->partOutH; ++ curH) { 
                    DataCopy(this->tmpBuf[curD * this->partOutH * partOut * this->blockLength + curH * partOut * this->blockLength], 
                             this->workspaceGm[curWorkspaceOffset + (curD * (this->heightOut + this->pH + this->ceilH) + curH) * (this->widthOut + this->pW + this->ceilW) * this->blockLength], partOut * this->blockLength); 
                }
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
    
    PipeBarrier<PIPE_V>();
    
    if constexpr (sizeof(T) == sizeof(S)) {
        if (!this->wAdd) {
            this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, partOut, part, addMulti, addTail, onlyTail, this->addParamsOverUint8);
        } else {
            this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, partOut, part, addMulti, addTail, onlyTail, this->addParamsOverUint8);
        }
    } else {
        if (!this->dAdd && !this->hAdd && !this->wAdd) {
            this->template Im2Col<false>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, partOut, part, addMulti, addTail, onlyTail, this->addParamsOverUint8);
        } else {
            this->template Im2Col<true>(dstOffset, sliceByWidthOffset, Im2ColTensor, curDepthOff_, partOut, part, addMulti, addTail, onlyTail, this->addParamsOverUint8);
        }
    }
    PipeBarrier<PIPE_V>();
    
    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd || this->wAdd) {
            auto event0 = this->pipe.FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(event0);
            WaitFlag<HardEvent::V_MTE3>(event0);
            for (uint64_t curD = 0; curD < this->partOutD; ++curD) {
                for (uint64_t curH = 0; curH < this->partOutH; ++ curH) { 
                    DataCopy(this->workspaceGm[curWorkspaceOffset + (curD * (this->heightOut + this->pH + this->ceilH) + curH) * (this->widthOut + this->pW + this->ceilW) * this->blockLength], 
                             this->tmpBuf[curD * this->partOutH * partOut * this->blockLength + curH * partOut * this->blockLength], partOut * this->blockLength); 
                }
            }  
            
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);

            event0 = this->pipe.FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(event0);
            WaitFlag<HardEvent::MTE3_V>(event0);
        } 
    }

    uint64_t tmpBufOffset = padDL * this->partOutH * partOut * this->blockLength;
    if (padHL || padWL || padHR || padWR) {
        tmpBufOffset = 0;
        // Recalc repeat params for splitW case
        this->stripPadRepeatTimes = (partOut - padWL - padWR) / this->vecProcBlocks;
        this->processedStripPadData = this->stripPadRepeatTimes * this->vecProcLength;
        this->stripPadTail = (partOut - padWL - padWR) * this->blockLength - this->processedStripPadData;
        this->PadStripParams = {1, 1, VEC_PROC_BLOCKS, VEC_PROC_BLOCKS};
        if (this->stripPadRepeatTimes == 0 || (this->stripPadRepeatTimes == 1 && this->stripPadTail == 0)) {
            this->PadStripParams.dstRepeatSize = this->stripPadRepeatTimes == 0 ? (this->stripPadTail / this->blockLength) : this->vecProcBlocks; 
            this->PadStripParams.srcRepeatSize = partOut;
        }
        
        bool smallSingleMask = ((this->stripPadRepeatTimes == 0) || (this->stripPadRepeatTimes == 1 && this->stripPadTail == 0)) ? true : false;
        bool doTail = ((this->stripPadRepeatTimes == 1 && this->stripPadTail == 0) || (this->stripPadTail == 0)) ? false : true;

        this->StripPad(padDL, padHL, padWL, this->partOutD - padDL - padDR, this->partOutH - padHL - padHR, partOut - padWL - padWR, this->partOutD, this->partOutH, partOut, smallSingleMask, doTail);
        PipeBarrier<PIPE_V>();
    }
    
    TransposeBack<S>(Im2ColTensor, this->tmpBuf[tmpBufOffset], partNC, this->partOutD - padDL - padDR, this->partOutH - padHL - padHR, partOut - padWL - padWR, realTransAlignRoundPartOutSize);
    PipeBarrier<PIPE_V>();
    
    if (computeTail) {
        this->gmHOff = this->sH * this->partOutW * this->blockLength;
        this->gmDOff = (this->sD * this->partOutH - this->partH * this->sH) * this->partOutW * this->blockLength;
        this->addParamsOverUint8 = (this->dH * this->partOutW) > UINT8_MAX;
        
        if (!this->im2colSingleWOut) {
            this->Im2colNewParamsTail.dstRepeatSize  = static_cast<uint16_t>(this->dH * this->partOutW);
            this->im2ColAddParamsTail.dstRepStride = static_cast<uint8_t>(this->dH * this->partOutW);
            this->im2ColAddParamsTail.src0RepStride = static_cast<uint8_t>(this->dH * this->partOutW);
            
            if (this->im2colAddRepeatTimes == 1) {
                this->Im2colNewParams.dstRepeatSize = this->dH * this->partOutW;
                
                if (!this->addParamsOverUint8) { 
                    this->im2ColAddParams.dstRepStride = this->dH * this->partOutW;
                    this->im2ColAddParams.src0RepStride = this->dH * this->partOutW;
                } else {
                    this->im2ColAddParams.dstRepStride = this->dW * VEC_PROC_BLOCKS;
                    this->im2ColAddParams.src0RepStride = this->dW * VEC_PROC_BLOCKS;
                    this->im2ColAddParams.src1RepStride = VEC_PROC_BLOCKS;
                }
            }
        } else {
            this->im2colCopyFullIters = this->partW / UINT8_MAX;
            this->processedIm2colCopyPartW = UINT8_MAX * this->im2colCopyFullIters;
            this->im2colCopyRepeatTimesTail = this->partW - this->processedIm2colCopyPartW;
        }
        this->transDataParams.repeatTimes = this->RoundUpBlock(this->partRoundDhwInp, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    }

    if constexpr (sizeof(T) == sizeof(S)) { // fp32/fp16 case 
        if (this->dAdd || this->hAdd || this->wAdd) {
            SetAtomicAdd<S>();
            CopyOut(this->gradInputGm[outputOffset], partNC, this->partOutD - padDL - padDR, this->partOutH - padHL - padHR, partOut - padWL - padWR);
            SetAtomicNone();
        } else {
            CopyOut(this->gradInputGm[outputOffset], partNC, this->partOutD - padDL - padDR, this->partOutH - padHL - padHR, partOut - padWL - padWR);
        }
    } else {
        CopyOut(this->gradInputGm[outputOffset], partNC, this->partOutD - padDL - padDR, this->partOutH - padHL - padHR, partOut - padWL - padWR);
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::ComputeAll(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD,
                                                                     const uint64_t& offH, const uint64_t& offW,
                                                                     bool& addMulti, bool& addTail, bool& onlyTail) {
    uint64_t d = 0, h = 0, w = 0;
    uint64_t dOut = 0, hOut = 0, wOut = 0;
    for (dOut = 0; dOut < this->roundDepthOut; dOut += offD, d += this->partD) {
        uint64_t padDL = ((int(this->pD - dOut) > 0) && (this->pD != 0)) ? this->pD - dOut : 0;
        uint64_t padDR = (((dOut + this->partOutD) > (this->depthOut + this->pD)) && (this->ceilD != 0)) ? (dOut + this->partOutD) - (this->depthOut + this->pD) : 0;
        uint64_t prevWorkspaceOffsetD = curWorkspaceOffset;
        h = 0;
        for (hOut = 0; hOut < this->roundHeightOut; hOut += offH, h += this->partH) {
            uint64_t padHL = ((int(this->pH - hOut) > 0) && (this->pH != 0)) ? this->pH - hOut : 0;
            uint64_t padHR = (((hOut + this->partOutH) > (this->heightOut + this->pH)) && (this->ceilH != 0)) ? (hOut + this->partOutH) - (this->heightOut + this->pH) : 0;
            uint64_t prevWorkspaceOffsetH = curWorkspaceOffset;
            w = 0;
            for (wOut = 0; wOut < this->roundWidthOut; wOut += offW, w += this->partW) {
                uint64_t padWL = ((int(this->pW - wOut) > 0) && (this->pW != 0)) ? this->pW - wOut : 0;
                uint64_t padWR = (((wOut + this->partOutW) > (this->widthOut + this->pW)) && (this->ceilW != 0)) ? (wOut + this->partOutW) - (this->widthOut + this->pW) : 0;
                uint64_t inputOffset = b * this->inputSize + d * this->hwInputSize + h * this->widthInput + w;
                uint64_t outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize + (hOut >= this->pH ? hOut - this->pH : 0) * this->widthOut + (wOut >= this->pW ? wOut - this->pW : 0);
                
                ComputeIteration(d * this->sD - this->pD, h * this->sH - this->pH, w * this->sW - this->pW, batchesPerIter, this->partOutW, this->partW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR, addMulti, addTail, onlyTail, false);
                curWorkspaceOffset += this->sW * this->partW * this->blockLength;
            }
            if (tailW >= 1) {
                uint64_t padWL = ((int(this->pW - wOut) > 0) && (this->pW != 0)) ? this->pW - wOut : 0;
                uint64_t padWR = (((wOut + this->tailOutW) > (this->widthOut + this->pW)) && (this->ceilW != 0)) ? (wOut + this->tailOutW) - (this->widthOut + this->pW) : 0;
                uint64_t inputOffset = b * this->inputSize + d * this->hwInputSize + h * this->widthInput + w;
                uint64_t outputOffset = b * this->outSize + (dOut >= this->pD ? dOut - this->pD : 0) * this->hwOutputSize + (hOut >= this->pH ? hOut - this->pH : 0) * this->widthOut + (wOut >= this->pW ? wOut - this->pW : 0);
                
                ComputeIteration(d * this->sD - this->pD, h * this->sH - this->pH, w * this->sW - this->pW, batchesPerIter, tailOutW, tailW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR, addMulti, addTail, onlyTail, true);
            }
            curWorkspaceOffset = prevWorkspaceOffsetH + this->sH * this->partH * (this->widthOut + this->pW + this->ceilW) * this->blockLength;
        }
        curWorkspaceOffset = prevWorkspaceOffsetD + this->sD * this->partD * (this->heightOut + this->pH + this->ceilH) * (this->widthOut + this->pW + this->ceilW) * this->blockLength;
    }
    curWorkspaceOffset = 0;
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::ProcessBatches(const uint64_t& b, const uint64_t& batchesPerIter, const uint64_t& offD, const uint64_t& offH, const uint64_t& offW) {
    bool singleAddOrCopy = this->im2colAddRepeatTimes == 1;
    bool multipleAddOrCopy = this->im2colAddRepeatTimes > 1;
    bool doTail = this->im2colAddTail != 0;
    bool onlyTail = this->im2colAddRepeatTimes == 0 && doTail;

    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd || this->wAdd) { 
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

    ComputeAll(b, batchesPerIter, offD, offH, offW, multipleAddOrCopy, doTail, onlyTail);

    if constexpr (sizeof(T) != sizeof(S)) {
        if (this->dAdd || this->hAdd || this->wAdd) { 
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2_workspace);
        }
    }
}

template <typename T, typename S> 
__aicore__ void KernelMaxPool3DGradWithArgmaxSplitW<T,S>::Compute() {
    this->CreateKernelIndexes();
    CreateGeneralOffsets();
    
    uint64_t offD = this->partOutD - (this->kD - 1) * this->dD + this->sD - 1;
    uint64_t offH = this->partOutH - (this->kH - 1) * this->dH + this->sH - 1;
    uint64_t offW = this->partOutW - (this->kW - 1) * this->dW + this->sW - 1;
    
    for (int64_t b = 0; b < this->roundBatches; b += this->blockLength) {
        ProcessBatches(b, this->blockLength, offD, offH, offW);
    }
    
    if (this->partNC != 0) {
        ProcessBatches(this->roundBatches, this->partNC, offD, offH, offW);
    }
}

#endif
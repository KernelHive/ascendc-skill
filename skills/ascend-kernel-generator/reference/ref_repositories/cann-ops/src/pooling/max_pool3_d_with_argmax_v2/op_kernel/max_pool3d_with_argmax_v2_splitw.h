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
 * \file max_pool3d_with_argmax_v2_splitw.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_WITH_ARGMAX_V2_SPLITW_H
#define OPP_MAX_POOL3D_WITH_ARGMAX_V2_SPLITW_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_v2_base.h"

using namespace AscendC;

template <typename T, typename S>
class KernelMaxPool3DWithArgmaxV2SplitW : public KernelMaxPool3DWithArgmaxV2Base<T,S> {
  private:
    uint32_t tailW;
    uint32_t tailOutW;
    uint32_t partsCurCore;
    uint32_t dStart = 0;
    uint32_t dOutStart = 0;
    uint32_t hStart = 0;
    uint32_t hOutStart = 0;
    uint32_t wStart = 0;
    uint32_t wOutStart = 0;

  public:
    __aicore__ KernelMaxPool3DWithArgmaxV2SplitW(const MaxPool3DWithArgmaxV2SplitWTilingData* __restrict tilingData_) :
                                                 KernelMaxPool3DWithArgmaxV2Base<T, S>(tilingData_->inputShapes[D_DIM], tilingData_->inputShapes[H_DIM], tilingData_->inputShapes[W_DIM],
                                                                                       tilingData_->outShapes[D_DIM], tilingData_->outShapes[H_DIM], tilingData_->outShapes[W_DIM],
                                                                                       tilingData_->kD, tilingData_->kH, tilingData_->kW,
                                                                                       tilingData_->sD, tilingData_->sH, tilingData_->sW,
                                                                                       tilingData_->pD, tilingData_->pH, tilingData_->pW,
                                                                                       tilingData_->dD, tilingData_->dH, tilingData_->dW,
                                                                                       tilingData_->partD, tilingData_->partH, tilingData_->partW,
                                                                                       tilingData_->minFloat, tilingData_->batchesPerCore, tilingData_->leftOverBatches,
                                                                                       tilingData_->ceilD, tilingData_->ceilH, tilingData_->ceilW,
                                                                                       tilingData_->sizeUb1, tilingData_->sizeUb2, tilingData_->sizeValues){
        const bool isSmallBatches = (sizeof(T) == sizeof(float)) && (tilingData_->batchesPerCore == 0) &&
                                    (tilingData_->sD == tilingData_->kD) && (tilingData_->sH == tilingData_->kH) && (tilingData_->sW == tilingData_->kW) &&
                                    (tilingData_->dD == 1) && (tilingData_->dH == 1) && (tilingData_->dW == 1) &&
                                    (tilingData_->pD == 0) && (tilingData_->pH == 0) && (tilingData_->pW == 0) &&
                                    (tilingData_->ceilD == 0) && (tilingData_->ceilH == 0) && (tilingData_->ceilW == 0) &&
                                    (tilingData_->kW <= tilingData_->partW);
        // use big enough partsCurCore to process all batches
        partsCurCore = this->batchesCurCore * this->outSize;

        if (isSmallBatches) {
            // we have less than 1 batch per core
            // so split data between cores according to parts fit to UB
            const uint32_t& partsPerCore = tilingData_->partsPerCore;
            const uint32_t& leftOverParts = tilingData_->leftOverParts;

            partsCurCore = partsPerCore + (this->coreIdx < leftOverParts ? 1 : 0);
            this->batchesCurCore = tilingData_->sizeIn[this->coreIdx];
            this->batchOffset = tilingData_->batchStart[this->coreIdx];

            dStart=tilingData_->splitPointDIn[this->coreIdx];
            dOutStart=tilingData_->splitPointDOut[this->coreIdx];
            hStart=tilingData_->splitPointHIn[this->coreIdx];
            hOutStart=tilingData_->splitPointHOut[this->coreIdx];
            wStart=tilingData_->splitPointWIn[this->coreIdx];
            wOutStart=tilingData_->splitPointWOut[this->coreIdx];
        }
    }

    __aicore__ void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe *pipe) {
        this->pipe = pipe;
        this->GMInit(x, y, indices, workspace);
        
        PrepareScalars();
        auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);
        
        this->UbInit();
    }

    __aicore__ void Process() {
        if (g_coreType != AIV || this->batchesCurCore == 0) {
            return;
        }
        MaxPool3DWithArgmaxV2Algorithm();
    }

  private:
    __aicore__ inline void DoMaxPool(int32_t dOutOff, int32_t hOutOff, int32_t wOutOff, 
                                     uint32_t partNC, uint32_t part, uint32_t partOut,
                                     uint32_t inputOffset, uint32_t outputOffset,
                                     uint32_t padDL, uint32_t padDR,
                                     uint32_t padHL, uint32_t padHR,
                                     uint32_t padWL, uint32_t padWR){
        CopyIn(this->xGm[inputOffset], partNC, part, padHL, padHR, padDL, padDR);
        
        Img2ColC0(partNC, dOutOff, hOutOff, wOutOff, partOut, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
        
        this->CopyOutInds(outputOffset, partNC, partOut);
    }

    __aicore__ void PrepareScalars(){
        this->roundDepthInput = this->sD * (this->depthOut - 1) + 1;
        this->roundHeightInput = this->sH * (this->heightOut - 1) + 1;
        
        auto itW = this->RoundDownBlock(this->widthOut, this->partOutW);
        this->roundWidthInput = this->sW * (itW - 1) + 1;
        tailOutW = this->widthOut - itW;

        this->roundW = this->RoundUpBlock(this->partW, this->blockLengthS);
        this->alContOutSize = this->OutputCalcW(this->roundW);
        tailW = this->dW * (this->kW - 1) + this->sW * (tailOutW - 1) + 1;
        
        this->partHwInp = this->roundW * this->partH;

        this->PrepareBaseScalars();
    }
    
    // Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
    __aicore__ void MaxPool3DWithArgmaxV2Algorithm();
    __aicore__ void Im2ColNoDilationW(const LocalTensor<T>& dst);
    __aicore__ void Im2ColNoDilationWBigKW(const LocalTensor<T>& dst);
    __aicore__ void Img2ColC0(const uint32_t partNC, const int32_t offD, const int32_t offH, const int32_t offW,
                              const uint32_t partWOutReal, // H and W values that is not rounded up
                              const uint32_t outputOffset,
                              const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
    __aicore__ inline void CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, const uint32_t& part, const uint32_t padL, const uint32_t padR, const uint32_t padDL=0, const uint32_t padDR=0);
    __aicore__ void Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb,
                            const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad,
                            const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
};

// Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitW<T,S>::MaxPool3DWithArgmaxV2Algorithm() {
    this->CreateKernelIndexes();

    auto hwOut = this->heightOut * this->widthOut;
    auto partNC = this->batchesCurCore % this->blockLength;
    auto roundBatches = this->batchesCurCore - partNC;

    uint32_t offD = this->partD - (this->kD - 1) * this->dD + this->sD - 1;
    uint32_t offH = this->partH - (this->kH - 1) * this->dH + this->sH - 1;
    uint32_t offW = this->partW - (this->kW - 1) * this->dW + this->sW - 1;
    auto part = this->partW;
    uint32_t d, h;
    // padDL - if we have left padding for this part
    // padDR - if we have right padding for this part (possibly ceilMode)
    // on the first iteration we have left padding (if pad have), on last right
    // in the middle no padding
    uint32_t padDL, padHL, padWL, padDR, padHR, padWR;
    uint32_t dOut=0, hOut=0, wOut=0;
    uint32_t w =0;
    uint32_t partCount = 0;
    for (int b=0; (b<roundBatches) && (partCount < this->partsCurCore); b+=this->blockLength){
        dOut = this->dOutStart;
        for (uint32_t d=this->dStart; (d<this->roundDepthInput) && (partCount < this->partsCurCore); d+=offD, dOut+=this->partOutD){
            padDL = ((int(this->pD - d) > 0) && (this->pD != 0)) ? this->pD - d : 0;  
            padDR = (((d + this->partD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (d + this->partD) - (this->depthInput + this->pD) : 0;
            hOut = this->hOutStart;
            for (uint32_t h=this->hStart; (h<this->roundHeightInput) && (partCount < this->partsCurCore); h+=offH, hOut+=this->partOutH){
                padHL = ((int(this->pH - h) > 0) && (this->pH != 0)) ? this->pH - h : 0;  
                padHR = (((h + this->partH) > (this->heightInput + this->pH)) && (this->ceilH != 0)) ? (h + this->partH) - (this->heightInput + this->pH) : 0;
                wOut = this->wOutStart;
                for (w=this->wStart; (w<this->roundWidthInput) && (partCount < this->partsCurCore); w+=offW, wOut+=this->partOutW){
                    padWL = ((int(this->pW - w) > 0) && (this->pW != 0)) ? this->pW - w : 0;  
                    padWR = (((w + this->partW) > (this->widthInput + this->pW)) && (this->ceilW != 0)) ? (w + this->partW) - (this->widthInput + this->pW) : 0;
                    part = this->partW - padWL - padWR;
                    uint32_t inputOffset = b * this->inputSize + (d >= this->pD ? d - this->pD : 0) * this->hwInputSize + (h >= this->pH ? h - this->pH : 0) * this->widthInput + (w >= this->pW ? w - this->pW : 0);
                    uint32_t outputOffset = b * this->outSize + dOut * hwOut + hOut * this->widthOut + wOut;
                    
                    // Offset here can be < 0 for case with padding
                    // This offset is start position of the first kernel for this iteration. If it starts on padding, offset will be < 0
                    DoMaxPool(dOut*this->sD-this->pD, hOut*this->sH-this->pH, wOut*this->sW-this->pW, this->blockLength, part, this->partOutW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
                    partCount++;
                }
                if ((tailOutW >= 1) && (partCount < this->partsCurCore)){
                    padWL = ((int(this->pW - w) > 0) && (this->pW != 0)) ? this->pW - w : 0;  
                    padWR = (((w + this->tailW) > (this->widthInput + this->pW)) && (this->ceilW != 0)) ? (w + this->tailW) - (this->widthInput + this->pW) : 0;
                    uint32_t inputOffset = b * this->inputSize + (d >= this->pD ? d - this->pD : 0) * this->hwInputSize + (h >= this->pH ? h - this->pH : 0) * this->widthInput + (w - this->pW);
                    uint32_t outputOffset = b * this->outSize + dOut * hwOut + hOut * this->widthOut + wOut;
                    
                    DoMaxPool(dOut*this->sD-this->pD, hOut*this->sH-this->pH, wOut*this->sW-this->pW, this->blockLength, tailW - padWR, tailOutW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
                    partCount++;
                }
                this->wStart = 0;
                this->wOutStart = 0;
            }
            this->hStart = 0;
            this->hOutStart = 0;
        }
        this->dStart = 0;
        this->dOutStart = 0;
    }
    if (partNC != 0){
        auto b = roundBatches;
        dOut = this->dOutStart;
        for (uint32_t d=this->dStart; (d<this->roundDepthInput) && (partCount < this->partsCurCore); d+=offD, dOut+=this->partOutD){
            padDL = ((int(this->pD - d) > 0) && (this->pD != 0)) ? this->pD - d : 0;  
            padDR = (((d + this->partD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (d + this->partD) - (this->depthInput + this->pD) : 0;
            hOut = this->hOutStart;
            for (uint32_t h=this->hStart; (h<this->roundHeightInput) && (partCount < this->partsCurCore); h+=offH, hOut+=this->partOutH){
                padHL = ((int(this->pH - h) > 0) && (this->pH != 0)) ? this->pH - h : 0;  
                padHR = (((h + this->partH) > (this->heightInput + this->pH)) && (this->ceilH != 0)) ? (h + this->partH) - (this->heightInput + this->pH) : 0;
                wOut = this->wOutStart;
                for (w=this->wStart; (w<this->roundWidthInput) && (partCount < this->partsCurCore); w+=offW, wOut+=this->partOutW){
                    padWL = ((int(this->pW - w) > 0) && (this->pW != 0)) ? this->pW - w : 0;  
                    padWR = (((w + this->partW) > (this->widthInput + this->pW)) && (this->ceilW != 0)) ? (w + this->partW) - (this->widthInput + this->pW) : 0;
                    part = this->partW - padWL - padWR;
                    uint32_t inputOffset = b * this->inputSize + (d >= this->pD ? d - this->pD : 0) * this->hwInputSize + (h >= this->pH ? h - this->pH : 0) * this->widthInput + (w >= this->pW ? w - this->pW : 0);
                    uint32_t outputOffset = b * this->outSize + dOut * hwOut + hOut * this->widthOut + wOut;
                    
                    // Offset here can be < 0 for case with padding
                    // This offset is start position of the first kernel for this iteration. If it starts on padding, offset will be < 0
                    DoMaxPool(dOut*this->sD-this->pD, hOut*this->sH-this->pH, wOut*this->sW-this->pW, partNC, part, this->partOutW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
                    partCount++;
                }
                if ((tailOutW >= 1) && (partCount < this->partsCurCore)){
                    padWL = ((int(this->pW - w) > 0) && (this->pW != 0)) ? this->pW - w : 0;  
                    padWR = (((w + this->tailW) > (this->widthInput + this->pW)) && (this->ceilW != 0)) ? (w + this->tailW) - (this->widthInput + this->pW) : 0;
                    uint32_t inputOffset = b * this->inputSize + (d >= this->pD ? d - this->pD : 0) * this->hwInputSize + (h >= this->pH ? h - this->pH : 0) * this->widthInput + (w - this->pW);
                    uint32_t outputOffset = b * this->outSize + dOut * hwOut + hOut * this->widthOut + wOut;
                    
                    DoMaxPool(dOut*this->sD-this->pD, hOut*this->sH-this->pH, wOut*this->sW-this->pW, partNC, tailW - padWR, tailOutW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
                    partCount++;
                }
                this->wStart = 0;
                this->wOutStart = 0;
            }
            this->hStart = 0;
            this->hOutStart = 0;
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitW<T,S>::Im2ColNoDilationW(const LocalTensor<T>& dst) {
    DataCopyParams Im2colParamsHD = {static_cast<uint16_t>(this->kH * this->kD), static_cast<uint16_t>(this->kW), 
                                    static_cast<uint16_t>(this->roundW - this->kW), 0};
    DataCopyParams Im2colParamsW = {static_cast<uint16_t>(this->partOutW), static_cast<uint16_t>(this->kW),
                                    0, static_cast<uint16_t>((this->kH * this->kD -1) * this->kW)};
    
    uint32_t sliceByWidthOffset = 0;
    uint32_t dOff = 0;
    if (this->kH * this->kD < this->alContOutSize && this->sW == this->kW){
        const uint32_t wholeRepeatTimes = this->partOutW / MAX_REPEAT_TIMES;
        const uint32_t tailRepeatTimes = this->partOutW % MAX_REPEAT_TIMES;
        const uint32_t wholeOffsetDst = wholeRepeatTimes * MAX_REPEAT_TIMES * this->blockKernelDHW;
        const uint32_t wholeOffsetSrc = wholeRepeatTimes * MAX_REPEAT_TIMES * this->kW * this->blockLength;

        for (uint32_t curWidthOut = 0; curWidthOut < this->kD * this->kH; curWidthOut++, dOff += (this->roundW * this->blockLength), sliceByWidthOffset += this->kW * this->blockLength) {
            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(dst[i * MAX_REPEAT_TIMES * this->blockKernelDHW + sliceByWidthOffset], this->transDataTensorUb1[i * MAX_REPEAT_TIMES * this->kW * this->blockLength + dOff],
                     this->kW * this->blockLength, MAX_REPEAT_TIMES, {1, 1, static_cast<uint16_t>(this->kernelDHW), static_cast<uint16_t>(this->kW)});
            }
            Copy(dst[wholeOffsetDst + sliceByWidthOffset], this->transDataTensorUb1[wholeOffsetSrc + dOff],
                 this->kW * this->blockLength, tailRepeatTimes, {1, 1, static_cast<uint16_t>(this->kernelDHW), static_cast<uint16_t>(this->kW)});
        }
    } else {
        for (uint32_t curWidthOut = 0; curWidthOut < this->partOutW; curWidthOut++, dOff += this->gmWOff, sliceByWidthOffset += this->blockKernelDHW) {
            Copy(dst[sliceByWidthOffset], this->transDataTensorUb1[dOff],
                 this->kW * this->blockLength, this->kH * this->kD, {1, 1, static_cast<uint16_t>(this->kW), static_cast<uint16_t>(this->roundW)});
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitW<T,S>::Im2ColNoDilationWBigKW(const LocalTensor<T>& dst) {
    DataCopyParams Im2colParamsHD = {static_cast<uint16_t>(this->kH * this->kD), static_cast<uint16_t>(this->kW), 
                                    static_cast<uint16_t>(this->roundW - this->kW), 0};
    DataCopyParams Im2colParamsW = {static_cast<uint16_t>(this->partOutW), static_cast<uint16_t>(this->kW),
                                    0, static_cast<uint16_t>((this->kH * this->kD -1) * this->kW)};
    
    uint32_t sliceByWidthOffset = 0;
    uint32_t dOff = 0;
    if (this->kH * this->kD < this->alContOutSize && this->sW == this->kW){
        for (uint32_t curWidthOut = 0; curWidthOut < this->kD * this->kH; curWidthOut++, dOff += (this->roundW * this->blockLength), sliceByWidthOffset += this->kW * this->blockLength) {
            DataCopy(dst[sliceByWidthOffset], this->transDataTensorUb1[dOff], Im2colParamsW);
        }
    } else {
        for (uint32_t curWidthOut = 0; curWidthOut < this->partOutW; curWidthOut++, dOff += this->gmWOff, sliceByWidthOffset += this->blockKernelDHW) {
            DataCopy(dst[sliceByWidthOffset], this->transDataTensorUb1[dOff], Im2colParamsHD);
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitW<T,S>::Img2ColC0(const uint32_t partNC, const int32_t offD, const int32_t offH, const int32_t offW,
                                                                  const uint32_t partWOutReal, const uint32_t outputOffset, const uint32_t pDL, const uint32_t pDR,
                                                                  const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    LocalTensor<T> curBuf;
    LocalTensor<T> maxTmp;
    
    LocalTensor<S> transDataTensorUb2Old = this->queIn.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2 = transDataTensorUb2Old. template ReinterpretCast<T>();
    
    // Cast for bf16
    this->CastBF16(transDataTensorUb2Old, transDataTensorUb2, partNC);

    maxTmp = this->transDataTensorUb1;
    curBuf = transDataTensorUb2;
    
    LocalTensor<uint8_t> maskIndex = maxTmp.template ReinterpretCast<uint8_t>();
	LocalTensor<uint8_t> nanMask = maskIndex[this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) * this->partOutW];
    
    // TransDataTo5HD (NCDHW -> NC1DHWC0)
    this->PrepareInput(transDataTensorUb2, this->partRoundDhwInp);
    
    // Add padding to data if needed
    if (pDL != 0 || pDR != 0 || pHL != 0 || pHR != 0 || pWL != 0 || pWR != 0) {
        uint32_t partDWoPad = this->partD - pDL - pDR;
        uint32_t partHWoPad = this->partH - pHL - pHR;
        uint32_t partWWoPad = this->partW - pWL - pWR;
        if (partWOutReal == tailOutW) {
            partWWoPad = tailW - pWL - pWR;
        } 
        Padding(this->transDataTensorUb1, transDataTensorUb2, partDWoPad, partHWoPad, partWWoPad, pDL, pDR, pHL, pHR, pWL, pWR);
    }
    
    // Im2ColC0 
    if (this->dD == 1 && this->dH == 1 && this->dW == 1){
        if (this->kW <= BLOCKS_IN_REP){
            Im2ColNoDilationW(curBuf);
        } else {
            Im2ColNoDilationWBigKW(curBuf);
        }
    } else {
        uint32_t splitWDOff = 0;
        this->Im2ColHWDDilation(curBuf, splitWDOff);
    }
    PipeBarrier<PIPE_V>();

    //Find Max for each row by width
    LocalTensor<S> reduceMaxResultBf16 = this->queOutVals.template AllocTensor<S>();
    LocalTensor<T> reduceMaxResult = reduceMaxResultBf16.template ReinterpretCast<T>();
    bool useRepeat = (this->kernelDHW - this->halfKernelDHW) < MAX_REPEAT_TIMES;
    uint32_t rowLen = this->partOutW;
    if (useRepeat){
        this->FindMaxRowPerRepeat(curBuf, reduceMaxResult, maxTmp, 0, rowLen);
    } else {
        this->FindMaxRowPerKernel(curBuf, reduceMaxResult, maxTmp, 0, rowLen);
    }
    
    LocalTensor<float> idxs = this->queOutInds.template AllocTensor<float>();
    LocalTensor<T> idxsT = idxs.ReinterpretCast<T>();

    Duplicate(idxsT, (T)1., rowLen * this->blockLength);
    // Find Indices for each row by width
    this->CompareSelect(reduceMaxResult, curBuf, maskIndex, nanMask, this->partOutW, 0);

    uint32_t alignedRowlen = this->RoundUpBlock((rowLen * this->blockLength) / BITS_UINT8, BLOCK_LEN_UINT8) * this->blockLength;
    Compare<T>(maskIndex, reduceMaxResult, reduceMaxResult, CMPMODE::EQ, alignedRowlen);
    PipeBarrier<PIPE_V>();

    Select<T>(idxsT, maskIndex, idxsT, -1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedRowlen);
    PipeBarrier<PIPE_V>();

    if (useRepeat){
        this->FindMinRowPerRepeat(curBuf, maxTmp, rowLen);
    } else {
        this->FindMinRowPerKernel(curBuf, maxTmp, rowLen);
    }

    Mul<T>(maxTmp, maxTmp, idxsT, rowLen * this->blockLength);  
    PipeBarrier<PIPE_V>();

    this->CastInds(idxs, maxTmp, 0, this->partOutW);
    
    // TransDataTo5HD (NC1DHWC0 -> NCDHW) for out
    this->template TransposeBack<S>(reduceMaxResult, idxs, partNC, outputOffset, partWOutReal);
    
    // TransDataTo5HD (NC1DHWC0 -> NCDHW) for indices
    LocalTensor<float> transDataTensorUb3 = this->transDataTensorUb1. template ReinterpretCast<float>();

    // Recount indeces
    auto tmpSize = this->RoundUpBlock(this->roundPartOutSize, BLOCK_LEN_FP32);
    uint32_t ncPartOutTrans = partNC * this->transAlignRoundPartOutSize;
    LocalTensor<float> tmpIdxs = transDataTensorUb2.template ReinterpretCast<float>();
    
    this->IndexRecalcFirst(transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);

    this->IndexRecalcSecond(transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);
    
    LocalTensor<int> generalOffsets = this->queOutVals.template AllocTensor<int>();
    LocalTensor<int> hTmp = generalOffsets[tmpSize];
    LocalTensor<int> wTmp = hTmp[tmpSize];

    Duplicate(generalOffsets, offD * (int)(this->hwInputSize), tmpSize);
    Duplicate(hTmp, offH * (int)(this->widthInput), tmpSize);
    ArithProgression<int>(wTmp, int(offW), this->sW, tmpSize);
    
    LocalTensor<int> transDataTensorUb3Int = transDataTensorUb3. template ReinterpretCast<int>();
    LocalTensor<int> idxsInt = idxs. template ReinterpretCast<int>();
    LocalTensor<int> tmpIdxsInt = tmpIdxs. template ReinterpretCast<int>();
    
    Cast(transDataTensorUb3Int, transDataTensorUb3, RoundMode::CAST_RINT, ncPartOutTrans);
    Cast(idxsInt, idxs, RoundMode::CAST_RINT, ncPartOutTrans);
    Cast(tmpIdxsInt, tmpIdxs, RoundMode::CAST_RINT, ncPartOutTrans);
    PipeBarrier<PIPE_V>();
    
    Muls(transDataTensorUb3Int, transDataTensorUb3Int, int(this->dD * this->hwInputSize), ncPartOutTrans);
    Muls(idxsInt, idxsInt, int(this->dH * this->widthInput), ncPartOutTrans);
    PipeBarrier<PIPE_V>();

    this->IndexRecalcThird(generalOffsets, hTmp, wTmp, transDataTensorUb3Int, idxsInt, tmpIdxsInt, idxs, ncPartOutTrans);

    this->queIn.template FreeTensor<S>(transDataTensorUb2Old);	
	this->queOutVals.template FreeTensor<int>(generalOffsets);
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2SplitW<T,S>::CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, const uint32_t& part, 
                                                                      const uint32_t padL, const uint32_t padR, const uint32_t padDL, const uint32_t padDR){
    uint32_t downPartW = this->RoundDownBlock(part, this->blockLengthS);
    uint32_t upPartW = this->roundW;
    
    uint32_t partDWoPad = this->partD - padDL - padDR;
    uint32_t tmpDInSize = partDWoPad * this->hwInputSize;
    
    uint32_t srcOff = 0;
    auto srcStride = this->widthInput - downPartW;
    LocalTensor<S> srcUb = this->queIn.template AllocTensor<S>();
    
    uint32_t partHWoPad = this->partH - padL - padR;
    
    if (this->widthInput % this->blockLengthS == 0 && downPartW == part){
        if (((this->widthInput - downPartW) / this->blockLengthS) < MAX_UINT16) {
            DataCopyParams params{static_cast<uint16_t>(partHWoPad), static_cast<uint16_t>(downPartW / this->blockLengthS), 
                                  static_cast<uint16_t>((this->widthInput - downPartW) / this->blockLengthS), 
                                  static_cast<uint16_t>((upPartW - downPartW) / this->blockLengthS)};
            for (uint32_t nc = 0; nc < partNC; nc++){
                auto tmp = srcGm[nc * this->inputSize];
                auto tmpDst = srcUb[this->partRoundDhwInp * nc];
                for (uint32_t d = 0, dstOff = 0; d < tmpDInSize; d+= this->hwInputSize, dstOff += upPartW * this->partH){
                    DataCopy(tmpDst[dstOff], tmp[d], params);
                }
            }
        } else {
            DataCopyParams params{static_cast<uint16_t>(1), static_cast<uint16_t>(downPartW / this->blockLengthS), 0, 0};
            for (uint32_t nc = 0; nc < partNC; nc++){
                for (uint32_t d = 0, dstOff = 0; d < tmpDInSize; d+= this->hwInputSize, dstOff += upPartW * this->partH){
                    auto tmp = srcGm[nc * this->inputSize + d];
                    auto tmpDst = srcUb[this->partRoundDhwInp * nc + dstOff];
                    for (uint32_t h = 0; h < partHWoPad; h++) {
                        DataCopy(tmpDst[h * upPartW], tmp[h *this->widthInput], params);
                    }
                }
            }
        }
    } else {
        DataCopyExtParams copyPadParamsInput{static_cast<uint16_t>(partHWoPad), static_cast<uint32_t>(part * sizeof(S)), 
                                             static_cast<uint32_t>((this->widthInput - part) * sizeof(S)), 
                                             static_cast<uint32_t>((upPartW - part) / this->blockLengthS), 0};
        DataCopyPadExtParams<S> paddingParamsInput{false, 0, 0, 0};
        for (uint32_t nc = 0; nc < partNC; nc++){
            auto tmp = srcGm[nc * this->inputSize];
            auto tmpDst = srcUb[this->partRoundDhwInp * nc];
            for (uint32_t d =0, dstOff=0; d < tmpDInSize; d+= this->hwInputSize, dstOff += upPartW * this->partH){
                DataCopyPad(tmpDst[dstOff], tmp[d], copyPadParamsInput, paddingParamsInput);
            }
        }
    }
    this->queIn.template EnQue<S>(srcUb);
}

template <typename T, typename S>
__aicore__ /*inline*/ void KernelMaxPool3DWithArgmaxV2SplitW<T,S>::Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb,
                                                                           const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad,
                                                                           const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    if (pWL != 0 || pWR != 0){
        Duplicate<T>(dstUb, this->minFloat, this->partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
        
        DataCopyParams padParams = {static_cast<uint16_t>(partHWoPad), static_cast<uint16_t>(partWWoPad), 
                                    static_cast<uint16_t>(this->roundW - partWWoPad), 
                                    static_cast<uint16_t>(this->roundW - partWWoPad)};
        for (uint32_t d = 0; d < partDWoPad; d++){
            DataCopy(dstUb[pWL * this->blockLength + d * this->partHwInp * this->blockLength], 
                     srcUb[d * this->partHwInp * this->blockLength], padParams);
        }
        PipeBarrier<PIPE_V>();
        
        DataCopy(srcUb, dstUb, this->partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
    }
    
    this->PaddingH(srcUb, dstUb, partDWoPad, partHWoPad, pHL, pHR, this->partHwInp - partHWoPad * this->roundW);

    this->PaddingD(srcUb, dstUb, partDWoPad, pDL, pDR);
}

#endif
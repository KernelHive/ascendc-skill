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
 * \file max_pool3d_with_argmax_v2_splitd.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_WITH_ARGMAX_V2_SPLITD_H
#define OPP_MAX_POOL3D_WITH_ARGMAX_V2_SPLITD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_v2_base.h"

using namespace AscendC;

template <typename T, typename S>
class KernelMaxPool3DWithArgmaxV2SplitD: public KernelMaxPool3DWithArgmaxV2Base<T, S> {
  private:
    uint32_t tailD;
    uint32_t tailOutD;
    uint32_t partsCurCore;
    uint32_t dStart = 0;
    uint32_t dOutStart = 0;

  public:
    __aicore__ KernelMaxPool3DWithArgmaxV2SplitD(const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData_) :
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
                                    (tilingData_->kD <= tilingData_->partD);
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
    __aicore__ inline void DoMaxPool(int32_t dOutOff, int32_t hOutOff, 
                                     uint32_t partNC, uint32_t part, uint32_t partOut, 
                                     uint32_t inputOffset, uint32_t outputOffset,
                                     uint32_t padDL, uint32_t padDR,
                                     uint32_t padHL, uint32_t padHR,
                                     uint32_t padWL, uint32_t padWR){
        CopyIn(this->xGm[inputOffset], partNC, part);
    
        Img2ColC0(partNC, dOutOff, hOutOff, -this->pW, partOut, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR); 

        this->CopyOutInds(outputOffset, partNC, partOut);
    }

    __aicore__ void PrepareScalars(){
        this->roundW = this->partW;
        
        auto itD = this->RoundDownBlock(this->depthOut, this->partOutD);
        this->roundDepthInput = this->sD * (itD - 1) + 1;
        tailOutD = this->depthOut - itD;
        tailD = this->dD * (this->kD - 1) + this->sD * (tailOutD - 1) + 1;
        
        this->partHwInp = this->roundW * this->partH;
        
        this->PrepareBaseScalars();
    }

    // Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
    __aicore__ void MaxPool3DWithArgmaxV2Algorithm();
    __aicore__ void Img2ColC0(const uint32_t partNC, const int32_t offD, const int32_t offH, const int32_t offW,
                              const uint32_t partOutReal, // H and W values that is not rounded up
                              const uint32_t outputOffset,
                              const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
    __aicore__ inline void CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, const uint32_t& part);
    __aicore__ inline void CreateGeneralOffsets();
    __aicore__ void Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb, 
                            const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad, 
                            const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
};

// Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitD<T,S>::MaxPool3DWithArgmaxV2Algorithm() {
    this->CreateKernelIndexes();
    CreateGeneralOffsets();
    
    auto hwOut = this->heightOut * this->widthOut;
    auto partNC = this->batchesCurCore % this->blockLength;
    auto roundBatches = this->batchesCurCore - partNC;

    uint32_t offD = this->partD - (this->kD - 1) * this->dD + this->sD - 1;
    auto part = this->partD;
    uint32_t dOut=0;
    uint32_t d, h;
    uint32_t padDL, padDR;
    uint32_t padWL = this->pW;
    uint32_t padWR = this->ceilW;
    uint32_t padHL = this->pH;
    uint32_t padHR = this->ceilH;
    
    uint32_t partCount = 0;
    for (int b=0; (b<roundBatches) && (partCount < this->partsCurCore); b+=this->blockLength){
        dOut = this->dOutStart;
        for (d=this->dStart; (d<this->roundDepthInput) && (partCount < this->partsCurCore); d+=offD, dOut+=this->partOutD){
            padDL = ((int(this->pD - d) > 0) && (this->pD != 0)) ? this->pD - d : 0;  
            padDR = (((d + this->partD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (d + this->partD) - (this->depthInput + this->pD) : 0;
            part = this->partD -  padDL - padDR;
            uint32_t inputOffset = b * this->inputSize + (d >= this->pD ? d - this->pD : 0) * this->hwInputSize;
            uint32_t outputOffset = b * this->outSize + dOut * hwOut;
            
            DoMaxPool(dOut*this->sD-this->pD, -this->pH, this->blockLength, part, this->roundPartOutSize, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
            partCount++;
        }
        
        if ((tailOutD >= 1) && (partCount < this->partsCurCore)){
            padDL = ((int(this->pD - d) > 0) && (this->pD != 0)) ? this->pD - d : 0;  
            padDR = (((d + this->tailD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (d + this->tailD) - (this->depthInput + this->pD) : 0;
            uint32_t inputOffset = b * this->inputSize + (d - this->pD) * this->hwInputSize;
            uint32_t outputOffset = b * this->outSize + dOut * hwOut;
            
            DoMaxPool(dOut*this->sD-this->pD, -this->pH, this->blockLength, part, tailOutD * this->partOutH * this->partOutW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
            partCount++;
        }
        this->dOutStart = 0;
        this->dStart = 0;
    }

    if (partNC != 0){
        dOut = this->dOutStart;
        for (d=this->dStart; (d<this->roundDepthInput) && (partCount < this->partsCurCore); d+=offD, dOut+=this->partOutD){
            padDL = ((int(this->pD - d) > 0) && (this->pD != 0)) ? this->pD - d : 0;  
            padDR = (((d + this->partD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (d + this->partD) - (this->depthInput + this->pD) : 0;
            part = this->partD -  padDL - padDR;
            uint32_t inputOffset = roundBatches * this->inputSize + (d >= this->pD ? d - this->pD : 0) * this->hwInputSize;
            uint32_t outputOffset = roundBatches * this->outSize + dOut * hwOut;
            
            DoMaxPool(dOut*this->sD-this->pD, -this->pH, partNC, part, this->roundPartOutSize, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
            partCount++;
        }

        if ((tailOutD >= 1) && (partCount < this->partsCurCore)) {
            padDL = ((int(this->pD - d) > 0) && (this->pD != 0)) ? this->pD - d : 0;  
            padDR = (((d + this->tailD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (d + this->tailD) - (this->depthInput + this->pD) : 0;
            uint32_t inputOffset = roundBatches * this->inputSize + (d - this->pD) * this->hwInputSize;
            uint32_t outputOffset = roundBatches * this->outSize + dOut * hwOut;
            
            DoMaxPool(dOut*this->sD-this->pD, -this->pH, partNC, part, tailOutD * this->partOutH * this->partOutW, inputOffset, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR);
            partCount++;
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitD<T,S>::Img2ColC0(const uint32_t partNC, const int32_t offD, const int32_t offH, const int32_t offW,
                                                                  const uint32_t partOutReal, const uint32_t outputOffset, const uint32_t pDL, const uint32_t pDR,
                                                                  const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    LocalTensor<T> curBuf;
    LocalTensor<T> maxTmp;
    
    LocalTensor<S> transDataTensorUb2Old = this->queIn.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2 = transDataTensorUb2Old. template ReinterpretCast<T>();

    // Cast for bf16 using transDataTensorUb1
    this->CastBF16(transDataTensorUb2Old, transDataTensorUb2, partNC);
    bool useRepeat = (this->kernelDHW - this->halfKernelDHW < MAX_REPEAT_TIMES) && (this->partOutH * this->partOutW * this->partOutD > 1 + (this->halfKernelDHW + 1) / MAX_DIV / BLOCKS_IN_REP);
    bool onlyRow = (this->roundPartOutSize * this->kernelDHW > this->partRoundDhwInp) || !useRepeat;
    uint32_t rowLen = onlyRow ? this->partOutW : this->partOutH * this->partOutW * this->partOutD;
    uint32_t curBufOffset = (this->maxTmpOff > MASK_COUNT * this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) / sizeof(float)) ?
                             this->maxTmpOff : MASK_COUNT * this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) / sizeof(float);

    if (this->partOutD * this->partOutH > 1 && onlyRow) {
        maxTmp = transDataTensorUb2;
        curBuf = transDataTensorUb2[curBufOffset * this->partOutW];
    } else {
        maxTmp = this->transDataTensorUb1;
        curBuf = transDataTensorUb2;
    }
    
    LocalTensor<uint8_t> maskIndex = maxTmp.template ReinterpretCast<uint8_t>();
	LocalTensor<uint8_t> nanMask = maskIndex[this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) * rowLen];
    
    // TransDataTo5HD (NCDHW -> NC1DHWC0)
    this->PrepareInput(transDataTensorUb2, this->partRoundDhwInp);
    
    // Add padding to data if needed
    if (pDL != 0 || pDR != 0 || pHL != 0 || pHR != 0 || pWL != 0 || pWR != 0) {
        uint32_t partDWoPad = this->partD - pDL - pDR;
        uint32_t partHWoPad = this->partH - pHL - pHR;
        uint32_t partWWoPad = this->partW - pWL - pWR;
        if (tailOutD >= 1 && partOutReal == tailOutD * this->partOutH * this->partOutW) {
            partDWoPad = tailD - pDL - pDR;
        }
        Padding(this->transDataTensorUb1, transDataTensorUb2, partDWoPad, partHWoPad, partWWoPad, pDL, pDR, pHL, pHR, pWL, pWR);
    }

    // Im2ColC0 
    uint32_t srcGmOffset = 0;
    uint32_t dhVal = 0;
    LocalTensor<T> reduceMaxResult = this->queOutVals.template AllocTensor<T>();
    uint32_t alignedRowlen = this->RoundUpBlock((rowLen * this->blockLength) / BITS_UINT8, BLOCK_LEN_UINT8) * this->blockLength;
    LocalTensor<float> idxs;
    LocalTensor<T> idxsT;
    if (!onlyRow){
        for (uint32_t curD = 0, srcGmOffset = 0, dstOff = 0; curD < this->partOutD; srcGmOffset += this->gmDOff, curD++) {
            for (uint32_t curHeightOut = 0; curHeightOut < this->partOutH; srcGmOffset += this->gmHOff, curHeightOut++, dstOff += this->partOutW * this->blockKernelDHW) {
                if ((this->dW == 1) && (this->dH == 1) && (this->kW <= BLOCKS_IN_REP)){
                    if(this->partOutW > this->kH){
                        this->Im2ColNoDilationPerRepeat(curBuf[dstOff], srcGmOffset);
                    } else {
                        this->Im2ColNoDilation(curBuf[dstOff], srcGmOffset);
                    }
                } else if ((this->dW == 1) && (this->dH == 1)){
                    this->Im2ColNoDilationBigKw(curBuf[dstOff], srcGmOffset);
                } else {
                    this->Im2ColHWDDilation(curBuf[dstOff], srcGmOffset);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        // Find Max for each row by width
        if (useRepeat){
            this->FindMaxRowPerRepeat(curBuf, reduceMaxResult, maxTmp, 0, rowLen);
        } else {
            this->FindMaxRowPerKernel(curBuf, reduceMaxResult, maxTmp, 0, rowLen);
        }

        idxs = this->queOutInds.template AllocTensor<float>();
        idxsT = idxs.ReinterpretCast<T>();

        Duplicate(idxsT, (T)1., rowLen * this->blockLength);
        // Find Indices for each row by width
        if (!this->kernelIsBlock){
            this->CompareSelect(reduceMaxResult, curBuf, maskIndex, nanMask, rowLen, 0);
        } else {
            this->CompareSelectBlockKernel(reduceMaxResult, curBuf, maskIndex, nanMask, rowLen, 0);
        }

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

        this->CastInds(idxs, maxTmp, 0, rowLen);
    } else {
        bool allocated = false;
        for (uint32_t curD = 0, srcGmOffset = 0; curD < this->partOutD; srcGmOffset += this->gmDOff, curD++) {
            for (uint32_t curHeightOut = 0; curHeightOut < this->partOutH; srcGmOffset += this->gmHOff, curHeightOut++) {
                if ((this->dW == 1) && (this->dH == 1) && (this->kW <= BLOCKS_IN_REP)){
                    this->Im2ColNoDilation(curBuf, srcGmOffset);
                } else if ((this->dW == 1) && (this->dH == 1)){
                    this->Im2ColNoDilationBigKw(curBuf, srcGmOffset);
                } else {
                    this->Im2ColHWDDilation(curBuf, srcGmOffset);
                }
                PipeBarrier<PIPE_V>();

                // Find Max for each row by width
                this->FindMaxRowPerKernel(curBuf, reduceMaxResult, maxTmp, dhVal, rowLen);

                if(!allocated){
                    idxs = this->queOutInds.template AllocTensor<float>();
                    allocated = true;
                }
                idxsT = idxs[dhVal].ReinterpretCast<T>();

                Duplicate(idxsT, (T)1., rowLen * this->blockLength);
                // Find Indices for each row by width
                if (!this->kernelIsBlock){
                    this->CompareSelect(reduceMaxResult, curBuf, maskIndex, nanMask, rowLen, dhVal);
                } else {
                    this->CompareSelectBlockKernel(reduceMaxResult, curBuf, maskIndex, nanMask, rowLen, dhVal);
                }

                Compare<T>(maskIndex, reduceMaxResult[dhVal], reduceMaxResult[dhVal], CMPMODE::EQ, alignedRowlen);
                PipeBarrier<PIPE_V>();

                Select<T>(idxsT, maskIndex, idxsT, -1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedRowlen);
                PipeBarrier<PIPE_V>();

                this->FindMinRowPerKernel(curBuf, maxTmp, rowLen);

                Mul<T>(maxTmp, maxTmp, idxsT, rowLen * this->blockLength);  
                PipeBarrier<PIPE_V>();

                this->CastInds(idxs, maxTmp, dhVal, this->partOutW);
                dhVal += this->partOutW * this->blockLength;
            }
        }
    }
    this->template TransposeBack<S>(reduceMaxResult, idxs, partNC, outputOffset, partOutReal);

    // TransDataTo5HD (NC1DHWC0 -> NCDHW) for indices
    LocalTensor<float> transDataTensorUb3 = this->transDataTensorUb1. template ReinterpretCast<float>();

    // Recount indeces
    auto tmpSize = this->RoundUpBlock(this->roundPartOutSize, BLOCK_LEN_FP32);
    uint32_t ncPartOutTrans = partNC * this->transAlignRoundPartOutSize;
    LocalTensor<float> tmpIdxs = transDataTensorUb2.template ReinterpretCast<float>();
    
    this->IndexRecalcFirst(transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);

    this->IndexRecalcSecond(transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);
    
    LocalTensor<int> generalOffsetsTmp = this->queOutVals.template AllocTensor<int>();
    int generalOffset = (offD + this->pD) * this->hwInputSize;

    LocalTensor<int> transDataTensorUb3Int = transDataTensorUb3. template ReinterpretCast<int>();
    LocalTensor<int> idxsInt = idxs. template ReinterpretCast<int>();
    LocalTensor<int> tmpIdxsInt = tmpIdxs. template ReinterpretCast<int>();

    Cast(transDataTensorUb3Int, transDataTensorUb3, RoundMode::CAST_RINT, ncPartOutTrans);
    Cast(idxsInt, idxs, RoundMode::CAST_RINT, ncPartOutTrans);
    Cast(tmpIdxsInt, tmpIdxs, RoundMode::CAST_RINT, ncPartOutTrans);
    PipeBarrier<PIPE_V>();
    
    Muls(transDataTensorUb3Int, transDataTensorUb3Int, int(this->dD * this->hwInputSize), ncPartOutTrans);
    Muls(idxsInt, idxsInt, int(this->dH * this->widthInput), ncPartOutTrans);
    Adds(generalOffsetsTmp, this->dKernelStart, generalOffset, tmpSize);
    PipeBarrier<PIPE_V>();
  
    this->IndexRecalcThird(generalOffsetsTmp, this->hKernelStart, this->wKernelStart, transDataTensorUb3Int, idxsInt, tmpIdxsInt, idxs, ncPartOutTrans);

    this->queIn.template FreeTensor<S>(transDataTensorUb2Old);	
	this->queOutVals.template FreeTensor<int>(generalOffsetsTmp);
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2SplitD<T,S>::CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, const uint32_t& part) {
    auto dhwInp = part * this->hwInputSize;
    auto downSize = this->RoundDownBlock(dhwInp, this->blockLengthS);
    auto roundSize = this->RoundUpBlock(this->partRoundDhwInp, this->blockLengthS);
    LocalTensor<S> srcUb = this->queIn.template AllocTensor<S>();
    
    if (this->inputSize % this->blockLengthS == 0 && dhwInp == downSize){
        if (((this->inputSize - downSize) / this->blockLengthS) < MAX_UINT16) {
            DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>(downSize/ this->blockLengthS), 
                                  static_cast<uint16_t>((this->inputSize - downSize) / this->blockLengthS), 
                                  static_cast<uint16_t>((roundSize - downSize) / this->blockLengthS)};
            DataCopy(srcUb, srcGm, params);
        } else {
            DataCopyParams params{static_cast<uint16_t>(1), 
                                  static_cast<uint16_t>(dhwInp / this->blockLengthS), 0, 0};
            for (uint32_t nc = 0; nc < partNC; nc++) {
                DataCopy(srcUb[roundSize * nc], srcGm[nc * this->inputSize], params);
            }
        }
    } else {
        DataCopyExtParams params{static_cast<uint16_t>(partNC), static_cast<uint32_t>(dhwInp * sizeof(S)), 
                                 static_cast<uint32_t>((this->inputSize - dhwInp) * sizeof(S)), 
                                 static_cast<uint32_t>((roundSize - dhwInp) / this->blockLengthS), 0};
        DataCopyPadExtParams<S> padParams{false, 0, 0, 0};
        DataCopyPad(srcUb, srcGm, params, padParams);
    }
    this->queIn.template EnQue<S>(srcUb);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2SplitD<T,S>::Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb,
                                                                const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad, 
                                                                const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    this->PaddingW(srcUb, dstUb, partDWoPad, partHWoPad, partWWoPad, pWL, pWR);

    this->PaddingH(srcUb, dstUb, partDWoPad, partHWoPad, pHL, pHR);

    this->PaddingD(srcUb, dstUb, partDWoPad, pDL, pDR);
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2SplitD<T,S>::CreateGeneralOffsets() {
    // Brcb sync
    PipeBarrier<PIPE_V>(); 
    auto event0 = this->pipe->FetchEventID(HardEvent::S_V); 
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);  
    
    // Calcucate generalOffsets
    auto partNC = this->batchesCurCore % this->blockLength;
    auto tmpSize = this->RoundUpBlock(this->roundPartOutSize, BLOCK_LEN_FP32);
    uint32_t ncPartOutTrans = partNC * this->transAlignRoundPartOutSize;
    
    LocalTensor<int> dTmp = this->dKernelStart;
    LocalTensor<int> hTmp = this->hKernelStart;
    LocalTensor<int> wTmp = this->wKernelStart;
    
    uint32_t i = 0;
    int32_t offD = -this->pD;
    int32_t offH = -this->pH;
    int32_t offW = -this->pW;

    if ((this->partOutD == 1) && (this->partOutW % BLOCK_LEN_FP32 == 0) && (sizeof(T) == BLOCK_SIZE / BLOCK_LEN_FP32)) {
        Duplicate<int>(dTmp, offD, tmpSize);
        ArithProgression(wTmp, offW, (int)this->sW, this->partOutW);
        for (int32_t curH = offH; curH < offH + (int)this->partOutH * (int)this->sH; curH+=(int)this->sH) {
            Duplicate<int>(hTmp[i], curH, this->partOutW);
            DataCopy(wTmp[i], wTmp, this->partOutW);
            i+=this->partOutW;
        }
    } else {
        for (int32_t curD = offD; curD < offD + (int)this->partOutD * (int)this->sD; curD+=(int)this->sD) {
            for (int32_t curH = offH; curH < offH + (int)this->partOutH * (int)this->sH; curH+=(int)this->sH) {
                for (int32_t curW = offW; curW < offW + (int)this->partOutW * (int)this->sW; curW+=(int)this->sW) {
                    dTmp.SetValue(i, curD);
                    hTmp.SetValue(i, curH);
                    wTmp.SetValue(i, curW);
                    i++;
                }
            }
        }
    }
    
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);
    event0 = this->pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
    
    Muls(hTmp, hTmp, int(this->widthInput), tmpSize);
    Muls(dTmp, dTmp, int(this->hwInputSize), tmpSize);
    PipeBarrier<PIPE_V>();
}
    
#endif
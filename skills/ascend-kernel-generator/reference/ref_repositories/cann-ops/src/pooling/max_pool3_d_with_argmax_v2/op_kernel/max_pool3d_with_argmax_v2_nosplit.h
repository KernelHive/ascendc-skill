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
 * \file max_pool3d_with_argmax_v2_nosplit.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_WITH_ARGMAX_V2_NOSPLIT_H
#define OPP_MAX_POOL3D_WITH_ARGMAX_V2_NOSPLIT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_v2_base.h"

using namespace AscendC;

template <typename T, typename S>
class KernelMaxPool3DWithArgmaxV2NoSplit : public KernelMaxPool3DWithArgmaxV2Base<T, S> {
  public:
    __aicore__ KernelMaxPool3DWithArgmaxV2NoSplit(const MaxPool3DWithArgmaxV2NoSplitTilingData* __restrict tilingData_) :
                                                  KernelMaxPool3DWithArgmaxV2Base<T, S>(tilingData_->inputShapes[D_DIM], tilingData_->inputShapes[H_DIM], tilingData_->inputShapes[W_DIM],
                                                                                        tilingData_->outShapes[D_DIM], tilingData_->outShapes[H_DIM], tilingData_->outShapes[W_DIM],
                                                                                        tilingData_->kD, tilingData_->kH, tilingData_->kW,
                                                                                        tilingData_->sD, tilingData_->sH, tilingData_->sW,
                                                                                        tilingData_->pD, tilingData_->pH, tilingData_->pW,
                                                                                        tilingData_->dD, tilingData_->dH, tilingData_->dW,
                                                                                        tilingData_->partD, tilingData_->partH, tilingData_->partW,
                                                                                        tilingData_->minFloat, tilingData_->batchesPerCore, tilingData_->leftOverBatches,
                                                                                        tilingData_->ceilD, tilingData_->ceilH, tilingData_->ceilW,
                                                                                        tilingData_->sizeUb1, tilingData_->sizeUb2, tilingData_->sizeValues){}
    
    __aicore__ void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe *pipe) {
        this->pipe = pipe;
        this->GMInit(x, y, indices, workspace);
        
        PrepareScalars();
        
        this->UbInit();
        auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);
    }
    
    __aicore__ void Process() {
        if ((g_coreType != AIV) || (this->batchesCurCore == 0)) {
            return;
        }
        MaxPool3DWithArgmaxV2Algorithm();
    }

  private:
    __aicore__ inline void DoMaxPool(int32_t dOutOff, int32_t hOutOff, 
                                     uint32_t partNC, 
                                     uint32_t inputOffset, uint32_t outputOffset,
                                     uint32_t padDL, uint32_t padDR,
                                     uint32_t padHL, uint32_t padHR,
                                     uint32_t padWL, uint32_t padWR){
        CopyIn(this->xGm[inputOffset], partNC, padDL, padDR, padHL, padHR, padWL, padWR);
    
        Img2ColC0(partNC, dOutOff, hOutOff, -this->pW, outputOffset, padDL, padDR, padHL, padHR, padWL, padWR); 

        CopyOutInds(outputOffset, partNC);
    }
    
    __aicore__ inline void CopyOutInds(const uint32_t& dstOffset, const uint32_t& partNC){
        const uint32_t partOutSize = this->roundPartOutSize;
        const uint32_t partOutSizeTrans = this->RoundUpBlock(partOutSize, MIN_TRANSPOSE_ROWS);
        LocalTensor<float> idxs = this->queOutInds.template DeQue<float>();
        
        if (partOutSizeTrans == partOutSize){
            DataCopy(this->indicesGm[dstOffset], idxs, partNC * partOutSize);
        } else {
            DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>((partOutSize) * sizeof(float)),
                                  static_cast<uint16_t>((partOutSizeTrans - partOutSize) / BLOCK_LEN_FP32), 0};
            DataCopyPad(this->indicesGm[dstOffset], idxs, params);
        }
        auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(event0);
        WaitFlag<HardEvent::MTE3_S>(event0);
        this->queOutInds.template FreeTensor<float>(idxs);
    }

    __aicore__ inline void PrepareScalars(){
        this->alContOutSize = this->roundPartOutSize;
        this->roundW = this->partW;
        this->partHwInp = this->roundW * this->partH;
        
        this->PrepareBaseScalars();
    }
    
    template <typename V>
    __aicore__ inline void TransposeBack(LocalTensor<half> &reduceMaxResult, LocalTensor<float> &idxs, uint32_t partNC, uint32_t outputOffset){
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

        this->TransposeBackVals(reduceMaxResult, partNC, dstLocalList, srcLocalList);
    
        CopyOutVals(outputOffset, partNC);
        
        this->TransposeBackIndsHalf(idxs, dstLocalList, srcLocalList);
    }
    
    template <typename V>
    __aicore__ inline void TransposeBack(LocalTensor<float> &reduceMaxResult, LocalTensor<float> &idxs, uint32_t partNC, uint32_t outputOffset){
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        
        this->template TransposeBackVals<V>(reduceMaxResult, partNC, dstLocalList, srcLocalList);
   
        CopyOutVals(outputOffset, partNC);
        
        this->TransposeBackInds(idxs, dstLocalList, srcLocalList);
    }

    __aicore__ inline void IndexRecalcNoSplit(LocalTensor<float> &dTmp, LocalTensor<float> &hTmp,
                                              LocalTensor<float> &dTensor, LocalTensor<float> &hTensor, 
                                              LocalTensor<float> &wTensor, uint32_t len) {
	    const auto tmpSize = this->RoundUpBlock(this->roundPartOutSize, BLOCK_LEN_FP32);
        for (uint32_t curChannel = 0; curChannel < len; curChannel+= this->transAlignRoundPartOutSize) {
            Add(dTensor[curChannel], dTensor[curChannel], dTmp, tmpSize);
            Add(hTensor[curChannel], hTensor[curChannel], hTmp, tmpSize);
        }
        PipeBarrier<PIPE_V>();
        auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);
    
        this->queOutVals.template FreeTensor<float>(dTmp);
    
        Maxs(dTensor, dTensor, 0.f, len);
        Maxs(hTensor, hTensor, 0.f, len);
        PipeBarrier<PIPE_V>();

        Add(wTensor, wTensor, hTensor, len);
        PipeBarrier<PIPE_V>();
        
        Add(dTensor, wTensor, dTensor, len);
        PipeBarrier<PIPE_V>();

        LocalTensor<int> indicesTmpLongInt = hTensor.ReinterpretCast<int>();
        Cast(indicesTmpLongInt, dTensor, RoundMode::CAST_RINT, len);
        PipeBarrier<PIPE_V>();
        event0 = this->pipe->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(event0);
        WaitFlag<HardEvent::V_S>(event0);
        
        this->queOutInds.template EnQue<float>(hTensor);
    }
    
    // Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
    __aicore__ void MaxPool3DWithArgmaxV2Algorithm(void);
    __aicore__ void Img2ColC0(const uint32_t partNC, const int32_t offD, const int32_t offH, const int32_t offW,
                              const uint32_t outputOffset,
                              const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
    __aicore__ inline void CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, const uint32_t pDL, const uint32_t pDR, 
                                  const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
    __aicore__ inline void CopyOutVals(const uint32_t& dstOffset, const uint32_t& partNC);
    
    __aicore__ void Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb,
                            const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad,
                            const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
};

// Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2NoSplit<T,S>::MaxPool3DWithArgmaxV2Algorithm(void) {
    this->CreateKernelIndexes();
    
    auto partNC = this->batchesCurCore % this->blockLength;
    auto roundBatches = this->batchesCurCore - partNC;

    for (int b=0; b<roundBatches; b+=this->blockLength){
        DoMaxPool(-this->pD, -this->pH, this->blockLength, b * this->inputSize, b * this->outSize, 
                  this->pD, this->ceilD, this->pH, this->ceilH, this->pW, this->ceilW);
    }

    if (partNC != 0){
        DoMaxPool(-this->pD, -this->pH, partNC, roundBatches * this->inputSize, roundBatches * this->outSize, 
                  this->pD, this->ceilD, this->pH, this->ceilH, this->pW, this->ceilW);
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2NoSplit<T,S>::Img2ColC0(const uint32_t partNC, const int32_t offD, const int32_t offH, const int32_t offW,
                                                                   const uint32_t outputOffset, const uint32_t pDL, const uint32_t pDR, 
                                                                   const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    LocalTensor<T> curBuf;
    LocalTensor<T> maxTmp;
    
    LocalTensor<S> transDataTensorUb2Old = this->queIn.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2 = transDataTensorUb2Old. template ReinterpretCast<T>();

    // Cast for bf16 using transDataTensorUb1
    this->CastBF16(transDataTensorUb2Old, transDataTensorUb2, partNC);
    uint32_t curBufOffset = (this->maxTmpOff > MASK_COUNT * this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) / sizeof(float)) ?
                             this->maxTmpOff : MASK_COUNT * this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) / sizeof(float);

    if (this->partOutD * this->partOutH > 1) {
        maxTmp = transDataTensorUb2;
        curBuf = transDataTensorUb2[curBufOffset * this->partOutW];
    } else {
        maxTmp = this->transDataTensorUb1;
        curBuf = transDataTensorUb2;
    }
    
    LocalTensor<uint8_t> maskIndex = maxTmp.template ReinterpretCast<uint8_t>();
	LocalTensor<uint8_t> nanMask = maskIndex[this->RoundUpBlock(this->blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8) * this->partOutW];
    
    // TransDataTo5HD (NCDHW -> NC1DHWC0)
    this->PrepareInput(transDataTensorUb2, this->partRoundDhwInp);
    
    // Add padding to data if needed
    if (pDL != 0 || pDR != 0 || pHL != 0 || pHR != 0 || pWL != 0 || pWR != 0) {
        uint32_t partDWoPad = this->partD - pDL - pDR;
        uint32_t partHWoPad = this->partH - pHL - pHR;
        uint32_t partWWoPad = this->partW - pWL - pWR;
        Padding(this->transDataTensorUb1, transDataTensorUb2, partDWoPad, partHWoPad, partWWoPad, pDL, pDR, pHL, pHR, pWL, pWR);
    }

    // Im2ColC0 
    uint32_t srcGmOffset = 0;
    uint32_t dhVal = 0;
    LocalTensor<T> reduceMaxResult = this->queOutVals.template AllocTensor<T>();
    LocalTensor<float> idxs = this->queOutInds.template AllocTensor<float>();
    uint32_t alignedRowlen = this->RoundUpBlock((this->partOutW * this->blockLength) / BITS_UINT8, BLOCK_LEN_UINT8) * this->blockLength;
    LocalTensor<T> idxsT;

    for (uint32_t curD = 0, srcGmOffset = 0; curD < this->partOutD; srcGmOffset += this->gmDOff, curD++) {
        for (uint32_t curHeightOut = 0; curHeightOut < this->partOutH; srcGmOffset += this->gmHOff, curHeightOut++) {
            if((this->dW != 1) || (this->dH != 1)) {
                this->Im2ColHWDDilation(curBuf, srcGmOffset);
            } else {
                if (this->kW <= BLOCKS_IN_REP){
                    if(this->partOutW > this->kH){
                        this->Im2ColNoDilationPerRepeat(curBuf, srcGmOffset);
                    } else {
                        this->Im2ColNoDilation(curBuf, srcGmOffset);
                    }
                } else {
                    this->Im2ColNoDilationBigKw(curBuf, srcGmOffset);
                }
            }
            PipeBarrier<PIPE_V>();

            // Find Max for each row by width
            this->FindMaxRowPerKernel(curBuf, reduceMaxResult, maxTmp, dhVal, this->partOutW);

            idxsT = idxs[dhVal].ReinterpretCast<T>();
            Duplicate(idxsT, (T)1., this->partOutW * this->blockLength);
            // Find Indices for each row by width
            this->CompareSelect(reduceMaxResult, curBuf, maskIndex, nanMask, this->partOutW, dhVal);

            Compare<T>(maskIndex, reduceMaxResult[dhVal], reduceMaxResult[dhVal], CMPMODE::EQ, alignedRowlen);
            PipeBarrier<PIPE_V>();

            Select<T>(idxsT, maskIndex, idxsT, -1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedRowlen);
            PipeBarrier<PIPE_V>();

            this->FindMinRowPerKernel(curBuf, maxTmp, this->partOutW);

            Mul<T>(maxTmp, maxTmp, idxsT, this->partOutW * this->blockLength);  
            PipeBarrier<PIPE_V>();
            
            this->CastInds(idxs, maxTmp, dhVal, this->partOutW);
            dhVal += this->partOutW * this->blockLength;
        }
    }
    this->template TransposeBack<S>(reduceMaxResult, idxs, partNC, outputOffset);
    
    // TransDataTo5HD (NC1DHWC0 -> NCDHW) for indices
    LocalTensor<float> transDataTensorUb3 = this->transDataTensorUb1. template ReinterpretCast<float>();
    
    // Recount indeces
    const auto tmpSize = this->RoundUpBlock(this->roundPartOutSize, BLOCK_LEN_FP32);
    uint32_t ncPartOutTrans = partNC * this->transAlignRoundPartOutSize;
    LocalTensor<float> tmpIdxs = transDataTensorUb2.template ReinterpretCast<float>();
        
    this->IndexRecalcFirst(transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);

    LocalTensor<float> generalOffsets = this->queOutVals.template AllocTensor<float>();
    LocalTensor<float> hTmp = generalOffsets[tmpSize]; 
    LocalTensor<float> wTmp = hTmp[tmpSize];
    float coeffH = 1.f / float(int(this->kW));
    float coeffD = coeffH / float(int(this->kH));
    uint32_t i = 0;
    for (int32_t curD = offD; curD < offD + (int)this->partOutD * (int)this->sD; curD+=(int)this->sD) {
        for (int32_t curH = offH; curH < offH + (int)this->partOutH * (int)this->sH; curH+=(int)this->sH) {
            for (int32_t curW = offW; curW < offW + (int)this->partOutW * (int)this->sW; curW+=(int)this->sW) {
                generalOffsets.SetValue(i, (float)curD);
                hTmp.SetValue(i, (float)curH);
                wTmp.SetValue(i, (float)curW);
                i++;
            }
        }
    }
    auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
    
    Muls(transDataTensorUb3, transDataTensorUb3, coeffD, ncPartOutTrans);
    Muls(hTmp, hTmp, float(int(this->widthInput)), tmpSize);
    Muls(generalOffsets, generalOffsets, float(int(this->hwInputSize)), tmpSize);
    Muls(idxs, idxs, coeffH, ncPartOutTrans);
    Muls(tmpIdxs, tmpIdxs, float(int(this->dW)), ncPartOutTrans);
    PipeBarrier<PIPE_V>();

    Floor(transDataTensorUb3, transDataTensorUb3, ncPartOutTrans);
    Floor(idxs, idxs, ncPartOutTrans);
    for (uint32_t curChannel = 0; curChannel < ncPartOutTrans; curChannel+= this->transAlignRoundPartOutSize) {
        Add(tmpIdxs[curChannel], tmpIdxs[curChannel], wTmp, tmpSize);
    }
    PipeBarrier<PIPE_V>();
    
    Muls(transDataTensorUb3, transDataTensorUb3, float(int(this->dD * this->hwInputSize)), ncPartOutTrans);
    Muls(idxs, idxs, float(int(this->dH * this->widthInput)), ncPartOutTrans);
    Maxs(tmpIdxs, tmpIdxs, 0.f, ncPartOutTrans);
    PipeBarrier<PIPE_V>();
  
	IndexRecalcNoSplit(generalOffsets, hTmp, transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);

    this->queIn.template FreeTensor<S>(transDataTensorUb2Old);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2NoSplit<T,S>::CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC,
                                                                const uint32_t pDL, const uint32_t pDR,
                                                                const uint32_t pHL, const uint32_t pHR,
                                                                const uint32_t pWL, const uint32_t pWR) {
    uint32_t partInputSize = partNC * this->partRoundDhwInp;
    LocalTensor<S> srcUb = this->queIn.template AllocTensor<S>();
    
    if (this->partRoundDhwInp % this->blockLengthS == 0 && !(pDL != 0 || pDR != 0 || pHL != 0 || pHR != 0 || pWL != 0 || pWR != 0)) {
        DataCopy(srcUb, srcGm, partInputSize);
    } else {
        // Copy data without padding but between channels can be free space
        uint32_t partRoundDhwInpWoPad = this->inputSize; 
        
        DataCopyParams copyPadParamsInput{static_cast<uint16_t>(partNC), static_cast<uint16_t>(partRoundDhwInpWoPad * sizeof(S)), 
                                          0, static_cast<uint16_t>((this->RoundUpBlock(this->partRoundDhwInp, this->blockLengthS) - partRoundDhwInpWoPad) / this->blockLengthS)};
        DataCopyPadParams paddingParamsInput{false, 0, 0, 0};
        DataCopyPad(srcUb, srcGm, copyPadParamsInput, paddingParamsInput);
    }
    this->queIn.template EnQue<S>(srcUb);
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2NoSplit<T,S>::CopyOutVals(const uint32_t& dstOffset, const uint32_t& partNC) {
    const uint32_t partOutSizeTrans = this->RoundUpBlock(this->roundPartOutSize, MIN_TRANSPOSE_ROWS);
    LocalTensor<S> reduceMaxResult = this->queOutVals.template DeQue<S>();

    if (partOutSizeTrans == this->roundPartOutSize){
        DataCopy(this->yGm[dstOffset], reduceMaxResult, partNC * this->roundPartOutSize);
    } else {
        DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>(this->roundPartOutSize * sizeof(S)), 
                              static_cast<uint16_t>((partOutSizeTrans - this->roundPartOutSize) / this->blockLengthS), 
                              0};
        DataCopyPad(this->yGm[dstOffset], reduceMaxResult, params);
    }
    auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
    SetFlag<HardEvent::MTE3_S>(event0);
    WaitFlag<HardEvent::MTE3_S>(event0);

    this->queOutVals.template FreeTensor<S>(reduceMaxResult);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2NoSplit<T,S>::Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb,
                                                                 const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad,
                                                                 const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    this->PaddingW(srcUb, dstUb, partDWoPad, partHWoPad, partWWoPad, pWL, pWR);

    this->PaddingH(srcUb, dstUb, partDWoPad, partHWoPad, pHL, pHR);

    this->PaddingD(srcUb, dstUb, partDWoPad, pDL, pDR);
}

#endif
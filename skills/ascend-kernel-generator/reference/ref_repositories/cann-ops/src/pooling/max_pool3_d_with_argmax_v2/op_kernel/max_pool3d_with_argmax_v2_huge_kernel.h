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
 * \file max_pool3d_with_argmax_v2_huge_kernel.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_WITH_ARGMAX_V2_HUGE_KERNEL_H
#define OPP_MAX_POOL3D_WITH_ARGMAX_V2_HUGE_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_v2_base.h"

using namespace AscendC;

template <typename T, typename S>
class KernelMaxPool3DWithArgmaxV2HugeKernel : public KernelMaxPool3DWithArgmaxV2Base<T, S> {
  public:
    __aicore__ KernelMaxPool3DWithArgmaxV2HugeKernel(const MaxPool3DWithArgmaxV2HugeKernelTilingData* __restrict tilingData_) :
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

    __aicore__ /*inline*/ void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe *pipe) {
        this->pipe = pipe;
        this->GMInit(x, y, indices, workspace);
        
        indexStart = ((this->kD + this->partD - 1) / this->partD) * ((this->kH + this->partH - 1) / this->partH) * ((this->kW + this->partW - 1) / this->partW) * this->blockLength + this->blockLength;
        wsGm.SetGlobalBuffer(((__gm__ T*)workspace + 2 * indexStart * this->coreIdx), 2 * indexStart);
        
        PrepareScalars();
        
        this->pipe->InitBuffer(this->queIn, 1, this->sizeUb2 * sizeof(float));
        this->pipe->InitBuffer(this->queOutVals, 1, this->valSize * sizeof(T));
        this->pipe->InitBuffer(this->queOutInds, 1, this->valSize * sizeof(float));
        
        TBuf<TPosition::VECCALC> worktBuf;
        
        // Both sizeUb1 and blockAlKernelDHW are power of blockLength
        auto maxUb = (this->partD * this->partH * this->partW * this->blockLength + this->sizeUb1) * sizeof(float);
        this->pipe->InitBuffer(worktBuf, maxUb);
        this->transDataTensorUb1 = worktBuf.template Get<T>();
        this->kernelIndexes = this->transDataTensorUb1[this->sizeUb1];
        
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
    __aicore__ inline void DoMaxPool(uint32_t partNC, 
                                     uint32_t partD, uint32_t partH, uint32_t partW,
                                     uint32_t inputOffset, uint32_t wsOffset,
                                     uint32_t padDL, uint32_t padDR,
                                     uint32_t padHL, uint32_t padHR,
                                     uint32_t padWL, uint32_t padWR){
        CopyIn(this->xGm[inputOffset], partNC, partD, partH, partW, padDL, padDR, padHL, padHR, padWL, padWR);
        Img2ColC0(partNC, partD, partH, partW, wsOffset, padDL, padDR, padHL, padHR, padWL, padWR);
    }
    
    __aicore__ inline void CopyOutInds(const uint32_t& dstOffset, const uint32_t& partNC){
        LocalTensor<float> idxs = this->queOutInds.template DeQue<float>();
        if (this->outSize * sizeof(T) < MAX_UINT16){
            DataCopyPad(this->indicesGm[dstOffset], idxs, 
                        {static_cast<uint16_t>(partNC), static_cast<uint16_t>(1 * sizeof(int)), 
                         (BLOCK_LEN_FP32 - 1) / BLOCK_LEN_FP32, 
                         static_cast<uint16_t>((this->outSize - 1)*sizeof(int))});
        } else {
            for (uint32_t b = 0; b < partNC; b++){
                DataCopyPad(this->indicesGm[(dstOffset + b * this->outSize)], idxs[b * this->blockLength], 
                            {static_cast<uint16_t>(1), static_cast<uint16_t>(1 * sizeof(int)), 0, 0});
            }
        }
        auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(event0);
        WaitFlag<HardEvent::MTE3_S>(event0);
        this->queOutInds.template FreeTensor<float>(idxs);
    }

    __aicore__ inline void PrepareScalars(){
        this->roundDepthInput = this->sD * (this->depthOut - 1) + 1;
        this->roundHeightInput = this->sH * (this->heightOut - 1) + 1;
        this->roundWidthInput = this->sW * (this->widthOut - 1) + 1;
        
        // length of cycle in kernel split
        auto partDWoDil = ((this->partD - 1) / this->dD + 1);
        auto partHWoDil = ((this->partH - 1) / this->dH + 1);
        auto partWWoDil = ((this->partW - 1) / this->dW + 1);
        dCycleMax = this->kD / partDWoDil + (this->kD % partDWoDil != 0 ? 1 : 0);
        hCycleMax = this->kH / partHWoDil + (this->kH % partHWoDil != 0 ? 1 : 0);
        wCycleMax = this->kW / partWWoDil + (this->kW % partWWoDil != 0 ? 1 : 0);

        this->roundW = this->RoundUpBlock(this->partW, this->blockLengthS);
        
        this->partHwInp = this->roundW * this->partH;
        
        this->PrepareBaseScalars();

        this->transDataParams.repeatTimes = this->RoundUpBlock(this->partD * this->partH * this->roundW, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.dstRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        this->transDataParams.srcRepStride = (this->transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / this->blockLength;
    }
    
    template <typename V>
    __aicore__ inline void TransposeBack(LocalTensor<half> &reduceMaxResult, LocalTensor<float> &idxs, uint32_t partNC, uint32_t outputOffset){
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

        this->template TransposeBackVals<V>(reduceMaxResult, partNC, dstLocalList, srcLocalList);
    
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

    __aicore__ inline void CreateKernelIndexes(uint32_t indStart) {
        ArithProgression<T>(this->transDataTensorUb1, (T)(1.f * indStart), (T)(1.f), this->RoundUpBlock(this->partW));
        PipeBarrier<PIPE_V>();
        auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);

        Brcb(this->kernelIndexes, this->transDataTensorUb1, this->RoundUpBlock(this->partW, BLOCKS_IN_REP) / BLOCKS_IN_REP, {1, BLOCKS_IN_REP});
        PipeBarrier<PIPE_V>();
        
        for(uint32_t i = 1; i < this->partH; i++){
            Adds(this->kernelIndexes[this->partW * this->blockLength * i], this->kernelIndexes, (T)(1.f * i * this->kW), this->partW * this->blockLength);
        }
        PipeBarrier<PIPE_V>();
        
        for(uint32_t i = 1; i < this->partD; i++){
            Adds(this->kernelIndexes[this->partH * this->partW * this->blockLength * i], this->kernelIndexes, (T)(1.f * i * this->kW * this->kH), this->partH * this->partW * this->blockLength);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename V>
    __aicore__ inline void CastBF16(LocalTensor<V> src, LocalTensor<float> dst, uint32_t partNC) {
        if (sizeof(V) != sizeof(float)) {
            const uint32_t partAlignDhwInp = partNC * this->RoundUpBlock(this->partRoundDhwInp, this->blockLengthS);
            const uint32_t repeatTimes = partAlignDhwInp / this->mask;
            const uint32_t wholeRepeatTimes = repeatTimes / MAX_REPEAT_TIMES;
            const uint32_t tailRepeatTimes = repeatTimes % MAX_REPEAT_TIMES;
            const uint32_t tailElemsPerRepeat = partAlignDhwInp % this->mask;

            const uint32_t wholeOffset = wholeRepeatTimes * MAX_REPEAT_TIMES * this->mask;
            const uint32_t tailOffset = wholeOffset + tailRepeatTimes * this->mask;

            Cast(this->transDataTensorUb1, src, RoundMode::CAST_NONE, partAlignDhwInp);
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(dst[i * MAX_REPEAT_TIMES * this->mask], this->transDataTensorUb1[i * MAX_REPEAT_TIMES * this->mask], this->mask, MAX_REPEAT_TIMES, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
            }
            Copy(dst[wholeOffset], this->transDataTensorUb1[wholeOffset], this->mask, tailRepeatTimes, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
            Copy(dst[tailOffset], this->transDataTensorUb1[tailOffset], tailElemsPerRepeat, 1, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename V>
    __aicore__ inline void CastBF16Back(LocalTensor<float> src, uint32_t partNC, bool isBF16){
        if (isBF16) {
            LocalTensor<V> bf16TmpTensor = src. template ReinterpretCast<V>(); 
            Cast(bf16TmpTensor, src, RoundMode::CAST_ROUND, this->blockLength * partNC);
            PipeBarrier<PIPE_V>();

            this->queOutVals.template EnQue<V>(bf16TmpTensor);
        } else {
            this->queOutVals.template EnQue<float>(src);
        }
    }
    // Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
    __aicore__ void MaxPool3DWithArgmaxV2Algorithm(void);
    __aicore__ void RemovingGaps(const LocalTensor<T>& dst, uint32_t partD, uint32_t partH, uint32_t partW);
    __aicore__ void Img2ColC0(const uint32_t partNC,
                              uint32_t partD, uint32_t partH, uint32_t partW,
                              const uint32_t outputOffset,
                              const uint32_t pDL, const uint32_t pDR,
                              const uint32_t pHL, const uint32_t pHR,
                              const uint32_t pWL, const uint32_t pWR);
    __aicore__  void FinalMax(uint32_t outputOffset, uint32_t partNC,
                              int32_t offD, int32_t offH, int32_t offW);
    __aicore__ inline void CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, 
                                  uint32_t partD, uint32_t partH, uint32_t partW,
                                  const uint32_t padDL, const uint32_t padDR,
                                  const uint32_t padHL, const uint32_t padHR,
                                  const uint32_t padWL, const uint32_t padWR);
    __aicore__ inline void CopyOutVals(const uint32_t& dstOffset, const uint32_t& partNC);
    
    __aicore__ void Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb, 
                            const uint32_t& partD, const uint32_t& partH, const uint32_t& partW,
                            const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad,
                            const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR);
                                   
    uint32_t dCycleMax;
    uint32_t hCycleMax;
    uint32_t wCycleMax;
    GlobalTensor<T> wsGm;
    uint32_t indexStart;
};

// Current implementation of MaxPool3DWithArgmaxV2 uses only Vector unit
template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::MaxPool3DWithArgmaxV2Algorithm(void) {
    auto hwOut = this->heightOut * this->widthOut;
    auto partNC = this->batchesCurCore % this->blockLength;
    auto roundBatches = this->batchesCurCore - partNC;

    uint32_t offD = this->sD;
    uint32_t offH = this->sH;
    uint32_t offW = this->sW;
    uint32_t d, h;
    // padDL - if we have left padding for this part
    // padDR - if we have right padding for this part (possibly ceilMode)
    // on the first itertion we have left padding (if pad have), on last right
    // in the middle no padding
    uint32_t padDL, padHL, padWL, padDR, padHR, padWR;
    uint32_t dOut=0, hOut=0, wOut=0;
    uint32_t w =0;
    uint32_t partCount = 0;
    for (int b=0; b < this->batchesCurCore; b+=this->blockLength) {
        dOut = 0;
        uint32_t batchLen = (b != roundBatches) ? this->blockLength : partNC;
		for (uint32_t d=0; d < this->roundDepthInput; d+=offD, dOut+=1) {
			hOut = 0;
			for (uint32_t h=0; h < this->roundHeightInput; h+=offH, hOut+=1) {
				wOut = 0;
				for (w=0; w < this->roundWidthInput; w+=offW, wOut+=1) {
					uint32_t outputOffset = b * this->outSize + dOut * hwOut + hOut * this->widthOut + wOut;
					Duplicate(this->transDataTensorUb1, (T)this->minFloat, indexStart);
					PipeBarrier<PIPE_V>();
					auto event0 = this->pipe->FetchEventID(HardEvent::V_MTE3);
					SetFlag<HardEvent::V_MTE3>(event0);
					WaitFlag<HardEvent::V_MTE3>(event0);

					DataCopy(wsGm, this->transDataTensorUb1, indexStart);
					event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
					SetFlag<HardEvent::MTE3_S>(event0);
					WaitFlag<HardEvent::MTE3_S>(event0);
                    
					for (uint32_t dKer = 0; dKer < dCycleMax; dKer++) {
                        uint32_t dInOff = d + dKer * (this->partD + this->dD - 1);
                        uint32_t curPartD = dKer != (dCycleMax - 1) ? this->partD : ((this->kD - 1) * this->dD + 1 - dKer * (this->partD + this->dD - 1));
			            padDL = ((int(this->pD - dInOff) > 0) && (this->pD != 0)) ? this->pD - dInOff : 0;
			            padDR = (((dInOff + curPartD) > (this->depthInput + this->pD)) && (this->ceilD != 0)) ? (dInOff + curPartD) - (this->depthInput + this->pD) : 0;
                        int32_t partDWoPad = (int32_t)curPartD - (int32_t)padDL - (int32_t)padDR;
						for (uint32_t hKer = 0; (hKer < hCycleMax) && (partDWoPad > 0); hKer++) {
                            uint32_t hInOff = h + hKer * (this->partH + this->dH - 1);
                            uint32_t curPartH = hKer != (hCycleMax - 1) ? this->partH : ((this->kH - 1) * this->dH + 1 - hKer * (this->partH + this->dH - 1));
                            padHL = ((int(this->pH - hInOff) > 0) && (this->pH != 0)) ? this->pH - hInOff : 0;
				            padHR = (((hInOff + curPartH) > (this->heightInput + this->pH)) && (this->ceilH != 0)) ? (hInOff + curPartH) - (this->heightInput + this->pH) : 0;
                            int32_t partHWoPad = (int32_t)curPartH - (int32_t)padHL - (int32_t)padHR;
							for (uint32_t wKer = 0; (wKer < wCycleMax) && (partHWoPad > 0); wKer++) {
                                uint32_t wInOff = w + wKer * (this->partW + this->dW - 1);
                                uint32_t curPartW = wKer != (wCycleMax - 1) ? this->partW : ((this->kW - 1) * this->dW + 1 - wKer * (this->partW + this->dW - 1));
                                padWL = ((int(this->pW - wInOff) > 0) && (this->pW != 0)) ? this->pW - wInOff : 0;
					            padWR = (((wInOff + curPartW) > (this->widthInput + this->pW)) && (this->ceilW != 0)) ? (wInOff + curPartW) - (this->widthInput + this->pW) : 0;

                                int32_t partWWoPad = (int32_t)curPartW - (int32_t)padWL - (int32_t)padWR;

								CreateKernelIndexes(dKer * ((this->partD - 1) / this->dD + 1) * this->kW * this->kH + hKer * ((this->partH - 1) / this->dH + 1)  * this->kW + wKer * (((this->partW - 1) / this->dW + 1) ));

								uint32_t inputOffset = b * this->inputSize + (dInOff >= this->pD ? dInOff - this->pD : 0) * this->hwInputSize + \
								                      (hInOff >= this->pH ? hInOff - this->pH : 0) * this->widthInput + \
													  (wInOff >= this->pW ? wInOff - this->pW : 0);
								uint32_t outInKer = (dKer * hCycleMax * wCycleMax + hKer * wCycleMax + wKer) * this->blockLength;
								// Offset here can be < 0 for case with padding
								// This offset is start position of the first kernel for this iteration. If it starts on padding, offset will be < 0
                                if (partWWoPad > 0) {
                                    DoMaxPool(batchLen, curPartD, curPartH, curPartW,
                                              inputOffset, outInKer,
                                              padDL, padDR, padHL, padHR, padWL, padWR);
                                }
							}
						}
					}
					FinalMax(outputOffset, batchLen, dOut*this->sD-this->pD, hOut*this->sH-this->pH, wOut*this->sW-this->pW);
					partCount++;
				}
			}
		}
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::FinalMax(uint32_t outputOffset, uint32_t partNC,
                                                                              int32_t offD, int32_t offH, int32_t offW) {
    LocalTensor<T> transDataTensorUb2 = this->queIn.template AllocTensor<T>();
    DataCopy(transDataTensorUb2, wsGm, indexStart * 2);
    this->queIn.template EnQue<T>(transDataTensorUb2);
    
    transDataTensorUb2 = this->queIn.template DeQue<T>();
    LocalTensor<T> maxResult = transDataTensorUb2[indexStart - this->blockLength];
    
    LocalTensor<uint8_t> maskIndex = this->transDataTensorUb1.template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> nanMask = transDataTensorUb2.template ReinterpretCast<uint8_t>();

    LocalTensor<T> reduceMaxResult = this->queOutVals.template AllocTensor<T>();
    
    Brcb(reduceMaxResult, maxResult, this->blockLength / BLOCKS_IN_REP, {1, BLOCKS_IN_REP});
    PipeBarrier<PIPE_V>();
	auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);
    
    Copy(this->transDataTensorUb1, reduceMaxResult, (this->blockLengthS / this->blockLength) * this->blockLength, this->blockLength, {1, 0, static_cast<uint16_t>(this->blockLengthS / this->blockLength), 1});
    PipeBarrier<PIPE_V>();
    Copy(reduceMaxResult, this->transDataTensorUb1, this->blockLength * BLOCKS_IN_REP, this->blockLengthS / BLOCKS_IN_REP, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
    PipeBarrier<PIPE_V>();
    
    CastBF16Back<S>(reduceMaxResult, this->blockLengthS, sizeof(S) == sizeof(bfloat16_t));
    
    CopyOutVals(outputOffset, partNC);

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(this->mask);
    Compare<T, uint8_t, false>(maskIndex, maxResult, transDataTensorUb2, CMPMODE::EQ, MASK_PLACEHOLDER, 
                               (dCycleMax * hCycleMax * wCycleMax * this->blockLength +  this->blockLength * BLOCKS_IN_REP - 1) / (this->blockLength * BLOCKS_IN_REP), 
                               {1, 0, 1, BLOCKS_IN_REP, 0, BLOCKS_IN_REP});
    Compare<T, uint8_t, false>(nanMask, transDataTensorUb2, transDataTensorUb2, CMPMODE::EQ, MASK_PLACEHOLDER,
                               (dCycleMax * hCycleMax * wCycleMax * this->blockLength +  this->blockLength * BLOCKS_IN_REP - 1) / (this->blockLength * BLOCKS_IN_REP),
                               {1, 1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP, BLOCKS_IN_REP});
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);

    LocalTensor<uint16_t> nanMasks16 = nanMask.ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> masks16 = maskIndex.ReinterpretCast<uint16_t>();

    Not(nanMasks16, nanMasks16, (dCycleMax * hCycleMax * wCycleMax * this->blockLength + this->blockLength) / (BITS_UINT8 * INT8_INT16));
    PipeBarrier<PIPE_V>();

    Or(masks16, masks16, nanMasks16, (dCycleMax * hCycleMax * wCycleMax * this->blockLength + this->blockLength) / (BITS_UINT8 * INT8_INT16));
    PipeBarrier<PIPE_V>();

    Select<T>(transDataTensorUb2, maskIndex, transDataTensorUb2[indexStart], 1.f + this->kernelDHW, SELMODE::VSEL_TENSOR_SCALAR_MODE, dCycleMax * hCycleMax * wCycleMax * this->blockLength);
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);
    
    uint32_t halfInd = (dCycleMax * hCycleMax * wCycleMax) / MAX_DIV;
    uint32_t halfOffset = halfInd * this->blockLength;

    Compare<T>(maskIndex, maxResult, maxResult, CMPMODE::EQ, this->blockLength, 1, {1,1,1,0,0,0});
    Duplicate(transDataTensorUb2[indexStart], (T)1., this->blockLength);  
    Min(transDataTensorUb2[halfOffset], transDataTensorUb2, transDataTensorUb2[halfOffset], halfOffset);
    PipeBarrier<PIPE_V>();

    auto curMax = transDataTensorUb2[halfOffset];
    for (uint32_t ind =dCycleMax * hCycleMax * wCycleMax - halfInd; ind > 1; ind -= halfInd) {
        halfInd = ind / MAX_DIV;
        halfOffset = halfInd * this->blockLength;

        Min(curMax[halfOffset], curMax, curMax[halfOffset], halfOffset);

        curMax = curMax[halfOffset];
        PipeBarrier<PIPE_V>();
    }

    Select<T>(transDataTensorUb2[indexStart], maskIndex, transDataTensorUb2[indexStart], -1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->blockLength, 1, {1,1,1,0,0,0});
    PipeBarrier<PIPE_V>();

    Mul<T>(curMax, curMax, transDataTensorUb2[indexStart], this->blockLength);  
    PipeBarrier<PIPE_V>();
    
    LocalTensor<float> floatUb2 = transDataTensorUb2.template ReinterpretCast<float>();
    LocalTensor<float> idxs = this->queOutInds.template AllocTensor<float>();
    this->CastInds(floatUb2, curMax, 0, 1);
    
    // TransDataTo5HD (NC1DHWC0 -> NCDHW) for out
    LocalTensor<float> transDataTensorUb3 = this->transDataTensorUb1. template ReinterpretCast<float>();
    
    Brcb(transDataTensorUb3, floatUb2, this->blockLength / BLOCKS_IN_REP, {1, BLOCKS_IN_REP});
    PipeBarrier<PIPE_V>();

    // Recount indeces
    auto tmpSize = this->blockLength;
    uint32_t ncPartOutTrans = partNC * tmpSize;
    LocalTensor<float> tmpIdxs = floatUb2;
    this->IndexRecalcFirst(transDataTensorUb3, idxs, tmpIdxs, ncPartOutTrans);
    
    float coeffH = 1.f / float(int(this->kW));
    float coeffD = coeffH / float(int(this->kH));
    float epsD = EPS_COEFF * coeffD;
    float epsH = EPS_COEFF * coeffH;

    // Calculation kernel's offset
    Adds(transDataTensorUb3, transDataTensorUb3, epsD, ncPartOutTrans);
    Adds(idxs, idxs, epsH, ncPartOutTrans);
    PipeBarrier<PIPE_V>();

    Muls(transDataTensorUb3, transDataTensorUb3, coeffD, ncPartOutTrans);
    Muls(idxs, idxs, coeffH, ncPartOutTrans);
    Muls(tmpIdxs, tmpIdxs, float(int(this->dW)), ncPartOutTrans);
    PipeBarrier<PIPE_V>();
    
    LocalTensor<int> generalOffsetsTmp = this->queOutVals.template AllocTensor<int>();
    LocalTensor<int> hTmp = generalOffsetsTmp[tmpSize];
    LocalTensor<int> wTmp = hTmp[tmpSize];

    Duplicate(generalOffsetsTmp, offD * (int)(this->hwInputSize), tmpSize);
    Duplicate(hTmp, offH * (int)(this->widthInput), tmpSize);
    Duplicate(wTmp, offW, tmpSize);
    Floor(transDataTensorUb3, transDataTensorUb3, ncPartOutTrans);
    Floor(idxs, idxs, ncPartOutTrans);
    PipeBarrier<PIPE_V>();
    
    LocalTensor<int> transDataTensorUb3Int = transDataTensorUb3. template ReinterpretCast<int>();
    LocalTensor<int> idxsInt = idxs. template ReinterpretCast<int>();
    LocalTensor<int> tmpIdxsInt = tmpIdxs. template ReinterpretCast<int>();

    Cast(transDataTensorUb3Int, transDataTensorUb3, RoundMode::CAST_RINT, ncPartOutTrans);
    Cast(idxsInt, idxs, RoundMode::CAST_RINT, ncPartOutTrans);
    Cast(tmpIdxsInt, tmpIdxs, RoundMode::CAST_RINT, ncPartOutTrans);
    PipeBarrier<PIPE_V>();

    Muls(transDataTensorUb3Int, transDataTensorUb3Int, (int)(this->dD * this->hwInputSize), ncPartOutTrans);
    Muls(idxsInt, idxsInt, (int)(this->dH * this->widthInput), ncPartOutTrans);
    PipeBarrier<PIPE_V>();

    Add(transDataTensorUb3Int, generalOffsetsTmp, transDataTensorUb3Int, this->blockLength, partNC, {1, 0, 1, 1, 0, 1});
    Add(idxsInt, hTmp, idxsInt, this->blockLength, partNC, {1, 0, 1, 1, 0, 1});
    Add(tmpIdxsInt, wTmp, tmpIdxsInt, this->blockLength, partNC, {1, 0, 1, 1, 0, 1});
    PipeBarrier<PIPE_V>();

    Maxs(transDataTensorUb3Int, transDataTensorUb3Int, 0, ncPartOutTrans);
    Maxs(idxsInt, idxsInt, 0, ncPartOutTrans);
    Maxs(tmpIdxsInt, tmpIdxsInt, 0, ncPartOutTrans);
    PipeBarrier<PIPE_V>();

    Add(tmpIdxsInt, tmpIdxsInt, idxsInt, ncPartOutTrans);
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(event0);
    WaitFlag<HardEvent::S_V>(event0);

    this->queOutVals.template FreeTensor<int>(generalOffsetsTmp);

    LocalTensor<int> indicesTmpInt = idxs.ReinterpretCast<int>();
    Add(indicesTmpInt, tmpIdxsInt, transDataTensorUb3Int, ncPartOutTrans);
    PipeBarrier<PIPE_V>();
    
    this->queOutInds.template EnQue<float>(idxs);
    this->queIn.template FreeTensor<T>(transDataTensorUb2);
    
    CopyOutInds(outputOffset, partNC);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::RemovingGaps(const LocalTensor<T>& dst, uint32_t partD, uint32_t partH, uint32_t partW) {
    uint32_t roundW = this->RoundUpBlock(partW, this->blockLengthS);
    DataCopyParams RemovingGapsParams = {static_cast<uint16_t>(partD * partH), static_cast<uint16_t>(partW),
                                         static_cast<uint16_t>(roundW - partW), 0};

    DataCopy(dst, this->transDataTensorUb1, RemovingGapsParams);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::Img2ColC0(const uint32_t partNC,
                                                                      uint32_t partD, uint32_t partH, uint32_t partW,
                                                                      const uint32_t outputOffset,
                                                                      const uint32_t pDL, const uint32_t pDR,
                                                                      const uint32_t pHL, const uint32_t pHR,
                                                                      const uint32_t pWL, const uint32_t pWR) {
    LocalTensor<T> maxTmp;
    
    LocalTensor<S> transDataTensorUb2Old = this->queIn.template DeQue<S>();
    LocalTensor<T> transDataTensorUb2 = transDataTensorUb2Old. template ReinterpretCast<T>();
    
    // Cast for bf16
    this->CastBF16<S>(transDataTensorUb2Old, transDataTensorUb2, partNC);

    maxTmp = this->transDataTensorUb1;
    uint32_t curKernelDHW = ((partD - 1) / this->dD + 1)  * ((partH - 1) / this->dH + 1)  * ((partW - 1) / this->dW + 1) ;
    LocalTensor<uint8_t> maskIndex = maxTmp.template ReinterpretCast<uint8_t>();
	LocalTensor<uint8_t> nanMask = maskIndex[this->RoundUpBlock((curKernelDHW * this->blockLength) / BITS_UINT8, BLOCK_LEN_UINT8)];

    // TransDataTo5HD (NCDHW -> NC1DHWC0)
    uint32_t roundW = this->RoundUpBlock(partW, this->blockLengthS);
    this->PrepareInput(transDataTensorUb2, partD * partH * roundW);

    // Add padding to data if needed
    if (pDL != 0 || pDR != 0 || pHL != 0 || pHR != 0 || pWL != 0 || pWR != 0) {
        uint32_t partDWoPad = partD - pDL - pDR;
        uint32_t partHWoPad = partH - pHL - pHR;
        uint32_t partWWoPad = partW - pWL - pWR;
        Padding(this->transDataTensorUb1, transDataTensorUb2, 
                partD, partH, partW,
                partDWoPad, partHWoPad, partWWoPad, 
                pDL, pDR, pHL, pHR, pWL, pWR);
    }

    // Removing gaps
    RemovingGaps(transDataTensorUb2, partD, partH, partW); 
    PipeBarrier<PIPE_V>();

    // Find Max for each row by width
    LocalTensor<S> reduceMaxResultBf16 = this->queOutVals.template AllocTensor<S>();
    LocalTensor<T> reduceMaxResult = reduceMaxResultBf16.template ReinterpretCast<T>();

    uint32_t halfInd = curKernelDHW / MAX_DIV;
    uint32_t halfOffset = halfInd * this->blockLength;
    
    Max(maxTmp, transDataTensorUb2, transDataTensorUb2[halfOffset], halfOffset);

    if (curKernelDHW % MAX_DIV == 1) {
        Copy(maxTmp[halfOffset], transDataTensorUb2[(curKernelDHW - 1) * this->blockLength], this->blockLength, 1,
             {1, 1, static_cast<uint16_t>(halfInd), static_cast<uint16_t>(curKernelDHW)});
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<T> curMax = maxTmp;
    for (uint32_t ind = curKernelDHW - halfInd; ind > MAX_DIV; ind -= halfInd) {
        halfInd = ind / MAX_DIV;
        halfOffset = halfInd * this->blockLength;

        Max(curMax[halfOffset], curMax, curMax[halfOffset], halfOffset);

        curMax = curMax[halfOffset];
        PipeBarrier<PIPE_V>();
    }
    
    Max(reduceMaxResult, curMax, curMax[this->blockLength], this->blockLength);
    PipeBarrier<PIPE_V>();
    this->queOutVals.template EnQue<T>(reduceMaxResult);
    
    reduceMaxResult = this->queOutVals.template DeQue<T>();
    DataCopy(wsGm[outputOffset], reduceMaxResult, this->blockLength);
    SetAtomicMax<T>();
    DataCopy(wsGm[indexStart - this->blockLength], reduceMaxResult, this->blockLength);
    auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
    SetFlag<HardEvent::MTE3_S>(event0);
    WaitFlag<HardEvent::MTE3_S>(event0);
    SetAtomicNone();
    
    // Find Indices for each row by width
    auto wholeRow = this->RoundUpBlock(curKernelDHW, this->blockLength) * this->blockLength;
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(this->mask);
    Compare<T, uint8_t, false>(maskIndex, reduceMaxResult, transDataTensorUb2, CMPMODE::EQ, MASK_PLACEHOLDER, wholeRow / this->mask,
                               {1, 0, 1, BLOCKS_IN_REP, 0, BLOCKS_IN_REP});
    Compare<T, uint8_t, false>(nanMask, transDataTensorUb2, transDataTensorUb2, CMPMODE::EQ, MASK_PLACEHOLDER, wholeRow / this->mask,
                               {1, 1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP, BLOCKS_IN_REP});
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);
    
    this->queOutVals.template FreeTensor<S>(reduceMaxResultBf16);

    Select<T>(transDataTensorUb2, maskIndex, this->kernelIndexes, (T)(1.f + this->kernelDHW), SELMODE::VSEL_TENSOR_SCALAR_MODE, wholeRow);
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);

    Muls(this->kernelIndexes, this->kernelIndexes, -1.f, wholeRow);
    PipeBarrier<PIPE_V>();

    Select<T>(transDataTensorUb2, nanMask, transDataTensorUb2, this->kernelIndexes, SELMODE::VSEL_TENSOR_TENSOR_MODE, wholeRow);
    PipeBarrier<PIPE_V>();
    event0 = this->pipe->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event0);
    WaitFlag<HardEvent::V_S>(event0);

    Muls(this->kernelIndexes, this->kernelIndexes, -1.f, wholeRow);

    halfInd = curKernelDHW / MAX_DIV;
    halfOffset = halfInd * this->blockLength;
    
    Min(transDataTensorUb2[halfOffset], transDataTensorUb2, transDataTensorUb2[halfOffset], halfOffset);
    PipeBarrier<PIPE_V>();

    curMax = transDataTensorUb2[halfOffset];
    for (uint32_t ind = curKernelDHW - halfInd; ind > MAX_DIV; ind -= halfInd) {
        halfInd = ind / MAX_DIV;
        halfOffset = halfInd * this->blockLength;

        Min(curMax[halfOffset], curMax, curMax[halfOffset], halfOffset);

        curMax = curMax[halfOffset];
        PipeBarrier<PIPE_V>();
    }

    this->queIn.template FreeTensor(transDataTensorUb2Old);

    LocalTensor<T> idxs = this->queOutInds.template AllocTensor<T>();
    Min(idxs, curMax, curMax[this->blockLength], this->blockLength);
    
    this->queOutInds.template EnQue<T>(idxs);
    
    idxs = this->queOutInds.template DeQue<T>();
    DataCopy(this->wsGm[outputOffset + indexStart], idxs, this->blockLength);
    event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
    SetFlag<HardEvent::MTE3_S>(event0);
    WaitFlag<HardEvent::MTE3_S>(event0);
    this->queOutInds.template FreeTensor<T>(idxs);
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::CopyIn(GlobalTensor<S> srcGm, const uint32_t& partNC, 
                                                                          uint32_t partD, uint32_t partH, uint32_t partW,
                                                                          const uint32_t padDL, const uint32_t padDR,
                                                                          const uint32_t padHL, const uint32_t padHR,
                                                                          const uint32_t padWL, const uint32_t padWR) {
    uint32_t partWWoPad = partW - padWL - padWR;
    uint32_t downPartW = this->RoundDownBlock(partWWoPad, this->blockLengthS);
    uint32_t upPartW = this->RoundUpBlock(partW, this->blockLengthS);
    
    uint32_t partDWoPad = partD - padDL - padDR;
    uint32_t tmpDInSize = partDWoPad * this->hwInputSize;
    
    uint32_t srcOff = 0;
    auto srcStride = this->widthInput - downPartW;
    LocalTensor<S> srcUb = this->queIn.template AllocTensor<S>();
    
    uint32_t partHWoPad = partH - padHL - padHR;
    
    if (this->widthInput % this->blockLengthS == 0 && downPartW == partWWoPad) {
        if (((this->widthInput - downPartW) / this->blockLengthS) < MAX_UINT16) {
            DataCopyParams params{static_cast<uint16_t>(partHWoPad), static_cast<uint16_t>(downPartW / this->blockLengthS),
                                  static_cast<uint16_t>((this->widthInput - downPartW) / this->blockLengthS),
                                  static_cast<uint16_t>((upPartW - downPartW) / this->blockLengthS)};
            for (uint32_t nc = 0; nc < partNC; nc++){
                auto tmp = srcGm[nc * this->inputSize];
                auto tmpDst = srcUb[partD * partH * partW * nc];
                for (uint32_t d = 0, dstOff = 0; d < tmpDInSize; d+= this->hwInputSize, dstOff += upPartW * partH) {
                    DataCopy(tmpDst[dstOff], tmp[d], params);
                }
            }
        } else {
            DataCopyParams params{static_cast<uint16_t>(1), static_cast<uint16_t>(downPartW / this->blockLengthS), 0, 0};
            for (uint32_t nc = 0; nc < partNC; nc++){
                for (uint32_t d = 0, dstOff = 0; d < tmpDInSize; d+= this->hwInputSize, dstOff += upPartW * partH){
                    auto tmp = srcGm[nc * this->inputSize + d];
                    auto tmpDst = srcUb[partD * partH * partW * nc + dstOff];
                    for (uint32_t h = 0; h < partHWoPad; h++) {
                        DataCopy(tmpDst[h * upPartW], tmp[h *this->widthInput], params);
                    }
                }
            }
        }
    } else {
        DataCopyExtParams copyPadParamsInput{static_cast<uint16_t>(partHWoPad), static_cast<uint32_t>(partWWoPad * sizeof(S)), 
                                             static_cast<uint32_t>((this->widthInput - partWWoPad) * sizeof(S)), 
                                             static_cast<uint32_t>((upPartW - partWWoPad) / this->blockLengthS), 0};
        DataCopyPadExtParams<S> paddingParamsInput{false, 0, 0, 0};
        for (uint32_t nc = 0; nc < partNC; nc++){
            auto tmp = srcGm[nc * this->inputSize];
            auto tmpDst = srcUb[partD * partH * upPartW * nc];
            for (uint32_t d =0, dstOff=0; d < tmpDInSize; d+= this->hwInputSize, dstOff += upPartW * partH){
                DataCopyPad(tmpDst[dstOff], tmp[d], copyPadParamsInput, paddingParamsInput);
            }
        }
    }
    this->queIn.template EnQue<S>(srcUb);
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::CopyOutVals(const uint32_t& dstOffset, const uint32_t& partNC) {
    LocalTensor<S> reduceMaxResult = this->queOutVals.template DeQue<S>();
    
    if (this->outSize * sizeof(S) < MAX_UINT16){
        DataCopyPad(this->yGm[dstOffset], reduceMaxResult, 
                    {static_cast<uint16_t>(partNC), static_cast<uint16_t>(1 * sizeof(S)), 0, 
                     static_cast<uint16_t>((this->outSize - 1) * sizeof(S))});
    } else {
        for (uint32_t b = 0; b < partNC; b++){
            DataCopyPad(this->yGm[dstOffset + b * this->outSize], reduceMaxResult[b * this->blockLengthS], 
                        {static_cast<uint16_t>(1), static_cast<uint16_t>(1 * sizeof(S)), 0, 0});
        }
    }
    auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
    SetFlag<HardEvent::MTE3_S>(event0);
    WaitFlag<HardEvent::MTE3_S>(event0);
    this->queOutVals.template FreeTensor<S>(reduceMaxResult);
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2HugeKernel<T,S>::Padding(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb,
                                                                    const uint32_t& partD, const uint32_t& partH, const uint32_t& partW,
                                                                    const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad,
                                                                    const uint32_t pDL, const uint32_t pDR, const uint32_t pHL, const uint32_t pHR, const uint32_t pWL, const uint32_t pWR) {
    uint32_t upPartW = this->RoundUpBlock(partW, this->blockLengthS);
    uint32_t partHwInp = partH * upPartW;
    uint32_t partRoundDhwInp = partD * partHwInp;
   if (pWL != 0 || pWR != 0){
        Duplicate<T>(dstUb, this->minFloat, partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
        
        DataCopyParams padParams = {static_cast<uint16_t>(partHWoPad), static_cast<uint16_t>(partWWoPad), 
                                    static_cast<uint16_t>(upPartW - partWWoPad), 
                                    static_cast<uint16_t>(upPartW - partWWoPad)};
        for (uint32_t d = 0; d < partDWoPad; d++){
            DataCopy(dstUb[pWL * this->blockLength + d * partHwInp * this->blockLength], 
                     srcUb[d * partHwInp * this->blockLength], padParams);
        }
        PipeBarrier<PIPE_V>();
        
        DataCopy(srcUb, dstUb, partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
    }
    
    if (pHL != 0 || pHR != 0) {
        Duplicate<T>(dstUb, this->minFloat, partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
        
        DataCopyParams padParams = {static_cast<uint16_t>(partDWoPad), static_cast<uint16_t>(partHWoPad * upPartW), 
                                    static_cast<uint16_t>(partHwInp - partHWoPad * upPartW),
                                    static_cast<uint16_t>(partHwInp - partHWoPad * upPartW)};
        DataCopy(dstUb[pHL * upPartW * this->blockLength], srcUb, padParams);
        PipeBarrier<PIPE_V>();
        
        DataCopy(srcUb, dstUb, partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
    }
    
    if (pDL != 0){
        Duplicate<T>(dstUb, this->minFloat, partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
       
        DataCopy(dstUb[pDL * partHwInp * this->blockLength], srcUb, partDWoPad * partHwInp * this->blockLength);
        PipeBarrier<PIPE_V>();

        DataCopy(srcUb, dstUb, partRoundDhwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
    }
    
    if (pDR != 0) {
        Duplicate<T>(srcUb[(partDWoPad + pDL) * partHwInp * this->blockLength], this->minFloat, pDR * partHwInp * this->blockLength);
        PipeBarrier<PIPE_V>();
    }
}

#endif
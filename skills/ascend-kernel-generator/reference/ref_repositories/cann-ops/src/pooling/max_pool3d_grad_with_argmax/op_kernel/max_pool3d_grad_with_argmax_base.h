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
 * \file max_pool3d_grad_with_argmax_base.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_BASE_H
#define OPP_MAX_POOL3D_GRAD_WITH_ARGMAX_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_grad_with_argmax_common.h"

using namespace MaxPool3DGradWithArgmaxComm;

constexpr uint64_t VEC_PROC_SIZE = 256;
constexpr uint64_t VEC_PROC_BLOCKS = VEC_PROC_SIZE / BLOCK_SIZE;
constexpr uint64_t BLOCK_LEN_FP32 = 8;
constexpr uint64_t BLOCK_LEN_FP16 = 16;
constexpr uint64_t BLOCK_LEN_UINT8 = 32;
constexpr uint64_t MIN_TRANSPOSE_ROWS = 16;
constexpr uint64_t BLOCKS_IN_REP = 8;

const uint64_t MAX_UINT16 = 65536;
const uint64_t INT64_FP32 = 2;
const uint64_t MAX_DIV = 2;
const uint64_t BITS_UINT8 = 8;
const uint64_t TMP_TENSORS_NUM = 3;
const uint64_t D_DIM = 0;
const uint64_t H_DIM = 1;
const uint64_t W_DIM = 2;

template <typename T, typename S>
class KernelMaxPool3DGradWithArgmaxBase {
  public:
    const MaxPool3DGradWithArgmaxTilingData* tilingData;

  protected:
    const uint64_t depthInput;
    const uint64_t heightInput;
    const uint64_t widthInput;
    
    const uint64_t depthOut;
    const uint64_t heightOut;
    const uint64_t widthOut;
    
    const uint64_t kD;
    const uint64_t kH;
    const uint64_t kW;
    const uint64_t sD;
    const uint64_t sH;
    const uint64_t sW;
    const uint64_t pD;
    const uint64_t pH;
    const uint64_t pW;
    const uint64_t dD;
    const uint64_t dH;
    const uint64_t dW;
    const uint64_t partD;
    const uint64_t partH;
    const uint64_t partW;
    const uint64_t partOutD;
    const uint64_t partOutH;
    const uint64_t partOutW;
    const uint64_t ceilD;
    const uint64_t ceilH;
    const uint64_t ceilW;
    const uint64_t sizeUb1;
    const uint64_t sizeUb2;
    const uint64_t valSize;

    static constexpr uint64_t blockLength = BLOCK_SIZE / sizeof(T);
    static constexpr uint64_t blockLengthS = BLOCK_SIZE / sizeof(S);
    static constexpr uint64_t vecProcBlocks = VEC_PROC_SIZE / BLOCK_SIZE;
    static constexpr uint64_t vecProcLength = VEC_PROC_SIZE / sizeof(T);
    static constexpr uint64_t vecProcLengthFloat = VEC_PROC_SIZE / sizeof(float);
    
    const uint64_t hwInputSize = heightInput * widthInput;
    const uint64_t inputSize = depthInput * hwInputSize;
    const uint64_t hwOutputSize = heightOut * widthOut;
    const uint64_t outSize = depthOut * hwOutputSize;
    const uint64_t partHwInp = partW * partH;
    const uint64_t partHwOut = partOutH * partOutW;
    const uint64_t partRoundDhwInp = partD * partHwInp;
    const uint64_t roundPartOutSize = partOutD * partHwOut;
    const uint64_t partAlignRoundDhwInp = RoundUpBlock(partRoundDhwInp, blockLength);
    const uint64_t blockAlPartDhwInp = partAlignRoundDhwInp * blockLength;
    const uint64_t transAlignRoundPartSize = RoundUpBlock(partRoundDhwInp, MIN_TRANSPOSE_ROWS);
    const uint64_t transAlignRoundPartOutSize = RoundUpBlock(roundPartOutSize, MIN_TRANSPOSE_ROWS);

    const float epsCoeffD = 0.5f / hwOutputSize;
    const float epsCoeffH = 0.5f / widthOut;
    
    const uint64_t gmWOff = sW * blockLength;
    const uint64_t kernelDHW = kD * kH * kW;
    const uint64_t halfKernelDHW = kernelDHW / MAX_DIV;
    const uint64_t alKernelDHW = RoundUpBlock(kernelDHW);
    const uint64_t blockKernelDHW = kernelDHW * blockLength;
    const uint64_t blockAlKernelDHW = alKernelDHW * blockLength;
    const bool kernelIsBlock = (kernelDHW <= blockLength) && (blockLength == BLOCK_LEN_FP32);
    const uint64_t mask = BLOCKS_IN_REP * blockLength;
    
    const uint64_t sliceOff = kH * kW * blockLength;
    
    uint64_t im2colCopyFullIters = partW / UINT8_MAX;
    uint64_t processedIm2colCopyPartW = UINT8_MAX * im2colCopyFullIters;
    uint8_t im2colCopyRepeatTimesTail = partW - processedIm2colCopyPartW;
    
    const uint8_t im2colAddRepeatTimes = kW / vecProcBlocks; // if equal to 1 it means that repeatTimes = kH can take place
    uint8_t stripPadRepeatTimes = widthOut / vecProcBlocks;
    
    const uint64_t processedIm2colAddDataSrc = im2colAddRepeatTimes * vecProcLength;
    const uint64_t processedIm2colAddDataDst = processedIm2colAddDataSrc * dW;
    
    uint64_t processedStripPadData = stripPadRepeatTimes * vecProcLength;

    const uint64_t im2colAddTail = kW * blockLength - processedIm2colAddDataSrc;
    uint64_t stripPadTail = widthOut * blockLength - processedStripPadData; 

    uint64_t partNC;
    uint64_t roundBatches;
    uint64_t roundDepthOut;
    uint64_t roundHeightOut;
    uint64_t roundWidthOut;
    uint64_t curDepthOff;
    uint64_t gmDOff;
    uint64_t gmHOff;
    uint64_t maskSize;
  
    const uint64_t coreNum = GetBlockNum();
    const uint64_t coreIdx = GetBlockIdx();

    CopyRepeatParams Im2colNewParams = {static_cast<uint16_t>(dW), 1, static_cast<uint16_t>(sW), static_cast<uint16_t>(kernelDHW)};
    CopyRepeatParams Im2colNewParamsTail = {static_cast<uint16_t>(dW), 1, static_cast<uint16_t>(dH * partOutW), static_cast<uint16_t>(kW)};

    CopyRepeatParams PadStripParams = {1, 1, VEC_PROC_BLOCKS, VEC_PROC_BLOCKS};
    
    BinaryRepeatParams im2ColAddParams; // src0 should be the same as dst 
    BinaryRepeatParams im2ColAddParamsTail = {static_cast<uint8_t>(dW), static_cast<uint8_t>(dW), 1, static_cast<uint8_t>(dH * partOutW), 
                                              static_cast<uint8_t>(dH * partOutW), static_cast<uint8_t>(kW)}; // src0 should be the same as dst, repeatTimes = kH always takes place

    bool addParamsOverUint8 = (dH * partOutW) > UINT8_MAX; 
    
    const bool dAdd = sD <= (dD * (kD - 1));
    const bool hAdd = sH <= (dH * (kH - 1));
    const bool wAdd = sW <= (dW * (kW - 1));
    
    const bool im2colSingleWOut = kW <= vecProcBlocks && !(dAdd || hAdd || wAdd); 

    GlobalTensor<S> gradOutputGm;
    GlobalTensor<int32_t> indicesGm; 
    GlobalTensor<S> gradInputGm;
    GlobalTensor<float> workspaceGm;
    
    LocalTensor<T> transDataTensorUb1;
    LocalTensor<T> kernelIndexes;
    LocalTensor<float> generalOffsets;
    LocalTensor<uint8_t> masks;
    LocalTensor<T> tmpBuf;
    
    TQue<TPosition::VECIN, 1> queInVals;
    TQue<TPosition::VECIN, 1> queInInds;
    TQue<TPosition::VECOUT, 1> queOutVals;
    
    TransDataTo5HDParams transDataParams;
    TransDataTo5HDParams transDataParamsReverse;
    
  protected:
    __aicore__ inline uint64_t RoundUpBlock(const uint64_t& src, const uint64_t& blockLen = BLOCK_SIZE / sizeof(T)) {
        if (blockLen != 0) {
            return src != 0 ? (src + blockLen - 1) / blockLen * blockLen : blockLen;
        }
        return blockLen;
    }

    __aicore__ inline uint64_t RoundDownBlock(const uint64_t& src, const uint64_t& blockLen = BLOCK_SIZE / sizeof(T)) {
        if (blockLen != 0) {
            return (src / blockLen) * blockLen;
        }
        return blockLen;
    }
  
    __aicore__ inline void CreateKernelIndexes() {
        ArithProgression<T>(tmpBuf, (T)0.f, (T)1.f, alKernelDHW);
        auto event0 = pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);
        PipeBarrier<PIPE_V>();
        
        Brcb(kernelIndexes, tmpBuf, alKernelDHW / BLOCKS_IN_REP, {1, BLOCKS_IN_REP});
    }
    
    __aicore__ inline void PrepareBaseScalars() {
        partNC = batchesCurCore % blockLength;
        roundBatches = batchesCurCore - partNC;
        curDepthOff = dD * partHwOut * blockLength;
        gmHOff = sH * partOutW * blockLength;
        gmDOff = (sD * partHwOut - partH * sH * partOutW) * blockLength;
        maskSize = RoundUpBlock(partRoundDhwInp * blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8);

        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = RoundUpBlock(partRoundDhwInp, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        transDataParams.dstRepStride = (transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        transDataParams.srcRepStride = (transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / blockLength;

        transDataParamsReverse.dstHighHalf = false;
        transDataParamsReverse.srcHighHalf = false;
        transDataParamsReverse.repeatTimes = transAlignRoundPartOutSize / NCHW_CONV_ADDR_LIST_SIZE;
        transDataParamsReverse.dstRepStride = (transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / blockLength;
        transDataParamsReverse.srcRepStride = (transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
    }

    template <typename V, bool indices = false>
    __aicore__ void CopyIn(TQue<TPosition::VECIN, 1>& inQue, GlobalTensor<V> srcGm, const uint64_t& partNC, const uint64_t& partIn) {
        constexpr uint64_t curBlockLen = BLOCK_SIZE / sizeof(V); 
        const uint64_t downPartIn = this->RoundDownBlock(partIn, curBlockLen);
        const uint64_t roundSize = this->RoundUpBlock(partIn, curBlockLen);
        const uint64_t roundFp16Size = this->RoundUpBlock(partIn, this->blockLength);
        LocalTensor<V> dstUb = inQue.template AllocTensor<V>();
        
        if constexpr (indices && (sizeof(T) == sizeof(S)) && (sizeof(T) == sizeof(half))) {
            if (downPartIn == partIn && this->inputSize % this->blockLength == 0) {
                if ((inputSize - partIn) < MAX_UINT16) {
                    DataCopyParams copyParams{static_cast<uint16_t>(partNC), static_cast<uint16_t>(partIn / curBlockLen),
                                              static_cast<uint16_t>((this->inputSize - partIn) / curBlockLen), 
                                              (partIn % this->blockLength) <= curBlockLen && (partIn % this->blockLength) != 0};
                    DataCopy(dstUb, srcGm, copyParams);
                } else {
                    DataCopyParams copyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(partIn / curBlockLen), 0, 
                                              (partIn % this->blockLength) <= curBlockLen && (partIn % this->blockLength) != 0};
                    for (uint64_t nc = 0; nc < partNC; ++nc) {
                        DataCopy(dstUb[nc * roundFp16Size], srcGm[nc * this->inputSize], copyParams);
                    }
                }
            } else {
                DataCopyExtParams copyPadParamsInput{static_cast<uint16_t>(partNC), static_cast<uint32_t>(partIn * sizeof(V)),
                                                     static_cast<uint32_t>((this->inputSize - partIn) * sizeof(V)), 
                                                     (partIn % this->blockLength) <= curBlockLen && (partIn % this->blockLength) != 0, 0};
                DataCopyPadExtParams<V> paddingParamsInput{false, 0, 0, 0};
                DataCopyPad(dstUb, srcGm, copyPadParamsInput, paddingParamsInput);
            }
        } else {
            if (downPartIn == partIn && this->inputSize % curBlockLen == 0) {
                if ((this->inputSize - partIn) / curBlockLen < MAX_UINT16) {
                    DataCopyParams copyParams{static_cast<uint16_t>(partNC), static_cast<uint16_t>(partIn / curBlockLen), 
                                              static_cast<uint16_t>((this->inputSize - partIn) / curBlockLen), 0};
                    DataCopy(dstUb, srcGm, copyParams);
                } else {
                    DataCopyParams copyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(partIn / curBlockLen), 0, 0};
                    for (uint64_t nc = 0; nc < partNC; ++nc) {
                        DataCopy(dstUb[nc * roundSize], srcGm[nc * this->inputSize], copyParams);
                    }
                }
            } else {
                DataCopyExtParams copyPadParamsInput{static_cast<uint16_t>(partNC), static_cast<uint32_t>(partIn * sizeof(V)),
                                                     static_cast<uint32_t>((this->inputSize - partIn) * sizeof(V)), 0, 0};
                DataCopyPadExtParams<V> paddingParamsInput{false, 0, 0, 0};
                DataCopyPad(dstUb, srcGm, copyPadParamsInput, paddingParamsInput);
            }
        }
        inQue.template EnQue<V>(dstUb);
    }
    
    template <bool indicesTrans = false>
    __aicore__ inline void PrepareInput(const LocalTensor<half>& srcUb, LocalTensor<half>& dstUb, const uint64_t& dataSize) {
        const uint64_t partAlignDhwInp = RoundUpBlock(dataSize, blockLength);
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        
        for (uint64_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; ++i) {
            dstLocalList[i] = (uint64_t)dstUb[i * blockLength].GetPhyAddr();
            srcLocalList[i] = (uint64_t)srcUb[i * partAlignDhwInp].GetPhyAddr();
        }
        
        TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        PipeBarrier<PIPE_V>();
    }
    
    template <bool indicesTrans = false>
    __aicore__ inline void PrepareInput(const LocalTensor<float>& srcUb, LocalTensor<float>& dstUb, const uint64_t& dataSize) {
        uint64_t partAlignDhwInp;
        if constexpr (indicesTrans) {
            partAlignDhwInp = RoundUpBlock(dataSize, blockLength);
        } else {
            partAlignDhwInp = RoundUpBlock(dataSize, blockLengthS);
        }
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        LocalTensor<float> tmp = dstUb[blockLength * blockLength];
        LocalTensor<float> tmp2 = srcUb[blockLength];
        
        for (int64_t i = 0; i < blockLength; i++) {
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i] = (uint64_t)dstUb[i * blockLength].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i + 1] = (uint64_t)tmp[i * blockLength].GetPhyAddr();
            srcLocalList[i] = (uint64_t)srcUb[i * partAlignDhwInp].GetPhyAddr();
            srcLocalList[i + blockLength] = (uint64_t)tmp2[i * partAlignDhwInp].GetPhyAddr();
        }

        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParams);
        PipeBarrier<PIPE_V>();
    }
 
    __aicore__ void CastInds(LocalTensor<float>& srcDst, const uint64_t& rowLen) {
        LocalTensor<int32_t> srcUbIndicesLong = srcDst.template ReinterpretCast<int32_t>();
        Cast(srcDst, srcUbIndicesLong, RoundMode::CAST_RINT, rowLen);
    }

    __aicore__ void CastIndsToHalf(LocalTensor<float>& srcDst, const uint64_t& rowLen) {
        LocalTensor<half> srcDstHalf = srcDst.template ReinterpretCast<half>();
        Cast(srcDstHalf, srcDst, RoundMode::CAST_RINT, rowLen);
    }
    
    template <bool subOffset = false> 
    __aicore__ inline void IndexRecalcFirst(LocalTensor<float>& dTensor, LocalTensor<float>& hTensor, LocalTensor<float>& wTensor, const uint64_t& len,
                                            const int64_t& dOff = 0, const int64_t& hOff = 0) { 
        float coeffH = 1.f / float((int)this->widthOut);
        float coeffD = float(1.f / float((int)this->widthOut)) / float((int)this->heightOut);
        float coeffW = coeffH;

        LocalTensor<float> tmp = dTensor[sizeUb1]; 
        
        Copy(tmp, dTensor, vecProcLengthFloat, RoundUpBlock(len, vecProcLengthFloat) / vecProcLengthFloat, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP}); 
        PipeBarrier<PIPE_V>();
        
        // d component
        if constexpr (subOffset) {
            float offsetD = -1.f * (dOff * uint32_t(hwOutputSize));
            Adds(tmp, tmp, offsetD, len);
            PipeBarrier<PIPE_V>();
        }

        Muls(dTensor, tmp, coeffD, len); 
        PipeBarrier<PIPE_V>();

        Adds(dTensor, dTensor, epsCoeffD, len);
        PipeBarrier<PIPE_V>();
        
        Floor(dTensor, dTensor, len);
        PipeBarrier<PIPE_V>();

        // h component
        Muls(hTensor, dTensor, -1.f * uint32_t(hwOutputSize), len); // hTensor contains -1.f * Hin * Win
        PipeBarrier<PIPE_V>();
        
        Add(hTensor, tmp, hTensor, len); // i % (Hin * Win)
        PipeBarrier<PIPE_V>();
        
        if constexpr (subOffset) {
            float offsetH = -1.f * hOff * widthOut;
            Adds(hTensor, hTensor, offsetH, len);
            PipeBarrier<PIPE_V>();
        }

        Copy(tmp, hTensor, vecProcLengthFloat, RoundUpBlock(len, vecProcLengthFloat) / vecProcLengthFloat, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP}); 
        PipeBarrier<PIPE_V>();

        Muls(hTensor, hTensor, coeffH, len); // (i % ((Hin * Win)) * 1 / Win
        PipeBarrier<PIPE_V>();
        
        Adds(hTensor, hTensor, epsCoeffH, len);
        PipeBarrier<PIPE_V>();
        
        Floor(hTensor, hTensor, len);
        PipeBarrier<PIPE_V>();
        
        // w component
        Muls(wTensor, hTensor, -1.f * widthOut, len);
        PipeBarrier<PIPE_V>();

        Add(wTensor, tmp, wTensor, len); // i % (Hin * Win) % Win
    }
    
    __aicore__ inline void IndexRecalcSecond(LocalTensor<float>& dTensor, LocalTensor<float>& hTensor, LocalTensor<float>& wTensor, const uint64_t& len) {
        Muls(dTensor, dTensor, 1.f / this->dD, len);
        Muls(hTensor, hTensor, 1.f / this->dH, len);
        Muls(wTensor, wTensor, 1.f / this->dW, len);
        PipeBarrier<PIPE_V>();
    
        Floor(dTensor, dTensor, len);
        Floor(hTensor, hTensor, len);
        Floor(wTensor, wTensor, len);
        PipeBarrier<PIPE_V>();
        
        Muls(dTensor, dTensor, 1.f * this->kH * this->kW, len);
        Muls(hTensor, hTensor, 1.f * this->kW, len);
        PipeBarrier<PIPE_V>();
        
        Add(dTensor, dTensor, hTensor, len);
        PipeBarrier<PIPE_V>();
        
        Add(dTensor, dTensor, wTensor, len);
        PipeBarrier<PIPE_V>();   
    }
    
    __aicore__ inline void CompareSelect(LocalTensor<T>& srcDst, LocalTensor<T>& dstTensor, const uint64_t& rowLen) {
        auto wholeRow = blockKernelDHW * rowLen;
        auto alInt8Offset = RoundUpBlock(blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8);
        LocalTensor<T> srcIndices = srcDst[sizeUb1];
        
        for (uint64_t dstOff = 0, blocks = 0; blocks < rowLen * blockLength; blocks += blockLength, dstOff += alInt8Offset) {
            Compare<T, uint8_t, false>(masks[dstOff], srcIndices[blocks], kernelIndexes, CMPMODE::EQ, mask, blockAlKernelDHW / mask,
                                       {1, 0, 1, BLOCKS_IN_REP, 0, BLOCKS_IN_REP});
        }
              
        PipeBarrier<PIPE_V>();
        auto event0 = pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(event0);
        WaitFlag<HardEvent::V_S>(event0);
        
        for (uint64_t srcOff = 0, dstOff = 0, maskOff = 0; dstOff < wholeRow; dstOff += blockKernelDHW, srcOff += blockLength, maskOff += alInt8Offset) {
            Select<T>(dstTensor[dstOff], masks[maskOff], srcDst[srcOff], 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, blockAlKernelDHW / mask, {1, 0, 0, BLOCKS_IN_REP, 0, 1});
        }
        PipeBarrier<PIPE_V>(); 
    }

    __aicore__ inline void CompareSelectBlockKernel(LocalTensor<T>& srcDst, LocalTensor<T>& dstTensor, const uint64_t& rowLen) {
        LocalTensor<T> srcIndices = srcDst[sizeUb1];
        auto maskOff = (MAX_REPEAT_TIMES + 1) * BLOCKS_IN_REP;
        
        uint64_t maxRep = 0;
        for (maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++) {
            Compare<T>(masks[maskOff * maxRep], srcIndices[MAX_REPEAT_TIMES * maxRep * blockLength],
                       kernelIndexes, CMPMODE::EQ, blockKernelDHW, MAX_REPEAT_TIMES,
                       {1, 0, 1, static_cast<uint8_t>(kernelDHW), 1, 0});
        }
        Compare<T>(masks[maskOff * maxRep], srcIndices[MAX_REPEAT_TIMES * maxRep * blockLength],
                   kernelIndexes, CMPMODE::EQ, blockKernelDHW, rowLen % MAX_REPEAT_TIMES,
                   {1, 0, 1, static_cast<uint8_t>(kernelDHW), 1, 0});
        
        PipeBarrier<PIPE_V>();
        auto event0 = pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);

        for (maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++) {
            Select<T>(dstTensor[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], masks[maskOff * maxRep],
                      srcDst[MAX_REPEAT_TIMES * maxRep * blockLength], 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, blockKernelDHW, MAX_REPEAT_TIMES, {1, 0, 0, static_cast<uint8_t>(kernelDHW), 1, 1});
        }
        Select<T>(dstTensor[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], masks[maskOff * maxRep],
                  srcDst[MAX_REPEAT_TIMES * maxRep * blockLength], 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, blockKernelDHW, rowLen % MAX_REPEAT_TIMES, {1, 0, 0, static_cast<uint8_t>(kernelDHW), 1, 1});
        PipeBarrier<PIPE_V>();
    }
      
    __aicore__ inline void Im2ColAddSingleKernel(const uint64_t& dOff, uint32_t& srcOffset, const LocalTensor<T>& srcTensor, const uint64_t& curDepthOff_, const uint64_t& curPartOutW,
                                                 bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8) {
        for (uint64_t curDepthOut = dOff; curDepthOut < dOff + kD * curDepthOff_; srcOffset += sliceOff, curDepthOut += curDepthOff_) {
            if (!onlyTail) {
                if (addMulti || overUint8) {
                    for (uint64_t curKhSlice = srcOffset, tmpBufOffset = curDepthOut; curKhSlice < srcOffset + sliceOff; curKhSlice += (kW * blockLength), tmpBufOffset += dH * curPartOutW * blockLength) {
                        Add(tmpBuf[tmpBufOffset], tmpBuf[tmpBufOffset], srcTensor[curKhSlice], vecProcLength, im2colAddRepeatTimes, im2ColAddParams);
                    } 
                } else { // kW can be passed in a single repeat => repeatTimes = kH can take place
                    Add(tmpBuf[curDepthOut], tmpBuf[curDepthOut], srcTensor[srcOffset], vecProcLength, kH, im2ColAddParams);
                }
            }

            if (addTail) {
                auto tailTmpBuf = tmpBuf[curDepthOut], tailSrcBuf = srcTensor[srcOffset];
                
                if (!onlyTail) {
                    tailTmpBuf = tailTmpBuf[processedIm2colAddDataDst];
                    tailSrcBuf = tailSrcBuf[processedIm2colAddDataSrc];
                }
                
                if (!overUint8) {
                    Add(tailTmpBuf, tailTmpBuf, tailSrcBuf, im2colAddTail, kH, im2ColAddParamsTail);
                } else {
                    for (uint64_t curHOff = 0, curSrcOff = 0; curHOff < (kH * dH * curPartOutW * blockLength); curHOff += dH * curPartOutW * blockLength, curSrcOff += (kW * blockLength)) {
                        Add(tailTmpBuf[curHOff], tailTmpBuf[curHOff], tailSrcBuf[curSrcOff], im2colAddTail, 1, im2ColAddParamsTail);
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
        }

        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void Im2ColCopySingleWOut(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint64_t& curDepthOff_, const uint64_t& curPartOutW,
                                                bool& addMulti, bool& addTail, bool& onlyTail) {
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        
        for (uint64_t curKD = 0; curKD < kD; ++curKD) {
            for (uint64_t curKH = 0; curKH < kH; ++curKH) {
                if (addMulti && !onlyTail) {
                    for (uint64_t curIter = 0; curIter < im2colCopyFullIters; ++curIter) {
                        Copy(dstTensor[curKD * curDepthOff_ + curKH * dH * curPartOutW * blockLength + curIter * UINT8_MAX * dW * kW * blockLength], srcTensor[curKD * kH * kW * blockLength + curKH * kW * blockLength + curIter * UINT8_MAX * kernelDHW * blockLength], kW * blockLength, UINT8_MAX, Im2colNewParams); 
                    }
                }
                
                if (addTail) {
                    if (!onlyTail) {
                        Copy(dstTensor[curKD * curDepthOff_ + curKH * dH * curPartOutW * blockLength + processedIm2colCopyPartW * dW * kW * blockLength], srcTensor[curKD * kH * kW * blockLength + curKH * kW * blockLength + processedIm2colCopyPartW * kernelDHW * blockLength], kW * blockLength, im2colCopyRepeatTimesTail, Im2colNewParams); 
                    } else {
                        Copy(dstTensor[curKD * curDepthOff_ + curKH * dH * curPartOutW * blockLength], srcTensor[curKD * kH * kW * blockLength + curKH * kW * blockLength], kW * blockLength, im2colCopyRepeatTimesTail, Im2colNewParams); 
                    }
                } 
            }
        }
    }
    
    __aicore__ inline void Im2ColCopySingleKernel(const uint64_t& dOff, uint32_t& srcOffset, const LocalTensor<T>& srcTensor, const uint64_t& curDepthOff_, const uint64_t& curPartOutW,
                                                  bool& addMulti, bool& addTail, bool& onlyTail) {
        for (uint64_t curDepthOut = dOff; curDepthOut < dOff + kD * curDepthOff_; srcOffset += sliceOff, curDepthOut += curDepthOff_) {
            if (!onlyTail) {
                if (addMulti) {
                    for (uint64_t curKhSlice = srcOffset, tmpBufOffset = curDepthOut; curKhSlice < srcOffset + sliceOff; curKhSlice += (kW * blockLength), tmpBufOffset += dH * curPartOutW * blockLength) {
                        Copy(tmpBuf[tmpBufOffset], srcTensor[curKhSlice], vecProcLength, im2colAddRepeatTimes, Im2colNewParams);
                    } 
                } else { // kW can be passed in a single repeat => repeatTimes = kH can take place
                    Copy(tmpBuf[curDepthOut], srcTensor[srcOffset], vecProcLength, kH, Im2colNewParams);
                }
            }

            if (addTail) {
                auto tailTmpBuf = tmpBuf[curDepthOut], tailSrcBuf = srcTensor[srcOffset];
                
                if (!onlyTail) {
                    tailTmpBuf = tailTmpBuf[processedIm2colAddDataDst];
                    tailSrcBuf = tailSrcBuf[processedIm2colAddDataSrc];
                }
                
                Copy(tailTmpBuf, tailSrcBuf, im2colAddTail, kH, Im2colNewParamsTail); 
            }
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool overlapKernels = false>
    __aicore__ inline void Im2Col(uint64_t dOff, uint32_t& srcOffset, const LocalTensor<T>& srcTensor, const uint64_t& curDepthOff_, const uint64_t& curPartOutW, const uint64_t& curPartW,
                                  bool& addMulti, bool& addTail, bool& onlyTail, bool& overUint8) {
        if constexpr (overlapKernels) {
            for (uint64_t i = 0; i < curPartW; ++i, dOff += gmWOff) {
                Im2ColAddSingleKernel(dOff, srcOffset, srcTensor, curDepthOff_, curPartOutW, addMulti, addTail, onlyTail, overUint8);
            } 
        } else {
            if (this->im2colSingleWOut) [[likely]] {
                Im2ColCopySingleWOut(tmpBuf[dOff], srcTensor[srcOffset], curDepthOff_, curPartOutW, addMulti, addTail, onlyTail);
                srcOffset += curPartW * blockKernelDHW;
            } else {
                for (uint64_t i = 0; i < curPartW; ++i, dOff += gmWOff) {
                    Im2ColCopySingleKernel(dOff, srcOffset, srcTensor, curDepthOff_, curPartOutW, addMulti, addTail, onlyTail);
                } 
            }
        }
    }
    
   __aicore__ inline void StripPad(const uint64_t pD, const uint64_t pH, const uint64_t pW,
                                   const uint64_t realPartOutD, const uint64_t realPartOutH, const uint64_t realPartOutW,
                                   const uint64_t origPartOutD, const uint64_t origPartOutH, const uint64_t origPartOutW,
                                   bool smallSingleMask, bool doTail) {
        uint64_t origPartOutHW = origPartOutH * origPartOutW;
        uint64_t dstOffset = 0, srcOffset = 0;

        auto event0 = pipe.FetchEventID(HardEvent::V_S);
        LocalTensor<T> srcTensor = tmpBuf[(pD * origPartOutHW + pH * origPartOutW + pW) * blockLength];
        for (uint64_t curD = 0; curD < realPartOutD; ++curD) {
            if (!smallSingleMask) {
                for (uint64_t curHOffset = srcOffset; curHOffset < (srcOffset + realPartOutH * origPartOutW * blockLength); curHOffset += (origPartOutW * blockLength)) {        
                    if constexpr (blockLength == BLOCK_LEN_FP16) {
                        SetFlag<HardEvent::V_S>(event0);
                        WaitFlag<HardEvent::V_S>(event0); 
                    }
                    Copy(tmpBuf[dstOffset], srcTensor[curHOffset], vecProcLength, stripPadRepeatTimes, PadStripParams);
                    if (doTail) {
                        Copy(tmpBuf[dstOffset + processedStripPadData], srcTensor[curHOffset + processedStripPadData], stripPadTail, 1, PadStripParams);
                    }
                    dstOffset += (realPartOutW * blockLength);
                }
            } else {
                const uint64_t copyLen = doTail ? stripPadTail : vecProcLength;
                const uint64_t wholeRepeatTimes = realPartOutH / MAX_REPEAT_TIMES;
                const uint64_t tailRepeatTimes = realPartOutH % MAX_REPEAT_TIMES;
                const uint64_t cycleOffsetDst = MAX_REPEAT_TIMES * PadStripParams.dstRepeatSize * blockLength;
                const uint64_t cycleOffsetSrc = MAX_REPEAT_TIMES * PadStripParams.srcRepeatSize * blockLength;
                const uint64_t wholeOffsetDst = wholeRepeatTimes * cycleOffsetDst;
                const uint64_t wholeOffsetSrc = wholeRepeatTimes * cycleOffsetSrc;
                
                for (uint64_t i = 0; i < wholeRepeatTimes; i++) {
                    Copy(tmpBuf[i * cycleOffsetDst + dstOffset], srcTensor[i * cycleOffsetSrc + srcOffset], copyLen, MAX_REPEAT_TIMES, PadStripParams);
                }
                Copy(tmpBuf[wholeOffsetDst + dstOffset], srcTensor[wholeOffsetSrc + srcOffset], copyLen, tailRepeatTimes, PadStripParams);

                dstOffset += realPartOutH * realPartOutW * blockLength;
            }
            srcOffset += origPartOutHW * blockLength;
        }
    }
    
    template <typename V>
    __aicore__ inline void TransposeBack(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint64_t& partNC, const uint64_t& dataLen) {
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        LocalTensor<float> tmp = srcTensor[blockLength];
        LocalTensor<float> tmp2 = dstTensor[blockLength];
        for (int64_t i = 0; i < blockLength; ++i) {
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i] = (uint64_t)srcTensor[NCHW_CONV_ADDR_LIST_SIZE * i].GetPhyAddr();
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i + 1] = (uint64_t)tmp[NCHW_CONV_ADDR_LIST_SIZE * i].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i] = (uint64_t)dstTensor[i * dataLen].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i + 1] = (uint64_t)tmp2[i * dataLen].GetPhyAddr();
        }

        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParamsReverse);
        PipeBarrier<PIPE_V>();
        CastBF16Back(dstTensor, partNC, dataLen, sizeof(V) == sizeof(bfloat16_t)); 
    }
    
    template <typename V>
    __aicore__ inline void TransposeBack(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor, const uint64_t& partNC, const uint64_t& dataLen) {
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        for (int64_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; ++i) {
            srcLocalList[i] = (uint64_t)srcTensor[i * blockLength].GetPhyAddr();
            dstLocalList[i] = (uint64_t)dstTensor[i * dataLen].GetPhyAddr();
        }

        TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParamsReverse);
        CastBF16Back(dstTensor, partNC);
    }
    
    template <typename V, typename L>
    __aicore__ inline void TransposeBack(const LocalTensor<L>& dstTensor, const uint64_t& partNC, const uint64_t& dataLen) {
        TransposeBack<V>(dstTensor, tmpBuf, partNC, dataLen); 
    }
    
    __aicore__ inline void CastBF16Back(LocalTensor<half> src, uint64_t partNC) {
        queOutVals.template EnQue<half>(src);
    }
    
    __aicore__ inline void CastBF16Back(LocalTensor<float> src, uint64_t partNC, const uint64_t& dataLen, bool isBF16) {
        if (isBF16) {
            LocalTensor<S> tmpTensor = src.template ReinterpretCast<S>();
            if constexpr (std::is_same<S, bfloat16_t>::value) {
                Cast(tmpTensor, src, RoundMode::CAST_RINT, dataLen * partNC);
            } else if constexpr (std::is_same<S, half>::value) {
                Cast(tmpTensor, src, RoundMode::CAST_NONE, dataLen * partNC);
            }
            
            queOutVals.template EnQue<S>(tmpTensor);
        } else {
            queOutVals.template EnQue<float>(src);
        }
    }
    
  public:
    TPipe pipe;
    uint64_t batchesCurCore;
    uint64_t batchOffset;

  public:
    __aicore__ inline KernelMaxPool3DGradWithArgmaxBase(const uint64_t& inShapeD, const uint64_t& inShapeH, const uint64_t& inShapeW,
                                                        const uint64_t& outShapeD, const uint64_t& outShapeH, const uint64_t& outShapeW,
                                                        const uint64_t& kD, const uint64_t& kH, const uint64_t& kW,
                                                        const uint64_t& sD, const uint64_t& sH, const uint64_t& sW,
                                                        const uint64_t& pD, const uint64_t& pH, const uint64_t& pW,
                                                        const uint64_t& dD, const uint64_t& dH, const uint64_t& dW,
                                                        const uint64_t& partD, const uint64_t& partH, const uint64_t& partW,
                                                        const uint64_t& partOutD, const uint64_t& partOutH, const uint64_t& partOutW,
                                                        const uint64_t& batchesPerCore, const uint64_t& leftOverBatches,
                                                        const uint64_t& ceilD, const uint64_t& ceilH, const uint64_t& ceilW,
                                                        const uint64_t& sizeUb1, const uint64_t& sizeUb2, const uint64_t& valSize) :
                                                        depthInput(inShapeD), heightInput(inShapeH), widthInput(inShapeW),
                                                        depthOut(outShapeD),heightOut(outShapeH), widthOut(outShapeW),
                                                        kD(kD), kH(kH), kW(kW), sD(sD), sH(sH), sW(sW),
                                                        pD(pD), pH(pH), pW(pW), dD(dD), dH(dH), dW(dW),
                                                        partD(partD), partH(partH), partW(partW),
                                                        partOutD(partOutD), partOutH(partOutH), partOutW(partOutW),
                                                        ceilD(ceilD), ceilH(ceilH), ceilW(ceilW),
                                                        sizeUb1(sizeUb1), sizeUb2(sizeUb2), valSize(valSize) {
        ASSERT(coreNum != 0 && "block dim can not be zero!");
        batchesCurCore = batchesPerCore + 
                         (blockLength * (coreIdx + 1) <= leftOverBatches ? blockLength : (blockLength * coreIdx < leftOverBatches ? leftOverBatches % blockLength : 0));
        batchOffset = batchesPerCore * coreIdx + ((coreIdx < (leftOverBatches + blockLength - 1) / blockLength) ? blockLength * coreIdx : leftOverBatches);
        
        im2ColAddParams.dstBlkStride = dW;
        im2ColAddParams.src0BlkStride = dW;
        
        if (im2colAddRepeatTimes != 1) {
            if (!im2colSingleWOut) {
                Im2colNewParams.dstRepeatSize = VEC_PROC_BLOCKS;
                Im2colNewParams.srcRepeatSize = VEC_PROC_BLOCKS;
            }
            
            im2ColAddParams.dstRepStride = dW * VEC_PROC_BLOCKS;
            im2ColAddParams.src0RepStride = dW * VEC_PROC_BLOCKS;
            im2ColAddParams.src1RepStride = VEC_PROC_BLOCKS;
        } else {
            if (!im2colSingleWOut) {
                Im2colNewParams.dstRepeatSize = dH * partOutW;
                Im2colNewParams.srcRepeatSize = kW;
            }
            
            im2ColAddParams.dstRepStride = dH * partOutW;
            im2ColAddParams.src0RepStride = dH * partOutW;
            im2ColAddParams.src1RepStride = kW;
        }

        if (addParamsOverUint8) {
            im2ColAddParams.dstRepStride = dW * VEC_PROC_BLOCKS;
            im2ColAddParams.src0RepStride = dW * VEC_PROC_BLOCKS;
            im2ColAddParams.src1RepStride = VEC_PROC_BLOCKS;
        }

        if (stripPadRepeatTimes == 0 || (stripPadRepeatTimes == 1 && stripPadTail == 0)) {
            PadStripParams.dstRepeatSize = stripPadRepeatTimes == 0 ? (stripPadTail / blockLength) : vecProcBlocks; 
            PadStripParams.srcRepeatSize = partOutW;
        }
    }
    
    __aicore__ inline void GMInit(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR indices, GM_ADDR gradInput, GM_ADDR workspace) {
        uint64_t inputOffsetPerCore = batchOffset * inputSize;
        uint64_t outputOffsetPerCore = batchOffset * outSize;
        uint64_t inputPerCore = batchesCurCore * inputSize;
        uint64_t outputPerCore = batchesCurCore * outSize;

        gradOutputGm.SetGlobalBuffer(((__gm__ S*)gradOutput + inputOffsetPerCore), inputPerCore);
        indicesGm.SetGlobalBuffer(((__gm__ int32_t*)indices + inputOffsetPerCore), inputPerCore);
        gradInputGm.SetGlobalBuffer(((__gm__ S*)gradInput + outputOffsetPerCore), outputPerCore);
       
        InitGlobalMemory(gradInputGm, outputPerCore, S(0.f));
        auto event0 = pipe.FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(event0);
        WaitFlag<HardEvent::MTE3_S>(event0);
    }

    __aicore__ inline void UbInit() {
        pipe.InitBuffer(queInVals, 1, sizeUb1 * sizeof(T));
        pipe.InitBuffer(queInInds, 1, sizeUb1 * INT64_FP32 * sizeof(float));
        pipe.InitBuffer(queOutVals, 1, sizeUb2 * sizeof(T));

        TBuf<TPosition::VECCALC> worktBuf;
        pipe.InitBuffer(worktBuf, valSize * sizeof(T));

        transDataTensorUb1 = worktBuf.Get<T>();
        kernelIndexes = transDataTensorUb1;
        generalOffsets = kernelIndexes[blockAlKernelDHW].template ReinterpretCast<float>();
        masks = generalOffsets[TMP_TENSORS_NUM * partAlignRoundDhwInp].template ReinterpretCast<uint8_t>();
        tmpBuf = masks[maskSize].template ReinterpretCast<T>();
    }

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR indices, GM_ADDR gradInput, GM_ADDR workspace) {
        GMInit(gradOutput, self, indices, gradInput, workspace);

        PrepareBaseScalars();
        auto event0 = pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);

        UbInit();
    }
};

#endif
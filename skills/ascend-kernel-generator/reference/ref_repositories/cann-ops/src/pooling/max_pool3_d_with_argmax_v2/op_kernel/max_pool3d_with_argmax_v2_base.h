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
 * \file max_pool3d_with_argmax_v2_base.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_WITH_ARGMAX_V2_BASE_H
#define OPP_MAX_POOL3D_WITH_ARGMAX_V2_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BLOCK_LEN_FP32 = 8;
constexpr uint32_t BLOCK_LEN_UINT8 = 32;
constexpr uint32_t MIN_TRANSPOSE_ROWS = 16;
constexpr uint32_t MAX_VEC_ELEMS_PER_REP_FP32 = 64;
constexpr uint32_t BLOCKS_IN_REP = 8;

const uint32_t MAX_UINT16 = 65536;
const uint32_t MAX_DIV = 2;
const uint32_t BITS_UINT8 = 8;
const uint32_t INT8_INT16 = 2;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t DIMS_COUNT = 3;
const uint32_t TRANSDATA_MUTLIPLER = 2;
const uint32_t MASK_COUNT = 2; // use 2 masks: mask for max values and for nan values
const float EPS_COEFF = 0.5f;

template <typename T, typename S>
class KernelMaxPool3DWithArgmaxV2Base {
  private:
    uint32_t curBufPos;
    const uint32_t coreNum = GetBlockNum();

  protected:
    const uint32_t coreIdx = GetBlockIdx();
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
    const float minFloat;
    const uint64_t ceilD;
    const uint64_t ceilH;
    const uint64_t ceilW;
    
    uint32_t roundW;
    uint32_t roundDepthInput;
    uint32_t roundHeightInput;
    uint32_t roundWidthInput;
    uint32_t alContOutSize;
    uint32_t partRoundDhwInp;
    uint32_t partHwInp;

    const uint64_t sizeUb1;
    const uint64_t sizeUb2;
    const uint64_t valSize;

    LocalTensor<T> transDataTensorUb1;
    LocalTensor<T> kernelIndexes;
    LocalTensor<int> dKernelStart;
    LocalTensor<int> hKernelStart;
    LocalTensor<int> wKernelStart;

    const uint32_t partOutD = OutputCalcD(partD);
    const uint32_t partOutH = OutputCalcH(partH);
    const uint32_t partOutW = OutputCalcW(partW);
    
    const uint32_t roundPartOutSize = partOutD * partOutH * partOutW;
    const uint32_t transAlignRoundPartOutSize = RoundUpBlock(roundPartOutSize, MIN_TRANSPOSE_ROWS);
    
    const uint32_t blockLength = BLOCK_SIZE / sizeof(T);
    const uint32_t blockLengthS = BLOCK_SIZE / sizeof(S);

    const uint32_t outSize = depthOut * heightOut * widthOut;
    const uint32_t hwInputSize = heightInput * widthInput;
    const uint32_t inputSize = depthInput * hwInputSize;
    const uint32_t kernelDHW = kD * kH * kW;
    const uint32_t alKernelDHW = RoundUpBlock(kernelDHW);
    const uint32_t blockKernelDHW = kernelDHW * blockLength;
    const uint32_t blockAlKernelDHW = alKernelDHW * blockLength;
    const bool kernelIsBlock = (kernelDHW <= blockLength) && (blockLength == BLOCK_LEN_FP32);
    
    const uint32_t halfKernelDHW = kernelDHW / MAX_DIV;
    const uint32_t gmWOff = sW * blockLength;
    uint32_t gmDOff;
    uint32_t gmHOff;
    const uint32_t maxTmpOff = (kernelDHW - halfKernelDHW) * blockLength;
    const uint32_t sliceOff = kH * kW * blockLength;
    uint32_t curDepthOff;
    TransDataTo5HDParams transDataParamsReverse;
    TransDataTo5HDParams transDataParams;

    GlobalTensor<S> xGm;
    GlobalTensor<S> yGm;
    GlobalTensor<float> indicesGm;

    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOutVals;
    TQue<TPosition::VECOUT, 1> queOutInds;
    TPipe *pipe = nullptr;

    uint64_t batchesCurCore;
    uint64_t batchOffset;
    const uint64_t mask = BLOCKS_IN_REP * blockLength;
  public:
    __aicore__ inline KernelMaxPool3DWithArgmaxV2Base(const uint64_t& inShapeD, const uint64_t& inShapeH, const uint64_t& inShapeW,
                                                      const uint64_t& outShapeD, const uint64_t& outShapeH, const uint64_t& outShapeW,
                                                      const uint64_t& kD, const uint64_t& kH, const uint64_t& kW,
                                                      const uint64_t& sD, const uint64_t& sH, const uint64_t& sW,
                                                      const uint64_t& pD, const uint64_t& pH, const uint64_t& pW,
                                                      const uint64_t& dD, const uint64_t& dH, const uint64_t& dW,
                                                      const uint64_t& partD, const uint64_t& partH, const uint64_t& partW,
                                                      const float& minFloat, const uint64_t& batchesPerCore, const uint64_t& leftOverBatches,
                                                      const uint64_t& ceilD, const uint64_t& ceilH, const uint64_t& ceilW,
                                                      const uint64_t& sizeUb1, const uint64_t& sizeUb2, const uint64_t& valSize
                                                      ) : depthInput(inShapeD), heightInput(inShapeH), widthInput(inShapeW),
                                                      depthOut(outShapeD), heightOut(outShapeH), widthOut(outShapeW),
                                                      kD(kD), kH(kH), kW(kW), sD(sD), sH(sH), sW(sW),
                                                      pD(pD), pH(pH), pW(pW), dD(dD), dH(dH), dW(dW),
                                                      partD(partD), partH(partH), partW(partW),
                                                      minFloat(minFloat), ceilD(ceilD), ceilH(ceilH), ceilW(ceilW),
                                                      sizeUb1(sizeUb1), sizeUb2(sizeUb2), valSize(valSize){
        ASSERT(coreNum != 0 && "block dim can not be zero!");
        
        // Split data betweeen cores :
        // 1) split blocks with 8 channels between cores
        // 2) then remaining channels give one of cores
        batchesCurCore = batchesPerCore +
                         (blockLength * (coreIdx + 1) <= leftOverBatches ? blockLength : (blockLength * coreIdx < leftOverBatches ? leftOverBatches % blockLength : 0));
        batchOffset = batchesPerCore * coreIdx + ((coreIdx < (leftOverBatches + blockLength - 1) / blockLength) ? blockLength * coreIdx : leftOverBatches);
    }

    __aicore__ inline void GMInit(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace) {
        uint64_t inputOffsetPerCore = batchOffset * inputSize;
        uint64_t outputOffsetPerCore = batchOffset * outSize;
        uint64_t inputPerCore = batchesCurCore * inputSize;
        uint64_t outputPerCore = batchesCurCore * outSize;
        
        xGm.SetGlobalBuffer(((__gm__ S*)x + inputOffsetPerCore), inputPerCore);
        yGm.SetGlobalBuffer(((__gm__ S*)y + outputOffsetPerCore), outputPerCore);
        indicesGm.SetGlobalBuffer(((__gm__ float*)indices + outputOffsetPerCore), outputPerCore);
    }
    
    __aicore__ inline void UbInit() {
        pipe->InitBuffer(queIn, 1, sizeUb2 * sizeof(float));
        pipe->InitBuffer(queOutVals, 1, valSize * sizeof(T));
        pipe->InitBuffer(queOutInds, 1, valSize * sizeof(float));
        
        TBuf<TPosition::VECCALC> worktBuf;
        
        // Both sizeUb1 and blockAlKernelDHW are power of blockLength
        auto kernelStartSize = RoundUpBlock(roundPartOutSize, blockLength);
        auto maxUb = (blockAlKernelDHW + sizeUb1 + DIMS_COUNT * kernelStartSize) * sizeof(float);
        auto elemsInFloat = blockLength / BLOCK_LEN_FP32;
        pipe->InitBuffer(worktBuf, maxUb);
        transDataTensorUb1 = worktBuf.Get<T>();
        kernelIndexes = transDataTensorUb1[sizeUb1 * elemsInFloat];
        dKernelStart = kernelIndexes[blockAlKernelDHW * elemsInFloat].template ReinterpretCast<int>();
        hKernelStart = dKernelStart[kernelStartSize];
        wKernelStart = hKernelStart[kernelStartSize];

		auto event0 = pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);
    }

  private:
    __aicore__ inline void FindMax(LocalTensor<T> src, LocalTensor<T> dst, uint32_t length, uint8_t inpStride, uint32_t rowLen) {
        uint32_t halfOffset = length * blockLength;
        BinaryRepeatParams pars{1, 1, 1, static_cast<uint8_t>(kernelDHW - halfKernelDHW), inpStride, inpStride};
        uint32_t dstOff = 0, srcOff=0;
        for (uint32_t maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++, dstOff += MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength, srcOff += MAX_REPEAT_TIMES * inpStride * blockLength){
            uint32_t rep = 0;
            for (rep=0; BLOCKS_IN_REP * (rep + 1) <= length; rep++){
                Max(dst[dstOff + rep * mask],
                    src[srcOff + rep * mask],
                    src[halfOffset + srcOff +  + rep * mask],
                    mask, MAX_REPEAT_TIMES, pars);
            }
            Max(dst[rep * mask + dstOff],
                src[rep * mask + srcOff],
                src[halfOffset + rep * mask + srcOff],
                (length - rep * BLOCKS_IN_REP) * blockLength, MAX_REPEAT_TIMES, pars);
        }
        uint32_t rep = 0;
        for (rep=0; BLOCKS_IN_REP * (rep + 1) <= length; rep++){
            Max(dst[rep * mask + dstOff],
                src[rep * mask + srcOff],
                src[halfOffset + rep * mask + srcOff],
                mask, rowLen % MAX_REPEAT_TIMES, pars);
        }
        Max(dst[rep * mask + dstOff],
            src[rep * mask + srcOff],
            src[halfOffset + rep * mask + srcOff],
            (length - rep * BLOCKS_IN_REP) * blockLength, rowLen % MAX_REPEAT_TIMES, pars);
    }

    __aicore__ inline void FindMin(LocalTensor<T> src, LocalTensor<T> dst, uint32_t length, uint8_t inpStride, uint32_t rowLen) {
        uint32_t halfOffset = length * blockLength;
        BinaryRepeatParams pars{1, 1, 1, static_cast<uint8_t>(kernelDHW - halfKernelDHW), inpStride, inpStride};
        uint32_t dstOff = 0, srcOff=0;
        for (uint32_t maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++, dstOff += MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength, srcOff += MAX_REPEAT_TIMES * inpStride * blockLength){
            uint32_t rep = 0;
            for (rep=0; BLOCKS_IN_REP * (rep + 1) <= length; rep++){
                Min(dst[rep * mask + dstOff],
                    src[rep * mask + srcOff],
                    src[halfOffset + rep * mask + srcOff],
                    mask, MAX_REPEAT_TIMES, pars);
            }
            Min(dst[rep * mask + dstOff],
                src[rep * mask + srcOff],
                src[halfOffset + rep * mask + srcOff],
                (length - rep * BLOCKS_IN_REP) * blockLength, MAX_REPEAT_TIMES, pars);
        }

        uint32_t rep = 0;
        for (rep=0; BLOCKS_IN_REP * (rep + 1) <= length; rep++){
            Min(dst[rep * mask + dstOff],
                src[rep * mask + srcOff],
                src[halfOffset + rep * mask + srcOff],
                mask, rowLen % MAX_REPEAT_TIMES, pars);
        }
        Min(dst[rep * mask + dstOff],
            src[rep * mask + srcOff],
            src[halfOffset + rep * mask + srcOff],
            (length - rep * BLOCKS_IN_REP) * blockLength, rowLen % MAX_REPEAT_TIMES, pars);
    }

    __aicore__ inline uint32_t OutputCalcD(const uint32_t& size) {
        return (size - dD * (kD - 1) + sD - 1) / sD;
    }

    __aicore__ inline uint32_t OutputCalcH(const uint32_t& size) {
        return (size - dH * (kH - 1) + sH - 1) / sH;
    }

    __aicore__ inline void CopyOutVals(const uint32_t& dstOffset,  const uint32_t& partNC, const uint32_t& partOut);

  protected:
    __aicore__ inline uint32_t RoundUpBlock(const uint32_t& src, const uint32_t& blockLen = BLOCK_SIZE / sizeof(T)) const {
        if (blockLen != 0) {
            return src != 0 ? (src + blockLen - 1) / blockLen * blockLen : blockLen;
        }
        return blockLen;
    }

    __aicore__ inline uint32_t RoundDownBlock(const uint32_t& src, const uint32_t& blockLen = BLOCK_SIZE / sizeof(T)) {
        if (blockLen != 0) {
            return (src / blockLen) * blockLen;
        }
        return blockLen;
    }

    __aicore__ inline uint32_t OutputCalcW(const uint32_t& size) {
        return (size - dW * (kW - 1) + sW - 1) / sW;
    }
    
    __aicore__ inline void CreateKernelIndexes() {
        ArithProgression<T>(transDataTensorUb1, (T)0.f, (T)1.f, alKernelDHW);
        PipeBarrier<PIPE_V>();
        auto event0 = this->pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);
        
        Brcb(kernelIndexes, transDataTensorUb1, alKernelDHW / BLOCKS_IN_REP, {1, BLOCKS_IN_REP}); // Can't be done in place
    }

    __aicore__ inline void PrepareInput(LocalTensor<half> srcUb, uint32_t dataSize) {
        const uint32_t partAlignDhwInp = RoundUpBlock(dataSize);
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        
        for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
            dstLocalList[i] = (uint64_t)transDataTensorUb1[i * blockLength].GetPhyAddr();
            srcLocalList[i] = (uint64_t)srcUb[i * partAlignDhwInp].GetPhyAddr();
        }

        TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams); // Result store in transDataTensorUb1
        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void PrepareInput(LocalTensor<float> srcUb, uint32_t dataSize){
        const uint32_t partAlignDhwInp = RoundUpBlock(dataSize, blockLengthS);
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
      
        LocalTensor<float> tmp = transDataTensorUb1[blockLength * blockLength];
        LocalTensor<float> tmp2 = srcUb[blockLength];
        for (int32_t i = 0; i < blockLength; i++) {
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i] = (uint64_t)transDataTensorUb1[i * blockLength].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i + 1] = (uint64_t)tmp[i * blockLength].GetPhyAddr();
            srcLocalList[i] = (uint64_t)srcUb[i * partAlignDhwInp].GetPhyAddr();
            srcLocalList[i + blockLength] = (uint64_t)tmp2[i * partAlignDhwInp].GetPhyAddr();
        }

        TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams); // Result store in transDataTensorUb1
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CastInds(LocalTensor<float> idxs, LocalTensor<float> src, uint32_t dhVal, uint32_t rowLen) {
        uint32_t rep = this->RoundDownBlock(rowLen * blockLength, MAX_VEC_ELEMS_PER_REP_FP32);
        uint32_t wholeReps = rep / MAX_VEC_ELEMS_PER_REP_FP32;
        uint32_t bigReps = wholeReps / MAX_REPEAT_TIMES;
        uint32_t remWholeReps = wholeReps - bigReps * MAX_REPEAT_TIMES;
        uint32_t remPart = rowLen * blockLength - rep;
        uint32_t maxRep = 0;

        for (maxRep = 0; maxRep + (MAX_REPEAT_TIMES - 1) * MAX_VEC_ELEMS_PER_REP_FP32 < rep; maxRep += MAX_REPEAT_TIMES * MAX_VEC_ELEMS_PER_REP_FP32){
            Copy(idxs[dhVal + maxRep], src[maxRep], MAX_VEC_ELEMS_PER_REP_FP32, MAX_REPEAT_TIMES, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        }
        Copy(idxs[dhVal + maxRep], src[maxRep], MAX_VEC_ELEMS_PER_REP_FP32, remWholeReps, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        Copy(idxs[dhVal + rep], src[rep], remPart, 1, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void CastInds(LocalTensor<float> idxs, LocalTensor<half> src, uint32_t dhVal, uint32_t rowLen) {
        Cast(idxs[dhVal], src, RoundMode::CAST_NONE, rowLen * blockLength);
        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void CastBF16(LocalTensor<bfloat16_t> src, LocalTensor<float> dst, uint32_t partNC) {
        const uint32_t partAlignDhwInp = partNC * RoundUpBlock(partRoundDhwInp, blockLengthS);
        const uint32_t repeatTimes = partAlignDhwInp / mask;
        const uint32_t wholeRepeatTimes = repeatTimes / MAX_REPEAT_TIMES;
        const uint32_t tailRepeatTimes = repeatTimes % MAX_REPEAT_TIMES;
        const uint32_t tailElemsPerRepeat = partAlignDhwInp % mask;

        const uint32_t wholeOffset = wholeRepeatTimes * MAX_REPEAT_TIMES * mask;
        const uint32_t tailOffset = wholeOffset + tailRepeatTimes * mask;

        Cast(transDataTensorUb1, src, RoundMode::CAST_NONE, partAlignDhwInp);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
            Copy(dst[i * MAX_REPEAT_TIMES * mask], transDataTensorUb1[i * MAX_REPEAT_TIMES * mask], mask, MAX_REPEAT_TIMES, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        }
        Copy(dst[wholeOffset], transDataTensorUb1[wholeOffset], mask, tailRepeatTimes, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        Copy(dst[tailOffset], transDataTensorUb1[tailOffset], tailElemsPerRepeat, 1, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CastBF16(LocalTensor<float> src, LocalTensor<float> dst, uint32_t partNC){}

    __aicore__ inline void CastBF16(LocalTensor<half> src, LocalTensor<half> dst, uint32_t partNC){}
    
    __aicore__ inline void CastBF16Back(LocalTensor<half> src, uint32_t partNC){
        queOutVals.EnQue<half>(src);
    }

    __aicore__ inline void CastBF16BackFloat(LocalTensor<float> src, uint32_t partNC){
        queOutVals.EnQue<float>(src);
    }

    __aicore__ inline void CastBF16BackBFloat(LocalTensor<float> src, uint32_t partNC){
        LocalTensor<bfloat16_t> bf16TmpTensor = src. template ReinterpretCast<bfloat16_t>(); 
        Cast(bf16TmpTensor, src, RoundMode::CAST_ROUND, transAlignRoundPartOutSize * partNC);
        PipeBarrier<PIPE_V>();

        queOutVals.EnQue<bfloat16_t>(bf16TmpTensor);
    }
    
    __aicore__ inline void PrepareBaseScalars(){
        partRoundDhwInp = partD * partHwInp;

        curDepthOff = dD * partHwInp * blockLength;
        
        gmDOff = (sD * partHwInp - partOutH * sH * partW) * blockLength;
        gmHOff = (sH * partW - partOutW * sW) * blockLength;
        
        transDataParamsReverse.dstHighHalf = false;
        transDataParamsReverse.srcHighHalf = false;
        transDataParamsReverse.repeatTimes = transAlignRoundPartOutSize / NCHW_CONV_ADDR_LIST_SIZE;
        transDataParamsReverse.dstRepStride = (transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / blockLength;
        transDataParamsReverse.srcRepStride = (transDataParamsReverse.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;

        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = RoundUpBlock(partRoundDhwInp, MIN_TRANSPOSE_ROWS) / NCHW_CONV_ADDR_LIST_SIZE;
        transDataParams.dstRepStride = (transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;
        transDataParams.srcRepStride = (transDataParams.repeatTimes == 1) ? 0 : NCHW_CONV_ADDR_LIST_SIZE / blockLength;
    }

    __aicore__ inline void FindMaxRowPerKernel(LocalTensor<T> &src, LocalTensor<T> &dst, LocalTensor<T> &tmp, uint32_t dhVal, uint32_t rowLen) {
        // Find Max for each row by width
        uint32_t halfInd = halfKernelDHW;
        uint32_t halfOffset = halfInd * blockLength;
        uint32_t wholeDataRow = blockKernelDHW * rowLen;
        auto maxTmpLen = maxTmpOff * rowLen;
        
        for (uint32_t curWidthOut = 0, dstOff = 0; curWidthOut < wholeDataRow; curWidthOut+=blockKernelDHW, dstOff+=maxTmpOff) {
            Max(tmp[dstOff], src[curWidthOut], src[curWidthOut + halfOffset], halfOffset);
        }

        if (kernelDHW % MAX_DIV == 1) {
            const uint32_t wholeRepeatTimes = rowLen / MAX_REPEAT_TIMES;
            const uint32_t tailRepeatTimes = rowLen % MAX_REPEAT_TIMES;
            const uint32_t wholeOffsetDst = wholeRepeatTimes * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength;
            const uint32_t wholeOffsetSrc = wholeRepeatTimes * MAX_REPEAT_TIMES * blockKernelDHW;

            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(tmp[i * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength + halfOffset], src[i * MAX_REPEAT_TIMES * blockKernelDHW + blockKernelDHW - blockLength], 
                     blockLength, MAX_REPEAT_TIMES, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
            }
            Copy(tmp[wholeOffsetDst + halfOffset], src[wholeOffsetSrc + blockKernelDHW - blockLength], 
                 blockLength, tailRepeatTimes, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
        }
        PipeBarrier<PIPE_V>();
            
        LocalTensor<T> curMax = tmp;
        for (uint32_t ind = kernelDHW - halfInd; ind > MAX_DIV; ind -= halfInd) {
            halfInd = ind / MAX_DIV;
            halfOffset = halfInd * blockLength;
            
            for (uint32_t curWidthOut = 0; curWidthOut < maxTmpLen; curWidthOut+=maxTmpOff) {
                Max(curMax[curWidthOut + halfOffset], curMax[curWidthOut], curMax[curWidthOut + halfOffset], halfOffset);
            }
            curMax = curMax[halfOffset];
            PipeBarrier<PIPE_V>();
        }
        if (kernelDHW > MAX_DIV) {
            for (uint32_t curWidthOut = 0, dstOff = dhVal; curWidthOut < maxTmpLen; curWidthOut+=maxTmpOff, dstOff+=blockLength) {
                Max(dst[dstOff], curMax[curWidthOut], curMax[curWidthOut + blockLength], blockLength);
            }
        } else {
            const uint32_t roundDownRepeatTimes = (blockLength * partOutW) / mask;
            const uint32_t tailElemsPerRepeat = (blockLength * partOutW) % mask;
            Copy(dst[dhVal], curMax, mask, roundDownRepeatTimes, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
            Copy(dst[dhVal + mask * roundDownRepeatTimes], curMax[mask * roundDownRepeatTimes], tailElemsPerRepeat, 1, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FindMaxRowPerRepeat(LocalTensor<T> &src, LocalTensor<T> &dst, LocalTensor<T> &tmp, uint32_t dhVal, uint32_t rowLen) {
        DataCopyParams copyEvenParams = {static_cast<uint16_t>(rowLen), 1, static_cast<uint16_t>(kernelDHW - 1), static_cast<uint16_t>((kernelDHW - 1) / MAX_DIV)};
        uint32_t halfInd = halfKernelDHW;
        uint32_t halfOffset = halfInd * blockLength;
        uint32_t wholeDataRow = blockKernelDHW * rowLen;
        auto maxTmpLen = maxTmpOff * rowLen;
        
        if (kernelDHW < MAX_REPEAT_TIMES && (rowLen > 1 + halfKernelDHW / BLOCKS_IN_REP)){
            FindMax(src, tmp, halfKernelDHW, static_cast<uint8_t>(kernelDHW), rowLen);
        } else {
            for (uint32_t curWidthOut = 0, dstOff = 0; curWidthOut < wholeDataRow; curWidthOut+=blockKernelDHW, dstOff+=maxTmpOff) {
                Max(tmp[dstOff], src[curWidthOut], src[curWidthOut + halfOffset], halfOffset);
            }
        }

        if (kernelDHW % MAX_DIV == 1) {
            const uint32_t wholeRepeatTimes = rowLen / MAX_REPEAT_TIMES;
            const uint32_t tailRepeatTimes = rowLen % MAX_REPEAT_TIMES;
            const uint32_t wholeOffsetDst = wholeRepeatTimes * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength;
            const uint32_t wholeOffsetSrc = wholeRepeatTimes * MAX_REPEAT_TIMES * blockKernelDHW;

            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(tmp[i * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength + halfOffset], src[i * MAX_REPEAT_TIMES * blockKernelDHW + blockKernelDHW - blockLength], 
                     blockLength, MAX_REPEAT_TIMES, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
            }
            Copy(tmp[wholeOffsetDst + halfOffset], src[wholeOffsetSrc + blockKernelDHW - blockLength], 
                 blockLength, tailRepeatTimes, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<T> curMax = tmp;
        for (uint32_t ind = kernelDHW - halfInd; ind > MAX_DIV; ind -= halfInd) {
            halfInd = ind / MAX_DIV;
            halfOffset = halfInd * blockLength;
            
            FindMax(curMax, curMax[halfOffset], halfInd, static_cast<uint8_t>(kernelDHW - halfKernelDHW), rowLen);
            
            curMax = curMax[halfOffset];
            PipeBarrier<PIPE_V>();
        }
        
        if (kernelDHW > MAX_DIV) {
            uint32_t maxRep = 0;
            for (maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++, dhVal+=MAX_REPEAT_TIMES * blockLength){
                Max(dst[dhVal], curMax[maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength],
                    curMax[blockLength + maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength], blockLength, MAX_REPEAT_TIMES,
                    {1, 1, 1, 1, static_cast<uint8_t>(kernelDHW - halfKernelDHW), static_cast<uint8_t>(kernelDHW - halfKernelDHW)});
            }
            Max(dst[dhVal], curMax[maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength],
                curMax[blockLength + maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength], blockLength, rowLen % MAX_REPEAT_TIMES,
                {1, 1, 1, 1, static_cast<uint8_t>(kernelDHW - halfKernelDHW), static_cast<uint8_t>(kernelDHW - halfKernelDHW)});
        } else {
            const uint32_t roundDownRepeatTimes = (blockLength * rowLen) / mask;
            const uint32_t tailElemsPerRepeat = (blockLength * rowLen) % mask;
            Copy(dst[dhVal], curMax, mask, roundDownRepeatTimes, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
            Copy(dst[dhVal + mask * roundDownRepeatTimes], curMax[mask * roundDownRepeatTimes], tailElemsPerRepeat, 1, {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FindMinRowPerKernel(LocalTensor<T> &src, LocalTensor<T> &tmp, uint32_t rowLen) {
        // Find Max for each row by width
        uint32_t halfInd = halfKernelDHW;
        uint32_t halfOffset = halfInd * blockLength;
        uint32_t wholeDataRow = blockKernelDHW * rowLen;
        auto maxTmpLen = maxTmpOff * rowLen;
        
        for (uint32_t curWidthOut = 0, dstOff = 0; curWidthOut < wholeDataRow; curWidthOut+=blockKernelDHW, dstOff+=maxTmpOff) {
            Min(tmp[dstOff], src[curWidthOut], src[curWidthOut + halfOffset], halfOffset);
        }

        if (kernelDHW % MAX_DIV == 1) {
            const uint32_t wholeRepeatTimes = rowLen / MAX_REPEAT_TIMES;
            const uint32_t tailRepeatTimes = rowLen % MAX_REPEAT_TIMES;
            const uint32_t wholeOffsetDst = wholeRepeatTimes * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength;
            const uint32_t wholeOffsetSrc = wholeRepeatTimes * MAX_REPEAT_TIMES * blockKernelDHW;

            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(tmp[i * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength + halfOffset], src[i * MAX_REPEAT_TIMES * blockKernelDHW + blockKernelDHW - blockLength], 
                     blockLength, MAX_REPEAT_TIMES, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
            }
            Copy(tmp[wholeOffsetDst + halfOffset], src[wholeOffsetSrc + blockKernelDHW - blockLength], 
                 blockLength, tailRepeatTimes, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
        }
        PipeBarrier<PIPE_V>();
            
        LocalTensor<T> curMax = tmp;
        for (uint32_t ind = kernelDHW - halfInd; ind > MAX_DIV; ind -= halfInd) {
            halfInd = ind / MAX_DIV;
            halfOffset = halfInd * blockLength;
            
            for (uint32_t curWidthOut = 0; curWidthOut < maxTmpLen; curWidthOut+=maxTmpOff) {
                Min(curMax[curWidthOut + halfOffset], curMax[curWidthOut], curMax[curWidthOut + halfOffset], halfOffset);
            }
            curMax = curMax[halfOffset];
            PipeBarrier<PIPE_V>();
        }
        if (kernelDHW > MAX_DIV) {
            for (uint32_t curWidthOut = 0, dstOff = 0; curWidthOut < maxTmpLen; curWidthOut+=maxTmpOff, dstOff+=blockLength) {
                Min(tmp[dstOff], curMax[curWidthOut], curMax[curWidthOut + blockLength], blockLength);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FindMinRowPerRepeat(LocalTensor<T> &src, LocalTensor<T> &tmp, uint32_t rowLen) {
        uint32_t halfInd = halfKernelDHW;
        uint32_t halfOffset = halfInd * blockLength;
        uint32_t wholeDataRow = blockKernelDHW * rowLen;
        auto maxTmpLen = maxTmpOff * rowLen;
        
        if (kernelDHW < MAX_REPEAT_TIMES && (rowLen > 1 + halfKernelDHW / BLOCKS_IN_REP)){
            FindMin(src, tmp, halfKernelDHW, static_cast<uint8_t>(kernelDHW), rowLen);
        } else {
            for (uint32_t curWidthOut = 0, dstOff = 0; curWidthOut < wholeDataRow; curWidthOut+=blockKernelDHW, dstOff+=maxTmpOff) {
                Min(tmp[dstOff], src[curWidthOut], src[curWidthOut + halfOffset], halfOffset);
            }
        }

        if (kernelDHW % MAX_DIV == 1) {
            const uint32_t wholeRepeatTimes = rowLen / MAX_REPEAT_TIMES;
            const uint32_t tailRepeatTimes = rowLen % MAX_REPEAT_TIMES;
            const uint32_t wholeOffsetDst = wholeRepeatTimes * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength;
            const uint32_t wholeOffsetSrc = wholeRepeatTimes * MAX_REPEAT_TIMES * blockKernelDHW;
            
            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(tmp[i * MAX_REPEAT_TIMES * (halfKernelDHW + 1) * blockLength + halfOffset], src[i * MAX_REPEAT_TIMES * blockKernelDHW + blockKernelDHW - blockLength], 
                     blockLength, MAX_REPEAT_TIMES, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
            }
            Copy(tmp[wholeOffsetDst + halfOffset], src[wholeOffsetSrc + blockKernelDHW - blockLength], 
                 blockLength, tailRepeatTimes, {1, 1, static_cast<uint16_t>(halfKernelDHW + 1), static_cast<uint16_t>(kernelDHW)});
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<T> curMax = tmp;
        for (uint32_t ind = kernelDHW - halfInd; ind > MAX_DIV; ind -= halfInd) {
            halfInd = ind / MAX_DIV;
            halfOffset = halfInd * blockLength;
            
            FindMin(curMax, curMax[halfOffset], halfInd, static_cast<uint8_t>(kernelDHW - halfKernelDHW), rowLen);
            
            curMax = curMax[halfOffset];
            PipeBarrier<PIPE_V>();
        }
        
        if (kernelDHW > MAX_DIV) {
            uint32_t maxRep = 0, dhVal = 0;
            for (maxRep = 0, dhVal=0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++, dhVal+=MAX_REPEAT_TIMES * blockLength){
                Min(tmp[dhVal], curMax[maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength],
                    curMax[blockLength + maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength], blockLength, MAX_REPEAT_TIMES,
                    {1, 1, 1, 1, static_cast<uint8_t>(kernelDHW - halfKernelDHW), static_cast<uint8_t>(kernelDHW - halfKernelDHW)});
            }
            Min(tmp[dhVal], curMax[maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength],
                curMax[blockLength + maxRep * MAX_REPEAT_TIMES * (kernelDHW - halfKernelDHW) * blockLength], blockLength, rowLen % MAX_REPEAT_TIMES,
                {1, 1, 1, 1, static_cast<uint8_t>(kernelDHW - halfKernelDHW), static_cast<uint8_t>(kernelDHW - halfKernelDHW)});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareSelect(LocalTensor<T> &maxResult, LocalTensor<T> &srcDst, LocalTensor<uint8_t> &masks, LocalTensor<uint8_t> &nanmasks, uint32_t rowLen, uint32_t dhVal) {
        auto wholeRow = blockKernelDHW * rowLen;
        auto alInt8Offset = RoundUpBlock(blockKernelDHW / BITS_UINT8, BLOCK_LEN_UINT8);
        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(mask);
        for (uint32_t curWidthOut = 0, dstOff = 0, blocks = dhVal; curWidthOut < wholeRow; curWidthOut += blockKernelDHW, blocks += blockLength, dstOff += alInt8Offset) {
            Compare<T, uint8_t, false>(masks[dstOff], maxResult[blocks], srcDst[curWidthOut], CMPMODE::EQ, MASK_PLACEHOLDER, blockAlKernelDHW / mask,
                                       {1, 0, 1, BLOCKS_IN_REP, 0, BLOCKS_IN_REP});
            Compare<T>(nanmasks[dstOff], srcDst[curWidthOut], srcDst[curWidthOut], CMPMODE::EQ, blockAlKernelDHW);
        }
        PipeBarrier<PIPE_V>();
        auto event0 = pipe->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(event0);
        WaitFlag<HardEvent::V_S>(event0);

        for (uint32_t curWidthOut = 0, srcOff = 0; curWidthOut < wholeRow; curWidthOut += blockKernelDHW, srcOff += alInt8Offset) {
            Select<T>(srcDst[curWidthOut], masks[srcOff], kernelIndexes, 1.f * kernelDHW, SELMODE::VSEL_TENSOR_SCALAR_MODE, blockKernelDHW);
        }
        PipeBarrier<PIPE_V>();

        Muls(kernelIndexes, kernelIndexes, (T)-1.f, blockKernelDHW);
        PipeBarrier<PIPE_V>();

        for (uint32_t curWidthOut = 0, srcOff = 0; curWidthOut < wholeRow; curWidthOut += blockKernelDHW, srcOff += alInt8Offset) {
            Select<T>(srcDst[curWidthOut], nanmasks[srcOff], srcDst[curWidthOut], kernelIndexes, SELMODE::VSEL_TENSOR_TENSOR_MODE, blockKernelDHW);
        }
        PipeBarrier<PIPE_V>();

        Muls(kernelIndexes, kernelIndexes, (T)-1.f, blockKernelDHW);
    }
    
    __aicore__ inline void CompareSelectBlockKernel(LocalTensor<T> &maxResult, LocalTensor<T> &srcDst, LocalTensor<uint8_t> &masks, LocalTensor<uint8_t> &nanmasks, uint32_t rowLen, uint32_t dhVal) {
        auto alInt8Offset = RoundUpBlock(blockKernelDHW, BLOCK_LEN_UINT8);
        auto maskOff = (MAX_REPEAT_TIMES + 1) * BLOCKS_IN_REP;

        uint32_t maxRep = 0;
        for (maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++) {
            Compare<T>(masks[maskOff * maxRep],
                       maxResult[dhVal + MAX_REPEAT_TIMES * maxRep * blockLength],
                       srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], CMPMODE::EQ, blockKernelDHW, MAX_REPEAT_TIMES,
                       {1, 0, 1, static_cast<uint8_t>(kernelDHW), 1, static_cast<uint8_t>(kernelDHW)});
            Compare<T>(nanmasks[maskOff * maxRep],
                       srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                       srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], CMPMODE::EQ, blockKernelDHW, MAX_REPEAT_TIMES,
                       {1, 1, 1, static_cast<uint8_t>(kernelDHW), static_cast<uint8_t>(kernelDHW), static_cast<uint8_t>(kernelDHW)});
        }
        Compare<T>(masks[maskOff * maxRep],
                   maxResult[dhVal + MAX_REPEAT_TIMES * maxRep * blockLength],
                   srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], CMPMODE::EQ, blockKernelDHW, rowLen % MAX_REPEAT_TIMES,
                   {1, 0, 1, static_cast<uint8_t>(kernelDHW), 1, static_cast<uint8_t>(kernelDHW)});
        Compare<T>(nanmasks[maskOff * maxRep],
                   srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                   srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], CMPMODE::EQ, blockKernelDHW, (rowLen % MAX_REPEAT_TIMES),
                   {1, 1, 1, static_cast<uint8_t>(kernelDHW), static_cast<uint8_t>(kernelDHW), static_cast<uint8_t>(kernelDHW)});
        PipeBarrier<PIPE_V>();
        auto event0 = pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);

        for (maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++) {
            Select<T>(srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                      masks[maskOff * maxRep],
                      kernelIndexes, 1.f * kernelDHW, SELMODE::VSEL_TENSOR_SCALAR_MODE, blockKernelDHW, MAX_REPEAT_TIMES, {1, 1, 1, static_cast<uint8_t>(kernelDHW), 0, 0});
        }
        Select<T>(srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                  masks[maskOff * maxRep],
                  kernelIndexes, 1.f * kernelDHW, SELMODE::VSEL_TENSOR_SCALAR_MODE, blockKernelDHW, rowLen % MAX_REPEAT_TIMES, {1, 1, 1, static_cast<uint8_t>(kernelDHW), 0, 0});
        PipeBarrier<PIPE_V>();

        Muls(kernelIndexes, kernelIndexes, (T)-1.f, blockKernelDHW);
        PipeBarrier<PIPE_V>();

        for (maxRep = 0; maxRep < rowLen / MAX_REPEAT_TIMES; maxRep++) {
            Select<T>(srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                      nanmasks[maskOff * maxRep],
                      srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW], 
                      kernelIndexes, SELMODE::VSEL_TENSOR_TENSOR_MODE, blockKernelDHW, MAX_REPEAT_TIMES, {1, 1, 1, static_cast<uint8_t>(kernelDHW), static_cast<uint8_t>(kernelDHW), 0});
        }
        Select<T>(srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                  nanmasks[maskOff * maxRep],
                  srcDst[MAX_REPEAT_TIMES * maxRep * blockKernelDHW],
                  kernelIndexes, SELMODE::VSEL_TENSOR_TENSOR_MODE, blockKernelDHW, rowLen % MAX_REPEAT_TIMES, {1, 1, 1, static_cast<uint8_t>(kernelDHW), static_cast<uint8_t>(kernelDHW), 0});
        PipeBarrier<PIPE_V>();

        Muls(kernelIndexes, kernelIndexes, (T)-1.f, blockKernelDHW);
    }

    __aicore__ inline void IndexRecalcFirst(LocalTensor<float> &dTensor, LocalTensor<float> &hTensor, LocalTensor<float> &wTensor, uint32_t len) {
        float coeffH = 1.f / float(int(kW));
        float coeffD = coeffH / float(int(kH));
        float coeffW = coeffH;
        float epsD = EPS_COEFF * coeffD;
        float epsW = EPS_COEFF * coeffW;

        // Calculation kernel's offset
        Adds(hTensor, dTensor, epsD, len);
        Adds(wTensor, dTensor, epsW, len);
        PipeBarrier<PIPE_V>();

        // h component
        Muls(hTensor, hTensor, coeffD, len);
        // w component
        Muls(wTensor, wTensor, coeffW, len);
        PipeBarrier<PIPE_V>();

        // Calculation general offset
        Floor(hTensor, hTensor, len);
        Floor(wTensor, wTensor, len);
        PipeBarrier<PIPE_V>();

        Muls(hTensor, hTensor, -1.f * kH * kW, len);
        Muls(wTensor, wTensor, -1.f * kW, len);
        PipeBarrier<PIPE_V>();

        Add(hTensor, dTensor, hTensor, len);
        Add(wTensor, dTensor, wTensor, len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void IndexRecalcSecond(LocalTensor<float> &dTensor, LocalTensor<float> &hTensor, 
                                             LocalTensor<float> &wTensor, uint32_t len) {
        float coeffH = 1.f / float(int(this->kW));
        float coeffD = coeffH / float(int(this->kH));
        float epsD = EPS_COEFF * coeffD;
        float epsH = EPS_COEFF * coeffH;

        // Calculation kernel's offset
        Adds(dTensor, dTensor, epsD, len);
        Adds(hTensor, hTensor, epsH, len);
        PipeBarrier<PIPE_V>();

        Muls(dTensor, dTensor, coeffD, len);
        Muls(hTensor, hTensor, coeffH, len);
        Muls(wTensor, wTensor, float(int(this->dW)), len);
        PipeBarrier<PIPE_V>();

        Floor(dTensor, dTensor, len);
        Floor(hTensor, hTensor, len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void IndexRecalcThird(LocalTensor<int> &dTmp, LocalTensor<int> &hTmp, LocalTensor<int> &wTmp,
                                            LocalTensor<int> &dTensor, LocalTensor<int> &hTensor, LocalTensor<int> &wTensor, 
                                            LocalTensor<float> &idxs, uint32_t len) {
        auto tmpSize = this->RoundUpBlock(this->roundPartOutSize, BLOCK_LEN_FP32);

        for (uint32_t curChannel = 0; curChannel < len; curChannel+= transAlignRoundPartOutSize) {
            Add(dTensor[curChannel], dTmp, dTensor[curChannel], tmpSize);
            Add(hTensor[curChannel], hTmp, hTensor[curChannel], tmpSize);
            Add(wTensor[curChannel], wTmp, wTensor[curChannel], tmpSize);
        }
        PipeBarrier<PIPE_V>();
		auto event0 = pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event0);
        WaitFlag<HardEvent::S_V>(event0);

        Maxs(dTensor, dTensor, 0, len);
        Maxs(hTensor, hTensor, 0, len);
        Maxs(wTensor, wTensor, 0, len);
        PipeBarrier<PIPE_V>();

        Add(wTensor, wTensor, hTensor, len);
        PipeBarrier<PIPE_V>();

        LocalTensor<int> indicesTmpInt = idxs.ReinterpretCast<int>();
        Add(indicesTmpInt, wTensor, dTensor, len);
        PipeBarrier<PIPE_V>();

        queOutInds.template EnQue<float>(idxs);
    }

    __aicore__ void Im2ColNoDilation(const LocalTensor<T>& dst, uint32_t& dOff);
    __aicore__ void Im2ColHWDDilation(const LocalTensor<T>& dst, uint32_t& dOff);
    __aicore__ void Im2ColNoDilationBigKw(const LocalTensor<T>& dst, uint32_t& dOff);
    __aicore__ void Im2ColNoDilationPerRepeat(const LocalTensor<T>& dst, uint32_t& dOff);

    template <typename V>
    __aicore__ inline void TransposeBackVals(LocalTensor<float> &reduceMaxResult, uint32_t partNC,
                                             uint64_t* dstLocalList, uint64_t* srcLocalList) {
        LocalTensor<float> tmp = reduceMaxResult[blockLength];
        LocalTensor<float> tmp2 = transDataTensorUb1[blockLength];
        for (int32_t i = 0; i < blockLength; i++) {
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i] = (uint64_t)reduceMaxResult[NCHW_CONV_ADDR_LIST_SIZE * i].GetPhyAddr();
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i + 1] = (uint64_t)tmp[NCHW_CONV_ADDR_LIST_SIZE * i].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i] = (uint64_t)transDataTensorUb1[i * transAlignRoundPartOutSize].GetPhyAddr();
            dstLocalList[NCHW_CONV_ADDR_LIST_SIZE / blockLength * i + 1] = (uint64_t)tmp2[i * transAlignRoundPartOutSize].GetPhyAddr();
        }

        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParamsReverse); // Result store in transDataTensorUb1
        PipeBarrier<PIPE_V>();

        Copy(reduceMaxResult, transDataTensorUb1, mask, RoundUpBlock(valSize, mask) / mask,
             {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        PipeBarrier<PIPE_V>();

        if constexpr (sizeof(V) == sizeof(bfloat16_t)) {
            CastBF16BackBFloat(reduceMaxResult, partNC);
        } else {
            CastBF16BackFloat(reduceMaxResult, partNC);
        }
    }

    __aicore__ inline void TransposeBackVals(LocalTensor<half> &reduceMaxResult, uint32_t partNC,
                                             uint64_t* dstLocalList, uint64_t* srcLocalList) {
        for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
            srcLocalList[i] = (uint64_t)reduceMaxResult[i * blockLength].GetPhyAddr();
            dstLocalList[i] = (uint64_t)transDataTensorUb1[i * transAlignRoundPartOutSize].GetPhyAddr();
        }

        TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParamsReverse); // Result store in transDataTensorUb1
        PipeBarrier<PIPE_V>();

        Copy(reduceMaxResult, transDataTensorUb1, mask, RoundUpBlock(valSize, mask) / mask,
             {1, 1, BLOCKS_IN_REP, BLOCKS_IN_REP});
        PipeBarrier<PIPE_V>();

        CastBF16Back(reduceMaxResult, partNC);
    }

    __aicore__ inline void TransposeBackInds(LocalTensor<float> &idxs,
                                             uint64_t* dstLocalList, uint64_t* srcLocalList) {
        LocalTensor<float> tmp2 = idxs[blockLength * blockLength];
        for (int32_t i = 0; i < blockLength; i++) {
            srcLocalList[i] = (uint64_t)idxs[i * blockLength].GetPhyAddr();
            srcLocalList[i + blockLength] = (uint64_t)tmp2[i * blockLength].GetPhyAddr();
        }

        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParamsReverse); // Result store in dst
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransposeBackIndsHalf(LocalTensor<float> &idxs,
                                                 uint64_t* dstLocalList, uint64_t* srcLocalList) {
        LocalTensor<float> dst = this->transDataTensorUb1.template ReinterpretCast<float>();

        TransDataTo5HDParams transDataParamsReverseHalf;
        transDataParamsReverseHalf.dstHighHalf = false;
        transDataParamsReverseHalf.srcHighHalf = false;
        transDataParamsReverseHalf.repeatTimes = this->transAlignRoundPartOutSize / NCHW_CONV_ADDR_LIST_SIZE * TRANSDATA_MUTLIPLER;
        transDataParamsReverseHalf.dstRepStride = (transDataParamsReverseHalf.repeatTimes == TRANSDATA_MUTLIPLER) ? 0 : 1;
        transDataParamsReverseHalf.srcRepStride = (transDataParamsReverseHalf.repeatTimes == TRANSDATA_MUTLIPLER) ? 0 : NCHW_CONV_ADDR_LIST_SIZE;

        for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER; i++) {
            dstLocalList[TRANSDATA_MUTLIPLER * i] = (uint64_t)dst[i * this->transAlignRoundPartOutSize].GetPhyAddr();
            srcLocalList[i] = (uint64_t)idxs[i * NCHW_CONV_ADDR_LIST_SIZE].GetPhyAddr();
        }
        for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER; i++) {
            dstLocalList[TRANSDATA_MUTLIPLER * i + 1] = (uint64_t)dst[(NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER + i) * this->transAlignRoundPartOutSize].GetPhyAddr();
            srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER + i] = (uint64_t)idxs[i * NCHW_CONV_ADDR_LIST_SIZE + NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER].GetPhyAddr();
        }

        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParamsReverseHalf); // Result store in dst
        PipeBarrier<PIPE_V>();

        if ((this->roundPartOutSize > (NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER)) & (this->roundPartOutSize <= NCHW_CONV_ADDR_LIST_SIZE)) {
            for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER; i++) {
                dstLocalList[TRANSDATA_MUTLIPLER * i] = (uint64_t)dst[i * this->transAlignRoundPartOutSize + BLOCK_LEN_FP32].GetPhyAddr();
                srcLocalList[i] = (uint64_t)idxs[i * NCHW_CONV_ADDR_LIST_SIZE + NCHW_CONV_ADDR_LIST_SIZE * BLOCK_LEN_FP32].GetPhyAddr();
            }
            for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER; i++) {
                dstLocalList[TRANSDATA_MUTLIPLER * i + 1] = (uint64_t)dst[(NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER + i) * this->transAlignRoundPartOutSize + BLOCK_LEN_FP32].GetPhyAddr();
                srcLocalList[NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER + i] = (uint64_t)idxs[i * NCHW_CONV_ADDR_LIST_SIZE + NCHW_CONV_ADDR_LIST_SIZE / TRANSDATA_MUTLIPLER + NCHW_CONV_ADDR_LIST_SIZE * BLOCK_LEN_FP32].GetPhyAddr();
            }

            TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParamsReverseHalf); // Result store in dst
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutInds(const uint32_t& dstOffset, const uint32_t& partNC, const uint32_t& partOut){
        const uint32_t downPartOut = this->RoundDownBlock(partOut);
        LocalTensor<float> idxs = this->queOutInds.template DeQue<float>();
        auto gmDst = this->indicesGm[dstOffset];

        if (downPartOut == partOut && (this->outSize) % BLOCK_LEN_FP32 == 0){
            if (this->outSize < MAX_UINT16 / sizeof(float)){
                DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>(downPartOut / BLOCK_LEN_FP32), 
                                      static_cast<uint16_t>((this->transAlignRoundPartOutSize - downPartOut) / BLOCK_LEN_FP32), 
                                      static_cast<uint16_t>((this->outSize - downPartOut) / BLOCK_LEN_FP32)};
                DataCopy(gmDst, idxs, params);
            } else {
                DataCopyParams params{static_cast<uint16_t>(1), static_cast<uint16_t>(downPartOut / BLOCK_LEN_FP32), 0, 0};
                for (uint32_t nc=0; nc<partNC; nc++){
                    DataCopy(gmDst[this->outSize * nc], idxs[this->transAlignRoundPartOutSize * nc], params);
                }
            }
        } else {
            if (this->outSize < MAX_UINT16 / sizeof(float)){
                DataCopyExtParams params{static_cast<uint16_t>(partNC), static_cast<uint32_t>(partOut * sizeof(float)), 
                                         static_cast<uint32_t>(((this->transAlignRoundPartOutSize - partOut)) / BLOCK_LEN_FP32), 
                                         static_cast<uint32_t>((this->outSize - partOut) * sizeof(float)), 0};    
                DataCopyPad(gmDst, idxs, params);
            } else {
                DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(partOut * sizeof(float)), 0, 0, 0};
                for (uint32_t nc=0; nc<partNC; nc++){
                    DataCopyPad(gmDst[this->outSize * nc], idxs[this->transAlignRoundPartOutSize * nc], params);
                }
            }
        }
        auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(event0);
        WaitFlag<HardEvent::MTE3_S>(event0);

        this->queOutInds.template FreeTensor<float>(idxs);
    }

    template <typename V>
    __aicore__ inline void TransposeBack(LocalTensor<half> &reduceMaxResult, LocalTensor<float> &idxs, uint32_t partNC, uint32_t outputOffset, uint32_t partOutReal){
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];

        this->TransposeBackVals(reduceMaxResult, partNC, dstLocalList, srcLocalList);
    
        this->CopyOutVals(outputOffset, partNC, partOutReal);
        
        this->TransposeBackIndsHalf(idxs, dstLocalList, srcLocalList);
    }

    template <typename V>
    __aicore__ inline void TransposeBack(LocalTensor<float> &reduceMaxResult, LocalTensor<float> &idxs, uint32_t partNC, uint32_t outputOffset, uint32_t partOutReal){
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        
        this->template TransposeBackVals<V>(reduceMaxResult, partNC, dstLocalList, srcLocalList);
   
        this->CopyOutVals(outputOffset, partNC, partOutReal);
        
        this->TransposeBackInds(idxs, dstLocalList, srcLocalList);
    }

    __aicore__ inline void PaddingD(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb, 
                                    const uint32_t& partDWoPad, const uint32_t pDL, const uint32_t pDR) {
        if (pDL != 0) {
            Duplicate<T>(dstUb, this->minFloat, this->partRoundDhwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
           
            DataCopy(dstUb[pDL * this->partHwInp * this->blockLength], srcUb, partDWoPad * this->partHwInp * this->blockLength);
            PipeBarrier<PIPE_V>();

            DataCopy(srcUb, dstUb, this->partRoundDhwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
        }
        
        if (pDR != 0) {
            Duplicate<T>(srcUb[(partDWoPad + pDL) * this->partHwInp * this->blockLength], this->minFloat, pDR * this->partHwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PaddingH(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb, 
                                    const uint32_t& partDWoPad, const uint32_t& partHWoPad,
                                    const uint32_t pHL, const uint32_t pHR, const uint32_t srcStride=0) {
        if (pHL != 0 || pHR != 0) {
            Duplicate<T>(dstUb, this->minFloat, this->partRoundDhwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
            
            DataCopyParams padParams = {static_cast<uint16_t>(partDWoPad), static_cast<uint16_t>(partHWoPad * this->roundW), 
                                        static_cast<uint16_t>(srcStride), static_cast<uint16_t>(this->partHwInp - partHWoPad * this->roundW)};
            DataCopy(dstUb[pHL * this->roundW * this->blockLength], srcUb, padParams);
            PipeBarrier<PIPE_V>();
            
            DataCopy(srcUb, dstUb, this->partRoundDhwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PaddingW(LocalTensor<T> &srcUb, LocalTensor<T> &dstUb, 
                                    const uint32_t& partDWoPad, const uint32_t& partHWoPad, const uint32_t& partWWoPad, 
                                    const uint32_t pWL, const uint32_t pWR) {
        if (pWL != 0 || pWR != 0) {
            Duplicate<T>(dstUb, this->minFloat, this->partRoundDhwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
            
            DataCopyParams padParams = {static_cast<uint16_t>(partDWoPad * partHWoPad), static_cast<uint16_t>(partWWoPad), 
                                        0, static_cast<uint16_t>(this->roundW - partWWoPad)};

            DataCopy(dstUb[pWL * this->blockLength], srcUb, padParams);
            PipeBarrier<PIPE_V>();
            
            DataCopy(srcUb, dstUb, this->partRoundDhwInp * this->blockLength);
            PipeBarrier<PIPE_V>();
        }
    }
};

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2Base<T,S>::Im2ColNoDilationPerRepeat(const LocalTensor<T>& dst, uint32_t& dOff) {
    uint32_t sliceByWidthOffset = 0;

    const uint32_t wholeRepeatTimes = partOutW / MAX_REPEAT_TIMES;
    const uint32_t tailRepeatTimes = partOutW % MAX_REPEAT_TIMES;
    const uint32_t wholeOffsetDst = wholeRepeatTimes * MAX_REPEAT_TIMES * blockKernelDHW;
    const uint32_t wholeOffsetSrc = wholeRepeatTimes * MAX_REPEAT_TIMES * gmWOff;

    for (uint32_t curDepthOut = dOff; curDepthOut < dOff + kD * curDepthOff; curDepthOut += curDepthOff) {
        for (uint32_t curHeightOut = curDepthOut; curHeightOut < curDepthOut + kH * partW * blockLength; 
             sliceByWidthOffset += kW * blockLength, curHeightOut += partW * blockLength) {
            for (uint32_t i = 0; i < wholeRepeatTimes; i++) {
                Copy(dst[i * MAX_REPEAT_TIMES * blockKernelDHW + sliceByWidthOffset], transDataTensorUb1[i * MAX_REPEAT_TIMES * gmWOff + curHeightOut], 
                     kW * blockLength, MAX_REPEAT_TIMES, {1, 1, static_cast<uint16_t>(kernelDHW), static_cast<uint16_t>(sW)});
            }
            Copy(dst[wholeOffsetDst + sliceByWidthOffset], transDataTensorUb1[wholeOffsetSrc + curHeightOut], 
                 kW * blockLength, tailRepeatTimes, {1, 1, static_cast<uint16_t>(kernelDHW), static_cast<uint16_t>(sW)});
        }
    }
    dOff += gmWOff * partOutW;
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2Base<T,S>::Im2ColNoDilation(const LocalTensor<T>& dst, uint32_t& dOff) {
    uint32_t sliceByWidthOffset = 0;
    for (uint32_t curWidthOut = 0; curWidthOut < partOutW; ++curWidthOut, dOff += gmWOff) {
        for (uint32_t curDepthOut = dOff; curDepthOut < dOff + kD * curDepthOff; sliceByWidthOffset += sliceOff, curDepthOut += curDepthOff) {
            Copy(dst[sliceByWidthOffset], transDataTensorUb1[curDepthOut], kW * blockLength, kH, {1, 1, static_cast<uint16_t>(kW), static_cast<uint16_t>(partW)});
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2Base<T,S>::Im2ColNoDilationBigKw(const LocalTensor<T>& dst, uint32_t& dOff) {
    uint32_t sliceByWidthOffset = 0;
    DataCopyParams Im2colParams = {static_cast<uint16_t>(kH), static_cast<uint16_t>(kW), static_cast<uint16_t>(partW - kW), 0};

    for (uint32_t curWidthOut = 0; curWidthOut < partOutW; ++curWidthOut, dOff += gmWOff) {
        for (uint32_t curDepthOut = dOff; curDepthOut < dOff + kD * curDepthOff; sliceByWidthOffset += sliceOff, curDepthOut += curDepthOff) {
            DataCopy(dst[sliceByWidthOffset], transDataTensorUb1[curDepthOut], Im2colParams);
        }
    }
}

template <typename T, typename S>
__aicore__ void KernelMaxPool3DWithArgmaxV2Base<T,S>::Im2ColHWDDilation(const LocalTensor<T>& dst, uint32_t& dOff) {
    DataCopyParams Im2colParams = {static_cast<uint16_t>(kW), static_cast<uint16_t>(1), static_cast<uint16_t>(dW-1), 0};
    auto hOff = dH * roundW * blockLength;
    
    uint32_t sliceByWidthOffset = 0;
    for (uint32_t curWidthOut = 0; curWidthOut < partOutW; ++curWidthOut, dOff += gmWOff) {
        for (uint32_t curDepthOut = dOff; curDepthOut < dOff + kD * curDepthOff; curDepthOut += curDepthOff) {
            auto ub1Pos = transDataTensorUb1[curDepthOut];
            for (uint32_t hSrc = 0; hSrc < kH * hOff; sliceByWidthOffset+=kW * blockLength, hSrc += hOff) {
                DataCopy(dst[sliceByWidthOffset], ub1Pos[hSrc], Im2colParams);
            }
        }
    }
}

template <typename T, typename S>
__aicore__ inline void KernelMaxPool3DWithArgmaxV2Base<T,S>::CopyOutVals(const uint32_t& dstOffset,  const uint32_t& partNC, const uint32_t& partOut){
    const uint32_t downPartOut = this->RoundDownBlock(partOut, this->blockLengthS);
    auto gmDst = this->yGm[dstOffset];
    LocalTensor<S> reduceMaxResult = this->queOutVals.template DeQue<S>();

    if (downPartOut == partOut && this->outSize % this->blockLengthS == 0){
        if (this->outSize < MAX_UINT16 / sizeof(S)){
            DataCopyParams params{static_cast<uint16_t>(partNC), static_cast<uint16_t>(downPartOut / this->blockLengthS), 
                                  static_cast<uint16_t>((this->transAlignRoundPartOutSize - downPartOut) / this->blockLengthS), 
                                  static_cast<uint16_t>((this->outSize - downPartOut) / this->blockLengthS)};
            DataCopy(gmDst, reduceMaxResult, params);
        } else {
            DataCopyParams params{static_cast<uint16_t>(1), static_cast<uint16_t>(partOut / this->blockLengthS), 0, 0};
            for (uint32_t nc=0; nc<partNC; nc++){
                DataCopy(gmDst[this->outSize * nc], reduceMaxResult[this->transAlignRoundPartOutSize * nc], params);
            }
        }
    } else {
        if (this->outSize < MAX_UINT16 / sizeof(S)){
            DataCopyExtParams params{static_cast<uint16_t>(partNC), static_cast<uint32_t>(partOut * sizeof(S)), 
                                     static_cast<uint32_t>((this->transAlignRoundPartOutSize - partOut) / this->blockLengthS), 
                                     static_cast<uint32_t>((this->outSize - partOut) * sizeof(S)), 0};
            DataCopyPad(gmDst, reduceMaxResult, params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(partOut * sizeof(S)), 0, 0, 0};
            for (uint32_t nc=0; nc<partNC; nc++){
                DataCopyPad(gmDst[this->outSize * nc], reduceMaxResult[this->transAlignRoundPartOutSize * nc], params);
            }
        }
    }
    auto event0 = this->pipe->FetchEventID(HardEvent::MTE3_S);
    SetFlag<HardEvent::MTE3_S>(event0);
    WaitFlag<HardEvent::MTE3_S>(event0);

    this->queOutVals.template FreeTensor<S>(reduceMaxResult);
}

#endif
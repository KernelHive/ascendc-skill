/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file rfft1_d_bluestein.h
 * \brief
 */

#ifndef OPP_RFFT1D_BLUESTEIN_H
#define OPP_RFFT1D_BLUESTEIN_H

#include "rfft1_d.h"

class KernelRfftBluestein
{
private:
    Rfft1DTilingData* tilingData;
    const uint32_t* factors = &(tilingData->factors[0]);
    const int32_t& norm = tilingData->normal;
    const uint8_t& isBluestein = tilingData->isBluestein;
    const int32_t& fftLength = tilingData->length;
    const int32_t& fftLengthPad = tilingData->lengthPad;
    const int32_t& outLength = tilingData->outLength; //other mentions must be multiplied by 2!
    const uint32_t& batchesPerCore = tilingData->batchesPerCore;
    const uint32_t& leftOverBatches = tilingData->leftOverBatches;
    const uint32_t& tmpLenPerBatch = tilingData->tmpLenPerBatch;
    
    const uint32_t& matmulTmpsLen = tilingData->matmulTmpsLen;
    const uint32_t& matmulTmpsSize = tilingData->matmulTmpsSize;
    
    const uint32_t& dftRealOverallSize = tilingData->dftRealOverallSize;
    const uint32_t& dftImagOverallSize = tilingData->dftImagOverallSize;
    const uint32_t& twiddleOverallSize = tilingData->twiddleOverallSize;
    const uint32_t& fftMatrOverallSize = tilingData->fftMatrOverallSize;
    
    GM_ADDR dftAddr;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    
    GlobalTensor<DTYPE_X> dftGmReal;
    GlobalTensor<DTYPE_X> dftGmImag;

    GlobalTensor<DTYPE_X> twiddleGmReal;
    GlobalTensor<DTYPE_X> twiddleGmImag;
    
    GlobalTensor<DTYPE_X> alphaGm;
    GlobalTensor<DTYPE_X> ctBetaRealGm;
    GlobalTensor<DTYPE_X> ctBetaImagGm;
    GlobalTensor<DTYPE_X> conjBetaRealGm;
    GlobalTensor<DTYPE_X> conjBetaImagGm;
    
    GlobalTensor<DTYPE_X> doubleXGm;
    GlobalTensor<DTYPE_X> ctResGm;
    
    const uint32_t coreIdx = GetBlockIdx() / CORE_IDX_DIV;
    const uint32_t subIdx = GetSubBlockIdx();
    const uint32_t coreNum = GetBlockNum();

    const uint32_t batchesCurCore = batchesPerCore + (coreIdx < leftOverBatches);
    const uint32_t batchesOffset = coreIdx * batchesPerCore + (coreIdx < leftOverBatches ? 
                                                                   coreIdx : leftOverBatches);
    const uint32_t inputOffset = batchesOffset * fftLength;
    const uint32_t outputOffset = batchesOffset * COMPLEX * outLength;
    GM_ADDR workspace;
    GM_ADDR y;
public:
    uint32_t curBufPos = 0;
    uint32_t maxBufSize;

    KernelRfftCooleyTukey ct;     
public:
    __aicore__ inline KernelRfftBluestein(Rfft1DTilingData* tilingData_) : tilingData(tilingData_), ct(tilingData_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace)
    {
        this->dftAddr = dft;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        dftGmReal.SetGlobalBuffer((__gm__ DTYPE_X*)dft, dftRealOverallSize);
        auto dftPointer = (__gm__ DTYPE_X*)dft + dftRealOverallSize;
        dftGmImag.SetGlobalBuffer((__gm__ DTYPE_X*)dftPointer, dftImagOverallSize);
        dftPointer += dftImagOverallSize;

        twiddleGmReal.SetGlobalBuffer((__gm__ DTYPE_X*)dftPointer, twiddleOverallSize); 
        dftPointer += twiddleOverallSize;
        twiddleGmImag.SetGlobalBuffer((__gm__ DTYPE_X*)dftPointer, twiddleOverallSize);
        
        auto bluesteinPointer = (__gm__ DTYPE_X*)dft + COMPLEX * fftMatrOverallSize;

        alphaGm.SetGlobalBuffer((__gm__ DTYPE_X*)bluesteinPointer, COMPLEX * fftLength);
        bluesteinPointer += COMPLEX * fftLength;
        conjBetaRealGm.SetGlobalBuffer((__gm__ DTYPE_X*)bluesteinPointer, COMPLEX * fftLengthPad);
        bluesteinPointer += COMPLEX * fftLengthPad;
        conjBetaImagGm.SetGlobalBuffer((__gm__ DTYPE_X*)bluesteinPointer, COMPLEX * fftLengthPad);
        bluesteinPointer += COMPLEX * fftLengthPad;
        ctBetaRealGm.SetGlobalBuffer((__gm__ DTYPE_X*)bluesteinPointer, COMPLEX * fftLengthPad);
        bluesteinPointer += COMPLEX * fftLengthPad;
        ctBetaImagGm.SetGlobalBuffer((__gm__ DTYPE_X*)bluesteinPointer, COMPLEX * fftLengthPad);

        uint32_t copyLenIn = fftLength * batchesCurCore;
        uint32_t copyLenOut = COMPLEX * outLength * batchesCurCore;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + inputOffset, copyLenIn);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + outputOffset, copyLenOut);
        auto doubleXPos = workspace + matmulTmpsSize;
        
        const uint32_t tmpResSize = COMPLEX * fftLengthPad + BLOCK_LEN_FP32; 

        doubleXGm.SetGlobalBuffer((__gm__ DTYPE_X*)doubleXPos + coreIdx * tmpResSize, tmpResSize);
        auto ctResPos = doubleXPos + (24 * tmpResSize + 1) * sizeof(DTYPE_X);
        ctResGm.SetGlobalBuffer((__gm__ DTYPE_X*)ctResPos + coreIdx * tmpResSize, tmpResSize);

        ct.Init(workspace + matmulTmpsSize, dft, ctResPos + sizeof(DTYPE_X), workspace);

        this->workspace = workspace;
        this->y = y;
    }

    __aicore__ inline void Process()
    {
        if (g_coreType == AIV && (coreIdx >= coreNum || subIdx != 0)){
            return;
        }
        
        Compute();
    }

private:
    __aicore__ inline void CalcInputCoeffs(const uint32_t elsPerIter, const uint32_t iter, const uint32_t iterLeftOver) {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        
        LocalTensor<DTYPE_X> xLocal = ct.AllocTmpTensor(elsPerIter);
        LocalTensor<DTYPE_X> alphaLoc = ct.AllocTmpTensor(elsPerIter);

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (uint32_t i = 0; i < iter; ++i) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(alphaLoc, alphaGm[i * elsPerIter], elsPerIter);
            DataCopy(xLocal, doubleXGm[i * elsPerIter], elsPerIter);

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            
            Mul(xLocal, xLocal, alphaLoc, elsPerIter);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(ctResGm[i * elsPerIter], xLocal, elsPerIter);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        if (iterLeftOver) {
            DataCopy(alphaLoc, alphaGm[iter * elsPerIter], iterLeftOver);
            DataCopy(xLocal, doubleXGm[iter * elsPerIter], iterLeftOver);

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            
            Mul(xLocal, xLocal, alphaLoc, iterLeftOver);
            
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(ctResGm[iter * elsPerIter], xLocal, iterLeftOver);
        }

        ct.FreeTmpTensor(UB_LEN);
    }

    __aicore__ inline void MulInputCoeffs(const uint32_t& curBatch)
    {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        
        const uint32_t leftOver = (2 * fftLength) % BLOCK_LEN_FP32;
        auto dataSize = leftOver == 0 ? 2 * fftLength : RoundUpBlock(2 * fftLength);
        
        const uint32_t elsPerIter = UB_LEN / 2;
        const uint32_t iter = dataSize / elsPerIter;
        const uint32_t iterLeftOver = dataSize % elsPerIter; 
        
        LocalTensor<DTYPE_X> doubleMatrix = ct.AllocTmpTensor(BLOCK_LEN_FP32);
        doubleMatrix.SetValue(0, 1.f);
        doubleMatrix.SetValue(1, 1.f);
        DataCopy(doubleXGm[matmulTmpsSize + (GetBlockNum() - 1) * (COMPLEX * fftLengthPad + BLOCK_LEN_FP32)], doubleMatrix, BLOCK_LEN_FP32);
        AscendC::PipeBarrier<PIPE_V>();
        ct.FreeTmpTensor(BLOCK_LEN_FP32);

        ct.matmulObj4.SetTensorA(xGm[curBatch * fftLength]);
        ct.matmulObj4.SetTensorB(doubleXGm[matmulTmpsSize + (GetBlockNum() - 1) * (COMPLEX * fftLengthPad + BLOCK_LEN_FP32)]);
        ct.matmulObj4.IterateAll(doubleXGm);
        
        CalcInputCoeffs(elsPerIter, iter, iterLeftOver);
        
        LocalTensor<DTYPE_X> dupZero = ct.AllocTmpTensor(UB_LEN);
        
        const uint32_t zeroToFill = 2 * (fftLengthPad - fftLength);
        const uint32_t zeroIter = zeroToFill / UB_LEN;
        const uint32_t zeroLeftOver = zeroToFill % UB_LEN;

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        
        Duplicate(dupZero, 0.f, UB_LEN);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3); 

        for (uint32_t i = 0; i < zeroIter; ++i){
            DataCopy(ctResGm[COMPLEX * fftLength + i * UB_LEN], dupZero, UB_LEN);
        }

        if (zeroLeftOver){
            uint32_t elsToCopy = RoundDownBlock(zeroLeftOver); 
            uint32_t elsToPadCopy = zeroLeftOver % BLOCK_LEN_FP32; 
            
            DataCopy(ctResGm[COMPLEX * fftLength + zeroIter * UB_LEN], dupZero, elsToCopy);
            if (elsToPadCopy){
                DataCopyParams copyPadParams{1, static_cast<uint16_t>(elsToPadCopy * sizeof(DTYPE_X)), 0, 0};
                DataCopyPad(ctResGm[COMPLEX * fftLength + zeroIter * UB_LEN + elsToCopy], dupZero, copyPadParams);
            }
        }
        ct.FreeTmpTensor(UB_LEN);
    }

    __aicore__ inline void TailingProcessing(LocalTensor<DTYPE_X>& ubInput, GlobalTensor<DTYPE_X>& dst, GlobalTensor<DTYPE_X>& src,
                                             LocalTensor<DTYPE_X>& tmpTensor, LocalTensor<DTYPE_X>& tmpTensor2, uint32_t tailing,
                                             uint32_t partUb, uint32_t it, const bool& backward, const uint32_t xLen, float norm) {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        if (tailing) {
            DataCopy(ubInput, src[partUb * it], tailing);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            
            if (backward){
                Muls(ubInput, ubInput, norm, tailing);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Copy(tmpTensor, ubInput, MAX_VEC_ELEMS_PER_REP, tailing / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            
            Muls(tmpTensor, ubInput, 0.f, (uint64_t*) MASK170, tailing / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            
            PairReduceSum(tmpTensor, tmpTensor, tailing / MAX_VEC_ELEMS_PER_REP, MAX_VEC_ELEMS_PER_REP, 1, 1, BLOCK_LEN_FP32);
            
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(dst[partUb / COMPLEX * it], tmpTensor, tailing / COMPLEX);
            
            Copy(tmpTensor2, ubInput, MAX_VEC_ELEMS_PER_REP, tailing / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            
            Muls(tmpTensor2, ubInput, 0.f, (uint64_t*) MASK85, tailing / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            
            PairReduceSum(tmpTensor2, tmpTensor2, tailing / MAX_VEC_ELEMS_PER_REP, MAX_VEC_ELEMS_PER_REP, 1, 1, BLOCK_LEN_FP32);
            
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(dst[partUb / COMPLEX * it + xLen / COMPLEX], tmpTensor2, tailing / COMPLEX);
        }
    }

    __aicore__ inline void TailProcessing(GlobalTensor<DTYPE_X>& input, GlobalTensor<DTYPE_X>& output,
                                          GlobalTensor<DTYPE_X>& curBetaReal, GlobalTensor<DTYPE_X>& curBetaImag,
                                          LocalTensor<DTYPE_X>& tmpResTensor,
                                          LocalTensor<DTYPE_X>& betaReal, LocalTensor<DTYPE_X>& betaImag,
                                          LocalTensor<DTYPE_X>& paddedResTensor, LocalTensor<DTYPE_X>& tmpTensor1,
                                          const uint32_t part, const uint32_t partSize, const uint32_t tail,
                                          const uint32_t tailSizePadded, DropOutShapeInfo info) {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                
        const uint32_t copy_els = RoundDownBlock(tail);
        const uint32_t copy_els_pad = tail % BLOCK_LEN_FP32;
        
        DataCopyPadParams paddingParams{false, 0, 0, 0};
        DataCopyParams copyPadParams{1, static_cast<uint16_t>(copy_els_pad * sizeof(DTYPE_X)), 0, 0};
        if (copy_els){
            DataCopy(tmpResTensor, input[1 + part * partSize], copy_els);
            DataCopy(betaReal, curBetaReal[part * partSize], copy_els);
            DataCopy(betaImag, curBetaImag[part * partSize], copy_els);
        }
        if (copy_els_pad) {
            DataCopyPad(tmpResTensor[copy_els], input[1 + part * partSize + copy_els], copyPadParams, paddingParams);
            DataCopyPad(betaReal[copy_els], curBetaReal[part * partSize + copy_els], copyPadParams, paddingParams);
            DataCopyPad(betaImag[copy_els], curBetaImag[part * partSize + copy_els], copyPadParams, paddingParams);
        }
        
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Mul(tmpResTensor, betaReal, tmpResTensor, tail);

        ct.InverseInput(input[part * partSize], paddedResTensor, tmpTensor1, tailSizePadded, info);
        AscendC::PipeBarrier<PIPE_V>();

        Mul(paddedResTensor, betaImag, paddedResTensor, tail);
        AscendC::PipeBarrier<PIPE_V>();

        Add(tmpResTensor, paddedResTensor, tmpResTensor, tail);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        
        if (copy_els) {
            DataCopy(output[part * partSize], tmpResTensor, copy_els);
        }
        if (copy_els_pad) {
            DataCopyPad(output[part * partSize + copy_els], tmpResTensor[copy_els], copyPadParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void SplitInput(GlobalTensor<DTYPE_X> src, GlobalTensor<DTYPE_X> dst, const bool& backward = false){ 
        const uint32_t xLen = COMPLEX * fftLengthPad;
        
        uint32_t partUb = RoundDownBlock((ct.maxBufSize - ct.curBufPos) / (3 * sizeof(DTYPE_X)), MAX_VEC_ELEMS_PER_REP);
        uint32_t it = partUb != 0 ? xLen / partUb : 0;
        uint32_t tailing = (xLen - partUb * it);
        float norm = 1.f / fftLengthPad;

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        
        LocalTensor<DTYPE_X> ubInput = ct.AllocTmpTensor(partUb);
        LocalTensor<DTYPE_X> tmpTensor = ct.AllocTmpTensor(partUb);
        LocalTensor<DTYPE_X> tmpTensor2 = ct.AllocTmpTensor(partUb);
        
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (uint32_t i = 0; i < it; i++){
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(ubInput, src[partUb * i], partUb);
            
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            
            if (backward){
                Muls(ubInput, ubInput, norm, partUb);
                AscendC::PipeBarrier<PIPE_V>();
            }
            
            Copy(tmpTensor, ubInput, MAX_VEC_ELEMS_PER_REP, partUb / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            
            Muls(tmpTensor, ubInput, 0.f, (uint64_t*) MASK170, partUb / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(tmpTensor, tmpTensor, partUb / MAX_VEC_ELEMS_PER_REP, MAX_VEC_ELEMS_PER_REP, 1, 1, BLOCK_LEN_FP32);
            
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(dst[partUb / COMPLEX * i], tmpTensor, partUb / COMPLEX);
            
            Copy(tmpTensor2, ubInput, MAX_VEC_ELEMS_PER_REP, partUb / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            
            Muls(tmpTensor2, ubInput, 0.f, (uint64_t*) MASK85, partUb / MAX_VEC_ELEMS_PER_REP, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            PairReduceSum(tmpTensor2, tmpTensor2, partUb / MAX_VEC_ELEMS_PER_REP, MAX_VEC_ELEMS_PER_REP, 1, 1, BLOCK_LEN_FP32);
            
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(dst[partUb / COMPLEX * i + xLen / COMPLEX], tmpTensor2, partUb / COMPLEX);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        
        TailingProcessing(ubInput, dst, src, tmpTensor, tmpTensor2,
                          tailing, partUb, it, backward, xLen, norm);

        ct.FreeTmpTensor(TMP_TENSOR_SIZE_MULTIPLIER * partUb);
    } 

   // input must be in format: 0 R1 I1 R2 I2 ...
   __aicore__ inline void MulsCoeffs(GlobalTensor<DTYPE_X> input, GlobalTensor<DTYPE_X> output, bool last=0){
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        
        const uint32_t whole = COMPLEX * (last ? outLength : fftLengthPad);
        const uint32_t partSize = RoundDownBlock((UB_LEN / 5) - 128, 64);
        const uint32_t partSizePadded = RoundUpBlock(partSize + 64, 64);
        const uint32_t part = partSize != 0 ? whole / partSize : 0;
        
        LocalTensor<DTYPE_X> tmpResTensor = ct.AllocTmpTensor(partSize);
        LocalTensor<DTYPE_X> betaReal = ct.AllocTmpTensor(partSize);
        LocalTensor<DTYPE_X> betaImag = ct.AllocTmpTensor(partSize);
        LocalTensor<DTYPE_X> paddedResTensor = ct.AllocTmpTensor(partSizePadded);
        LocalTensor<DTYPE_X> tmpTensor1 = ct.AllocTmpTensor(partSizePadded);
        
        auto curBetaReal = last ? conjBetaRealGm : ctBetaRealGm;
        auto curBetaImag = last ? conjBetaImagGm : ctBetaImagGm;

        const uint32_t tail = whole - partSize * part;//RoundDownBlock(whole - partSize * part, 32);
        const uint32_t tailSizePadded = RoundUpBlock(tail, 64);

        DropOutShapeInfo info;
        info.firstAxis = BLOCK_LEN_FP32;
        info.srcLastAxis = partSizePadded/ BLOCK_LEN_FP32;
        info.maskLastAxis = 1;
        
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (int p = 0; p<part; p++){
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(tmpResTensor, input[1 + p * partSize], partSize);
            DataCopy(betaReal, curBetaReal[p * partSize], partSize);
            DataCopy(betaImag, curBetaImag[p * partSize], partSize);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Mul(tmpResTensor, betaReal, tmpResTensor, partSize);

            ct.InverseInput(input[p * partSize], paddedResTensor, tmpTensor1, partSizePadded, info);

            AscendC::PipeBarrier<PIPE_V>();

            Mul(paddedResTensor, betaImag, paddedResTensor, partSize);
            AscendC::PipeBarrier<PIPE_V>();

            Add(tmpResTensor, paddedResTensor, tmpResTensor, partSize);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            
            DataCopy(output[p * partSize], tmpResTensor, partSize);
            
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        
        info.srcLastAxis = tailSizePadded / BLOCK_LEN_FP32;

        if (tail) {
            TailProcessing(input, output, curBetaReal, curBetaImag, tmpResTensor, betaReal, betaImag,
                           paddedResTensor, tmpTensor1, part, partSize, tail, tailSizePadded, info);
        }
        
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        ct.FreeTmpTensor(TMP_TENSOR_SIZE_MULTIPLIER * partSize + COMPLEX * partSizePadded);
    }

    // Depending on the input length perform an FFT computation
    __aicore__ inline void Compute() {
        const uint32_t tmpResSize = 2 * fftLengthPad + BLOCK_LEN_FP32;
        
        GM_ADDR doubleXPos = workspace + matmulTmpsSize;  
        GM_ADDR ctXPos = doubleXPos + (24 * tmpResSize + 1) * sizeof(DTYPE_X);

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (uint32_t curBatch = 0; curBatch < batchesCurCore; ++curBatch){ 
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            MulInputCoeffs(curBatch);
            
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            
            SplitInput(ctResGm, doubleXGm);
  
            ct.ReInit(doubleXPos, dftAddr, ctXPos + sizeof(DTYPE_X));
            AscendC::PipeBarrier<PIPE_ALL>();
            this->ct.Process();
              
            AscendC::PipeBarrier<PIPE_ALL>();
            MulsCoeffs(ctResGm, ctResGm[1]);
  
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            
            SplitInput(ctResGm[1], doubleXGm);
  
            ct.ReInit(doubleXPos, dftAddr + fftMatrOverallSize * sizeof(DTYPE_X), doubleXPos + sizeof(DTYPE_X));
            AscendC::PipeBarrier<PIPE_ALL>();
            this->ct.Process();
  
            AscendC::PipeBarrier<PIPE_ALL>();
            MulsCoeffs(doubleXGm, yGm[curBatch * COMPLEX * outLength], 1);
            
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
};

#endif
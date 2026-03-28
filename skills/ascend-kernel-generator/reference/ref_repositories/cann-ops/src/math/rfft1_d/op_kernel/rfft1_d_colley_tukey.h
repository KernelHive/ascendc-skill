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
 * \file rfft1_d_colley_tukey.h
 * \brief
 */

#ifndef OPP_RFFT1D_COLLEY_TUKEY_H
#define OPP_RFFT1D_COLLEY_TUKEY_H

#include "rfft1_d.h"

class KernelRfftCooleyTukey {
   public:
    Rfft1DTilingData* tilingData;
   private:
    
    const uint32_t* factors = &(tilingData->factors[0]);
    const int32_t& norm = tilingData->normal;
    const uint8_t& isBluestein = tilingData->isBluestein;
    const int32_t& fftLength = isBluestein ? tilingData->lengthPad : tilingData->length;
    const int32_t& outLength = isBluestein ? tilingData->lengthPad : tilingData->outLength;
    const uint32_t& batchesPerCore = isBluestein ? BATCHES_PER_CORE_BLUESTEIN : tilingData->batchesPerCore;
    const uint32_t& leftOverBatches = isBluestein ? LEFT_OVER_BATCHES_BLUESTEIN : tilingData->leftOverBatches;
    const uint32_t& tmpLenPerBatch = tilingData->tmpLenPerBatch;
    const uint32_t& tailSize = tilingData->tailSize;
  
    const uint32_t& matmulTmpsLen = tilingData->matmulTmpsLen;
    const uint32_t& matmulTmpsSize = tilingData->matmulTmpsSize;
  
    const uint32_t& dftRealOverallSize = tilingData->dftRealOverallSize;
    const uint32_t& dftImagOverallSize = tilingData->dftImagOverallSize;
    const uint32_t& twiddleOverallSize = tilingData->twiddleOverallSize;
    const uint32_t& fftMatrOverallSize = tilingData->fftMatrOverallSize;
  
    const uint32_t* dftRealOffsets = &(tilingData->dftRealOffsets[0]);
    const uint32_t* dftImagOffsets = &(tilingData->dftImagOffsets[0]);
    const uint32_t* twiddleOffsets = &(tilingData->twiddleOffsets[0]);
  
    TBuf<TPosition::VECCALC> workspaceBuf;
  
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
  
    GlobalTensor<DTYPE_X> dftGmReal;
    GlobalTensor<DTYPE_X> dftGmImag;
  
    GlobalTensor<DTYPE_X> twiddleGmReal;
    GlobalTensor<DTYPE_X> twiddleGmImag;
  
    const uint32_t coreIdx = GetBlockIdx() / CORE_IDX_DIV;
    const uint32_t subIdx = GetSubBlockIdx();
    const uint32_t coreNum = GetBlockNum();
    
    const uint8_t isLarge = (fftLength == MAX_FFT_LEN);
    const uint8_t notLargeCase = (!isBluestein && !isLarge);
    const uint8_t largeCase = (!isBluestein && isLarge);

    const uint32_t batchesCurCore = batchesPerCore + (coreIdx < leftOverBatches);
    const uint32_t batchesOffset =
        coreIdx * batchesPerCore + (coreIdx < leftOverBatches ? coreIdx : leftOverBatches);
    const uint32_t inputOffset = batchesOffset * ((isBluestein + 1) * fftLength + isBluestein * BLOCK_LEN_FP32);
    const uint32_t outputOffset = batchesOffset * (COMPLEX * outLength + isBluestein * BLOCK_LEN_FP32);
    const uint32_t copyLenIn = (isBluestein + 1) * fftLength * batchesCurCore;
    const uint32_t copyLenOut = COMPLEX * outLength * batchesCurCore;
  
    public:
    
    GlobalTensor<DTYPE_X> matmulTmps;
    
    uint32_t curBufPos = 0;
    uint32_t maxBufSize = UB_SIZE;
  
    TPipe pipe;
    using inputAType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X, true>;
    using inputBType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X, true>;
    using outputCType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    Matmul<inputAType, inputBType, outputCType> matmulObj1, matmulObj2, matmulObj3, matmulObj4;
  
    public:
    __aicore__ inline KernelRfftCooleyTukey(Rfft1DTilingData* tilingData_) : tilingData(tilingData_) {
    }
  
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    
        dftGmReal.SetGlobalBuffer((__gm__ DTYPE_X*)dft, dftRealOverallSize);
        dftGmImag.SetGlobalBuffer((__gm__ DTYPE_X*)dft + dftRealOverallSize, dftImagOverallSize);
    
        twiddleGmReal.SetGlobalBuffer((__gm__ DTYPE_X*)dft + (dftRealOverallSize + dftImagOverallSize),
                                       twiddleOverallSize);
        twiddleGmImag.SetGlobalBuffer(
            (__gm__ DTYPE_X*)dft + (dftRealOverallSize + dftImagOverallSize + twiddleOverallSize),
            twiddleOverallSize);
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + inputOffset, copyLenIn);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + outputOffset, copyLenOut);
        
        matmulTmps.SetGlobalBuffer((__gm__ DTYPE_X*)workspace + coreIdx * tmpLenPerBatch,
                                    tmpLenPerBatch);
    
        auto pointer = workspace + matmulTmpsSize;
    
        pipe.InitBuffer(workspaceBuf, UB_SIZE);
    }
  
    __aicore__ inline void ReInit(GM_ADDR x, GM_ADDR dft, GM_ADDR y) {
        dftGmReal.SetGlobalBuffer((__gm__ DTYPE_X*)dft, dftRealOverallSize);
        auto dftPointer = dft + dftRealOverallSize * sizeof(DTYPE_X);
        dftGmImag.SetGlobalBuffer((__gm__ DTYPE_X*)dftPointer, dftImagOverallSize);
        dftPointer += dftImagOverallSize * sizeof(DTYPE_X);
    
        twiddleGmReal.SetGlobalBuffer((__gm__ DTYPE_X*)dftPointer, twiddleOverallSize); 
        dftPointer += twiddleOverallSize * sizeof(DTYPE_X);
        twiddleGmImag.SetGlobalBuffer((__gm__ DTYPE_X*)dftPointer, twiddleOverallSize);
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + inputOffset, copyLenIn);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + outputOffset, copyLenOut);
    }
  
    __aicore__ inline void Compute() {
        CooleyTukeyAlgorithm();
    }
  
    __aicore__ inline void Process() {
        if (g_coreType == AIV && (coreIdx >= coreNum || subIdx != 0)) {
            return;
        }
    
        Compute();
    }
  
    __aicore__ inline LocalTensor<DTYPE_X> AllocTmpTensor(const uint32_t& size) {
        const uint32_t sizeUB = RoundUpBlock(size);
        curBufPos += sizeUB;
        return workspaceBuf.Get<DTYPE_X>(curBufPos)[curBufPos - sizeUB];
    }
  
    __aicore__ inline void FreeTmpTensor(const uint32_t& size) {
        curBufPos -= size;
    }
  
    // GM structured as follows: 0 R1 I1 R2 I2 R3...
  
    __aicore__ inline void InverseInput(GlobalTensor<DTYPE_X> matmulTmps, LocalTensor<DTYPE_X>& resTensor,
                                        LocalTensor<DTYPE_X>& tmpTensor, const uint32_t& sizePadded,
                                        DropOutShapeInfo& info, const bool& negative_imag = false) {
        DataCopy(resTensor, matmulTmps, sizePadded);     // 0 R1 I1 R2 I2 R3 ...
        DataCopy(tmpTensor, matmulTmps[2], sizePadded);  // I1 R2 I2 R3 ...
    
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    
        const uint32_t iter = sizePadded / (MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP);
        const uint32_t tail = (sizePadded % (MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP)) / MAX_VEC_ELEMS_PER_REP;
    
        if (negative_imag) {
            Muls(tmpTensor, tmpTensor, -1.f, sizePadded);
            AscendC::PipeBarrier<PIPE_V>();
        }
        
        for (uint32_t i=0; i<iter; ++i){
            Muls(tmpTensor[i * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], tmpTensor[i * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], 0.f, (uint64_t*)MASK170, MAX_VEC_REP, {1, 1, 8, 8});
            Muls(resTensor[i * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], resTensor[i * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], 0.f, (uint64_t*)MASK85, MAX_VEC_REP, {1, 1, 8, 8});
        }
        if (tail) {
            Muls(tmpTensor[iter * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], tmpTensor[iter * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], 0.f, (uint64_t*)MASK170, tail, {1, 1, 8, 8});
            Muls(resTensor[iter * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], resTensor[iter * MAX_VEC_REP * MAX_VEC_ELEMS_PER_REP], 0.f, (uint64_t*)MASK85, tail, {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
    
        Add(resTensor, resTensor, tmpTensor, sizePadded);
    }
   
    private:
    __aicore__ inline void FirstIterCalc(const uint32_t& curRadix, uint32_t curBatch) {
        if (isBluestein) {
            if (curRadix != 0) {
                matmulObj1.SetOrgShape(fftLength / curRadix, COMPLEX * curRadix, COMPLEX * curRadix);
                matmulObj1.SetSingleShape(fftLength / curRadix, COMPLEX * curRadix, COMPLEX * curRadix);
            }
        }

        matmulObj1.SetTensorA(xGm[curBatch * fftLength], true);
        matmulObj1.SetTensorB(dftGmReal, false);
        matmulObj1.IterateAll(matmulTmps[1]);
    }

    __aicore__ inline void LastIterFirstStep(GlobalTensor<DTYPE_X>& twiddleTensorRe, GlobalTensor<DTYPE_X>& twiddleTensorIm,
                                             const uint32_t curRadix, const uint32_t prevRadices, const uint32_t curSize, 
                                             const uint32_t elsToCopy, const uint32_t elsToPadCopy, const uint32_t curSizePadded, 
                                             const uint32_t firstSize, DataCopyParams copyPadParams, DropOutShapeInfo info) {
        LocalTensor<DTYPE_X> paddedResTensor = AllocTmpTensor(curSizePadded);
        LocalTensor<DTYPE_X> tmpTensor1 = AllocTmpTensor(curSizePadded);
        LocalTensor<DTYPE_X> tmpResTensor = AllocTmpTensor(curSize);
        LocalTensor<DTYPE_X> twiddleTensorReal = AllocTmpTensor(curSize);
        LocalTensor<DTYPE_X> twiddleTensorImag = AllocTmpTensor(curSize);
        
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (uint32_t curInnerIter = 0; curInnerIter < (COMPLEX - !isLarge); ++curInnerIter) {
            const uint32_t innerIterOffset = curInnerIter * curRadix * prevRadices;
            const uint32_t innerIterTailOffset = curInnerIter * curRadix;
            
            for (uint32_t i = 0; i < curRadix; i++) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                DataCopy(twiddleTensorReal, twiddleTensorRe[innerIterOffset + i * curSize], curSize);
                DataCopy(twiddleTensorImag, twiddleTensorIm[innerIterOffset + i * curSize], curSize);

                DataCopy(tmpResTensor, matmulTmps[1 + innerIterOffset + i * curSize], curSize);

                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                Mul(tmpResTensor, twiddleTensorReal, tmpResTensor, curSize);
                InverseInput(matmulTmps[innerIterOffset + i * curSize], paddedResTensor, tmpTensor1, curSizePadded, info);
                AscendC::PipeBarrier<PIPE_V>();

                Mul(paddedResTensor, twiddleTensorImag, paddedResTensor, curSize);
                AscendC::PipeBarrier<PIPE_V>();

                Add(tmpResTensor, paddedResTensor, tmpResTensor, curSize);

                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                DataCopy(matmulTmps[1 + innerIterOffset + i * curSize], tmpResTensor, curSize);
                if (notLargeCase) {
                    if (elsToCopy) {
                        DataCopy(matmulTmps[1 + firstSize + i * tailSize], tmpResTensor, elsToCopy);
                    }
                    if (elsToPadCopy) {
                        DataCopyPad(matmulTmps[1 + firstSize + i * tailSize + elsToCopy], tmpResTensor[elsToCopy], copyPadParams);
                    }
                }
                if (largeCase && i % COMPLEX == 0) {
                    if (elsToCopy) {
                        DataCopy(matmulTmps[1 + firstSize + (i / COMPLEX) * tailSize + curInnerIter * (curRadix / COMPLEX) * tailSize], tmpResTensor, elsToCopy);
                    }
                    if (elsToPadCopy) {
                        DataCopyPad(matmulTmps[1 + firstSize + (i / COMPLEX) * tailSize + curInnerIter * (curRadix / COMPLEX) * tailSize + elsToCopy], tmpResTensor[elsToCopy], copyPadParams);
                    }
                }
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
        }
        FreeTmpTensor(TMP_TENSOR_SIZE_MULTIPLIER * curSize + COMPLEX * curSizePadded);
    }

    __aicore__ inline void LastIterSecondStep(GlobalTensor<DTYPE_X>& curYGm, GlobalTensor<DTYPE_X>& dftTensorReal, 
                                              GlobalTensor<DTYPE_X>& dftTensorImag, LocalTensor<DTYPE_X>& matmulPaddedResTensor, 
                                              LocalTensor<DTYPE_X>& tmpTensor2, const uint32_t curRadix, const uint32_t halfCurRadix, 
                                              const uint32_t trueCurSize, const uint32_t elsToCopy, const uint32_t elsToPadCopy, 
                                              const uint32_t imagOffset, const uint32_t trueCurSizePadded, 
                                              const uint32_t firstSize, DataCopyParams copyPadParams, DropOutShapeInfo info) {
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        if (!isBluestein) {   
            matmulObj3.SetOrgShape(1, tailSize, curRadix);
            matmulObj3.SetSingleShape(1, tailSize, curRadix);

            matmulObj3.SetTensorA(dftTensorReal[halfCurRadix * curRadix], false);
            matmulObj3.SetTensorB(matmulTmps[1 + firstSize], false);
            matmulObj3.IterateAll<false>(curYGm[halfCurRadix * trueCurSize]);
        }
        
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (uint32_t i = 0; i < curRadix; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            
            InverseInput(matmulTmps[i * trueCurSize], matmulPaddedResTensor, tmpTensor2, trueCurSizePadded, info, true);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            DataCopy(matmulTmps[imagOffset + i * trueCurSize], matmulPaddedResTensor, trueCurSize);
            if (!isBluestein) {
                if (elsToCopy) {
                    DataCopy(matmulTmps[imagOffset + firstSize + i * tailSize], matmulPaddedResTensor, elsToCopy);
                }
                if (elsToPadCopy) {
                    DataCopyPad(matmulTmps[imagOffset + firstSize + i * tailSize + elsToCopy], matmulPaddedResTensor[elsToCopy], copyPadParams);
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        
        if (!isBluestein) {
            matmulObj3.SetTensorA(dftTensorImag[halfCurRadix * curRadix], false);
            matmulObj3.SetTensorB(matmulTmps[imagOffset + firstSize], false);
            matmulObj3.IterateAll<false>(curYGm[halfCurRadix * trueCurSize], true);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        
        if (!isBluestein) {
            matmulObj3.SetOrgShape(halfCurRadix, trueCurSize, curRadix);
            matmulObj3.SetSingleShape(halfCurRadix, trueCurSize, curRadix);
        } else {
            matmulObj3.SetOrgShape(curRadix, trueCurSize, curRadix);
            matmulObj3.SetSingleShape(curRadix, trueCurSize, curRadix);
        }

        matmulObj3.SetTensorA(dftTensorReal, false);
        matmulObj3.SetTensorB(matmulTmps[1], false);
        matmulObj3.IterateAll(curYGm);
        AscendC::PipeBarrier<PIPE_ALL>();

        matmulObj3.SetTensorA(dftTensorImag, false);
        matmulObj3.SetTensorB(matmulTmps[imagOffset], false);
        matmulObj3.IterateAll(curYGm, true);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void IntermediateIterFirstStep(const uint32_t prevRadices, const uint32_t nextRadices, 
                                                     const uint32_t firstSize, const uint32_t curRadix) {
        const uint32_t curSize = curRadix * (COMPLEX - isLarge) * prevRadices;
        const uint32_t trueCurSize = COMPLEX * curRadix * prevRadices;

        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        const uint32_t blockLen = prevRadices * COMPLEX / BLOCK_LEN_FP32;
        LocalTensor<DTYPE_X> tmpMoveTensor = AllocTmpTensor(trueCurSize);
                        
        DataCopyParams copyInfo = {(uint16_t)curRadix, (uint16_t)blockLen, uint16_t((nextRadices - 1) * blockLen), 0};
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        for (uint32_t i = 0; i < nextRadices; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(tmpMoveTensor, matmulTmps[1 + i * COMPLEX * prevRadices], copyInfo);

            event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);

            DataCopy(matmulTmps[1 + firstSize + i * trueCurSize], tmpMoveTensor, trueCurSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }

        FreeTmpTensor(trueCurSize);
    }

    __aicore__ inline void IntermediateIterSecondStep(GlobalTensor<DTYPE_X>& twiddleTensorRe, GlobalTensor<DTYPE_X>& twiddleTensorIm,
                                                      const uint32_t prevRadices, const uint32_t nextRadices, 
                                                      const uint32_t firstSize, const uint32_t curRadix) {
        const uint32_t curSize = curRadix * (COMPLEX - isLarge) * prevRadices;
        const uint32_t trueCurSize = COMPLEX * curRadix * prevRadices;
        const uint32_t curSizePadded = RoundUpBlock(curSize + 1, 64);

        LocalTensor<DTYPE_X> paddedResTensor = AllocTmpTensor(curSizePadded);
        LocalTensor<DTYPE_X> tmpTensor1 = AllocTmpTensor(curSizePadded);
        LocalTensor<DTYPE_X> tmpResTensor = AllocTmpTensor(curSize);
        LocalTensor<DTYPE_X> twiddleTensorReal = AllocTmpTensor(curSize);
        LocalTensor<DTYPE_X> twiddleTensorImag = AllocTmpTensor(curSize);
        
        DropOutShapeInfo info;
        info.firstAxis = BLOCK_LEN_FP32;
        info.srcLastAxis = curSizePadded / BLOCK_LEN_FP32;
        info.maskLastAxis = 1;

        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        for (uint32_t curInnerIter = 0; curInnerIter < (COMPLEX - !isLarge); ++curInnerIter) {
            const uint32_t innerIterOffset = curInnerIter * curSize;
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            
            DataCopy(twiddleTensorReal, twiddleTensorRe[innerIterOffset], curSize);
            DataCopy(twiddleTensorImag, twiddleTensorIm[innerIterOffset], curSize);

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            for (uint32_t i = 0; i < nextRadices; i++) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                DataCopy(tmpResTensor, matmulTmps[1 + innerIterOffset + firstSize + i * trueCurSize], curSize);

                SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);

                DataCopy(matmulTmps[1 + innerIterOffset + i * trueCurSize], tmpResTensor, curSize);

                event_t eventIdMte3ToMte2Second = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Second);

                event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

                Mul(tmpResTensor, twiddleTensorReal, tmpResTensor, curSize);

                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Second);
                InverseInput(matmulTmps[innerIterOffset + i * trueCurSize], paddedResTensor, tmpTensor1, curSizePadded, info);
                AscendC::PipeBarrier<PIPE_V>();

                Mul(paddedResTensor, twiddleTensorImag, paddedResTensor, curSize);
                AscendC::PipeBarrier<PIPE_V>();

                Add(tmpResTensor, paddedResTensor, tmpResTensor, curSize);

                event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                DataCopy(matmulTmps[1 + innerIterOffset + i * trueCurSize], tmpResTensor, curSize);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
        }
        
        FreeTmpTensor(TMP_BUF_NUM_FIRST_PART * curSize + TMP_BUF_NUM_SECOND_PART * curSizePadded);
    }

    __aicore__ inline void IntermediateIterThirdStep(GlobalTensor<DTYPE_X>& dftTensorReal, GlobalTensor<DTYPE_X>& dftTensorImag,
                                                     const uint32_t prevRadices, const uint32_t nextRadices, 
                                                     const uint32_t firstSize, const uint32_t curRadix) {
        const uint32_t trueCurSize = COMPLEX * curRadix * prevRadices;
        const uint32_t trueCurSizePadded = RoundUpBlock(trueCurSize + 1, 64);

        DropOutShapeInfo info;
        info.firstAxis = BLOCK_LEN_FP32;
        info.srcLastAxis = trueCurSizePadded / BLOCK_LEN_FP32;
        info.maskLastAxis = 1;
            
        LocalTensor<DTYPE_X> matmulPaddedResTensor = AllocTmpTensor(trueCurSizePadded);
        LocalTensor<DTYPE_X> tmpTensor2 = AllocTmpTensor(trueCurSizePadded);
        
        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        for (uint32_t i = 0; i < nextRadices; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            InverseInput(matmulTmps[i * trueCurSize], matmulPaddedResTensor, tmpTensor2, trueCurSizePadded, info, true);

            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            DataCopy(matmulTmps[1 + firstSize + i * trueCurSize], matmulPaddedResTensor, trueCurSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        for (uint32_t i = 0; i < nextRadices; i++) {
            matmulObj2.SetTensorA(dftTensorReal, false);
            matmulObj2.SetTensorB(matmulTmps[1 + i * trueCurSize], false);
            matmulObj2.IterateAll(matmulTmps[1 + i * trueCurSize]);
            AscendC::PipeBarrier<PIPE_ALL>();

            matmulObj2.SetTensorA(dftTensorImag, false);
            matmulObj2.SetTensorB(matmulTmps[1 + firstSize + i * trueCurSize], false);
            matmulObj2.IterateAll(matmulTmps[1 + i * trueCurSize], true);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        FreeTmpTensor(COMPLEX * trueCurSizePadded);
    }
    __aicore__ inline uint32_t CalculatePrevRadices(const uint32_t factors[], uint32_t curFactorIndex) {
        uint32_t prevRadices = 1;
        for (uint8_t i = 0; i < MAX_FACTORS_LEN; i++) {
            if (i < curFactorIndex) {
                prevRadices *= factors[i];
            }
        }
        return prevRadices;
    }

    __aicore__ inline uint32_t CalculateNextRadices(const uint32_t factors[], uint32_t curFactorIndex) {
        uint32_t nextRadices = 1;
        for (uint8_t i = 0; i < MAX_FACTORS_LEN; i++) {
            if (i > curFactorIndex) {
                nextRadices *= factors[i];
            }
        }
        return nextRadices;
    }

    // Current implementation of Cooley-Tukey algorithm uses only Vector unit
    __aicore__ inline void CooleyTukeyAlgorithm() {
        const uint32_t firstSize = factors[0] * 2 * RoundUpBlock(fftLength / factors[0]);
    
        for (uint32_t curBatch = 0; curBatch < batchesCurCore; ++curBatch) {
            for (uint8_t curFactorIndex = 0; curFactorIndex < MAX_FACTORS_LEN; curFactorIndex++) {
                const uint32_t& curRadix = factors[curFactorIndex];
        
                if (curRadix == 1) {
                    break;
                } 
        
                uint32_t prevRadices = CalculatePrevRadices(factors, curFactorIndex);
                uint32_t nextRadices = CalculateNextRadices(factors, curFactorIndex);

                const bool firstIter = (curFactorIndex == 0);
                const bool lastIter = (nextRadices == 1);
        
                auto dftTensorReal = dftGmReal[dftRealOffsets[curFactorIndex]];
                auto dftTensorImag = dftGmImag[dftImagOffsets[curFactorIndex]];
        
                auto twiddleTensorRe = twiddleGmReal[twiddleOffsets[curFactorIndex]];
                auto twiddleTensorIm = twiddleGmImag[twiddleOffsets[curFactorIndex]];
        
                if (firstIter) {
                    FirstIterCalc(curRadix, curBatch);
                } else if (lastIter) {
                    const uint32_t curSize = (2 - isLarge) * prevRadices;
                    const uint32_t trueCurSize = 2 * prevRadices;
                    const uint32_t curSizePadded = RoundUpBlock(curSize + 1, 64);
                    const uint32_t trueCurSizePadded = RoundUpBlock(trueCurSize + 1, 64);
                    const uint32_t halfCurRadix = curRadix / COMPLEX;
                    const uint32_t imagOffset = 1 + firstSize + curRadix * tailSize;

                    const uint32_t elsToCopy = RoundDownBlock(tailSize);
                    const uint32_t elsToPadCopy = tailSize - elsToCopy;
        
                    DataCopyParams copyPadParams{1, static_cast<uint16_t>(elsToPadCopy * sizeof(DTYPE_X)), 0, 0};

                    DropOutShapeInfo info;
                    info.firstAxis = BLOCK_LEN_FP32;
                    info.srcLastAxis = curSizePadded / BLOCK_LEN_FP32;
                    info.maskLastAxis = 1;

                    LastIterFirstStep(twiddleTensorRe, twiddleTensorIm, curRadix, prevRadices, curSize, 
                                      elsToCopy, elsToPadCopy, curSizePadded, firstSize, copyPadParams, info);

                    auto curYGm = yGm[curBatch * COMPLEX * outLength];
                    LocalTensor<DTYPE_X> matmulPaddedResTensor = AllocTmpTensor(trueCurSizePadded);
                    LocalTensor<DTYPE_X> tmpTensor2 = AllocTmpTensor(trueCurSizePadded);
                    
                    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

                    LastIterSecondStep(curYGm, dftTensorReal, dftTensorImag, matmulPaddedResTensor,
                                       tmpTensor2, curRadix, halfCurRadix, trueCurSize, elsToCopy, elsToPadCopy,
                                       imagOffset, trueCurSizePadded, firstSize, copyPadParams, info);
                    
                    FreeTmpTensor(COMPLEX * trueCurSizePadded);
                } else {
                    IntermediateIterFirstStep(prevRadices, nextRadices, firstSize, curRadix);

                    IntermediateIterSecondStep(twiddleTensorRe, twiddleTensorIm, prevRadices, nextRadices, firstSize, curRadix);
                                    
                    IntermediateIterThirdStep(dftTensorReal, dftTensorImag, prevRadices, nextRadices, firstSize, curRadix);
                }
            }
        }
    }
};

#endif
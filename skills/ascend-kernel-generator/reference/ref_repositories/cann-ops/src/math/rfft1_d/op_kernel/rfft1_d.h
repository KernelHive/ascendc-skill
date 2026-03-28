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
 * \file rfft1_d.h
 * \brief
 */

#ifndef OPP_RFFT1D_H
#define OPP_RFFT1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/pad/kernel_operator_pad_intf.h"

#define CUBE_DFT

#ifdef __CCE_AICORE__
#include "lib/matmul_intf.h"
using namespace matmul;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_LEN_FP32 = 8;
constexpr uint32_t BLOCK_LEN_FP32_ROW = 16;
constexpr uint32_t MIN_TRANSPOSE_ROWS = 16;
constexpr uint32_t MIN_TRANSPOSE = MIN_TRANSPOSE_ROWS * BLOCK_LEN_FP32;
constexpr uint32_t MAX_VEC_ELEMS_PER_REP = 64;
constexpr uint32_t TMP_BUF_NUM_FIRST_PART = 3;
constexpr uint32_t TMP_BUF_NUM_SECOND_PART = 2;
constexpr uint32_t CORE_IDX_DIV = 2;
constexpr uint32_t MAX_FACTORS_LEN = 3;
constexpr uint32_t RFFT_HALF = 2;

constexpr uint32_t NZ_BORDER = 8;
constexpr uint32_t NZ_BLOCK = 16;

constexpr uint32_t TILING_FAST_BASE_M = 128;
constexpr uint32_t TILING_FAST_BASE_N = 128;
constexpr uint32_t TILING_FAST_BASE_K = 64;
constexpr uint32_t TILING_FAST_BASE_DEFAULT = 16;
constexpr uint32_t TILING_FAST_DEPTH_A = 8;
constexpr uint32_t TILING_FAST_DEPTH_B = 7;

constexpr uint32_t TILING_SECOND_BASE_M = 64;
constexpr uint32_t TILING_SECOND_BASE_N = 128;
constexpr uint32_t TILING_SECOND_BASE_K = 64;
constexpr uint32_t TILING_THIRD_BASE_M = 16;
constexpr uint32_t TILING_THIRD_BASE_N = 256;
constexpr uint32_t TILING_THIRD_BASE_K = 16;

constexpr uint32_t MAX_VEC_REP = 255;

constexpr uint32_t BATCHES_PER_CORE_BLUESTEIN = 1;
constexpr uint32_t LEFT_OVER_BATCHES_BLUESTEIN = 0;

static const uint8_t MASK85[16] = {85, 85, 85, 85, 85, 85, 85, 85, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t MASK170[16] = {170, 170, 170, 170, 170, 170, 170, 170, 0, 0, 0, 0, 0, 0, 0, 0};


const uint32_t MAX_WHOLE_DFT = 128;
const uint32_t MAX_CT_DFT = 64;
const uint32_t COMPLEX = 2;
const uint8_t TMP_TENSOR_SIZE_MULTIPLIER = 3;

const uint32_t UB_SIZE = 190 * 1024;
const uint32_t UB_LEN = UB_SIZE / sizeof(DTYPE_X);

constexpr uint32_t MAX_FFT_LEN = 64 * 64 * 64 * 2;

class KernelRfftFastDFT {
   private:
    const uint32_t fftLength;
    const uint8_t norm;
    const uint32_t batchesPerCore;
    const uint32_t leftOverBatches;
  
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_X> dftGm;
  
    uint32_t dftOverallSize;
    const __gm__ float* dftMatrix;
  
   public:
    TPipe pipe;
    using inputAType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X>;
    using inputBType = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_X>;
    using outputCType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    Matmul<inputAType, inputAType, outputCType> matmulObj;
    Matmul<inputAType, inputBType, outputCType> matmulObjNZ;
  
    uint32_t batches;
    uint32_t advancedBatches;
    uint32_t totalBatches;
    uint32_t modeLength;
    uint32_t cores;
    uint32_t batchesPerCoreCeil;
   public:
    __aicore__ inline KernelRfftFastDFT(const uint32_t& length, const uint32_t& batchesPerCore,
                                        const uint32_t& leftOverBatches, const uint32_t& norm,
                                        const uint32_t& dftOverallSize, uint32_t factors[MAX_FACTORS_LEN])
        : fftLength(length),
          batchesPerCore(batchesPerCore),
          leftOverBatches(leftOverBatches),
          norm(norm),
          dftOverallSize(dftOverallSize) {
        modeLength = (fftLength / RFFT_HALF + 1) * COMPLEX;
    
        cores = GetBlockNum();
        ASSERT(cores != 0 && "block dim can not be zero!");
    
        batches = batchesPerCore * cores + leftOverBatches;
        advancedBatches = leftOverBatches == 0 ? 0 : cores - leftOverBatches;
        totalBatches = batches + advancedBatches;
        batchesPerCoreCeil = (batches + advancedBatches) / GetBlockNum();
    }
  
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dftMatrix, GM_ADDR y, GM_ADDR workspace) {
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalBatches * fftLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, batches * modeLength);
    
        dftGm.SetGlobalBuffer((__gm__ DTYPE_X*)dftMatrix, dftOverallSize);
    }
  
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }
  
   private:
    // Copies the input from GM, normalizes it and splits the input
    __aicore__ inline void CopyIn() {
        if (g_coreType != AIV or GetSubBlockIdx() != 0) {
            return;
        }
    }
  
    __aicore__ void CubeDftMul(GlobalTensor<DTYPE_X>& a1, GlobalTensor<DTYPE_X>& b1, GlobalTensor<DTYPE_Y>& c1,
                                  bool accumulate = false) {
        matmulObj.SetTensorA(a1);
        matmulObj.SetTensorB(b1);
        matmulObj.IterateAll(c1, accumulate);
    
        matmulObj.End();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    
    __aicore__ void CubeDftMul1(GlobalTensor<DTYPE_X>& a1, GlobalTensor<DTYPE_X>& b1, GlobalTensor<DTYPE_Y>& c1,
                                  bool accumulate = false) {
        matmulObjNZ.SetTensorA(a1);
        matmulObjNZ.SetTensorB(b1);
        matmulObjNZ.IterateAll(c1, accumulate);
    
        matmulObjNZ.End();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
  
    // Count DFT directly (the entire input is multiplied on 1 DFT matrix)
    __aicore__ inline void CountWholeDft() {
        if (g_coreType == AIV and GetSubBlockIdx() != 0) {
            return;
        }
        int64_t offset;
        if (batches <= GetBlockNum()) {
            if (GetBlockIdx() / CORE_IDX_DIV < batches) {
                offset = totalBatches / cores * (GetBlockIdx() / CORE_IDX_DIV);
            } else {
                return;
            }
        } else if (GetBlockIdx() / CORE_IDX_DIV < batches * GetBlockNum() / totalBatches + 1) {
            offset = (GetBlockIdx() / CORE_IDX_DIV != batches * GetBlockNum() / totalBatches) ? 
                      totalBatches / cores * (GetBlockIdx() / CORE_IDX_DIV) :
                      totalBatches / cores * (GetBlockIdx() / CORE_IDX_DIV - 1) + batches % batchesPerCoreCeil;
        } else {
            return;
        }
        int64_t offsetX = fftLength * offset;
        int64_t offsetY = modeLength * offset;
    
        GlobalTensor<DTYPE_X> inputX = xGm[offsetX];
        GlobalTensor<DTYPE_X> inputDFT = dftGm;
        GlobalTensor<DTYPE_X> outputY = yGm[offsetY];
        if (fftLength % NZ_BLOCK && fftLength % NZ_BLOCK <= NZ_BORDER) {
            CubeDftMul(inputX, inputDFT, outputY);
        } else {
            CubeDftMul1(inputX, inputDFT, outputY);
        }
    }
  
    // Depending on the input length perform an FFT computation
    __aicore__ inline void Compute() {
        CountWholeDft();
    }
  
    // Merges real and imaginary parts into complex represintation and moves to GM
    __aicore__ inline void CopyOut() {
        if (g_coreType != AIV or GetSubBlockIdx() != 0) {
            return;
        }
    }
};

#endif //#ifdef __CCE_AICORE__

__aicore__ inline uint32_t RoundUpBlock(const uint32_t& src, const uint32_t& blockLen = BLOCK_LEN_FP32)
{
    if (blockLen != 0) {
        return src != 0 ? src + (blockLen - src % blockLen) % blockLen : blockLen;
    }
    return blockLen;
}

__aicore__ inline uint32_t RoundDownBlock(const uint32_t& src, const uint32_t& blockLen = BLOCK_LEN_FP32)
{
    if (blockLen != 0) {
        return (src / blockLen) * blockLen;
    }
    return blockLen;
}

__aicore__ inline TCubeTiling PrepareTiling(unsigned int m, unsigned int n, unsigned int k) {
    TCubeTiling tiling;
  
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    if(k % NZ_BLOCK > 0 && k % NZ_BLOCK <= NZ_BORDER)
    {
        tiling.Kb = k;
    }
    else
    {
        tiling.Kb = k + (BLOCK_LEN_FP32 - k % BLOCK_LEN_FP32);
    }
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    bool good_tiling = (k > MAX_VEC_ELEMS_PER_REP) || ((k & (k - 1)) == 0);
    tiling.baseM = good_tiling ? TILING_FAST_BASE_M : TILING_FAST_BASE_DEFAULT;
    tiling.baseN = good_tiling ? TILING_FAST_BASE_N : TILING_FAST_BASE_DEFAULT;
    tiling.baseK = good_tiling ? TILING_FAST_BASE_K : TILING_FAST_BASE_DEFAULT;
    tiling.depthA1 = TILING_FAST_DEPTH_A;
    tiling.depthB1 = TILING_FAST_DEPTH_B;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
  
    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling1(unsigned int m, unsigned int n, unsigned int k) {
    TCubeTiling tiling;
  
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = TILING_FAST_BASE_M;
    tiling.baseN = TILING_FAST_BASE_N;
    tiling.baseK = TILING_FAST_BASE_K;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
  
    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling2(unsigned int m, unsigned int n, unsigned int k) {
    TCubeTiling tiling;
  
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = TILING_SECOND_BASE_M;
    tiling.baseN = TILING_SECOND_BASE_N;
    tiling.baseK = TILING_SECOND_BASE_K;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
  
    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling3(unsigned int m, unsigned int n, unsigned int k) {
    TCubeTiling tiling;
  
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = TILING_THIRD_BASE_M;
    tiling.baseN = TILING_THIRD_BASE_N;
    tiling.baseK = TILING_THIRD_BASE_K;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
  
    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling4(unsigned int m, unsigned int n, unsigned int k){
    TCubeTiling tiling;
    
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = TILING_THIRD_BASE_M;
    tiling.baseN = TILING_THIRD_BASE_N;
    tiling.baseK = TILING_THIRD_BASE_K;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
    
    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling5(unsigned int m, unsigned int n, unsigned int k){
    TCubeTiling tiling;

    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = 64;
    tiling.baseN = 128;
    tiling.baseK = 64;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
    
    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling6(unsigned int m, unsigned int n, unsigned int k){
    TCubeTiling tiling;

    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = 16;
    tiling.baseN = 256;
    tiling.baseK = 16;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;

    return tiling;
}

__aicore__ inline TCubeTiling PrepareTiling7(unsigned int m, unsigned int n, unsigned int k){
    TCubeTiling tiling;

    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    tiling.baseM = 256;
    tiling.baseN = 16;
    tiling.baseK = 16;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;

    return tiling;
}

#endif
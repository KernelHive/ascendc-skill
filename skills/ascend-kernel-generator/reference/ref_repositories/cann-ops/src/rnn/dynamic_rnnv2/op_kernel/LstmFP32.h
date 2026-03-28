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
 * \file LstmFP32.h
 * \brief
 */
#ifndef _ASCENDC_LSTMFP32_H_
#define _ASCENDC_LSTMFP32_H_

#include "dynamic_rnn_common.h"

template <typename T>
class LstmMmSplitNDNDFP32 {
 public:
  __aicore__ inline LstmMmSplitNDNDFP32() = default;
  __aicore__ inline void Process();
  __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                              GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                              GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                              GM_ADDR outputO, GM_ADDR outputTanhC,
                              const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace);
  __aicore__ inline void InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                                GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf,
                                GM_ADDR wCo, GM_ADDR mask,
                                GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ,
                                GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace);
  __aicore__ inline void InitVars();
  __aicore__ inline void InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                                     GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                     GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                     GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                     GM_ADDR workspace);
  __aicore__ inline void InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                                       GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf,
                                       GM_ADDR wCo, GM_ADDR mask,
                                       GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                       GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                       GM_ADDR workspace);

 protected:
  struct tailSize {
    int64_t tailSingleCoreN;
    int64_t tailSingleCoreM;
    int64_t notTailNCoreCount;
    int64_t notTailMCoreCount;
    int32_t nCoreLoop;
    int32_t mCoreLoop;
    int64_t nCoreIndx;
    int64_t mCoreIndx;
  };

  __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail,
                                      int64_t cNdFormat, int32_t kSize);
  __aicore__ inline void InitQue();
  __aicore__ inline void ProcessInputMM();
  __aicore__ inline void ProcessHiddenMM(int64_t tIdx);
  __aicore__ inline void ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<T>& mixGm);
  __aicore__ inline void ProcessVectorInitHC(int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<T>& mixGm);
  __aicore__ inline void ProcessVector(int64_t tIdx);
  __aicore__ inline void ProcessInitalT();
  __aicore__ inline void CopyGate(AscendC::LocalTensor<T>& ub, AscendC::GlobalTensor<T>& gm, int64_t mIdx,
                                  int64_t nIdx, int64_t gateOffset);
  __aicore__ inline void CopyWithSigmoid(AscendC::LocalTensor<T>& dstUb, AscendC::GlobalTensor<T>& mixGm, int64_t mIdx,
                                        int64_t nIdx, int64_t gateOffset);
  __aicore__ inline void CopyWithSigmoidAddBias(AscendC::LocalTensor<float>& dstUb, AscendC::GlobalTensor<float>& mixGm, int64_t mIdx,
                                                int64_t nIdx, int64_t gateOffset);
  __aicore__ inline void CopyWithTanh(AscendC::LocalTensor<T>& dstUb, AscendC::GlobalTensor<T>& mixGm, int64_t mIdx,
                                      int64_t nIdx, int64_t gateOffset);
  __aicore__ inline void CopyWithMul(AscendC::LocalTensor<T>& dstUb, AscendC::LocalTensor<T>& other,
                                     AscendC::GlobalTensor<T>& mixGm, int64_t mIdx, int64_t nIdx);
  __aicore__ inline void CopyInHC(AscendC::LocalTensor<T>& dstUb, AscendC::GlobalTensor<T>& mixGm,
                                  int64_t tIdx, int64_t mIdx, int64_t nIdx);
  __aicore__ inline void CopyInSeq(AscendC::LocalTensor<T>& dstUb, AscendC::GlobalTensor<T>& mixGm,
                                   int64_t tIdx, int64_t mIdx, int64_t nIdx);
  __aicore__ inline void CopyOutput(AscendC::GlobalTensor<T>& gm, AscendC::LocalTensor<T>& ub, int64_t tIdx,
                                    int64_t mIdx, int64_t nIdx);
  __aicore__ inline void GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail, int32_t kSize);
  __aicore__ inline int64_t Ceil(int64_t x, int64_t y);

 public:
  AscendC::TPipe pipe;

  // describe Matmul input/output dtype&format
  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
      inputMM;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float>>
      hiddenMM;

 protected:
  // output GlobalTensors
  struct OutputGm {
    __aicore__ inline OutputGm() = default;
    AscendC::GlobalTensor<T> outYGm;
    AscendC::GlobalTensor<T> outHGm;
    AscendC::GlobalTensor<T> outCGm;
    AscendC::GlobalTensor<T> outIGm;
    AscendC::GlobalTensor<T> outJGm;
    AscendC::GlobalTensor<T> outFGm;
    AscendC::GlobalTensor<T> outOGm;
    AscendC::GlobalTensor<T> outTanhCGm;
    AscendC::GlobalTensor<T> workspace;
  };

  // input GlobalTensors
  struct InputGm {
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> weightInputGm;
    AscendC::GlobalTensor<T> weightHiddenGm;
    AscendC::GlobalTensor<T> biasGm;
    AscendC::GlobalTensor<T> seqLengthGm;
    AscendC::GlobalTensor<T> initHGm;
    AscendC::GlobalTensor<T> initCGm;
    AscendC::GlobalTensor<T> wciGm;
    AscendC::GlobalTensor<T> wcfGm;
    AscendC::GlobalTensor<T> wcoGm;
    AscendC::GlobalTensor<T> maskGm;
  };

  // Queue
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qidVecOut;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

  // LocalTensor
  AscendC::LocalTensor<float> ubLocal1, ubLocal2, ubLocal3, ubLocal4;

  OutputGm outputGm;
  InputGm inputGm;

  int64_t inputMKAllSize;
  int64_t iOffset;
  int64_t oOffset;
  int64_t jOffset;
  int64_t fOffset;
  int64_t tailSingleCoreN;
  int64_t tailSingleCoreM;
  int64_t notTailNCoreCount;
  int64_t notTailMCoreCount;
  int32_t nCoreLoop;
  int32_t mCoreLoop;
  TRnnOffsets inputOffsets;
  TRnnOffsets hiddenOffsets;

  int64_t allCellSize;
  int64_t oneCellSize;
  tailSize hiddenTail;
  tailSize inputTail;
  int32_t oriSingleCoreN;
  TRnnOffsets oriInputOffsets;
  TRnnOffsets oriHiddenOffsets;

  AscendC::GlobalTensor<int32_t> sync_gm;
  const DynamicRNNTilingData* __restrict tiling;
  TCubeTiling inputMMTiling;
  TCubeTiling hiddenMMTiling;
  AscendC::LocalTensor<int> sync_buf;

  int64_t blockSize;
  int64_t calBlockSize;
  int64_t vectorCoreM;
  int64_t vectorTailM;
  int64_t vectorCoreNum;
  int64_t vectorBaseM;
  int64_t vectorBaseTailM;
  int64_t vectorTailTailM;
  int64_t baseVector;
  int64_t calcSize;
  int64_t calcSizeAlign;
  int64_t blockIdx;
  int64_t vectorSplitM;
  int64_t vectorSplitN;
  int64_t vectorTailSplitM;
  int64_t vectorTailN;
  int64_t vectorBaseN;
  int64_t calcM;
  int64_t calcN;
  int64_t coreCalcM;
};

#endif

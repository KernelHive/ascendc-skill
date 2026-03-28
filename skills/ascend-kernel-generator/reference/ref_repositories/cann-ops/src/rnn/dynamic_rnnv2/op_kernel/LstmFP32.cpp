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
 * \file LstmFP32.cpp
 * \brief
 */
#include "LstmFP32.h"

using namespace AscendC;

template <typename T>
__aicore__ inline int64_t LstmMmSplitNDNDFP32<T>::Ceil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength,
                                                    GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                    GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                    GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                    GM_ADDR outputTanhC, const DynamicRNNTilingData* __restrict rnnTiling,
                                                    GM_ADDR workspace) {
  tiling = rnnTiling;
  inputMMTiling = tiling->inputMMParam;
  hiddenMMTiling = tiling->hiddenMMParam;
  InitBuffers(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI,
              outputJ, outputF, outputO, outputTanhC, workspace);
  InitVars();
  InitQue();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                              GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi,
                              GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH,
                              GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                              GM_ADDR outputTanhC, const DynamicRNNTilingData* __restrict rnnTiling,
                              GM_ADDR workspace) {
  tiling = rnnTiling;
  inputMMTiling = tiling->inputMMParam;
  hiddenMMTiling = tiling->hiddenMMParam;
  InitBuffersV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC, wCi, wCf, wCo, mask,
                outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, workspace);
  InitVars();
  InitQue();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::InitQue() {
  pipe.InitBuffer(qidVecIn, 1, baseVector * sizeof(float));
  pipe.InitBuffer(qidVecOut, 1, baseVector * sizeof(T));
  pipe.InitBuffer(calcBuf, 4 * baseVector * sizeof(float));
  // Init Local Tensors
  ubLocal1 = calcBuf.Get<float>(4 * baseVector);
  ubLocal2 = ubLocal1[baseVector];
  ubLocal3 = ubLocal2[baseVector];
  ubLocal4 = ubLocal3[baseVector];
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail,
                                                            int32_t kSize) {
  auto temp0 = Ceil(param.M, param.singleCoreM);
  auto temp1 = Ceil(param.N, param.singleCoreN);
  auto temp2 = Ceil(kSize, param.singleCoreK);  // 不切K, 应该=1
  if (temp0 == 0) {
    temp0 = 1;
  }
  if (temp2 == 0) {
    temp2 = 1;
  }
  auto divideKcoreNum = param.usedCoreNum / temp2;
  mmTail.mCoreIndx = (GetBlockIdx() % divideKcoreNum) % temp0;
  mmTail.nCoreIndx = (GetBlockIdx() % divideKcoreNum) / temp0;
  subKIndx = GetBlockIdx() / divideKcoreNum;  // 缺省为0
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail,
                                                            int64_t cNdFormat, int32_t kSize) {
  int32_t subKIndx;
  GetCoreIndex(param, subKIndx, mmTail, kSize);
  offset.AOffset = mmTail.mCoreIndx * kSize * param.singleCoreM;
  offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
  offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

  mmTail.nCoreLoop = Ceil(param.N, param.singleCoreN);
  mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;
  mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;
  mmTail.mCoreLoop = Ceil(param.M, param.singleCoreM);
  mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
  mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;

  if (cNdFormat == 1) {
    offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias,
                                                    GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi,
                                                    GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                                                    GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                                    GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                    GM_ADDR outputTanhC, GM_ADDR workspace) {
  CalcGMOffset(hiddenMMTiling, hiddenOffsets, hiddenTail, 1, static_cast<int32_t>(tiling->hiddenSize));
  CalcGMOffset(inputMMTiling, inputOffsets, inputTail, 1, static_cast<int32_t>(tiling->inputSize));
  oneCellSize = tiling->batch * tiling->hiddenSize;
  allCellSize = oneCellSize * LSTM_GATE_SIZE;
  oriHiddenOffsets = hiddenOffsets;
  oriInputOffsets = inputOffsets;

  inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                              tiling->timeStep * tiling->batch * tiling->inputSize);
  inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight),
                                   tiling->inputSize * LSTM_GATE_SIZE * tiling->hiddenSize);
  inputGm.weightHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(
                                         weight + tiling->inputSize * LSTM_GATE_SIZE * tiling->hiddenSize * sizeof(T)),
                                         tiling->hiddenSize * LSTM_GATE_SIZE * tiling->hiddenSize);
                                  
  inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * tiling->hiddenSize);

  if (tiling->isSeqLength != 0) {
    inputGm.seqLengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(seqLength),
                                tiling->timeStep * tiling->batch * tiling->hiddenSize);
  }

  if (tiling->isInithc != 0) {
    inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH), tiling->batch * tiling->hiddenSize);
    inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC), tiling->batch * tiling->hiddenSize);
  }
  outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  outputGm.outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  if (tiling->isTraining == 1) {
    outputGm.outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outTanhCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputTanhC),
                                        tiling->timeStep * tiling->batch * tiling->hiddenSize);
  }
  outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace),
                                     tiling->timeStep * tiling->batch * LSTM_GATE_SIZE * tiling->hiddenSize);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                    GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC,
                                                    GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                                    GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                                    GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                    GM_ADDR outputTanhC, GM_ADDR workspace) {
  CalcGMOffset(hiddenMMTiling, hiddenOffsets, hiddenTail, 1, static_cast<int32_t>(tiling->hiddenSize));
  CalcGMOffset(inputMMTiling, inputOffsets, inputTail, 1, static_cast<int32_t>(tiling->inputSize));
  oneCellSize = tiling->batch * tiling->hiddenSize;
  allCellSize = oneCellSize * LSTM_GATE_SIZE;
  oriHiddenOffsets = hiddenOffsets;
  oriInputOffsets = inputOffsets;

  inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                              tiling->timeStep * tiling->batch * tiling->inputSize);
  inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weightInput),
                                   tiling->inputSize * LSTM_GATE_SIZE * tiling->hiddenSize);
  inputGm.weightHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weightHidden),
                                   tiling->hiddenSize * LSTM_GATE_SIZE * tiling->hiddenSize);
                                  
  if (tiling->isBias == 1) {
    inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * tiling->hiddenSize);
  }
  if (tiling->isSeqLength != 0) {
    inputGm.seqLengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(seqLength),
                                tiling->timeStep * tiling->batch * tiling->hiddenSize);
  }
  if (tiling->isInithc != 0) {
    inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH), tiling->batch * tiling->hiddenSize);
    inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC), tiling->batch * tiling->hiddenSize);
  }
  outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  outputGm.outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  if (tiling->isTraining == 1) {
    outputGm.outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outTanhCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputTanhC),
                                        tiling->timeStep * tiling->batch * tiling->hiddenSize);
  }
  outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace),
                                     tiling->timeStep * tiling->batch * LSTM_GATE_SIZE * tiling->hiddenSize);
}
template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::InitVars() {
  int64_t ubSize = 16384; // after div the max num of exiting node at the same time,include 4 gate
  int64_t calcMaxSize = ubSize / sizeof(float);
  int64_t blockN = tiling->usedCoreNum;
  blockSize = 32 / sizeof(T);
  calBlockSize = 32 / sizeof(float);
  vectorCoreM = Ceil(tiling->batch, blockN);
  vectorCoreNum = Ceil(tiling->batch, vectorCoreM);
  vectorTailM = tiling->batch % vectorCoreM ? tiling->batch % vectorCoreM : vectorCoreM; 

  vectorSplitN = Ceil(tiling->hiddenSize, calcMaxSize);
  if (vectorSplitN == 1) {
    vectorBaseN = tiling->hiddenSize;
    vectorTailN = 0;
  } else {
    vectorBaseN = Ceil(Ceil(tiling->hiddenSize, vectorSplitN), calBlockSize) * calBlockSize;
    vectorTailN = tiling->hiddenSize - vectorBaseN * (vectorSplitN - 1);
  }

  vectorBaseM = ((calcMaxSize / vectorBaseN) > vectorCoreM) ? vectorCoreM : (calcMaxSize / vectorBaseN);

  vectorBaseTailM = vectorCoreM % vectorBaseM;
  vectorTailTailM = vectorTailM % vectorBaseM;

  vectorSplitM = Ceil(vectorCoreM, vectorBaseM);
  vectorTailSplitM = Ceil(vectorTailM, vectorBaseM);

  baseVector = vectorBaseM * Ceil(vectorBaseN, calBlockSize) * calBlockSize;

  iOffset = 0;
  jOffset = tiling->gateOrder == 0 ? tiling->hiddenSize : 2 * tiling->hiddenSize;
  fOffset = tiling->gateOrder == 0 ? 2 * tiling->hiddenSize : tiling->hiddenSize;
  oOffset = 3 * tiling->hiddenSize;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInputMM() {
  if (GetBlockIdx() < inputMMTiling.usedCoreNum) {
    inputMM.SetTensorA(inputGm.xGm[inputOffsets.AOffset]);
    inputMM.SetTensorB(inputGm.weightInputGm[inputOffsets.BOffset]);
    if (tiling->isBias == 1) {
      inputMM.SetBias(inputGm.biasGm[inputOffsets.BOffset]);
    }

    if (inputTail.nCoreIndx == inputTail.notTailNCoreCount && inputTail.mCoreIndx == inputTail.notTailMCoreCount) {
      inputMM.SetTail(inputTail.tailSingleCoreM, inputTail.tailSingleCoreN);
    } else if (inputTail.nCoreIndx == inputTail.notTailNCoreCount) {
      inputMM.SetTail(inputMMTiling.singleCoreM, inputTail.tailSingleCoreN);
    } else if (inputTail.mCoreIndx == inputTail.notTailMCoreCount) {
      inputMM.SetTail(inputTail.tailSingleCoreM, inputMMTiling.singleCoreN);
    }
    inputMM.IterateAll(outputGm.workspace[inputOffsets.COffset], false);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessHiddenMM(int64_t tIdx) {
  if (GetBlockIdx() < hiddenMMTiling.usedCoreNum) {
    if (tiling->direction == 1) {
      hiddenOffsets.COffset = oriHiddenOffsets.COffset + (tiling->timeStep - 1 - tIdx) * allCellSize;
    } else {
      hiddenOffsets.COffset = oriHiddenOffsets.COffset + tIdx * allCellSize;
    }
    hiddenMM.SetTensorA(inputGm.initHGm[hiddenOffsets.AOffset]);
    hiddenMM.SetTensorB(inputGm.weightHiddenGm[hiddenOffsets.BOffset]);
    if (hiddenTail.nCoreIndx == hiddenTail.notTailNCoreCount && hiddenTail.mCoreIndx == hiddenTail.notTailMCoreCount) {
      hiddenMM.SetTail(hiddenTail.tailSingleCoreM, hiddenTail.tailSingleCoreN);
    } else if (hiddenTail.nCoreIndx == hiddenTail.notTailNCoreCount) {
      hiddenMM.SetTail(hiddenMMTiling.singleCoreM, hiddenTail.tailSingleCoreN);
    } else if (hiddenTail.mCoreIndx == hiddenTail.notTailMCoreCount) {
      hiddenMM.SetTail(hiddenTail.tailSingleCoreM, hiddenMMTiling.singleCoreN);
    }
    hiddenMM.IterateAll(outputGm.workspace[hiddenOffsets.COffset], true);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyGate(LocalTensor<T>& ub, GlobalTensor<T>& gm, int64_t mIdx,
                                                        int64_t nIdx, int64_t gateOffset) {
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(T);
  dataCopyParams.srcStride = (4 * tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams.dstStride = 0;

  DataCopyPadParams padParams;
  padParams.isPad = false;
  padParams.leftPadding = 0;
  padParams.rightPadding = Ceil(calcN, blockSize) * blockSize - calcN;
  padParams.paddingValue = 0;

  DataCopyPad(ub,
              gm[gateOffset + blockIdx * vectorCoreM * tiling->hiddenSize * 4 +
                 mIdx * vectorBaseM * tiling->hiddenSize * 4 +
                 nIdx * vectorBaseN],
              dataCopyParams,
              padParams);
  qidVecIn.EnQue(ub);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithSigmoid(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                               int64_t mIdx, int64_t nIdx, int64_t gateOffset) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
  ubLocalIn = qidVecIn.DeQue<T>();
  Sigmoid(dstUb, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithSigmoidAddBias(LocalTensor<float>& dstUb, GlobalTensor<float>& mixGm,
                                                               int64_t mIdx, int64_t nIdx, int64_t gateOffset) {
  LocalTensor<float> ubLocalIn = qidVecIn.AllocTensor<float>();
  CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
  ubLocalIn = qidVecIn.DeQue<float>();
  Adds(ubLocalIn, ubLocalIn, (float)tiling->forgetBias, calcSizeAlign);
  PipeBarrier<PIPE_V>();
  Sigmoid(dstUb, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithTanh(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                            int64_t mIdx, int64_t nIdx, int64_t gateOffset) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
  ubLocalIn = qidVecIn.DeQue<T>();
  Tanh(dstUb, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithMul(LocalTensor<T>& dstUb, LocalTensor<T>& other,
                                                           GlobalTensor<T>& mixGm, int64_t mIdx, int64_t nIdx) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(T);
  dataCopyParams.srcStride = (tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams.dstStride = 0;

  DataCopyPadParams padParams;
  padParams.isPad = false;
  padParams.leftPadding = 0;
  padParams.rightPadding = Ceil(calcN, blockSize) * blockSize - calcN;
  padParams.paddingValue = 0;

  DataCopyPad(ubLocalIn,
              mixGm[blockIdx * vectorCoreM * tiling->hiddenSize + mIdx * vectorBaseM * tiling->hiddenSize +
                    nIdx * vectorBaseN],
              dataCopyParams,
              padParams);
  qidVecIn.EnQue(ubLocalIn);
  ubLocalIn = qidVecIn.DeQue<T>();
  Mul(dstUb, ubLocalIn, other, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyInHC(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm, int64_t tIdx, int64_t mIdx, int64_t nIdx) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(T);
  dataCopyParams.srcStride = (tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams.dstStride = 0;

  DataCopyPadParams padParams;
  padParams.isPad = false;
  padParams.leftPadding = 0;
  padParams.rightPadding = Ceil(calcN, blockSize) * blockSize - calcN;
  padParams.paddingValue = 0;

  DataCopyPad(ubLocalIn,
              mixGm[blockIdx * vectorCoreM * tiling->hiddenSize + mIdx * vectorBaseM * tiling->hiddenSize +
                    nIdx * vectorBaseN],
              dataCopyParams,
              padParams);
  qidVecIn.EnQue(ubLocalIn);
  ubLocalIn = qidVecIn.DeQue<T>();
  PipeBarrier<PIPE_V>();
  Adds(dstUb, ubLocalIn, (float)0.0, calcSizeAlign);
  PipeBarrier<PIPE_V>();
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyInSeq(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm, int64_t tIdx, int64_t mIdx, int64_t nIdx) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(T);
  dataCopyParams.srcStride = (tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams.dstStride = 0;

  DataCopyPadParams padParams;
  padParams.isPad = false;
  padParams.leftPadding = 0;
  padParams.rightPadding = Ceil(calcN, blockSize) * blockSize - calcN;
  padParams.paddingValue = 0;

  int64_t tOffset = tIdx * tiling->batch * tiling->hiddenSize;

  if (tiling->direction == 1) {
      tOffset = (tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize;
  }

  DataCopyPad(ubLocalIn,
              mixGm[tOffset + blockIdx * vectorCoreM * tiling->hiddenSize + mIdx * vectorBaseM * tiling->hiddenSize +
                    nIdx * vectorBaseN],
              dataCopyParams,
              padParams);
  qidVecIn.EnQue(ubLocalIn);
  ubLocalIn = qidVecIn.DeQue<T>();
  PipeBarrier<PIPE_V>();
  Adds(dstUb, ubLocalIn, (float)0.0, calcSizeAlign);
  PipeBarrier<PIPE_V>();
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyOutput(GlobalTensor<T>& gm, LocalTensor<T>& ub, int64_t tIdx,
                                                          int64_t mIdx, int64_t nIdx) {
  LocalTensor<T> outLocal = qidVecOut.AllocTensor<T>();
  PipeBarrier<PIPE_V>();
  Muls(outLocal, ub, (float)1.0, calcSizeAlign);
  qidVecOut.EnQue(outLocal);
  outLocal = qidVecOut.DeQue<T>();
  int64_t offset;
  if (tiling->direction == 1) {
    offset = (tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize +
             blockIdx * vectorCoreM * tiling->hiddenSize +
             mIdx * vectorBaseM * tiling->hiddenSize +
             nIdx * vectorBaseN;
  } else {
    offset = tIdx * tiling->batch * tiling->hiddenSize +
             blockIdx * vectorCoreM * tiling->hiddenSize +
             mIdx * vectorBaseM * tiling->hiddenSize +
             nIdx * vectorBaseN;
  }
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(T);
  dataCopyParams.srcStride = 0;
  dataCopyParams.dstStride = (tiling->hiddenSize - calcN) * sizeof(T);

  DataCopyPadParams padParams;
  padParams.isPad = false;
  padParams.leftPadding = 0;
  padParams.rightPadding = 0;
  padParams.paddingValue = 0;
  
  DataCopyPad(gm[offset],
              outLocal,
              dataCopyParams);
  qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx,
                                                                 GlobalTensor<T>& mixGm) {
  blockIdx = GetBlockIdx();
  // get n size
  if ((vectorTailN > 0) && (nIdx == vectorSplitN - 1)) {
    calcN = vectorTailN;
  } else {
    calcN = vectorBaseN;
  }
  // get m size
  calcM = vectorBaseM;
  if ((blockIdx < vectorCoreNum - 1) && (vectorBaseTailM > 0) && (mIdx == vectorSplitM - 1)) {
    // Calc block's m_size in the base core last block.
    calcM = vectorBaseTailM;
  }
  if ((blockIdx == vectorCoreNum - 1) &&
      (vectorTailTailM > 0) && (mIdx == vectorTailSplitM - 1)) {
    // Calc block's m_size in the last core last block.
    calcM = vectorTailTailM;
  }

  // get calc once block size
  calcSize = calcM * calcN;
  calcSizeAlign = calcM * Ceil(calcN, calBlockSize) * calBlockSize;

  PipeBarrier<PIPE_V>();

  // f 1 2 3 4 -> [1] 2 3 4
  auto fSigmoid = ubLocal1;
  LocalTensor<float> ubLocalIn = qidVecIn.AllocTensor<float>();
  CopyGate(ubLocalIn, mixGm, mIdx, nIdx, fOffset);
  ubLocalIn = qidVecIn.DeQue<float>();
  Adds(ubLocalIn, ubLocalIn, (float)tiling->forgetBias, calcSizeAlign);
  PipeBarrier<PIPE_V>();
  Sigmoid(fSigmoid, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outFGm, fSigmoid, tIdx, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();

  // [1] 2 3 4 -> 1 [2] 3 4
  auto cTmp1 = ubLocal2;
  CopyWithMul(cTmp1, fSigmoid, inputGm.initCGm, mIdx, nIdx);
  PipeBarrier<PIPE_V>();
  // i 1 [2] 3 4 -> [1] [2] 3 4
  auto iSigmoid = ubLocal1;
  CopyWithSigmoid(iSigmoid, mixGm, mIdx, nIdx, iOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outIGm, iSigmoid, tIdx, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();
  // j [1] [2] 3 4 -> [1] [2] [3] 4
  auto jTanh = ubLocal3;
  CopyWithTanh(jTanh, mixGm, mIdx, nIdx, jOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outJGm, jTanh, tIdx, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();
  // i * j [1] [2] [3] 4 -> 1 [2] 3 [4]
  auto cTmp2 = ubLocal4;
  Mul(cTmp2, jTanh, iSigmoid, calcSizeAlign);
  PipeBarrier<PIPE_V>();

  // i * j + f * c 1 [2] 3 [4] -> [1] 2 3 4
  auto updateC = ubLocal1;
  Add(updateC, cTmp1, cTmp2, calcSizeAlign);

  if (tiling->isSeqLength == 1) {
    auto initC = ubLocal2;
    auto seqLength = ubLocal4;
    CopyInHC(initC, inputGm.initCGm, 0, mIdx, nIdx);
    Sub(updateC, updateC, initC, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    CopyInSeq(seqLength, inputGm.seqLengthGm, tIdx, mIdx, nIdx);
    Mul(updateC, updateC, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Add(updateC, updateC, initC, calcSizeAlign);
    PipeBarrier<PIPE_V>();
  }

  if (tiling->cellClip > 0) {
    PipeBarrier<PIPE_V>();
    Mins(updateC, updateC, (float)tiling->cellClip, calcSizeAlign);
  }

  CopyOutput(outputGm.outCGm, updateC, tIdx, mIdx, nIdx);
  PipeBarrier<PIPE_V>();
  
  // tanh(c) 1 [2] 3 4 -> 1 [2] 3 4
  auto cTanh = ubLocal2;
  Tanh(cTanh, updateC, calcSizeAlign);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outTanhCGm, cTanh, tIdx, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();
  // o 1 [2] 3 4 -> [1] [2] 3 4
  auto oSigmoid = ubLocal1;
  CopyWithSigmoid(oSigmoid, mixGm, mIdx, nIdx, oOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outOGm, oSigmoid, tIdx, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();

  // o * Tanh(c) [1] [2] 3 4 -> 1 2 [3] 4
  auto updateH = ubLocal3;
  Mul(updateH, oSigmoid, cTanh, calcSizeAlign);

  if (tiling->isSeqLength == 1) {
    PipeBarrier<PIPE_V>();
    auto updateY = ubLocal1;
    auto initH = ubLocal2;
    auto seqLength = ubLocal4;
    Mul(updateY, updateH, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    CopyInHC(initH, inputGm.initHGm, 0, mIdx, nIdx);
    Sub(updateH, updateH, initH, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(updateH, updateH, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Add(updateH, updateH, initH, calcSizeAlign);
    CopyOutput(outputGm.outYGm, updateY, tIdx, mIdx, nIdx);
  } else {
    CopyOutput(outputGm.outYGm, updateH, tIdx, mIdx, nIdx);
  }

  CopyOutput(outputGm.outHGm, updateH, tIdx, mIdx, nIdx);
}


template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVectorInitHC(int64_t mIdx, int64_t nIdx,
                                                                 GlobalTensor<T>& mixGm) {
  blockIdx = GetBlockIdx();
  // get n size
  if ((vectorTailN > 0) && (nIdx == vectorSplitN - 1)) {
    calcN = vectorTailN;
  } else {
    calcN = vectorBaseN;
  }

  // get m size
  calcM = vectorBaseM;
  if ((blockIdx < vectorCoreNum - 1) && (vectorBaseTailM > 0) && (mIdx == vectorSplitM - 1)) {
    // Calc block's m_size in the base core last block.
    calcM = vectorBaseTailM;
  }
  if ((blockIdx == vectorCoreNum - 1) &&
      (vectorTailTailM > 0) && (mIdx == vectorTailSplitM - 1)) {
    // Calc block's m_size in the last core last block.
    calcM = vectorTailTailM;
  }

  // get calc once block size
  calcSize = calcM * calcN;
  calcSizeAlign = calcM * Ceil(calcN, calBlockSize) * calBlockSize;

  PipeBarrier<PIPE_V>();

  // f 1 2 3 4 -> [1] 2 3 4
  auto fSigmoid = ubLocal1;
  CopyWithSigmoidAddBias(fSigmoid, mixGm, mIdx, nIdx, iOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outFGm, fSigmoid, 0, mIdx, nIdx);
  }

  PipeBarrier<PIPE_V>();
  
  // i 1 [2] 3 4 -> [1] [2] 3 4
  auto iSigmoid = ubLocal1;
  CopyWithSigmoid(iSigmoid, mixGm, mIdx, nIdx, iOffset);
  
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outIGm, iSigmoid, 0, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();

  // j [1] [2] 3 4 -> [1] [2] [3] 4
  auto jTanh = ubLocal3;
  CopyWithTanh(jTanh, mixGm, mIdx, nIdx, jOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outJGm, jTanh, 0, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();

  // i * j [1] [2] [3] 4 -> 1 [2] 3 [4]
  auto cTmp2 = ubLocal4;
  Mul(cTmp2, jTanh, iSigmoid, calcSizeAlign);
  PipeBarrier<PIPE_V>();

  // i * j + f * c 1 [2] 3 [4] -> [1] 2 3 4
  auto updateC = cTmp2;

  if (tiling->isSeqLength == 1) {
    auto seqLength = ubLocal3;
    CopyInSeq(seqLength, inputGm.seqLengthGm, 0, mIdx, nIdx);
    Mul(updateC, updateC, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
  }

  if (tiling->cellClip > 0) {
    PipeBarrier<PIPE_V>();
    Mins(updateC, updateC, (float)tiling->cellClip, calcSizeAlign);
  }

  CopyOutput(outputGm.outCGm, updateC, 0, mIdx, nIdx);
  PipeBarrier<PIPE_V>();
  
  // tanh(c) 1 [2] 3 4 -> 1 [2] 3 4
  auto cTanh = ubLocal2;
  Tanh(cTanh, updateC, calcSizeAlign);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outTanhCGm, cTanh, 0, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();
  // o 1 [2] 3 4 -> [1] [2] 3 4
  auto oSigmoid = ubLocal1;
  CopyWithSigmoid(oSigmoid, mixGm, mIdx, nIdx, oOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outOGm, oSigmoid, 0, mIdx, nIdx);
  }
  PipeBarrier<PIPE_V>();

  // o * Tanh(c) [1] [2] 3 4 -> 1 2 [3] 4
  auto updateH = ubLocal4;
  Mul(updateH, oSigmoid, cTanh, calcSizeAlign);

  if (tiling->isSeqLength == 1) {
    auto seqLength = ubLocal3;
    PipeBarrier<PIPE_V>();
    Mul(updateH, updateH, seqLength, calcSizeAlign);
  }

  CopyOutput(outputGm.outHGm, updateH, 0, mIdx, nIdx);
  CopyOutput(outputGm.outYGm, updateH, 0, mIdx, nIdx);
}


template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVector(int64_t tIdx) {
  auto mCoreIndex = GetBlockIdx();
  if (mCoreIndex < vectorCoreNum) {
    auto coreLoopM = vectorSplitM;
    if (mCoreIndex == vectorCoreNum - 1) {
      // Calc the last core.
      coreLoopM = vectorTailSplitM;
    }
    int64_t offset;
    if (tiling->direction == 1) {
      offset = (tiling->timeStep - 1 - tIdx) * allCellSize;
    } else {
      offset = tIdx * allCellSize;
    }
    for (int64_t j = 0; j < coreLoopM; ++j) {
      for (int64_t k = 0; k < vectorSplitN; ++k) {
        auto mixGm = outputGm.workspace[offset];
        ProcessVectorOnce(tIdx, j, k, mixGm);
      }
    }
  }
}


template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInitalT() {
  auto mCoreIndex = GetBlockIdx();
  if (mCoreIndex < vectorCoreNum) {
    auto coreLoopM = vectorSplitM;
    if (mCoreIndex == vectorCoreNum - 1) {
      // Calc the last core.
      coreLoopM = vectorTailSplitM;
    }
    int64_t offset;
    if (tiling->direction == 1) {
      offset = (tiling->timeStep - 1) * allCellSize;
    } else {
      offset = 0;
    }
    for (int64_t j = 0; j < coreLoopM; ++j) {
      for (int64_t k = 0; k < vectorSplitN; ++k) {
        auto mixGm = outputGm.workspace[offset];
        ProcessVectorInitHC(j, k, mixGm);
      }
    }
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::Process() {
  ProcessInputMM();
  if (tiling->isInithc == 0) {
    SyncAll();
    ProcessInitalT();
    if (tiling->direction == 1) {
      inputGm.initCGm = outputGm.outCGm[(tiling->timeStep - 1) * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[(tiling->timeStep - 1) * tiling->batch * tiling->hiddenSize];
    } else {
      inputGm.initCGm = outputGm.outCGm;
      inputGm.initHGm = outputGm.outHGm;
    }
  }

  

  for (int64_t tIdx = tiling->isInithc == 0 ? 1 : 0; tIdx < tiling->timeStep; tIdx++) {
    SyncAll();

    ProcessHiddenMM(tIdx);

    SyncAll();

    ProcessVector(tIdx);

    SyncAll();

    if (tiling->direction == 1) {
      inputGm.initCGm = outputGm.outCGm[(tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[(tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize];
    } else {
      inputGm.initCGm = outputGm.outCGm[tIdx * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[tIdx * tiling->batch * tiling->hiddenSize];
    }
  }
}

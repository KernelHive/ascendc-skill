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
 * \file LstmFP16.cpp
 * \brief
 */
#include "LstmFP16.h"

using namespace AscendC;
constexpr static const uint32_t FLOAT_BLOCK_NUM = 8;
constexpr static const uint32_t IJFO_GATE_NUM = 4;

template <typename T>
__aicore__ inline int64_t LstmMmSplitNDNDFP16<T>::Ceil(int64_t x, int64_t y) {
  if (y == 0) {
    return x;
  }
  return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength,
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
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC,
                                                GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                                GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                                                GM_ADDR outputO, GM_ADDR outputTanhC,
                                                const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace) {
  tiling = rnnTiling;
  inputMMTiling = tiling->inputMMParam;
  hiddenMMTiling = tiling->hiddenMMParam;
  InitBuffersV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC,
            wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, workspace);
  InitVars();
  InitQue();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitQue() {
  pipe.InitBuffer(qidCIn, 1, baseVector * sizeof(T));
  pipe.InitBuffer(qidVecIn, 1, baseVector * sizeof(float));
  pipe.InitBuffer(qidVecIn2, 1, baseVector * sizeof(float));
  pipe.InitBuffer(qidVecOut, 1, baseVector * sizeof(T));
  pipe.InitBuffer(calcBuf, 4 * baseVector * sizeof(float));
  // Init Local Tensors
  ubLocal1 = calcBuf.Get<float>(4 * baseVector);
  ubLocal2 = ubLocal1[baseVector];
  ubLocal3 = ubLocal2[baseVector];
  ubLocal4 = ubLocal3[baseVector];
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail,
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
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail,
                                                            int32_t kSize) {
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

  offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias,
                                                           GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi,
                                                           GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                                                           GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                                           GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                           GM_ADDR outputTanhC, GM_ADDR workspace) {
  CalcGMOffset(hiddenMMTiling, hiddenOffsets, hiddenTail, static_cast<int32_t>(tiling->hiddenSize));
  CalcGMOffset(inputMMTiling, inputOffsets, inputTail, static_cast<int32_t>(tiling->inputSize));
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
  outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                     tiling->timeStep * tiling->batch * LSTM_GATE_SIZE * tiling->hiddenSize);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                             GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                                                             GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                             GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH,
                                                             GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ,
                                                             GM_ADDR outputF, GM_ADDR outputO,
                                                             GM_ADDR outputTanhC, GM_ADDR workspace) {
  CalcGMOffset(hiddenMMTiling, hiddenOffsets, hiddenTail, static_cast<int32_t>(tiling->hiddenSize));
  CalcGMOffset(inputMMTiling, inputOffsets, inputTail, static_cast<int32_t>(tiling->inputSize));
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
  outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                     tiling->timeStep * tiling->batch * LSTM_GATE_SIZE * tiling->hiddenSize);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitVars() {
  int64_t ubSize = 21504; // 170/8 float nodes [4 float Tbuff + 2 float enque + 1 fp16 enque + 1 fp16 outque + 1 float sigmoid buff]
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
    vectorBaseN = Ceil(Ceil(tiling->hiddenSize, vectorSplitN), blockSize) * blockSize;
    vectorTailN = tiling->hiddenSize - vectorBaseN * (vectorSplitN - 1);
  }

  vectorBaseM = ((calcMaxSize / vectorBaseN) > vectorCoreM) ? vectorCoreM : (calcMaxSize / vectorBaseN);

  vectorBaseTailM = vectorCoreM % vectorBaseM;
  vectorTailTailM = vectorTailM % vectorBaseM;

  vectorSplitM = Ceil(vectorCoreM, vectorBaseM);
  vectorTailSplitM = Ceil(vectorTailM, vectorBaseM);

  baseVector = vectorBaseM * Ceil(vectorBaseN, blockSize) * blockSize;

  iOffset = 0;
  jOffset = tiling->gateOrder == 0 ? tiling->hiddenSize : 2 * tiling->hiddenSize;
  fOffset = tiling->gateOrder == 0 ? 2 * tiling->hiddenSize : tiling->hiddenSize;
  oOffset = 3 * tiling->hiddenSize;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessInputMM() {
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
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessHiddenMM(int64_t tIdx) {
  if (GetBlockIdx() < hiddenMMTiling.usedCoreNum) {
    if (tiling->direction == 1) {
      hiddenOffsets.COffset = oriHiddenOffsets.COffset + (tiling->timeStep - 1 - tIdx) * allCellSize;
    } else {
      hiddenOffsets.COffset = oriHiddenOffsets.COffset + tIdx * allCellSize;
    }
    hiddenMM.SetTensorA(inputGm.initHGm[hiddenOffsets.AOffset]);
    hiddenMM.IterateAll(outputGm.workspace[hiddenOffsets.COffset], true);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInHCSeq(LocalTensor<float>& dstUb, GlobalTensor<T>& mixGm,
                                                           int64_t offset) {
  LocalTensor<T> ubLocalIn = qidCIn.AllocTensor<T>();
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

  DataCopyPad(ubLocalIn, mixGm[offset], dataCopyParams, padParams);
  qidCIn.EnQue(ubLocalIn);
  ubLocalIn = qidCIn.DeQue<T>();
  Cast(dstUb, ubLocalIn, RoundMode::CAST_NONE, calcSizeAlign);
  qidCIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyOutput(GlobalTensor<T>& gm, LocalTensor<float>& ub, int64_t offset) {
  auto outLocal = qidVecOut.AllocTensor<T>();
  PipeBarrier<PIPE_V>();
  Cast(outLocal, ub, RoundMode::CAST_ROUND, calcSizeAlign);
  qidVecOut.EnQue(outLocal);

  outLocal = qidVecOut.DeQue<T>();
  
  DataCopyParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(T);
  dataCopyParams.srcStride = 0;
  dataCopyParams.dstStride = (tiling->hiddenSize - calcN) * sizeof(T);

  DataCopyPad(gm[offset], outLocal, dataCopyParams);

  qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalcVecScaler(int64_t tIdx, int64_t mIdx, int64_t nIdx,
                                                             int64_t& ijfoBaseOffset, int64_t& initcOffset,
                                                             int64_t& offset) {
  blockIdx = GetBlockIdx();

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
  calcSizeAlign = calcM * Ceil(calcN, blockSize) * blockSize;

  ijfoBaseOffset = blockIdx * vectorCoreM * tiling->hiddenSize * IJFO_GATE_NUM +
                   mIdx * vectorBaseM * tiling->hiddenSize * IJFO_GATE_NUM + nIdx * vectorBaseN;
  initcOffset = blockIdx * vectorCoreM * tiling->hiddenSize +
                mIdx * vectorBaseM * tiling->hiddenSize + nIdx * vectorBaseN;
  if (tiling->direction == 1) {
    offset = (tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize + initcOffset;
  } else {
    offset = tIdx * tiling->batch * tiling->hiddenSize + initcOffset;
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInFJ(LocalTensor<float>& dstUb, GlobalTensor<float>& mixGm,
                                                        int64_t GmOffset) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(float);
  dataCopyParams.srcStride = (IJFO_GATE_NUM * tiling->hiddenSize - calcN) * sizeof(float);
  dataCopyParams.dstStride = (Ceil(calcN, blockSize) * blockSize - calcN) / FLOAT_BLOCK_NUM;

  DataCopyPadExtParams<float> padParams{false, 0, (uint8_t)(Ceil(calcN, calBlockSize) * calBlockSize - calcN), 0};

  dstUb = qidVecIn.AllocTensor<float>();
  DataCopyPad(dstUb, mixGm[GmOffset], dataCopyParams, padParams);
  qidVecIn.EnQue(dstUb);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInC(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                       const int64_t initcOffset) {
  DataCopyExtParams dataCopyParams1;
  dataCopyParams1.blockCount = calcM;
  dataCopyParams1.blockLen = calcN * sizeof(T);
  dataCopyParams1.srcStride = (tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams1.dstStride = 0;

  DataCopyPadExtParams<T> padParams1{false, 0, (uint8_t)(Ceil(calcN, blockSize) * blockSize - calcN), 0};

  dstUb = qidCIn.AllocTensor<T>();
  DataCopyPad(dstUb, mixGm[initcOffset], dataCopyParams1, padParams1);
  qidCIn.EnQue(dstUb);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInIO(LocalTensor<float>& dstUb, GlobalTensor<float>& mixGm,
                                                        int64_t GmOffset) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = calcM;
  dataCopyParams.blockLen = calcN * sizeof(float);
  dataCopyParams.srcStride = (IJFO_GATE_NUM * tiling->hiddenSize - calcN) * sizeof(float);
  dataCopyParams.dstStride = (Ceil(calcN, blockSize) * blockSize - calcN) / FLOAT_BLOCK_NUM;

  DataCopyPadExtParams<float> padParams{false, 0, (uint8_t)(Ceil(calcN, calBlockSize) * calBlockSize - calcN), 0};

  dstUb = qidVecIn2.AllocTensor<float>();
  DataCopyPad(dstUb, mixGm[GmOffset], dataCopyParams, padParams);
  qidVecIn2.EnQue(dstUb);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::AddfSigmoid(LocalTensor<float>& fSigmoid,
                                                           LocalTensor<float>& ubLocalIn, int64_t offset) {
  ubLocalIn = qidVecIn.DeQue<float>();
  Adds(ubLocalIn, ubLocalIn, (float)tiling->forgetBias, calcSizeAlign);
  PipeBarrier<PIPE_V>();
  Sigmoid(fSigmoid, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
  
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outFGm, fSigmoid, offset);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitCMulfSigmoid(LocalTensor<float>& cTmp1, LocalTensor<T>& ubLocalIn,
                                                                LocalTensor<float>& fSigmoid) {
  ubLocalIn = qidCIn.DeQue<T>();
  Cast(ubLocal3, ubLocalIn, RoundMode::CAST_NONE, calcSizeAlign);
  qidCIn.FreeTensor(ubLocalIn);

  PipeBarrier<PIPE_V>();
  Mul(cTmp1, ubLocal3, fSigmoid, calcSizeAlign);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CaliSigmoid(LocalTensor<float>& iSigmoid, LocalTensor<float>& ubLocalIn,
                                                           int64_t offset) {
  ubLocalIn = qidVecIn2.DeQue<float>();
  PipeBarrier<PIPE_V>();
  Sigmoid(iSigmoid, ubLocalIn, calcSizeAlign);
  qidVecIn2.FreeTensor(ubLocalIn);

  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outIGm, iSigmoid, offset);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CaljTanh(LocalTensor<float>& jTanh, LocalTensor<float>& ubLocalIn,
                                                        int64_t offset) {
  ubLocalIn = qidVecIn.DeQue<float>();
  PipeBarrier<PIPE_V>();
  Tanh(jTanh, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);

  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outJGm, jTanh, offset);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CaloSigmoid(LocalTensor<float>& oSigmoid, LocalTensor<float>& ubLocalIn,
                                                           int64_t offset) {
  ubLocalIn = qidVecIn2.DeQue<float>();
  PipeBarrier<PIPE_V>();
  Sigmoid(oSigmoid, ubLocalIn, calcSizeAlign);
  qidVecIn2.FreeTensor(ubLocalIn);

  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outOGm, oSigmoid, offset);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyOutYH(LocalTensor<float>& updateH, int64_t offset,
                                                         int64_t initcOffset) {
  DataCopyExtParams dataCopyParams1;
  dataCopyParams1.blockCount = calcM;
  dataCopyParams1.blockLen = calcN * sizeof(T);
  dataCopyParams1.srcStride = (tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams1.dstStride = 0;

  DataCopyPadExtParams<T> padParams1{false, 0, (uint8_t)(Ceil(calcN, blockSize) * blockSize - calcN), 0};

  if (tiling->isSeqLength == 1) {
    PipeBarrier<PIPE_V>();
    auto updateY = ubLocal1;
    auto initH = ubLocal2;
    auto seqLength = ubLocal4;
    Mul(updateY, updateH, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    CopyInHCSeq(initH, inputGm.initHGm, initcOffset);
    PipeBarrier<PIPE_V>();
    Sub(updateH, updateH, initH, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(updateH, updateH, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Add(updateH, updateH, initH, calcSizeAlign);
    CopyOutput(outputGm.outYGm, updateY, offset);
    CopyOutput(outputGm.outHGm, updateH, offset);
  } else {
    LocalTensor<T> outLocal = qidVecOut.AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    Cast(outLocal, updateH, RoundMode::CAST_ROUND, calcSizeAlign);
    qidVecOut.EnQue(outLocal);
    outLocal = qidVecOut.DeQue<T>();

    DataCopyPad(outputGm.outYGm[offset], outLocal, dataCopyParams1);
    DataCopyPad(outputGm.outHGm[offset], outLocal, dataCopyParams1);

    qidVecOut.FreeTensor(outLocal);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyOutYHt0(LocalTensor<float>& updateH, int64_t offset) {
  DataCopyExtParams dataCopyParams1;
  dataCopyParams1.blockCount = calcM;
  dataCopyParams1.blockLen = calcN * sizeof(T);
  dataCopyParams1.srcStride = (tiling->hiddenSize - calcN) * sizeof(T);
  dataCopyParams1.dstStride = 0;

  DataCopyPadExtParams<T> padParams1{false, 0, (uint8_t)(Ceil(calcN, blockSize) * blockSize - calcN), 0};

  if (tiling->isSeqLength == 1) {
    auto seqLength = ubLocal3;
    PipeBarrier<PIPE_V>();
    Mul(updateH, updateH, seqLength, calcSizeAlign);
  }

  LocalTensor<T> outLocal = qidVecOut.AllocTensor<T>();
  PipeBarrier<PIPE_V>();
  Cast(outLocal, updateH, RoundMode::CAST_ROUND, calcSizeAlign);
  qidVecOut.EnQue(outLocal);
  outLocal = qidVecOut.DeQue<T>();

  DataCopyPad(outputGm.outYGm[offset], outLocal, dataCopyParams1);
  DataCopyPad(outputGm.outHGm[offset], outLocal, dataCopyParams1);

  qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalAddTanh(LocalTensor<float>& cTanh, LocalTensor<float>& ubLocal2,
                                                          LocalTensor<float>& cTmp2, int64_t offset,
                                                          int64_t initcOffset) {
  PipeBarrier<PIPE_V>();

  auto updateC = ubLocal1;
  Add(updateC, ubLocal2, cTmp2, calcSizeAlign);

  if (tiling->isSeqLength == 1) {
    PipeBarrier<PIPE_V>();
    auto initC = ubLocal2;
    auto seqLength = ubLocal4;
    CopyInHCSeq(initC, inputGm.initCGm, initcOffset);
    PipeBarrier<PIPE_V>();
    Sub(updateC, updateC, initC, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    CopyInHCSeq(seqLength, inputGm.seqLengthGm, offset);
    PipeBarrier<PIPE_V>();
    Mul(updateC, updateC, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Add(updateC, updateC, initC, calcSizeAlign);
    PipeBarrier<PIPE_V>();
  }

  if (tiling->cellClip > 0) {
    PipeBarrier<PIPE_V>();
    Mins(updateC, updateC, (float)tiling->cellClip, calcSizeAlign);
  }
  CopyOutput(outputGm.outCGm, updateC, offset);
  PipeBarrier<PIPE_V>();
  Tanh(cTanh, updateC, calcSizeAlign);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outTanhCGm, cTanh, offset);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalAddTanht0(LocalTensor<float>& cTanh, LocalTensor<float>& ubLocal2,
                                                            LocalTensor<float>& cTmp2, int64_t offset,
                                                            int64_t initcOffset) {
  auto updateC = ubLocal4;

  if (tiling->isSeqLength == 1) {
    auto seqLength = ubLocal3;
    CopyInHCSeq(seqLength, inputGm.seqLengthGm, offset);
    PipeBarrier<PIPE_V>();
    Mul(updateC, updateC, seqLength, calcSizeAlign);
    PipeBarrier<PIPE_V>();
  }

  if (tiling->cellClip > 0) {
    PipeBarrier<PIPE_V>();
    Mins(updateC, updateC, (float)tiling->cellClip, calcSizeAlign);
  }

  CopyOutput(outputGm.outCGm, updateC, offset);

  PipeBarrier<PIPE_V>();
  Tanh(cTanh, updateC, calcSizeAlign);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outTanhCGm, cTanh, offset);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx,
                                                                 GlobalTensor<float>& mixGm) {
  int64_t ijfoBaseOffset, initcOffset, offset = 0;
  CalcVecScaler(tIdx, mIdx, nIdx, ijfoBaseOffset, initcOffset, offset);

  LocalTensor<T> initCVec;
  LocalTensor<float> fInput, jInput, iInput, oInput;
  auto fSigmoid = ubLocal1;
  auto iSigmoid = ubLocal1;
  auto oSigmoid = ubLocal1;
  auto jTanh = ubLocal3;
  auto cTanh = ubLocal2;
  auto updateH = ubLocal3;
  /*
  initc          f           i           j            o
    |            |           |           |            |
    |         1.sigmoid   3.sigmoid   4.tanh      7.sigmoid
    |            |           |           |            |
     ----2.mul---             ---5.mul---             |
           |                         |                |
           ----------6.add------------                |
                        |                             |
                     7.tanh                           |
                        |                             |
                        -------------8.mul------------
  */
  CopyInFJ(fInput, mixGm, fOffset + ijfoBaseOffset);

  CopyInC(initCVec, inputGm.initCGm, initcOffset);

  CopyInIO(iInput, mixGm, iOffset + ijfoBaseOffset);

  AddfSigmoid(fSigmoid, fInput, offset);

  CopyInFJ(jInput, mixGm, jOffset + ijfoBaseOffset);

  InitCMulfSigmoid(ubLocal2, initCVec, fSigmoid);

  CaliSigmoid(iSigmoid, iInput, offset);

  CopyInIO(oInput, mixGm, oOffset + ijfoBaseOffset);

  CaljTanh(jTanh, jInput, offset);

  PipeBarrier<PIPE_V>();
  Mul(ubLocal4, jTanh, iSigmoid, calcSizeAlign);

  CalAddTanh(cTanh, ubLocal2, ubLocal4, offset, initcOffset);

  CaloSigmoid(oSigmoid, oInput, offset);

  PipeBarrier<PIPE_V>();
  Mul(updateH, oSigmoid, cTanh, calcSizeAlign);

  CopyOutYH(updateH, offset, initcOffset);
}


template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessVectorInitHC(int64_t mIdx, int64_t nIdx,
                                                                 GlobalTensor<float>& mixGm) {
  int64_t ijfoBaseOffset, initcOffset, offset = 0;
  CalcVecScaler(0, mIdx, nIdx, ijfoBaseOffset, initcOffset, offset);

  LocalTensor<float> fInput, jInput, iInput, oInput;
  auto fSigmoid = ubLocal1;
  auto iSigmoid = ubLocal1;
  auto jTanh = ubLocal3;
  auto cTanh = ubLocal2;
  auto oSigmoid = ubLocal1;
  auto updateH = ubLocal4;
  /*
   null          f           i           j            o
    |            |           |           |            |
    |         1.sigmoid   3.sigmoid   4.tanh      7.sigmoid
    |                        |           |            |
                              ---5.mul---             |
                                     |                |
                         -------------                |
                        |                             |
                     7.tanh                           |
                        |                             |
                        -------------8.mul------------
  */
  CopyInFJ(fInput, mixGm, fOffset + ijfoBaseOffset);

  CopyInIO(iInput, mixGm, iOffset + ijfoBaseOffset);

  AddfSigmoid(fSigmoid, fInput, offset);

  CopyInFJ(jInput, mixGm, jOffset + ijfoBaseOffset);

  CaliSigmoid(iSigmoid, iInput, offset);

  CopyInIO(oInput, mixGm, oOffset + ijfoBaseOffset);

  CaljTanh(jTanh, jInput, offset);

  PipeBarrier<PIPE_V>();
  Mul(ubLocal4, jTanh, iSigmoid, calcSizeAlign);

  CalAddTanht0(cTanh, ubLocal2, ubLocal4, offset, initcOffset);

  CaloSigmoid(oSigmoid, oInput, offset);

  PipeBarrier<PIPE_V>();
  Mul(updateH, oSigmoid, cTanh, calcSizeAlign);

  CopyOutYHt0(updateH, offset);
}


template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessVector(int64_t tIdx) {
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
    auto mixGm = outputGm.workspace[offset];
    for (int64_t j = 0; j < coreLoopM; ++j) {
      for (int64_t k = 0; k < vectorSplitN; ++k) {
        ProcessVectorOnce(tIdx, j, k, mixGm);
      }
    }
  }
}


template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessInitalT() {
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
    auto mixGm = outputGm.workspace[offset];
    for (int64_t j = 0; j < coreLoopM; ++j) {
      for (int64_t k = 0; k < vectorSplitN; ++k) {
        ProcessVectorInitHC(j, k, mixGm);
      }
    }
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::Process() {
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
  
  hiddenMM.SetTensorB(inputGm.weightHiddenGm[hiddenOffsets.BOffset]);
  if (hiddenTail.nCoreIndx == hiddenTail.notTailNCoreCount && hiddenTail.mCoreIndx == hiddenTail.notTailMCoreCount) {
    hiddenMM.SetTail(hiddenTail.tailSingleCoreM, hiddenTail.tailSingleCoreN);
  } else if (hiddenTail.nCoreIndx == hiddenTail.notTailNCoreCount) {
    hiddenMM.SetTail(hiddenMMTiling.singleCoreM, hiddenTail.tailSingleCoreN);
  } else if (hiddenTail.mCoreIndx == hiddenTail.notTailMCoreCount) {
    hiddenMM.SetTail(hiddenTail.tailSingleCoreM, hiddenMMTiling.singleCoreN);
  }
  SyncAll();
  for (int64_t tIdx = tiling->isInithc == 0 ? 1 : 0; tIdx < tiling->timeStep; tIdx++) {
    ProcessHiddenMM(tIdx);

    SyncAll();

    ProcessVector(tIdx);

    if (tiling->direction == 1) {
      inputGm.initCGm = outputGm.outCGm[(tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[(tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize];
    } else {
      inputGm.initCGm = outputGm.outCGm[tIdx * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[tIdx * tiling->batch * tiling->hiddenSize];
    }
    SyncAll();
  }
}

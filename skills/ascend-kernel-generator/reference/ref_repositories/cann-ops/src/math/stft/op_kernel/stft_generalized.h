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
 * \file stft_generalized.h
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace STFTND {
using namespace AscendC;
using namespace matmul;

template<typename T, int32_t bufferNum, const MatmulConfig& MM_CFG>
class STFTGeneralized {
 public:
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> aType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true> bType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> cType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> biasType;

  Matmul<aType, bType, cType, biasType, MM_CFG> mm0;
  Matmul<aType, bType, cType, biasType, MM_CFG> mm1;
  Matmul<aType, bType, cType, biasType, MM_CFG> mm2;
  Matmul<aType, bType, cType, biasType, MM_CFG> mm3;
  Matmul<aType, bType, cType, biasType, MM_CFG> *mm;

  __aicore__ inline STFTGeneralized() {};

  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR plan,
                              GM_ADDR window,
                              GM_ADDR y,
                              GM_ADDR workspace,
                              STFTGeneralizedTilingData* tilingData,
                              TPipe *pipeIn) {
    pipe = pipeIn;
    tiling = tilingData;
    if (tiling->usedCoreNum <= GetBlockIdx()) {
      return;
    }

    uint64_t inputGmSize = tiling->batch * ((tiling->inputSize + tiling->nfft) * sizeof(T) + BLOCK_SIZE - 1)
                           / BLOCK_SIZE * BLOCK_SIZE / sizeof(T);
    inputGm.SetGlobalBuffer((__gm__ T*)x, inputGmSize);

    uint64_t splitWindowGmSize = (uint64_t)tiling->batch * tiling->matmulN * tiling->nfftAlign;
    splitWindowGm.SetGlobalBuffer((__gm__ T*)workspace, splitWindowGmSize);

    uint64_t splitWindowWorkspaceSize = (splitWindowGmSize * sizeof(T) + WORKSPACE_ALIGN_SIZE - 1)
                                        / WORKSPACE_ALIGN_SIZE * WORKSPACE_ALIGN_SIZE / sizeof(T);
    uint64_t realGmSize = (uint64_t)tiling->batch * tiling->matmulM * tiling->matmulN;
    uint64_t imagGmSize = realGmSize;
    uint64_t outputGmSize = IMAG_AND_REAL * realGmSize;
    matmulOutputGm.SetGlobalBuffer((__gm__ T*)workspace + splitWindowWorkspaceSize, outputGmSize);
    outputGm.SetGlobalBuffer((__gm__ T*)y, outputGmSize);

    uint64_t realWindowSize = (uint64_t)tiling->matmulM * tiling->nfftAlign;
    uint64_t imagWindowSize = realWindowSize;
    if (window == nullptr) {
      planGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(plan), realWindowSize * IMAG_AND_REAL);
    } else {
      uint64_t matmulWorkspaceSize =
          (((uint64_t)tiling->batch * tiling->matmulN * tiling->matmulM * sizeof(T) * IMAG_AND_REAL +
            WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) * WORKSPACE_ALIGN_SIZE / sizeof(T);
      uint64_t planOffset = splitWindowWorkspaceSize + matmulWorkspaceSize;
      planGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + planOffset, realWindowSize * IMAG_AND_REAL);
    }

    bufferSize = tiling->copyUBSize / IMAG_AND_REAL / bufferNum / BLOCK_SIZE * BLOCK_SIZE;
    if (tiling->nfft == tiling->nfftAlign && (tiling->hopLength * sizeof(T)) % BLOCK_SIZE == 0 &&
        tiling->nfftAlign * sizeof(T) <= bufferSize && tiling->hopLength * sizeof(T) <= bufferSize) {
      pipe->InitBuffer(outCopy, bufferNum, bufferSize);
      pipe->InitBuffer(inCopy, bufferNum, bufferSize);
      isAligned = true;
    } else {
      if (tiling->splitWindowSkipNum > 0) {
        if (tiling->splitWindowSkipNum * tiling->hopLength >= tiling->nfftAlign) {
          uint32_t oneBufferSize = (tiling->copyUBSize / 2) / BLOCK_SIZE * BLOCK_SIZE;
          pipe->InitBuffer(inCopy, bufferNum, oneBufferSize);
          pipe->InitBuffer(outCopy, bufferNum, oneBufferSize);
        } else {
          uint32_t skipHopLength = tiling->splitWindowSkipNum * tiling->hopLength;
          uint32_t onceNInUB = (tiling->copyUBSize / sizeof(T) - tiling->nfftAlign) /
                               (skipHopLength + tiling->nfftAlign);
          uint32_t inputSize = (onceNInUB * skipHopLength + tiling->nfftAlign) * sizeof(T);
          uint32_t outputSize = onceNInUB * tiling->nfftAlign * sizeof(T);
          pipe->InitBuffer(inCopy, bufferNum, inputSize);
          pipe->InitBuffer(outCopy, bufferNum, outputSize);
        }
      } else {
        bufferSize *= 2;
        pipe->InitBuffer(inCopy, bufferNum, bufferSize);
        isAligned = false;
      }
    }

    maskCount = tiling->maskUBSize / BLOCK_SIZE * BLOCK_SIZE / sizeof(int32_t);
    pipe->InitBuffer(gatherIn, GATHER_BUFFER_NUM, maskCount * sizeof(T));
    pipe->InitBuffer(gatherOut, GATHER_BUFFER_NUM, maskCount * sizeof(T));
    pipe->InitBuffer(maskUB, tiling->maskUBSize);
    pipe->InitBuffer(tempUB, tiling->maskUBSize);

    if (tiling->normalized) {
      SetRootNfft();
    }
  }

  __aicore__ inline void Process() {
    SyncAll();
    auto blockIdx = GetBlockIdx();
    if (tiling->usedCoreNum <= blockIdx) {
      return;
    }
    uint32_t nIdx = blockIdx % tiling->matmulNCoreNum;
    uint32_t nFactor = tiling->matmulNCoreFactor;
    uint32_t nOffset = nIdx * nFactor;
    uint32_t mIdx = (blockIdx / tiling->matmulNCoreNum) % tiling->matmulMCoreNum;
    uint32_t mFactor = tiling->matmulMCoreFactor;
    uint32_t mOffset = mIdx * mFactor;
    uint32_t bIdx = (blockIdx / tiling->matmulNCoreNum / tiling->matmulMCoreNum) % tiling->batchCoreNum;
    uint32_t bFactor = tiling->batchCoreFactor;
    uint32_t bOffset = bIdx * bFactor;
    bool isTailM = false;
    bool isTailN = false;

    if (nIdx >= tiling->matmulNCoreNum - tiling->matmulNTailCoreNum) {
      nOffset = (tiling->matmulNCoreNum - tiling->matmulNTailCoreNum) * nFactor;
      nFactor = tiling->matmulN % nFactor;
      isTailN = true;
    }
    if (mIdx >= tiling->matmulMCoreNum - tiling->matmulMTailCoreNum) {
      mOffset = (tiling->matmulMCoreNum - tiling->matmulMTailCoreNum) * mFactor +
                (mIdx + tiling->matmulMTailCoreNum - tiling->matmulMCoreNum) * (mFactor - 1);
      mFactor = mFactor - 1;
      isTailM = true;
    }
    if (bIdx >= tiling->batchCoreNum - tiling->batchTailCoreNum) {
      bOffset = (tiling->batchCoreNum - tiling->batchTailCoreNum) * bFactor +
                (bIdx + tiling->batchTailCoreNum - tiling->batchCoreNum) * (bFactor - 1);
      bFactor = bFactor - 1;
    }

    if (!isTailM) {
      mm = !isTailN ? &mm0 : &mm1;
    } else {
      mm = !isTailN ? &mm2 : &mm3;
    }

    for (int i = 0; i < bFactor; i++) {
      int64_t inputOffset = (bOffset + i) * (tiling->inputSize + tiling->nfft) + nOffset * tiling->hopLength;
      int64_t splitWindowOffset = ((bOffset + i) * tiling->matmulN + nOffset) * tiling->nfftAlign;
      int64_t outputOffset = (((bOffset + i) * tiling->matmulM + mOffset) * tiling->matmulN + nOffset) * IMAG_AND_REAL;
      int64_t realOffset = (bOffset + i) * tiling->matmulM * tiling->matmulN + mOffset * tiling->matmulN +
                           nIdx * mFactor * tiling->matmulNCoreFactor;
      int64_t imagOffset = realOffset;
      int64_t a1Offset = mOffset * tiling->nfftAlign;
      int64_t a2Offset = a1Offset;

      int64_t planOffset = IMAG_AND_REAL * mOffset * tiling->nfftAlign;
      int64_t matmulOutputOffset = IMAG_AND_REAL * realOffset;

      SplitWindows(inputOffset, splitWindowOffset, nFactor);
      if (i == 0) {
        GenerateGatherMask(nFactor);
      }

      StftMatmul(splitWindowOffset, planOffset, matmulOutputOffset);

      GatherRealAndImag(matmulOutputOffset, outputOffset, mFactor, nFactor);
    }
  }

 private:
  __aicore__ inline void GenerateGatherMask(uint32_t nFactor) {
    uint32_t alignNum = BLOCK_SIZE / sizeof(T);
    uint32_t nAlign = (nFactor + alignNum - 1) / alignNum * alignNum;
    if (nAlign * IMAG_AND_REAL < maskCount) {
      GenerateGatherMaskSmallN(nFactor);
    } else {
      GenerateGatherMaskLargeN();
    }
  }

  __aicore__ inline void GenerateGatherMaskLargeN() {
    LocalTensor<int32_t> maskTemp = maskUB.template Get<int32_t>(maskCount);
    // real or img must align to 64
    uint32_t alignNum = REPEAT_SIZE / sizeof(T);
    uint32_t nAlignInUB = (maskCount / IMAG_AND_REAL) / alignNum * alignNum;
    uint32_t maskTotalNum = nAlignInUB * IMAG_AND_REAL;

    // 生成等差数列 0, 2, 4, 6, 8, 10, 12, 14,...
    ArithProgression<int32_t>(maskTemp, (int32_t)0, (int32_t)2, maskTotalNum);
    PipeBarrier<PIPE_V>();

    // 奇数index清零：0, 0, 4, 0, 8, 0, 12, 0,...
    uint64_t maskBit1[2] = {0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA};
    // float32, mask_h should be zero
    if (sizeof(T) == DWORD_SIZE) {
      maskBit1[1] = 0;
    }

    UnaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.srcBlkStride = 1;
    repeatParams.dstRepStride = BLOCK_NUM;
    repeatParams.srcRepStride = BLOCK_NUM;
    Muls(maskTemp, maskTemp, 0, maskBit1, maskTotalNum * sizeof(int32_t) / REPEAT_SIZE, repeatParams);

    //生成等差数列：-2， 0， 2， 4， 6， 8， 10，12，...
    LocalTensor<int32_t> temp = tempUB.template Get<int32_t>(maskCount);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int32_t START = -2;
    int32_t STEP = 2;
    ArithProgression<int32_t>(temp, START, STEP, maskTotalNum);
    PipeBarrier<PIPE_V>();

    // 加上地址偏移offset = imagbase-realbase：offset-2, offset, offset+2, offset+4, ..., offset+12, ...
    int32_t offset = nAlignInUB * sizeof(T);

    Adds(temp, temp, offset, maskTotalNum);
    PipeBarrier<PIPE_V>();

    // 偶数index清零： 0， offset, 0, offset+4, 0, offset+8, 0, offset+12,...
    uint64_t maskBit2[2] = {0x5555555555555555, 0x5555555555555555};
    // float32, mask_h should be zero
    if (sizeof(T) == DWORD_SIZE) {
      maskBit2[1] = 0;
    }
    Muls(temp, temp, 0, maskBit2, maskTotalNum * sizeof(int32_t) / REPEAT_SIZE, repeatParams);
    PipeBarrier<PIPE_V>();

    // 相加: 0, offset, 4, offset+4, 8, offset+8, 12, offset+12,...
    Add(maskTemp, maskTemp, temp, maskTotalNum);
    PipeBarrier<PIPE_V>();
    mask = maskTemp.ReinterpretCast<uint32_t>();
  }

  __aicore__ inline void GenerateGatherMaskSmallN(uint32_t nFactor) {
    LocalTensor<int32_t> maskTemp = maskUB.template Get<int32_t>(maskCount);
    uint32_t alignNum = BLOCK_SIZE / sizeof(T);
    uint32_t nAlign = (nFactor + alignNum - 1) / alignNum * alignNum;
    uint32_t onceMInUB = maskCount / nAlign / IMAG_AND_REAL;
    int32_t offset = nAlign * sizeof(T);
    uint32_t blocksForRealAndImg = nAlign * IMAG_AND_REAL / alignNum;

    // for one row mask
    for (int32_t i = 0; i < alignNum / IMAG_AND_REAL; i++) {
      maskTemp.SetValue(i * IMAG_AND_REAL, i * sizeof(T));
      maskTemp.SetValue(i * IMAG_AND_REAL + 1, i * sizeof(T) + offset);
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    uint32_t i = 1;
    for (; i <= (blocksForRealAndImg / BINARY_ADD_COEF); i *= IMAG_AND_REAL) {
      uint32_t maskAddrOffset = i * alignNum;
      int32_t maskValOffset = i * (alignNum / IMAG_AND_REAL) * sizeof(T);
      Adds(maskTemp[maskAddrOffset], maskTemp, maskValOffset, maskAddrOffset);
      PipeBarrier<PIPE_V>();
    }
    if (i > (blocksForRealAndImg / BINARY_ADD_COEF) && i < blocksForRealAndImg) {
      int32_t lastMaskAddrOffset = i * alignNum;
      uint32_t lastMaskCalcNum = (nAlign * IMAG_AND_REAL) - lastMaskAddrOffset;
      int32_t lastMaskValOffset = i * (alignNum / IMAG_AND_REAL) * sizeof(T);
      Adds(maskTemp[lastMaskAddrOffset], maskTemp, lastMaskValOffset, lastMaskCalcNum);
      PipeBarrier<PIPE_V>();
    }
    // for multi row mask
    i = 1;
    for (; i <= (onceMInUB / BINARY_ADD_COEF); i *= IMAG_AND_REAL) {
      uint32_t maskAddrOffset = i * nAlign * IMAG_AND_REAL;
      int32_t maskValOffset = maskAddrOffset * sizeof(T);
      Adds(maskTemp[maskAddrOffset], maskTemp, maskValOffset, maskAddrOffset);
      PipeBarrier<PIPE_V>();
    }
    if (i > (onceMInUB / BINARY_ADD_COEF) && i < onceMInUB) {
      uint32_t lastMaskAddrOffset = i * nAlign * IMAG_AND_REAL;
      int32_t lastMaskValOffset = lastMaskAddrOffset * sizeof(T);
      uint32_t lastMaskCalcNum = (onceMInUB - i) * nAlign * IMAG_AND_REAL;
      Adds(maskTemp[lastMaskAddrOffset], maskTemp, lastMaskValOffset, lastMaskCalcNum);
      PipeBarrier<PIPE_V>();
    }
    mask = maskTemp.ReinterpretCast<uint32_t>();
  }

  __aicore__ inline void GatherRealAndImag(int64_t matmulOffset, int64_t outputOffset,
                                           uint32_t mFactor, uint32_t nFactor) {
    uint32_t alignNum = BLOCK_SIZE / sizeof(T);
    uint32_t nAlign = (nFactor + alignNum - 1) / alignNum * alignNum;
    if (nAlign * IMAG_AND_REAL < maskCount) {
      GatherRealAndImagSmallN(matmulOffset, outputOffset, mFactor, nFactor);
    } else {
      GatherRealAndImagLargeN(matmulOffset, outputOffset, mFactor, nFactor);
    }
  }

  __aicore__ inline void SetRootNfft() {
    if (sizeof(T) == DWORD_SIZE) {
      // float32
      rootNfft = tiling->rootNfft;
    } else {
      // float16
      LocalTensor<float> src = tempUB.template Get<float>(BLOCK_SIZE / sizeof(T));
      LocalTensor<T> dst = tempUB.template Get<T>(BLOCK_SIZE / sizeof(T));
      src.SetValue(0, tiling->rootNfft);
      event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
      SetFlag<HardEvent::S_V>(eventIdSToV);
      WaitFlag<HardEvent::S_V>(eventIdSToV);
      Cast(dst, src, RoundMode::CAST_ROUND, 1);
      event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      SetFlag<HardEvent::V_S>(eventIdVToS);
      WaitFlag<HardEvent::V_S>(eventIdVToS);
      rootNfft = dst.GetValue(0);
    }
  }

  __aicore__ inline void GatherRealAndImagSmallN(int64_t matmulOffset, int64_t outputOffset,
                                                 uint32_t mFactor, uint32_t nFactor) {
    uint32_t alignNum = BLOCK_SIZE / sizeof(T);
    uint32_t nAlign = (nFactor + alignNum - 1) / alignNum * alignNum;
    uint32_t onceM = maskCount / nAlign / IMAG_AND_REAL;
    uint32_t mQuotient = mFactor / onceM;
    uint32_t mRemainder = mFactor % onceM;

    int64_t copyInGMOffsetCoef = onceM * nFactor * IMAG_AND_REAL;
    int64_t copyOutGMOffsetCoef = onceM * tiling->matmulN * IMAG_AND_REAL;

    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = onceM * IMAG_AND_REAL;
    copyInParams.blockLen = nFactor * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = onceM;
    copyOutParams.blockLen = nFactor * IMAG_AND_REAL * sizeof(T);
    copyOutParams.srcStride = (IMAG_AND_REAL * (nAlign - nFactor)) / alignNum;
    copyOutParams.dstStride = IMAG_AND_REAL * (tiling->matmulN - nFactor) * sizeof(T);
    for (uint32_t i = 0; i < mQuotient; i++) {
      LocalTensor<T> inputLocal = gatherIn.template AllocTensor<T>();
      DataCopyPad(inputLocal, matmulOutputGm[matmulOffset + i * copyInGMOffsetCoef], copyInParams, dataCopyPadExtParams);
      gatherIn.EnQue(inputLocal);
      inputLocal = gatherIn.template DeQue<T>();
      LocalTensor<T> outputLocal = gatherOut.template AllocTensor<T>();
      Gather(outputLocal, inputLocal, mask, 0, onceM * nAlign * IMAG_AND_REAL);
      if (tiling->normalized) {
        PipeBarrier<PIPE_V>();
        Muls(outputLocal, outputLocal, rootNfft, onceM * nAlign * IMAG_AND_REAL);
      }
      gatherOut.EnQue(outputLocal);
      gatherIn.FreeTensor(inputLocal);
      outputLocal = gatherOut.template DeQue<T>();
      DataCopyPad(outputGm[outputOffset + i * copyOutGMOffsetCoef], outputLocal, copyOutParams);
      gatherOut.FreeTensor(outputLocal);
    }
    if (mRemainder > 0) {
      DataCopyExtParams copyInParamsRemainder;
      copyInParamsRemainder.blockCount = mRemainder * IMAG_AND_REAL;
      copyInParamsRemainder.blockLen = nFactor * sizeof(T);
      copyInParamsRemainder.srcStride = 0;
      copyInParamsRemainder.dstStride = 0;
      DataCopyExtParams copyOutParamsRemainder;
      copyOutParamsRemainder.blockCount = mRemainder;
      copyOutParamsRemainder.blockLen = nFactor * IMAG_AND_REAL * sizeof(T);
      copyOutParamsRemainder.srcStride = (IMAG_AND_REAL * (nAlign - nFactor)) / alignNum;
      copyOutParamsRemainder.dstStride = IMAG_AND_REAL * (tiling->matmulN - nFactor) * sizeof(T);
      LocalTensor<T> inputLocal = gatherIn.template AllocTensor<T>();
      DataCopyPad(inputLocal, matmulOutputGm[matmulOffset + mQuotient * copyInGMOffsetCoef],
                  copyInParamsRemainder, dataCopyPadExtParams);
      gatherIn.EnQue(inputLocal);
      inputLocal = gatherIn.template DeQue<T>();
      LocalTensor<T> outputLocal = gatherOut.template AllocTensor<T>();
      Gather(outputLocal, inputLocal, mask, 0, mRemainder * nAlign * IMAG_AND_REAL);
      if (tiling->normalized) {
        PipeBarrier<PIPE_V>();
        Muls(outputLocal, outputLocal, rootNfft, mRemainder * nAlign * IMAG_AND_REAL);
      }
      gatherOut.EnQue(outputLocal);
      gatherIn.FreeTensor(inputLocal);
      outputLocal = gatherOut.template DeQue<T>();
      DataCopyPad(outputGm[outputOffset + mQuotient * copyOutGMOffsetCoef], outputLocal, copyOutParamsRemainder);
      gatherOut.FreeTensor(outputLocal);
    }
  }

  __aicore__ inline void GatherRealAndImagLargeN(int64_t matmulOffset, int64_t outputOffset,
                                                 uint32_t mFactor, uint32_t nFactor) {
    uint32_t alignNum = REPEAT_SIZE / sizeof(T);
    uint32_t nAlignInUB = (maskCount / IMAG_AND_REAL) / alignNum * alignNum;
    uint32_t nQuotient = nFactor / nAlignInUB;
    uint32_t nRemainder = nFactor % nAlignInUB;

    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = IMAG_AND_REAL;
    copyInParams.blockLen = nAlignInUB * sizeof(T);
    copyInParams.srcStride = (nFactor - nAlignInUB) * sizeof(T);
    copyInParams.dstStride = 0;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = nAlignInUB * IMAG_AND_REAL * sizeof(T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyExtParams copyInParamsRemainder;
    copyInParamsRemainder.blockCount = IMAG_AND_REAL;
    copyInParamsRemainder.blockLen = nRemainder * sizeof(T);
    copyInParamsRemainder.srcStride = (nFactor - nRemainder) * sizeof(T);
    copyInParamsRemainder.dstStride = ((nAlignInUB - nRemainder) * sizeof(T)) / BLOCK_SIZE;
    DataCopyExtParams copyOutParamsRemainder;
    copyOutParamsRemainder.blockCount = 1;
    copyOutParamsRemainder.blockLen = nRemainder * IMAG_AND_REAL * sizeof(T);
    copyOutParamsRemainder.srcStride = 0;
    copyOutParamsRemainder.dstStride = 0;

    for (int64_t i = 0; i < mFactor; i++) {
      int64_t curMatmulOffset = matmulOffset + i * nFactor * IMAG_AND_REAL;
      int64_t curOutputOffset = outputOffset + i * tiling->matmulN * IMAG_AND_REAL;
      for (int64_t j = 0; j < nQuotient; j++) {
        LocalTensor<T> inputLocal = gatherIn.template AllocTensor<T>();
        DataCopyPad(inputLocal, matmulOutputGm[curMatmulOffset + j * nAlignInUB], copyInParams, dataCopyPadExtParams);
        gatherIn.EnQue(inputLocal);
        inputLocal = gatherIn.template DeQue<T>();
        LocalTensor<T> outputLocal = gatherOut.template AllocTensor<T>();
        Gather(outputLocal, inputLocal, mask, 0, nAlignInUB * IMAG_AND_REAL);
        if (tiling->normalized) {
          PipeBarrier<PIPE_V>();
          Muls(outputLocal, outputLocal, rootNfft, nAlignInUB * IMAG_AND_REAL);
        }
        gatherOut.EnQue(outputLocal);
        gatherIn.FreeTensor(inputLocal);
        outputLocal = gatherOut.template DeQue<T>();
        DataCopyPad(outputGm[curOutputOffset + j * nAlignInUB * IMAG_AND_REAL], outputLocal, copyOutParams);
        gatherOut.FreeTensor(outputLocal);
      }
      if (nRemainder > 0) {
        LocalTensor<T> inputLocal = gatherIn.template AllocTensor<T>();
        DataCopyPad(inputLocal, matmulOutputGm[curMatmulOffset + nQuotient * nAlignInUB],
                    copyInParamsRemainder, dataCopyPadExtParams);
        gatherIn.EnQue(inputLocal);
        inputLocal = gatherIn.template DeQue<T>();
        LocalTensor<T> outputLocal = gatherOut.template AllocTensor<T>();
        Gather(outputLocal, inputLocal, mask, 0, nAlignInUB * IMAG_AND_REAL);
        if (tiling->normalized) {
          PipeBarrier<PIPE_V>();
          Muls(outputLocal, outputLocal, rootNfft, nRemainder * IMAG_AND_REAL);
        }
        gatherOut.EnQue(outputLocal);
        gatherIn.FreeTensor(inputLocal);
        outputLocal = gatherOut.template DeQue<T>();
        DataCopyPad(outputGm[curOutputOffset + nQuotient * nAlignInUB * IMAG_AND_REAL],
                    outputLocal, copyOutParamsRemainder);
        gatherOut.FreeTensor(outputLocal);
      }
    }
  }

  __aicore__ inline void StftMatmul(int64_t splitWindowOffset, int64_t planOffset, int64_t matmulOutputOffset) {
    mm->SetTensorA(planGm[planOffset]);
    mm->SetTensorB(splitWindowGm[splitWindowOffset], true);
    mm->IterateAll(matmulOutputGm[matmulOutputOffset]);
    mm->End();
  }

  __aicore__ inline void SplitWindowsUseVecSkip(int64_t inputOffset, int64_t outputOffset, int32_t nFactor) {
    uint32_t splitWindowSkipNum = tiling->splitWindowSkipNum;
    uint32_t windowsForOnceSkip = (nFactor + splitWindowSkipNum - 1) / splitWindowSkipNum;
    uint32_t tailSkipIdx = splitWindowSkipNum - ((windowsForOnceSkip * splitWindowSkipNum) - nFactor);
    uint32_t skipHopLength = splitWindowSkipNum * tiling->hopLength;
    uint32_t onceNInUB = (tiling->copyUBSize / sizeof(T) - tiling->nfftAlign) / (skipHopLength + tiling->nfftAlign);
    uint32_t burstLen = tiling->nfftAlign * sizeof(T);
    uint32_t copyOutDstGap = (splitWindowSkipNum - 1) * tiling->nfftAlign * sizeof(T);
    uint32_t onceSkipInInputGM = onceNInUB * skipHopLength;
    uint32_t copyLength = (onceNInUB - 1) * skipHopLength + tiling->nfft;
    int64_t onceSkipInOutputGm = onceNInUB * splitWindowSkipNum * tiling->nfftAlign;
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    DataCopyExtParams copyOutParam;
    copyOutParam.blockCount = onceNInUB;
    copyOutParam.blockLen = burstLen;
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = copyOutDstGap;
    for (int64_t i = 0; i < splitWindowSkipNum; i++) {
      uint32_t curSplitN = i >= tailSkipIdx ? windowsForOnceSkip - 1 : windowsForOnceSkip;
      uint32_t splitQuotient = curSplitN / onceNInUB;
      uint32_t splitRemainder = curSplitN % onceNInUB;
      for (int64_t j = 0; j < splitQuotient; j++) {
        LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
        DataCopyExtParams copyInParam;
        copyInParam.blockLen = copyLength * sizeof(T);
        copyInParam.blockCount = 1;
        copyInParam.srcStride = 0;
        copyInParam.dstStride = 0;
        DataCopyPad(inputLocal, inputGm[inputOffset + i * tiling->hopLength + j * onceSkipInInputGM],
                    copyInParam, dataCopyPadExtParams);
        inCopy.EnQue(inputLocal);
        inputLocal = inCopy.template DeQue<T>();
        LocalTensor<T> outputLocal = outCopy.template AllocTensor<T>();
        for (uint32_t k = 0; k < onceNInUB; k++) {
          DataCopy(outputLocal[k * tiling->nfftAlign], inputLocal[k * skipHopLength], tiling->nfftAlign);
        }
        outCopy.EnQue(outputLocal);
        inCopy.FreeTensor(inputLocal);
        outputLocal = outCopy.template DeQue<T>();
        DataCopyPad(splitWindowGm[outputOffset + i * tiling->nfftAlign + j * onceSkipInOutputGm],
                    outputLocal, copyOutParam);
        outCopy.FreeTensor(outputLocal);
      }
      if (splitRemainder > 0) {
        DataCopyExtParams copyOutParamRemainder;
        copyOutParamRemainder.blockCount = splitRemainder;
        copyOutParamRemainder.blockLen = burstLen;
        copyOutParamRemainder.srcStride = 0;
        copyOutParamRemainder.dstStride = copyOutDstGap;
        uint32_t copyLength = (splitRemainder - 1) * skipHopLength + tiling->nfft;
        LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
        DataCopyExtParams copyInParam;
        copyInParam.blockLen = copyLength * sizeof(T);
        copyInParam.blockCount = 1;
        copyInParam.srcStride = 0;
        copyInParam.dstStride = 0;
        DataCopyPad(inputLocal, inputGm[inputOffset + i * tiling->hopLength + splitQuotient * onceSkipInInputGM],
                    copyInParam, dataCopyPadExtParams);
        inCopy.EnQue(inputLocal);
        inputLocal = inCopy.template DeQue<T>();
        LocalTensor<T> outputLocal = outCopy.template AllocTensor<T>();
        for (uint32_t k = 0; k < splitRemainder; k++) {
          DataCopy(outputLocal[k * tiling->nfftAlign], inputLocal[k * skipHopLength], tiling->nfftAlign);
        }
        outCopy.EnQue(outputLocal);
        inCopy.FreeTensor(inputLocal);
        outputLocal = outCopy.template DeQue<T>();
        DataCopyPad(splitWindowGm[outputOffset + i * tiling->nfftAlign + splitQuotient * onceSkipInOutputGm],
                    outputLocal, copyOutParamRemainder);
        outCopy.FreeTensor(outputLocal);
      }
    }
  }

  __aicore__ inline void SplitWindowsUseMTESkip(int64_t inputOffset, int64_t outputOffset, int32_t nFactor) {
    uint32_t splitWindowSkipNum = tiling->splitWindowSkipNum;
    uint32_t windowsForOnceSkip = (nFactor + splitWindowSkipNum - 1) / splitWindowSkipNum;
    uint32_t tailSkipIdx = splitWindowSkipNum - ((windowsForOnceSkip * splitWindowSkipNum) - nFactor);
    uint32_t skipHopLength = splitWindowSkipNum * tiling->hopLength;
    uint32_t oneBufferSize = (tiling->copyUBSize / 2) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t onceNInUB = oneBufferSize / (tiling->nfftAlign * sizeof(T));
    uint32_t burstLenIn = tiling->nfft * sizeof(T);
    uint32_t burstLenOut = tiling->nfftAlign * sizeof(T);
    uint32_t copyInSrcGap = (skipHopLength - tiling->nfft) * sizeof(T);
    // nfftAlign - nfft: less than one block part, use pad
    uint32_t padLength = ((tiling->nfftAlign - tiling->nfft) * sizeof(T) % BLOCK_SIZE) / sizeof(T);
    // nfftAlign - nfft: greater than one block part, use stride
    uint32_t copyInDstGap = (tiling->nfftAlign - tiling->nfft) * sizeof(T) / BLOCK_SIZE;
    uint32_t copyOutDstGap = (splitWindowSkipNum - 1) * tiling->nfftAlign * sizeof(T);
    uint32_t onceSkipInInputGM = onceNInUB * skipHopLength;
    int64_t onceSkipInOutputGm = onceNInUB * splitWindowSkipNum * tiling->nfftAlign;

    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = padLength > 0 ? true : false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = padLength;
    dataCopyPadExtParams.paddingValue = 0;
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = onceNInUB;
    copyInParams.blockLen = burstLenIn;
    copyInParams.srcStride = copyInSrcGap;
    copyInParams.dstStride = copyInDstGap;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = onceNInUB;
    copyOutParams.blockLen = burstLenOut;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = copyOutDstGap;
    for (int64_t i = 0; i < splitWindowSkipNum; i++) {
      uint32_t curSplitN = i >= tailSkipIdx ? windowsForOnceSkip - 1 : windowsForOnceSkip;
      uint32_t splitQuotient = curSplitN / onceNInUB;
      uint32_t splitRemainder = curSplitN % onceNInUB;
      for (int64_t j = 0; j < splitQuotient; j++) {
        LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
        DataCopyPad(inputLocal, inputGm[inputOffset + i * tiling->hopLength + j * onceSkipInInputGM],
                    copyInParams, dataCopyPadExtParams);
        inCopy.EnQue(inputLocal);
        inputLocal = inCopy.template DeQue<T>();
        LocalTensor<T> outputLocal = outCopy.template AllocTensor<T>();
        DataCopy(outputLocal, inputLocal, onceNInUB * tiling->nfftAlign);
        outCopy.EnQue(outputLocal);
        inCopy.FreeTensor(inputLocal);
        outputLocal = outCopy.template DeQue<T>();
        DataCopyPad(splitWindowGm[outputOffset + i * tiling->nfftAlign + j * onceSkipInOutputGm],
                    outputLocal, copyOutParams);
        outCopy.FreeTensor(outputLocal);
      }
      if (splitRemainder > 0) {
        DataCopyExtParams copyInParamsRemainder;
        copyInParamsRemainder.blockCount = splitRemainder;
        copyInParamsRemainder.blockLen = burstLenIn;
        copyInParamsRemainder.srcStride = copyInSrcGap;
        copyInParamsRemainder.dstStride = copyInDstGap;
        DataCopyExtParams copyOutParamsRemainder;
        copyOutParamsRemainder.blockCount = splitRemainder;
        copyOutParamsRemainder.blockLen = burstLenOut;
        copyOutParamsRemainder.srcStride = 0;
        copyOutParamsRemainder.dstStride = copyOutDstGap;
        LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
        DataCopyPad(inputLocal, inputGm[inputOffset + i * tiling->hopLength + splitQuotient * onceSkipInInputGM],
                    copyInParamsRemainder, dataCopyPadExtParams);
        inCopy.EnQue(inputLocal);
        inputLocal = inCopy.template DeQue<T>();
        LocalTensor<T> outputLocal = outCopy.template AllocTensor<T>();
        DataCopy(outputLocal, inputLocal, splitRemainder * tiling->nfftAlign);
        outCopy.EnQue(outputLocal);
        inCopy.FreeTensor(inputLocal);
        outputLocal = outCopy.template DeQue<T>();
        DataCopyPad(splitWindowGm[outputOffset + i * tiling->nfftAlign + splitQuotient * onceSkipInOutputGm],
                    outputLocal, copyOutParamsRemainder);
        outCopy.FreeTensor(outputLocal);
      }
    }
  }

  __aicore__ inline void SplitWindowsNormal(int64_t inputOffset, int64_t outputOffset, int32_t nFactor) {
    int32_t loopCount = tiling->nfft * sizeof(T) / bufferSize;
    int32_t tailCount = (tiling->nfft * sizeof(T) % bufferSize) / sizeof(T);
    if (!isAligned) {
      // not align
      for (int32_t i = 0; i < nFactor; i++) {
        for (int32_t j = 0; j < loopCount; j++) {
          LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
          DataCopy(inputLocal, inputGm[inputOffset + i * tiling->hopLength + j * bufferSize / sizeof(T)],
                   bufferSize / sizeof(T));
          event_t eventIdMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
          SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
          WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
          DataCopy(splitWindowGm[outputOffset + i * tiling->nfftAlign + j * bufferSize / sizeof(T)], inputLocal,
                   bufferSize / sizeof(T));
          event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
          SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
          WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
          inCopy.FreeTensor(inputLocal);
        }

        if (tailCount > 0) {
          DataCopyPadExtParams<T> dataCopyPadExtParams;
          dataCopyPadExtParams.isPad = false;
          DataCopyExtParams copyParams;
          copyParams.blockCount = 1;
          copyParams.blockLen = tailCount * sizeof(T);
          copyParams.srcStride = 0;
          copyParams.dstStride = 0;
          LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
          DataCopyPad(inputLocal, inputGm[inputOffset + i * tiling->hopLength + loopCount * bufferSize / sizeof(T)], copyParams, dataCopyPadExtParams);
          event_t eventIdMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
          SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
          WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
          DataCopyPad(splitWindowGm[outputOffset + i * tiling->nfftAlign + loopCount * bufferSize / sizeof(T)], inputLocal, copyParams);
          event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
          SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
          WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
          inCopy.FreeTensor(inputLocal);
        }
      }
    } else { // align
      int32_t total = 0;
      while (total < nFactor) {
        LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
        int32_t inputLeft = (tiling->batch * (tiling->inputSize + tiling->nfft) - total * tiling->hopLength - inputOffset) * sizeof(T);
        int32_t copyLength = inputLeft > bufferSize ? bufferSize / sizeof(T) : inputLeft / sizeof(T);
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = copyLength * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(inputLocal, inputGm[inputOffset + total * tiling->hopLength], copyParams, dataCopyPadExtParams);
        inCopy.EnQue(inputLocal);
        inputLocal = inCopy.template DeQue<T>();
        LocalTensor<T> outputLocal = outCopy.template AllocTensor<T>();
        int32_t bufferLeft = copyLength;
        int32_t outputBufferLeft = bufferSize / sizeof(T);
        int32_t i = 0;
        while (bufferLeft >= (int32_t)tiling->nfftAlign && outputBufferLeft >= (int32_t) tiling->nfftAlign && total + i < nFactor) {
          DataCopy(outputLocal[i * tiling->nfftAlign], inputLocal[i * tiling->hopLength], tiling->nfftAlign);
          i++;
          bufferLeft -= tiling->hopLength;
          outputBufferLeft -= tiling->nfftAlign;
        }
        outCopy.EnQue(outputLocal);
        inCopy.FreeTensor(inputLocal);
        outputLocal = outCopy.template DeQue<T>();
        DataCopy(splitWindowGm[outputOffset + total * tiling->nfftAlign], outputLocal, i * tiling->nfftAlign);
        outCopy.FreeTensor(outputLocal);
        total += i;
      }
    }
  }

  __aicore__ inline void SplitWindows(int64_t inputOffset, int64_t outputOffset, int32_t nFactor) {
    if (tiling->splitWindowSkipNum > 0) {
      if (tiling->splitWindowSkipNum * tiling->hopLength >= tiling->nfftAlign) {
        return SplitWindowsUseMTESkip(inputOffset, outputOffset, nFactor);
      }
      return SplitWindowsUseVecSkip(inputOffset, outputOffset, nFactor);
    }
    return SplitWindowsNormal(inputOffset, outputOffset, nFactor);
  }

  uint32_t BLOCK_NUM = 8;
  uint32_t BLOCK_SIZE = 32;
  uint32_t WORKSPACE_ALIGN_SIZE = 512;
  uint32_t REPEAT_SIZE = 256;
  uint32_t IMAG_AND_REAL = 2;
  uint32_t DWORD_SIZE = 4;
  uint32_t GATHER_BUFFER_NUM = 2;
  uint32_t BINARY_ADD_COEF = 2;
  int32_t maskCount;
  int32_t bufferSize;
  bool isAligned;
  T rootNfft;

  STFTGeneralizedTilingData *tiling;

  TPipe *pipe;
  TQue<QuePosition::VECIN, bufferNum> inCopy;
  TQue<QuePosition::VECOUT, bufferNum> outCopy;

  TQue<QuePosition::VECIN, bufferNum> gatherIn;
  TQue<QuePosition::VECOUT, bufferNum> gatherOut;

  GlobalTensor<T> inputGm;
  GlobalTensor<T> outputGm;
  GlobalTensor<T> splitWindowGm;
  GlobalTensor<T> matmulOutputGm; // real part of matmul workspace
  GlobalTensor<T> planGm; // real part of dft matrix

  TBuf<> maskUB; // gather mask in ub
  TBuf<> tempUB; // temp buffer in ub

  LocalTensor<uint32_t> mask;
};
}  // namespace STFTND
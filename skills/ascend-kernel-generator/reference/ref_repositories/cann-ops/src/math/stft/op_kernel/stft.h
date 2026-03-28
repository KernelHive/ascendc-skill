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
 * \file stft.h
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"
#include "lib/pad/kernel_operator_pad_intf.h"
#include "kernel_operator_mm_intf.h"
#include "kernel_operator_fixpipe_intf.h"

namespace STFTND {
using namespace AscendC;
using namespace matmul;

constexpr int32_t ALIGNMENT_SIZE = 32;
constexpr int64_t CONTINUOUS_DATA = 256;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t FLOAT_MASK = 64;
constexpr uint32_t BLOCK_NUM = 8;
constexpr int32_t REAL_IMAG_PER_ROW = 4;
constexpr int32_t REAL_IMAG_COLS = 8;
constexpr int32_t REAL_IMAG = 2;
constexpr int PING_PONG_NUM = 2;
constexpr int BLOCK_BYTES = 32;
constexpr int BLOCK_ALIGN_NUM = 8;
constexpr int MODE_MOVE_LEFT = 4;
constexpr int FLAG_MOVE_LEFT = 8;
constexpr int TAIL_ROW_NUM = 2;
constexpr int TWO_BLOCKS = 2;
constexpr int FOUR_BLOCKS = 4;
constexpr int EIGHT_BLOCKS = 8;
constexpr int DIFFS_ONE_BLOCK = 4;
constexpr int C_V_DOUBLE = 2;
constexpr int WORKSPACE_SIZE_ALIGN = 512;

template <typename T, int32_t bufferNum>
class StftND {
 public:
  __aicore__ inline StftND(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR window, GM_ADDR y, GM_ADDR workspace, STFTTilingData* tilingData) {
    inTilingData = tilingData;
    size_t windowSplitWorkspaceSize =
        (((inTilingData->blockNum * inTilingData->frameCount * inTilingData->nfft * sizeof(T) + 511) / 512) * 512) *
        inTilingData->aivBatchLoop / sizeof(T);
    if (g_coreType == AIV) {
      inputGm.SetGlobalBuffer((__gm__ T*)x, inTilingData->batch * inTilingData->inputSize);
      windowSplitWorkspaceGm.SetGlobalBuffer((__gm__ T*)workspace, inTilingData->blockNum * inTilingData->aivBatchLoop *
                                                                       inTilingData->frameCount * inTilingData->nfft);
      outputGm.SetGlobalBuffer((__gm__ T*)y,
                               inTilingData->matmulM * inTilingData->frameCount * inTilingData->batch * 2);
      gmReal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + windowSplitWorkspaceSize,
                             inTilingData->batch * inTilingData->matmulM * inTilingData->frameCount * 2);
      gmImag.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + windowSplitWorkspaceSize,
                             inTilingData->batch * inTilingData->matmulM * inTilingData->frameCount * 2);
      pipe.InitBuffer(
          inQueueInput, bufferNum,
          (inTilingData->blkFrame * inTilingData->hop + (inTilingData->nfft - inTilingData->hop)) * sizeof(T));
      pipe.InitBuffer(outQueueOutput, bufferNum, inTilingData->blkFrame * inTilingData->nfft * sizeof(T));
    }

    if (g_coreType == AIC) {
      auto blockIdx = GetBlockIdx();
      a1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(window), inTilingData->matmulM * inTilingData->nfft * 2);
      bGlobal.SetGlobalBuffer(
          reinterpret_cast<__gm__ T*>(workspace),
          inTilingData->blockNum * inTilingData->nfft * inTilingData->frameCount * inTilingData->aivBatchLoop);
      matMulWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + windowSplitWorkspaceSize,
                                        inTilingData->matmulM * inTilingData->frameCount * inTilingData->batch * 2);

      curCoreM_ = inTilingData->aicTotalLen;
      curCoreN_ = inTilingData->mmTilingData.N;
      curCoreK_ = inTilingData->mmTilingData.singleCoreK;
      if (blockIdx % inTilingData->aicMatmulMCore >= inTilingData->aicMTailIdx) {
        curCoreM_ = inTilingData->aicTailLen;
      }
      baseN_ = 96;
      baseMK_ = baseM_ * baseK_;
      baseKN_ = baseK_ * baseN_;
      baseMN_ = baseM_ * baseN_;
      L1N_ = baseN_;
      midTailM_ = curCoreM_ % L1M_;
      midTailN_ = curCoreN_ % L1N_;
      if (midTailM_ == 0) {
        midTailM_ = L1M_;
      }
      if (midTailN_ == 0) {
        midTailN_ = L1N_;
      }

      pipe.InitBuffer(inQueueL1, 3, baseMK_ * sizeof(T) * inK_);
      pipe.InitBuffer(inQueueA0, depthA0_, baseMK_ * sizeof(T));
      pipe.InitBuffer(inQueueB0, depthB0_, baseKN_ * sizeof(T));
      pipe.InitBuffer(outQueueCO, depthC0_, baseMN_ * sizeof(T));
    }
  }

  __aicore__ inline void Process() {
    // 分帧
    uint32_t totalMLen = inTilingData->aicTotalLen;
    uint32_t frameCount = inTilingData->frameCount;
    uint32_t aicTailLen = inTilingData->aicTailLen;
    uint32_t aicBatchLoop = inTilingData->aicBatchLoop;
    uint32_t matmulM = inTilingData->matmulM;

    if (g_coreType == AIV) {
      auto blockIdx = GetBlockIdx();
      SplitFrameNormal(0, blockIdx);

      pipe.InitBuffer(realInQueue, PING_PONG_NUM, inTilingData->sizePerRepeat);
      pipe.InitBuffer(imagInQueue, PING_PONG_NUM, inTilingData->sizePerRepeat);
      pipe.InitBuffer(complexOutQueue, PING_PONG_NUM, REAL_IMAG * inTilingData->sizePerRepeat);
      pipe.InitBuffer(mask, REAL_IMAG * inTilingData->sizePerRepeat);

      uint32_t gatherSizePerRepeat = inTilingData->sizePerRepeat;
      uint32_t windowLoop = inTilingData->aivWindowLoop;
      uint32_t outUbGap = inTilingData->aivGatherUbGap;
      uint32_t aivTotalEvenMLen = inTilingData->aivTotalEvenRow;
      uint32_t aivTailEvenMLen = inTilingData->aivTailEvenRow;
      uint32_t aivBatchLoop = inTilingData->aivBatchLoop;
      uint32_t frameCountAlign = inTilingData->frameCountAlign;
      uint32_t frameCountDiff = frameCountAlign - frameCount;
      uint32_t aivMTailIdx = inTilingData->aivMTailIdx;
      uint32_t aivTailLoop = inTilingData->aivTailLoop;
      uint32_t aivBatchTailIdx = inTilingData->aivBatchTailIdx;
      int32_t numPerRepeat = gatherSizePerRepeat / sizeof(T);
      int32_t maskCol = (numPerRepeat * REAL_IMAG) / BLOCK_ALIGN_NUM;

      rowEven = inTilingData->aivTotalEvenRow;
      rowOdd = inTilingData->aivTotalOddRow;
      if (blockIdx % inTilingData->aivWindowLoop >= inTilingData->aivMTailIdx) {
        rowEven = inTilingData->aivTailEvenRow;
        rowOdd = inTilingData->aivTailOddRow;
      }
      if (blockIdx % 2 == 0) {
        N = rowEven * inTilingData->frameCountAlign;
      } else {
        N = rowOdd * inTilingData->frameCountAlign;
      }

      LocalTensor<int32_t> maskTemp = mask.template Get<int32_t>();
      LocalTensor<uint32_t> maskBuf;

      uint64_t flag_id_mte3 = 3;
      uint64_t mode = 2;
      uint64_t config = 1 | (mode << MODE_MOVE_LEFT) | (flag_id_mte3 << FLAG_MOVE_LEFT);
      ffts_cross_core_sync(PIPE_MTE3, config);
      for (int32_t i = 1; i < aivBatchLoop; i++) {
        if (i >= aivTailLoop && blockIdx >= aivBatchTailIdx) {
          continue;
        }

        SplitFrameNormal(i, blockIdx);
      }
      CreateGatherMask(maskTemp, maskCol, REAL_IMAG_COLS, 0, REAL_IMAG * gatherSizePerRepeat);

      // 生成mask等差数列
      int32_t offsetBase = 0;
      if (unlikely(blockIdx % windowLoop < aivMTailIdx)) {
        offsetBase = ((blockIdx % windowLoop) / C_V_DOUBLE) * totalMLen * frameCount +
                     (blockIdx % C_V_DOUBLE) * (aivTotalEvenMLen * REAL_IMAG) * frameCount;
      } else {
        offsetBase =
            (aivMTailIdx / C_V_DOUBLE) * totalMLen * frameCount +
            (((blockIdx % windowLoop) - aivMTailIdx) % C_V_DOUBLE) * (aivTailEvenMLen * REAL_IMAG) * frameCount +
            ((blockIdx % windowLoop) - aivMTailIdx) / C_V_DOUBLE * aicTailLen * frameCount;
      }

      int repeats = (N * sizeof(T) + gatherSizePerRepeat - 1) / gatherSizePerRepeat;
      maskBuf = maskTemp.ReinterpretCast<uint32_t>();

      for (int32_t i = 0; i < aivBatchLoop; i++) {
        if (i >= aivTailLoop && blockIdx >= aivBatchTailIdx) {
          continue;
        }

        uint64_t flag_id_fix = 1;
        wait_flag_dev(flag_id_fix);

        int32_t curBatch = (blockIdx / windowLoop) * aicBatchLoop + i;
        int32_t gmRealOffset = curBatch * matmulM * frameCount * 2;
        int32_t gmImagOffset = gmRealOffset + frameCount;
        int32_t outputOffset = curBatch * matmulM * frameCount * 2;

        gmRealOffset = gmRealOffset + offsetBase;
        gmImagOffset = gmImagOffset + offsetBase;
        outputOffset = outputOffset + offsetBase;
        auto gmRealLocal = gmReal[gmRealOffset];
        auto gmImagLocal = gmImag[gmImagOffset];
        auto outputGmLocal = outputGm[outputOffset];
        int32_t len = numPerRepeat;
        int32_t nBurst = len / frameCountAlign;
        int32_t offset = nBurst * frameCount * 2;
        for (int j = 0; j < repeats; j++) {
          if (j == repeats - 1) {
            len = N - numPerRepeat * j;
            nBurst = len / frameCountAlign;
          }
          LocalTensor<T> realBuf = realInQueue.template AllocTensor<T>();
          DataCopyExtParams copyGmToUbParams{static_cast<uint16_t>(nBurst),
                                             static_cast<uint32_t>(frameCount * sizeof(T)),
                                             static_cast<uint32_t>(frameCount * sizeof(T)), 0, 0};
          DataCopyPadExtParams<T> padParams{false, 0, static_cast<uint8_t>(frameCountDiff), 0};
          DataCopyPad(realBuf, gmRealLocal[j * offset], copyGmToUbParams, padParams);
          realInQueue.EnQue(realBuf);

          LocalTensor<T> imagBuf = imagInQueue.template AllocTensor<T>();
          DataCopyPad(imagBuf, gmImagLocal[j * offset], copyGmToUbParams, padParams);
          imagInQueue.EnQue(imagBuf);

          realBuf = realInQueue.template DeQue<T>();
          imagBuf = imagInQueue.template DeQue<T>();
          LocalTensor<T> complexBuf = complexOutQueue.template AllocTensor<T>();
          Gather(complexBuf, realBuf, maskBuf, 0, REAL_IMAG * len);
          complexOutQueue.EnQue(complexBuf);
          realInQueue.FreeTensor(realBuf);
          imagInQueue.FreeTensor(imagBuf);

          complexBuf = complexOutQueue.template DeQue<T>();
          DataCopyExtParams copyUbToGmParams{static_cast<uint16_t>(nBurst),
                                             static_cast<uint32_t>(frameCount * sizeof(T) * REAL_IMAG), outUbGap, 0, 0};
          DataCopyPad(outputGmLocal[j * offset], complexBuf, copyUbToGmParams);
          complexOutQueue.FreeTensor(complexBuf);
        }
      }
    }

    // Matmul
    if (g_coreType == AIC) {
      int blockIdx = GetBlockIdx();
      int32_t aicMatmulMCore = inTilingData->aicMatmulMCore;
      uint32_t aicTailLoop = inTilingData->aicTailLoop;
      uint32_t aicBatchTailIdx = inTilingData->aicBatchTailIdx;
      int32_t curBatchBase = (blockIdx / aicMatmulMCore) * aicBatchLoop;
      int32_t innerReminder = blockIdx % aicMatmulMCore;
      int32_t aicMTailIdx = inTilingData->aicMTailIdx;
      int32_t nfft = inTilingData->nfft;

      globalM_ = inTilingData->mmTilingData.M;
      globalN_ = inTilingData->mmTilingData.N;
      globalK_ = inTilingData->mmTilingData.Ka;

      int32_t a1GlobalOffset = 0;
      int32_t outputOffsetBase = 0;
      if (unlikely(blockIdx % aicMatmulMCore < aicMTailIdx)) {
        a1GlobalOffset = innerReminder * totalMLen * nfft;
        outputOffsetBase = innerReminder * totalMLen * frameCount;
      } else {
        a1GlobalOffset = aicMTailIdx * totalMLen * nfft + (innerReminder - aicMTailIdx) * aicTailLen * nfft;
        outputOffsetBase =
            aicMTailIdx * totalMLen * frameCount + (innerReminder - aicMTailIdx) * aicTailLen * frameCount;
      }

      uint64_t flag_id_mte3 = 3;
      wait_flag_dev(flag_id_mte3);
      for (int i = 0; i < aicBatchLoop; i++) {
        if (i >= aicTailLoop && blockIdx >= aicBatchTailIdx) {
          continue;
        }

        int32_t curBatch = curBatchBase + i;
        int32_t bGlobalOffset =
            blockIdx * aicBatchLoop * frameCount * inTilingData->nfft + i * frameCount * inTilingData->nfft;
        int32_t outputOffset = curBatch * matmulM * frameCount * REAL_IMAG + outputOffsetBase;
        bGM = bGlobal[bGlobalOffset];
        aGM = a1Global[a1GlobalOffset];
        cGM = matMulWorkspaceGm[outputOffset];
        IterateAllTB();
        uint64_t flag_id_fix = 1;
        uint64_t mode_fix = 2;
        uint64_t config_fix = 1 | (mode_fix << 4) | (flag_id_fix << 8);
        ffts_cross_core_sync(PIPE_FIX, config_fix);
      }
    }
  }

  __aicore__ inline void CreateGatherMask(LocalTensor<int32_t>& dst, const int32_t row, const int32_t col,
                                          const int32_t startValue, const int32_t diffValue) {
    int32_t repeatNum = row / BLOCK_NUM;
    int32_t tailNum = row % BLOCK_NUM;
    int32_t realNum = BLOCK_BYTES / (sizeof(T) * REAL_IMAG);
    for (int32_t i = 0; i < REAL_IMAG_COLS / REAL_IMAG; i++) {
      dst.SetValue(i * REAL_IMAG, startValue + i * realNum);
      dst.SetValue(i * REAL_IMAG + 1, startValue + diffValue + i * realNum);
    }
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    if (repeatNum == 0) {
      AddsComputeAlignment(dst, dst, row, col, realNum);
      return;
    }
    AddsComputeAlignment(dst, dst, REAL_IMAG, col, realNum);
    PipeBarrier<PIPE_V>();
    AddsComputeAlignment(dst, dst, REAL_IMAG, col * TWO_BLOCKS, TWO_BLOCKS * realNum);
    PipeBarrier<PIPE_V>();
    AddsComputeAlignment(dst, dst, REAL_IMAG, col * FOUR_BLOCKS, FOUR_BLOCKS * realNum);
    PipeBarrier<PIPE_V>();
    AddsComputeAlignment(dst, dst, repeatNum, col * EIGHT_BLOCKS, EIGHT_BLOCKS * realNum);
    PipeBarrier<PIPE_V>();
    if (tailNum != 0) {
      int32_t lastPrevOffset = repeatNum * EIGHT_BLOCKS * (BLOCK_BYTES / sizeof(T)) - BLOCK_BYTES / sizeof(T);
      auto lastPrevDst = dst[lastPrevOffset];
      AddsComputeAlignment(lastPrevDst, lastPrevDst, tailNum + 1, col, realNum);
    }
  }

  __aicore__ inline void AddsComputeAlignment(LocalTensor<int32_t>& dst, LocalTensor<int32_t>& src,
                                              const int32_t rowCount, const int32_t colCount, const int32_t scalar) {
    int32_t repeatTimesOneRow = (colCount * FLOAT_BYTES + CONTINUOUS_DATA - 1) / CONTINUOUS_DATA;
    int32_t countOneRow32B = (colCount * FLOAT_BYTES + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
    UnaryRepeatParams repeatParams;
    for (int32_t i = 0; i < rowCount; i++) {
      repeatParams.dstBlkStride = 1;
      repeatParams.srcBlkStride = 1;
      repeatParams.dstRepStride = CONTINUOUS_DATA / ALIGNMENT_SIZE;
      repeatParams.srcRepStride = CONTINUOUS_DATA / ALIGNMENT_SIZE;

      if ((repeatTimesOneRow - 1) > 0) {
        Adds(dst[i * ((countOneRow32B * ALIGNMENT_SIZE) / FLOAT_BYTES)], src[0], static_cast<int32_t>(i * sizeof(T)),
             FLOAT_MASK, repeatTimesOneRow - 1, repeatParams);
      }
      int32_t mask = FLOAT_MASK - (((repeatTimesOneRow * CONTINUOUS_DATA) - (colCount * FLOAT_BYTES)) / FLOAT_BYTES);
      Adds(dst[i * (countOneRow32B * ALIGNMENT_SIZE / FLOAT_BYTES) +
               (repeatTimesOneRow - 1) * (CONTINUOUS_DATA / FLOAT_BYTES)],
           src[(repeatTimesOneRow - 1) * (CONTINUOUS_DATA / FLOAT_BYTES)], static_cast<int32_t>(i * sizeof(T) * scalar),
           mask, 1, repeatParams);
    }
  }

  __aicore__ inline void SplitFrameNormal(int32_t batchLoopIndex, int32_t blockIdx) {
    int32_t windowLength = inTilingData->frameCount;
    int32_t windowLoop = inTilingData->aivWindowLoop;
    int32_t batchLoop = inTilingData->aivBatchLoop;
    int32_t nfft = inTilingData->nfft;
    int32_t blkFrame = inTilingData->blkFrame;
    int32_t curBatch = (blockIdx / windowLoop) * batchLoop + batchLoopIndex;
    int32_t batchOffset = blockIdx / 2;

    int32_t loopBlkFrame = windowLength / blkFrame;
    int32_t remainingFrame = windowLength % blkFrame;
    int32_t halfBlkFrame = loopBlkFrame / 2;
    int32_t inputGmOffset = curBatch * (inTilingData->inputSize + nfft);
    int32_t windowSplitOffset = batchOffset * batchLoop * nfft * windowLength + batchLoopIndex * nfft * windowLength;

    // 判断BlockIdx是否为偶数
    if (GetBlockIdx() % 2 == 0) {
      for (int32_t i = 0; i < halfBlkFrame; i++) {
        CopyIn(i, blkFrame, inputGmOffset);
        Compute(i, blkFrame);
        CopyOut(i, blkFrame, windowSplitOffset);
      }
    } else {
      for (int32_t i = halfBlkFrame; i < loopBlkFrame; i++) {
        CopyIn(i, blkFrame, inputGmOffset);
        Compute(i, blkFrame);
        CopyOut(i, blkFrame, windowSplitOffset);
      }
      if (remainingFrame != 0) {
        CopyIn(loopBlkFrame, remainingFrame, inputGmOffset);
        Compute(loopBlkFrame, remainingFrame);
        CopyOut(loopBlkFrame, remainingFrame, windowSplitOffset);
      }
    }
  }

  __aicore__ inline int MatmulCeil(int num1, int num2) {
    ASSERT(num2 > 0);
    return (num1 + num2 - 1) / num2;
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress, int32_t remainingFrame, int32_t inputBaseOffset) {
    int inputGmStart = progress * inTilingData->blkFrame * inTilingData->hop;
    int copySize = remainingFrame * inTilingData->hop + (inTilingData->nfft - inTilingData->hop);
    LocalTensor<T> inputLocal = inQueueInput.template AllocTensor<T>();
    DataCopy(inputLocal, inputGm[inputGmStart + inputBaseOffset], copySize);
    inQueueInput.EnQue(inputLocal);
  }

  __aicore__ inline void Compute(int32_t progress, int32_t remainingFrame) {
    LocalTensor<T> inputLocal = inQueueInput.template DeQue<T>();
    LocalTensor<T> outputLocal = outQueueOutput.template AllocTensor<T>();

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = inTilingData->nfft / BLOCK_NUM;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    for (int32_t j = 0; j < remainingFrame; j++) {
      DataCopy(outputLocal[j * inTilingData->nfft], inputLocal[j * inTilingData->hop], dataCopyParams);
    }

    outQueueOutput.EnQue(outputLocal);
    inQueueInput.FreeTensor(inputLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress, int32_t remainingFrame, int32_t outputBaseOffset) {
    LocalTensor<T> outputLocal2 = outQueueOutput.template DeQue<T>();
    DataCopy(windowSplitWorkspaceGm[progress * inTilingData->blkFrame * inTilingData->nfft + outputBaseOffset],
             outputLocal2, remainingFrame * inTilingData->nfft);
    outQueueOutput.FreeTensor(outputLocal2);
  }

  __aicore__ inline void IterateAllTB() {
    int midM = MatmulCeil(curCoreM_, baseM_);
    int midN = MatmulCeil(curCoreN_, baseN_);
    int iterAM = 0;
    int iterBN = 0;
    int iterK = 0;
    int useL1M = curCoreM_ < L1M_ ? curCoreM_ : L1M_;
    int useL1N = curCoreN_ < L1N_ ? curCoreN_ : L1N_;
    LocalTensor<T> tmpB;
    LocalTensor<T> tmpA;

    LocalTensor<T> b1Local;
    LocalTensor<T> a1Local;

    tmpA = inQueueL1.AllocTensor<T>();
    CopyGMND2LMNZ(tmpA, aGM, iterAM, iterK, useL1M, L1K_, globalK_);
    tmpB = inQueueL1.AllocTensor<T>();
    CopyGMND2LMNZTranspose(tmpB, bGM, iterK, iterBN, L1K_, useL1N, globalK_);
    inQueueL1.EnQue(tmpA);
    inQueueL1.EnQue(tmpB);
    for (int idxAMOut = 0; idxAMOut < midM; ++idxAMOut) {
      if (idxAMOut == midM - 1) {
        useL1M = midTailM_;
      }
      iterAM = idxAMOut * baseM_;
      iterBN = 0;
      a1Local = inQueueL1.DeQue<T>();
      for (int idxBNOut = 0; idxBNOut < midN - 1; ++idxBNOut) {
        if (unlikely(idxBNOut == midN - 2)) {
          tmpB = inQueueL1.AllocTensor<T>();
          CopyGMND2LMNZTranspose(tmpB, bGM, iterK, iterBN + baseN_, L1K_, midTailN_, globalK_);
          inQueueL1.EnQue(tmpB);
        } else {
          tmpB = inQueueL1.AllocTensor<T>();
          CopyGMND2LMNZTranspose(tmpB, bGM, iterK, iterBN + baseN_, L1K_, useL1N, globalK_);
          inQueueL1.EnQue(tmpB);
        }
        b1Local = inQueueL1.DeQue<T>();
        computeTail(a1Local, b1Local, iterAM, iterBN, useL1M, useL1N, L1K_, false);
        inQueueL1.FreeTensor(b1Local);
        iterBN += baseN_;
      }
      if (likely(idxAMOut < midM - 1)) {
        int nextM = iterAM + baseM_;
        int nextK = 0;
        int nextN = 0;
        int nextUseL1M = L1M_;
        if (unlikely(idxAMOut == midM - 2)) {
          nextUseL1M = midTailM_;
        }
        tmpA = inQueueL1.AllocTensor<T>();
        CopyGMND2LMNZ(tmpA, aGM, nextM, nextK, nextUseL1M, L1K_, globalK_);
        inQueueL1.EnQue(tmpA);

        b1Local = inQueueL1.DeQue<T>();
        computeTail(a1Local, b1Local, iterAM, iterBN, useL1M, midTailN_, L1K_, false);
        inQueueL1.FreeTensor(b1Local);
        inQueueL1.FreeTensor(a1Local);

        tmpB = inQueueL1.AllocTensor<T>();
        if (midN != 1) {
          CopyGMND2LMNZTranspose(tmpB, bGM, nextK, nextN, L1K_, useL1N, globalK_);
        }
        inQueueL1.EnQue(tmpB);
      } else {
        b1Local = inQueueL1.DeQue<T>();
        computeTail(a1Local, b1Local, iterAM, iterBN, useL1M, midTailN_, L1K_, false);
        inQueueL1.FreeTensor(b1Local);
        inQueueL1.FreeTensor(a1Local);
      }
    }
  }

  __aicore__ inline void computeTail(LocalTensor<T>& a1Local, LocalTensor<T>& b1Local, int curAM, int curBN, int useL1M,
                                     int useL1N, int useL1K, bool fixpipeBias) {
    int nCount = MatmulCeil(useL1N, baseN_);
    int kCount = MatmulCeil(useL1K, baseK_);
    int sizeInM = useL1M;
    int sizeInN = useL1N;
    int sizeInK = baseK_;
    int tailSizeInK = useL1K - (kCount - 1) * baseK_;

    LocalTensor<T> a0Local;
    LocalTensor<T> b0Local;
    LocalTensor<T> c0Local = outQueueCO.AllocTensor<T>();

    LoadDataA0v5(useL1M, 0, sizeInM, sizeInK, a1Local);
    LoadDataB0v5(useL1K, 0, sizeInK, sizeInN, b1Local);
    a0Local = inQueueA0.DeQue<T>();
    b0Local = inQueueB0.DeQue<T>();
    Mmad(c0Local, a0Local, b0Local,
         {static_cast<uint16_t>(sizeInM), static_cast<uint16_t>(sizeInN), static_cast<uint16_t>(sizeInK), 0, false,
          true});

    PipeBarrier<PIPE_M>();
    inQueueA0.FreeTensor(a0Local);
    inQueueB0.FreeTensor(b0Local);

    LoadDataA0v5(useL1M, 1, sizeInM, sizeInK, a1Local);
    LoadDataB0v5(useL1K, 1, sizeInK, sizeInN, b1Local);
    a0Local = inQueueA0.DeQue<T>();
    b0Local = inQueueB0.DeQue<T>();
    Mmad(c0Local, a0Local, b0Local,
         {static_cast<uint16_t>(sizeInM), static_cast<uint16_t>(sizeInN), static_cast<uint16_t>(sizeInK), 0, false,
          false});

    PipeBarrier<PIPE_M>();
    inQueueA0.FreeTensor(a0Local);
    inQueueB0.FreeTensor(b0Local);

    LoadDataA0v5(useL1M, 2, sizeInM, sizeInK, a1Local);
    LoadDataB0v5(useL1K, 2, sizeInK, sizeInN, b1Local);
    a0Local = inQueueA0.DeQue<T>();
    b0Local = inQueueB0.DeQue<T>();

    Mmad(c0Local, a0Local, b0Local,
         {static_cast<uint16_t>(sizeInM), static_cast<uint16_t>(sizeInN), static_cast<uint16_t>(sizeInK), 0, false,
          false});
    PipeBarrier<PIPE_M>();
    inQueueA0.FreeTensor(a0Local);
    inQueueB0.FreeTensor(b0Local);

    LoadDataA0v5(useL1M, 3, sizeInM, sizeInK, a1Local);
    LoadDataB0v5(useL1K, 3, sizeInK, sizeInN, b1Local);
    a0Local = inQueueA0.DeQue<T>();
    b0Local = inQueueB0.DeQue<T>();
    Mmad(c0Local, a0Local, b0Local,
         {static_cast<uint16_t>(sizeInM), static_cast<uint16_t>(sizeInN), static_cast<uint16_t>(sizeInK), 0, false,
          false});

    PipeBarrier<PIPE_M>();
    inQueueA0.FreeTensor(a0Local);
    inQueueB0.FreeTensor(b0Local);

    LoadDataA0v5(useL1M, 4, sizeInM, tailSizeInK, a1Local);
    LoadDataB0v5(useL1K, 4, tailSizeInK, sizeInN, b1Local);
    a0Local = inQueueA0.DeQue<T>();
    b0Local = inQueueB0.DeQue<T>();
    Mmad(c0Local, a0Local, b0Local,
         {static_cast<uint16_t>(sizeInM), static_cast<uint16_t>(sizeInN), static_cast<uint16_t>(tailSizeInK), 0, false,
          false});

    PipeBarrier<PIPE_M>();
    inQueueA0.FreeTensor(a0Local);
    inQueueB0.FreeTensor(b0Local);

    outQueueCO.EnQue<T>(c0Local);
    FixPipeC0Tail(curAM, 0, curBN, 0, sizeInM, sizeInN, fixpipeBias);
  }

  __aicore__ inline void LoadDataA0v5(int useL1M, int idxInK, int sizeInM, int sizeInK, const LocalTensor<T>& a1Local) {
    LocalTensor<T> a0Local = inQueueA0.AllocTensor<T>();

    int useL1MCeil = MatmulCeil(useL1M, cubeCol_);
    int sizeInMCeil = MatmulCeil(sizeInM, cubeCol_);
    int sizeInKCeil = MatmulCeil(sizeInK, cubeRow_);

    LoadData3DParamsV2<T> param;
    param.l1H = 1;
    param.l1W = useL1MCeil * cubeCol_;
    param.kExtension = cubeRow_ * sizeInKCeil;
    param.mExtension = cubeCol_ * sizeInMCeil;
    param.strideW = 1;
    param.strideH = 1;
    param.filterW = 1;
    param.filterH = 1;
    param.dilationFilterW = 1;
    param.dilationFilterH = 1;
    param.channelSize = cubeRow_ * sizeInKCeil;
    LoadData(a0Local, a1Local[cubeCol_ * useL1MCeil * baseK_ * idxInK], param);
    inQueueA0.EnQue(a0Local);
  }

  __aicore__ inline void LoadDataB0v5(int useL1N, int idxInK, int sizeInK, int sizeInN, const LocalTensor<T>& b1Local) {
    LocalTensor<T> b0Local = inQueueB0.AllocTensor<T>();

    uint8_t RepeatCeil = MatmulCeil(sizeInK, cubeRow_) * MatmulCeil(sizeInN, cubeCol_);
    int sizeInNCeil = MatmulCeil(sizeInN, cubeCol_) * cubeCol_;
    RepeatCeil = RepeatCeil == 1 ? 2 : RepeatCeil;
    LoadData2dParams param = {0, RepeatCeil, 1, 0, 0, false, inc};
    LoadData(b0Local, b1Local[idxInK * baseK_ * sizeInNCeil], param);
    inQueueB0.EnQue(b0Local);
  }

  __aicore__ inline void FixPipeC0Tail(int aM, int inM, int bN, int inN, int sizeInM, int sizeInN, bool setAtomicAdd) {
    LocalTensor<T> c0Local = outQueueCO.DeQue<T>();
    if (setAtomicAdd) {
      SetAtomicAdd<T>();
    }
    FixpipeParams<T> param;
    param.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(sizeInN)};
    param.cburstNum = 0;
    param.dstStride = globalN_;
    param.srcStride = 0;
    param.burstLen = sizeInM * REAL_IMAG;
    Fixpipe(cGM[(aM + inM * baseM_) * globalN_ + bN + inN * baseN_], c0Local, param);

    if (setAtomicAdd) {
      SetAtomicNone();
    }

    outQueueCO.FreeTensor(c0Local);
  }

  __aicore__ inline void CopyGMND2LMNZ(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint16_t row,
                                       const uint16_t col, const uint16_t height, const uint16_t width,
                                       const uint16_t gCol) {
    Nd2NzParams param = {1, height, width, 0, gCol, static_cast<uint16_t>(MatmulCeil(height, cubeCol_) * cubeCol_),
                         1, 0};
    DataCopy(dst, src[row * gCol + col], param);
  }

  __aicore__ inline void CopyGMND2LMNZTranspose(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                                const uint16_t row, const uint16_t col, const uint16_t height,
                                                const uint16_t width, const uint16_t gCol) {
    Nd2NzParams param = {1, width, height, 0, gCol, static_cast<uint16_t>(MatmulCeil(width, cubeCol_) * cubeCol_),
                         1, 0};
    DataCopy(dst, src[col * gCol + row], param);
  }

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, bufferNum> inQueueInput;
  TQue<QuePosition::VECOUT, bufferNum> outQueueOutput;
  TQue<QuePosition::VECIN, PING_PONG_NUM> realInQueue;
  TQue<QuePosition::VECIN, PING_PONG_NUM> imagInQueue;
  TQue<QuePosition::VECOUT, PING_PONG_NUM> complexOutQueue;

  GlobalTensor<T> inputGm;
  GlobalTensor<T> outputGm;
  GlobalTensor<T> windowSplitWorkspaceGm;
  GlobalTensor<T> matMulWorkspaceGm;

  GlobalTensor<T> gmReal;
  GlobalTensor<T> gmImag;

  GlobalTensor<T> aGM;
  GlobalTensor<T> bGM;
  GlobalTensor<T> cGM;

  TBuf<> mask;

  constexpr static int baseM_ = 96;
  int baseN_;
  constexpr static int baseK_ = 80;

  constexpr static int cubeRow_ = 8;
  constexpr static int cubeCol_ = 16;
  int baseMK_;
  int baseKN_;
  int baseMN_;
  constexpr static int depthA0_ = 2;
  constexpr static int depthB0_ = 2;
  constexpr static int depthC0_ = 2;

  constexpr static int inK_ = 5;
  constexpr static int L1M_ = baseM_;
  int L1N_;
  constexpr static int L1K_ = inK_ * baseK_;

  int globalM_;
  int globalN_;
  int globalK_;

  int curCoreM_;
  int curCoreN_;
  int curCoreK_;

  int midTailM_;
  int midTailN_;

  TQue<QuePosition::A1, 3> inQueueL1;
  TQue<QuePosition::A2, 2> inQueueA0;
  TQue<QuePosition::B2, 2> inQueueB0;
  TQue<QuePosition::CO1, 2> outQueueCO;

  GlobalTensor<T> a1Global;
  GlobalTensor<T> bGlobal;

  int32_t N;
  int32_t rowEven;
  int32_t rowOdd;
  STFTTilingData* inTilingData;
};
}  // namespace STFTND

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
  * \file stft_plan_mul.h
  * \brief
  */

#ifndef STFT_PLAN_MUL_H
#define STFT_PLAN_MUL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"
#include "lib/pad/kernel_operator_pad_intf.h"

namespace STFTND {
using namespace AscendC;
template <typename T, int32_t bufferNum>
class StftPlanMul {
 public:
  __aicore__ inline StftPlanMul(){};
  __aicore__ inline void Init(GM_ADDR plan, GM_ADDR window, GM_ADDR workspace, STFTPlanTilingData* tilingData,
                              TPipe* pipeIn) {
    pipe = pipeIn;
    inTilingData = tilingData;
    if (g_coreType == AIV) {
      auto blockIdx = GetBlockIdx();
      totalRow = inTilingData->totalLine;
      tailRow = inTilingData->tailLine;
      col = inTilingData->oneRowLen;
      ubMaxLine = inTilingData->ubMaxLine;
      row = totalRow;
      int32_t planGlobalOffset = blockIdx * totalRow * col;
      if (blockIdx >= inTilingData->tailBlockIdx) {
        planGlobalOffset =
            inTilingData->tailBlockIdx * totalRow * col + (blockIdx - inTilingData->tailBlockIdx) * tailRow * col;
        row = tailRow;
      }
      uint64_t splitWindowWorkspaceSize =
          (((uint64_t)inTilingData->batch * inTilingData->matmulN * col * sizeof(T)  +WORKSPACE_ALIGN_SIZE - 1) /
           WORKSPACE_ALIGN_SIZE) * WORKSPACE_ALIGN_SIZE / sizeof(T);
      uint64_t matmulWorkspaceSize =
          (((uint64_t)inTilingData->batch * inTilingData->matmulN * inTilingData->matmulM * IMAG_AND_REAL * sizeof(T)
            + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) * WORKSPACE_ALIGN_SIZE / sizeof(T);
      uint64_t planOffset = splitWindowWorkspaceSize + matmulWorkspaceSize;
      planGm.SetGlobalBuffer((__gm__ T*)plan + planGlobalOffset, row * col);
      windowGm.SetGlobalBuffer((__gm__ T*)window, col);
      outputGm.SetGlobalBuffer((__gm__ T*)workspace + planOffset + planGlobalOffset, row * col);
      pipe->InitBuffer(planInQue, bufferNum, ubMaxLine * col * sizeof(T));
      pipe->InitBuffer(planOutQue, bufferNum, ubMaxLine * col * sizeof(T));
      pipe->InitBuffer(windowInQue, 1, col * sizeof(T));
    }
  }

  __aicore__ inline void Process() {
    if (g_coreType == AIV) {
      int32_t loopCnt = (row + ubMaxLine - 1) / ubMaxLine;
      int32_t copyLen = ubMaxLine * col;
      int32_t onceRow = ubMaxLine;
      CopyWindowIn(col);
      windowInput = windowInQue.template DeQue<T>();
      for (size_t i = 0; i < loopCnt; i++) {
        if (i == loopCnt - 1) {
          copyLen = (row - i * ubMaxLine) * col;
          int32_t onceRow1 = (row - i * ubMaxLine);
          CopyPlanIn(copyLen, i, onceRow);
          Compute(onceRow1);
          CopyPlanOut(copyLen, i, onceRow);
        } else {
          CopyPlanIn(copyLen, i, onceRow);
          Compute(onceRow);
          CopyPlanOut(copyLen, i, onceRow);
        }
      }
      windowInQue.FreeTensor(windowInput);
    }
  }

 private:
  __aicore__ inline void CopyPlanIn(int32_t copyLen, int32_t progress, int32_t onceRow) {
    planInput = planInQue.template AllocTensor<T>();
    DataCopy(planInput, planGm[progress * onceRow * col], copyLen);
    planInQue.EnQue(planInput);
  }

  __aicore__ inline void CopyWindowIn(int32_t copyLen) {
    windowInput = windowInQue.template AllocTensor<T>();
    DataCopy(windowInput, windowGm, copyLen);
    windowInQue.EnQue(windowInput);
  }

  __aicore__ inline void Compute(int32_t onceRow) {
    planInput = planInQue.template DeQue<T>();

    int32_t totalInCol = inTilingData->totalInCol;
    int32_t blkInOneRow = col * sizeof(T) / BLOCK_SIZE;
    int32_t tailOffset = totalInCol * NUM_PER_REPEAT;

    planOutput = planOutQue.template AllocTensor<T>();
    BinaryRepeatParams totalParam;

    if (totalInCol < onceRow && blkInOneRow < MAX_REP_NUM) {
      totalParam.src1RepStride = 0;
      totalParam.dstRepStride = blkInOneRow;
      totalParam.src0RepStride = blkInOneRow;
      for (size_t i = 0; i < totalInCol; i++) {
        Mul(planOutput[i * NUM_PER_REPEAT], planInput[i * NUM_PER_REPEAT], windowInput[i * NUM_PER_REPEAT],
            NUM_PER_REPEAT, onceRow, totalParam);
      }
      if (inTilingData->tailInCol != 0) {
        BinaryRepeatParams tailParam;
        tailParam.src1RepStride = 0;
        tailParam.dstRepStride = blkInOneRow;
        tailParam.src0RepStride = blkInOneRow;
        Mul(planOutput[tailOffset], planInput[tailOffset], windowInput[tailOffset], inTilingData->tailInCol, onceRow,
            tailParam);
      }
    } else {
      int32_t blkInOneRepeat = NUM_PER_REPEAT * sizeof(T) / BLOCK_SIZE;
      for (size_t i = 0; i < onceRow; i++) {
        totalParam.src1RepStride = blkInOneRepeat;
        totalParam.dstRepStride = blkInOneRepeat;
        totalParam.src0RepStride = blkInOneRepeat;
        Mul(planOutput[i * col], planInput[i * col], windowInput, NUM_PER_REPEAT, totalInCol, totalParam);
        if (inTilingData->tailInCol != 0) {
          int32_t tailOffset = totalInCol * NUM_PER_REPEAT;
          Mul(planOutput[i * col + tailOffset], planInput[i * col + tailOffset], windowInput[tailOffset],
              inTilingData->tailInCol);
        }
      }
    }

    planOutQue.EnQue(planOutput);
    planInQue.FreeTensor(planInput);
  }

  __aicore__ inline void CopyPlanOut(int32_t copyLen, int32_t progress, int32_t onceRow) {
    planOutput = planOutQue.template DeQue<T>();
    DataCopy(outputGm[progress * onceRow * col], planOutput, copyLen);

    planOutQue.FreeTensor(planOutput);
  }

  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> planInQue;
  TQue<QuePosition::VECIN, 1> windowInQue;
  TQue<QuePosition::VECOUT, 1> planOutQue;
  GlobalTensor<T> planGm;
  GlobalTensor<T> windowGm;
  GlobalTensor<T> outputGm;

  LocalTensor<T> planInput;
  LocalTensor<T> planOutput;
  LocalTensor<T> windowInput;

  int32_t row;
  int32_t col;
  int32_t totalRow;
  int32_t tailRow;
  int32_t tailBlockIdx;
  int32_t ubMaxLine;
  STFTPlanTilingData* inTilingData;

  uint32_t BLOCK_SIZE = 32;
  uint32_t MAX_REP_NUM = 255;
  uint32_t SIZE_PER_REPEAT = 256;
  uint32_t WORKSPACE_ALIGN_SIZE = 512;
  uint32_t IMAG_AND_REAL = 2;
  uint32_t NUM_PER_REPEAT = SIZE_PER_REPEAT / sizeof(T);
};
} // namespace STFTND

#endif STFT_PLAN_MUL_H
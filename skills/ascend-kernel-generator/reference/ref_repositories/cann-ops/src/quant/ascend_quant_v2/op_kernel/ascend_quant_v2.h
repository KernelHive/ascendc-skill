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
 * \file ascend_quant_v2.h
 * \brief
 */

#ifndef ASCEND_QUANT_V2_H
#define ASCEND_QUANT_V2_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace AscendQuantV2 {
using namespace AscendC;

template <typename T>
class AscendQuantV2Base {
public:
  __aicore__ inline AscendQuantV2Base(){};

protected:
  __aicore__ inline void ParseTilingData(const AscendQuantV2TilingData* tilingData,
                                         AscendQuantV2TilingData &runTilingData);
  __aicore__ inline void ParseCoreBlocks(const AscendQuantV2TilingData &runTilingData, int32_t blockIdx,
                                         int64_t &blockN, int64_t &blockLen);
  __aicore__ inline void GetXInCopyParams(const AscendQuantV2TilingData &runTilingData, int64_t xN, int64_t xLen,
                                          int64_t lastDimLen, DataCopyExtParams &copyParams);
  __aicore__ inline void GetOutCopyParams(const AscendQuantV2TilingData &runTilingData, int64_t yN, int64_t yLen,
                                          int64_t lastDimLen, DataCopyExtParams &copyParams);
  __aicore__ inline void CastOut(const AscendQuantV2TilingData &runTilingData, int64_t dataCount, int64_t outOffset,
                                 int64_t inOffset, LocalTensor<int8_t>& outLocal, LocalTensor<float>& xLocal);
  __aicore__ inline void MoveOutY(LocalTensor<int8_t>& outLocal, GlobalTensor<int8_t>& outGm,
                                  LocalTensor<int8_t>& outTempLocal, AscendQuantV2TilingData &runTilingData,
                                  int64_t yN, int64_t yLen, int64_t yOutOffset);
  __aicore__ inline int64_t CeilAlign(int64_t a, int64_t b);

 protected:
  constexpr static int32_t BLOCK_SIZE = 32;
  constexpr static int32_t FLOAT_NUMS_IN_ONE_BLOCK = 8;
  constexpr static int32_t VEC_INTRI_FP16_NUM = 128;
  constexpr static int16_t MODE_ROUND = 0;
  constexpr static int16_t MODE_FLOOR = 1;
  constexpr static int16_t MODE_CEIL = 2;
  constexpr static int16_t MODE_TRUNC = 3;
  constexpr static int64_t INT4_NUMS_IN_INT8_SPACE = 2;
};

template <typename T>
__aicore__ inline void AscendQuantV2Base<T>::ParseTilingData(const AscendQuantV2TilingData* tilingData,
                                                             AscendQuantV2TilingData &runTilingData) {
  runTilingData = *tilingData;
}

template <typename T>
__aicore__ inline void AscendQuantV2Base<T>::ParseCoreBlocks(const AscendQuantV2TilingData &runTilingData,
                                                             int32_t blockIdx, int64_t &blockN, int64_t &blockLen) {
  if (runTilingData.blockUnion > 1) {
    if (blockIdx >= runTilingData.numCore) {
      blockN = 0;
    } else {
      int64_t yCoreNum = runTilingData.numCore / runTilingData.dim0;
      if (blockIdx % runTilingData.blockUnion >= yCoreNum) {
        blockN = 0;
      } else if (blockIdx % runTilingData.blockUnion == yCoreNum - 1) {
        blockN = 1;
        blockLen = runTilingData.blockTailFactor;
      } else {
        blockN = 1;
        blockLen = runTilingData.blockFactor;
      }
    }
  } else {
    if (runTilingData.blockAxis == 0) {
      if (blockIdx == runTilingData.numCore - 1) {
        blockN = runTilingData.blockTailFactor;
      } else {
        blockN = runTilingData.blockFactor;
      }
      blockLen = runTilingData.dim1;
    } else if (runTilingData.blockAxis == 1) {
      blockN = runTilingData.dim0;
      if (blockIdx == runTilingData.numCore - 1) {
        blockLen = runTilingData.blockTailFactor;
      } else {
        blockLen = runTilingData.blockFactor;
      }
    }
  }
}

template <typename T>
__aicore__ inline void AscendQuantV2Base<T>::GetXInCopyParams(const AscendQuantV2TilingData &runTilingData,
                                                              int64_t xN, int64_t xLen, int64_t lastDimLen,
                                                              DataCopyExtParams &copyParams) {
  copyParams.blockCount = xN;
  copyParams.blockLen = xLen * sizeof(T);
  if (runTilingData.baseLen > xLen) {
    copyParams.dstStride = (runTilingData.baseLen - xLen) * sizeof(T) / BLOCK_SIZE;
  } else {
    copyParams.dstStride = 0;
  }
  if (lastDimLen > xLen) {
    copyParams.srcStride = (lastDimLen - xLen) * sizeof(T);
  } else {
    copyParams.srcStride = 0;
  }
}

template <typename T>
__aicore__ inline void AscendQuantV2Base<T>::GetOutCopyParams(const AscendQuantV2TilingData &runTilingData,
                                                              int64_t yN, int64_t yLen, int64_t lastDimLen,
                                                              DataCopyExtParams &copyParams) {
  int64_t yLenReal = yLen;
  if (ORIG_DTYPE_Y == DT_INT4) {
    yLenReal = yLenReal / INT4_NUMS_IN_INT8_SPACE;
  }

  copyParams.blockCount = yN;
  copyParams.blockLen = yLenReal * sizeof(int8_t);
  if (lastDimLen > yLen) {
    if (ORIG_DTYPE_Y == DT_INT4) {
      copyParams.dstStride = (lastDimLen - yLen) * sizeof(int8_t) / INT4_NUMS_IN_INT8_SPACE;
    } else {
      copyParams.dstStride = (lastDimLen - yLen) * sizeof(int8_t);
    }
  } else {
    copyParams.dstStride = 0;
  }
  if (runTilingData.baseLen > yLenReal) {
    copyParams.srcStride = (runTilingData.baseLen - yLenReal) * sizeof(int8_t) / BLOCK_SIZE;
  } else {
    copyParams.srcStride = 0;
  }
}

template <typename T>
__aicore__ inline void AscendQuantV2Base<T>::CastOut(const AscendQuantV2TilingData &runTilingData,
                                                     int64_t dataCount, int64_t outOffset, int64_t inOffset,
                                                     LocalTensor<int8_t>& outLocal, LocalTensor<float>& xLocal) {
  if (runTilingData.roundMode == MODE_ROUND) {
    Cast(xLocal[inOffset].ReinterpretCast<int32_t>(), xLocal[inOffset], RoundMode::CAST_RINT, dataCount);
  } else if (runTilingData.roundMode == MODE_FLOOR) {
    Cast(xLocal[inOffset].ReinterpretCast<int32_t>(), xLocal[inOffset], RoundMode::CAST_FLOOR, dataCount);
  } else if (runTilingData.roundMode == MODE_CEIL) {
    Cast(xLocal[inOffset].ReinterpretCast<int32_t>(), xLocal[inOffset], RoundMode::CAST_CEIL, dataCount);
  } else if (runTilingData.roundMode == MODE_TRUNC) {
    Cast(xLocal[inOffset].ReinterpretCast<int32_t>(), xLocal[inOffset], RoundMode::CAST_TRUNC, dataCount);
  }
  PipeBarrier<PIPE_V>();
  SetDeqScale((half)1.000000e+00f);
  PipeBarrier<PIPE_V>();
  Cast(xLocal[inOffset].ReinterpretCast<half>(), xLocal[inOffset].ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
       dataCount);
  PipeBarrier<PIPE_V>();

  if (ORIG_DTYPE_Y == DT_INT4) {
    Cast(outLocal[outOffset].ReinterpretCast<int4b_t>(),
        xLocal[inOffset].ReinterpretCast<half>(),
        RoundMode::CAST_RINT, dataCount);
    return ;
  }

  if (runTilingData.roundMode == MODE_ROUND) {
#if defined(__DAV_M200__)
    Cast(outLocal[outOffset], xLocal[inOffset].ReinterpretCast<half>(), RoundMode::CAST_ROUND, dataCount);
#else
    Cast(outLocal[outOffset], xLocal[inOffset].ReinterpretCast<half>(), RoundMode::CAST_RINT, dataCount);
#endif
  } else if (runTilingData.roundMode == MODE_FLOOR) {
    Cast(outLocal[outOffset], xLocal[inOffset].ReinterpretCast<half>(), RoundMode::CAST_FLOOR, dataCount);
  } else if (runTilingData.roundMode == MODE_CEIL) {
    Cast(outLocal[outOffset], xLocal[inOffset].ReinterpretCast<half>(), RoundMode::CAST_CEIL, dataCount);
  } else if (runTilingData.roundMode == MODE_TRUNC) {
    Cast(outLocal[outOffset], xLocal[inOffset].ReinterpretCast<half>(), RoundMode::CAST_TRUNC, dataCount);
  }
  PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AscendQuantV2Base<T>::MoveOutY(LocalTensor<int8_t>& outLocal, GlobalTensor<int8_t>& outGm,
    LocalTensor<int8_t>& outTempLocal, AscendQuantV2TilingData &runTilingData,
    int64_t yN, int64_t yLen, int64_t yOutOffset) {
#if defined(__DAV_M200__)
  DataCopyParams copyParams;
  copyParams.blockCount = 1;
  copyParams.blockLen = (yLen * sizeof(int8_t) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  copyParams.dstStride = 0;
  copyParams.srcStride = 0;
  __ubuf__ int8_t* outTempBuf = (__ubuf__ int8_t*)outTempLocal.GetPhyAddr();
  __ubuf__ int8_t* outBuf = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
  int64_t totalDim1 = BLOCK_SIZE / yLen;
  if (yLen * sizeof(int8_t) % BLOCK_SIZE == 0 || yN <= totalDim1) {
    for (int64_t nIdx = 0; nIdx < yN; ++nIdx) {
      DataCopy(outGm[yOutOffset + nIdx * runTilingData.dim1], outLocal[nIdx * runTilingData.baseLen], copyParams);
      PipeBarrier<PIPE_MTE3>();
    }
  } else {
    int64_t lastDim = totalDim1 == 0 ? yN - 1 : yN - totalDim1;
    for (int64_t nIdx = 0; nIdx < lastDim; ++nIdx) {
      if (runTilingData.blockAxis == 0) {
        DataCopy(outGm[yOutOffset + nIdx * runTilingData.dim1], outLocal[nIdx * runTilingData.baseLen], copyParams);
        PipeBarrier<PIPE_MTE3>();
      } else {
        DataCopy(outGm[yOutOffset + nIdx * runTilingData.dim1], outLocal[nIdx * runTilingData.baseLen],
                 {1, static_cast<uint16_t>(yLen / BLOCK_SIZE), 0, 0});
        PipeBarrier<PIPE_ALL>();
        for (uint64_t i = 0; i < BLOCK_SIZE; i++) {
          *(outTempBuf + i) = *(outBuf + nIdx * runTilingData.baseLen + yLen - BLOCK_SIZE + i);
          }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm[yOutOffset + nIdx * runTilingData.dim1 + yLen - BLOCK_SIZE], outTempLocal, {1, 1, 0, 0});
        PipeBarrier<PIPE_MTE3>();
      }
    }
    int64_t yTempOffset = yOutOffset + (yN - 1) * runTilingData.dim1;
    int64_t yLastOffset = yTempOffset + yLen;
    int64_t outTempOffset = (yN - 1) * runTilingData.baseLen;
    int64_t outLastOffset = outTempOffset + yLen;
    PipeBarrier<PIPE_ALL>();
    // back to yN - totalDim1
    if (yLen / BLOCK_SIZE > 0) {
      DataCopy(outGm[yTempOffset], outLocal[outTempOffset], {1, static_cast<uint16_t>(yLen / BLOCK_SIZE), 0, 0});
      PipeBarrier<PIPE_ALL>();
      for (uint64_t i = 0; i < BLOCK_SIZE; i++) {
        *(outTempBuf + i) = *(outBuf + outLastOffset - BLOCK_SIZE + i);
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(outGm[yLastOffset - BLOCK_SIZE], outTempLocal, {1, 1, 0, 0});
    } else {
      int64_t modeLen = BLOCK_SIZE % yLen;
      if (modeLen != 0) {
        for (uint64_t i = 0; i < modeLen; i++) {
          *(outTempBuf + i) = *(outBuf + outTempOffset - totalDim1 * runTilingData.baseLen +
                                runTilingData.dim1 - modeLen + i);
        }
        for (uint64_t i = 0; i < totalDim1; i++) {
          for (uint64_t j = 0; j < runTilingData.dim1; j++) {
            // outTempOffset - totalDim1 * runTilingData.baseLen + runTilingData.baseLen + i * runTilingData.baseLen
            *(outTempBuf + modeLen + i * runTilingData.dim1 + j) =
              *(outBuf + outTempOffset + (i + 1 - totalDim1) * runTilingData.baseLen + j);
          }
        }
      } else {
        for (uint64_t i = 0; i < totalDim1; i++) {
          for (uint64_t j = 0; j < runTilingData.dim1; j++) {
            *(outTempBuf + i * runTilingData.dim1 + j) =
              *(outBuf + outTempOffset + (i + 1 - totalDim1) * runTilingData.baseLen + j);
          }
        }
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(outGm[yLastOffset - BLOCK_SIZE], outTempLocal, {1, 1, 0, 0});
    }
  }
#endif
}

template <typename T>
__aicore__ inline int64_t AscendQuantV2Base<T>::CeilAlign(int64_t a, int64_t b) {
  if (b == 0) {
    return 0;
  }
  return (a + b - 1) / b * b;
}
} // namespace AscendQuantV2

#endif
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
 * \file ascend_quant_v2_fp32.h
 * \brief
 */

#ifndef ASCEND_QUANT_V2_FP32_H
#define ASCEND_QUANT_V2_FP32_H

#include "ascend_quant_v2.h"

namespace AscendQuantV2 {
using namespace AscendC;
template <typename T>
class AscendQuantV2PerChannelFP32 : public AscendQuantV2Base<T> {
public:
  __aicore__ inline AscendQuantV2PerChannelFP32(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
                              const AscendQuantV2TilingData* tilingData) {
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));

    this->ParseTilingData(tilingData, tilingData_);
    this->ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockLen_);

    // calc n size to alloc queue
    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_.baseN * tilingData_.baseLen * sizeof(T));
    pipe_.InitBuffer(inQueueScale_, bufferNum_, tilingData_.baseLen * sizeof(T));
    if (tilingData_.hasOffset) {
      pipe_.InitBuffer(inQueueOffset_, bufferNum_, tilingData_.baseLen * sizeof(T));
    }
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_.baseN * tilingData_.baseLen);
#if defined(__DAV_M200__)
    pipe_.InitBuffer(outTempQueue_, bufferNum_, this->BLOCK_SIZE);
#endif
  }

  __aicore__ inline void Process() {
    if (blockIdx_ >= tilingData_.numCore || blockN_ == 0) {
      return;
    }
    if (tilingData_.blockAxis == 0) {
      gmXOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.dim1;
      gmSOffset_ = 0;
    } else if (tilingData_.blockAxis == 1) {
      gmXOffset_ = blockIdx_ * tilingData_.blockFactor;
      gmSOffset_ = blockIdx_ * tilingData_.blockFactor;
    } else {
      gmXOffset_ = (blockIdx_ / tilingData_.blockUnion * tilingData_.dim1 +
                    blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor);
      gmSOffset_ = blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor;
    }

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_.baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_.baseLen;
    for (int64_t i = 0; i < lenLoopNum; ++i) {
      CopyInScaleAndOffset(tilingData_.baseLen, gmSOffset_ + i * tilingData_.baseLen);
      CopyXAndCompute(tilingData_.baseLen, gmXOffset_ + i * tilingData_.baseLen);
      FreeScaleAndOffset();
    }
    if (lenLoopTail != 0) {
      CopyInScaleAndOffset(lenLoopTail, gmSOffset_ + lenLoopNum * tilingData_.baseLen);
      CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_.baseLen);
      FreeScaleAndOffset();
    }
  }

private:
  __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset) {
    int64_t nLoopNum = blockN_ / tilingData_.baseN;
    int64_t nLoopTail = blockN_ % tilingData_.baseN;
    int64_t xoffset = offset;
    for (int64_t nIdx = 0; nIdx < nLoopNum; ++nIdx) {
      xoffset = offset + nIdx * tilingData_.baseN * tilingData_.dim1;
      CopyInX(tilingData_.baseN, dataCount, xoffset);
      Compute(tilingData_.baseN, dataCount);
      CopyOutY(tilingData_.baseN, dataCount, xoffset);
    }
    if (nLoopTail != 0) {
      xoffset = offset + nLoopNum * tilingData_.baseN * tilingData_.dim1;
      CopyInX(nLoopTail, dataCount, xoffset);
      Compute(nLoopTail, dataCount);
      CopyOutY(nLoopTail, dataCount, xoffset);
    }
  }

  __aicore__ inline void CopyInScaleAndOffset(int64_t sLen, int64_t sInOffset) {
    LocalTensor<T> sLocal = inQueueScale_.AllocTensor<T>();
#if defined(__DAV_M200__)
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = (sLen * sizeof(T) + this->BLOCK_SIZE - 1) / this->BLOCK_SIZE;
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopy(sLocal, scaleGm_[sInOffset], copyParams);
#else
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopyPad(sLocal, scaleGm_[sInOffset], copyParams, {false, 0, 0, 0});
#endif
    inQueueScale_.EnQue(sLocal);

    if (tilingData_.hasOffset != 0) {
      LocalTensor<T> oLocal = inQueueOffset_.AllocTensor<T>();
#if defined(__DAV_M200__)
      DataCopy(oLocal, offsetGm_[sInOffset], copyParams);
#else
      DataCopyPad(oLocal, offsetGm_[sInOffset], copyParams, {false, 0, 0, 0});
#endif
      inQueueOffset_.EnQue(oLocal);
    }
  }
  __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset) {
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
#if defined(__DAV_M200__)
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = (xLen * sizeof(T) + this->BLOCK_SIZE - 1) / this->BLOCK_SIZE;
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    for (int64_t nIdx = 0; nIdx < xN; ++nIdx) {
      DataCopy(xLocal[nIdx * tilingData_.baseLen], xGm_[xInOffset + nIdx * tilingData_.dim1], copyParams);
    }
#else
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim1, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);
#endif
    inQueueX_.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset) {
    LocalTensor<int8_t> outLocal = outQueueY_.DeQue<int8_t>();
#if defined(__DAV_M200__)
    LocalTensor<int8_t> outTemp = outTempQueue_.AllocTensor<int8_t>();
    this->MoveOutY(outLocal, yGm_, outTemp, tilingData_, yN, yLen, yOutOffset);
    outTempQueue_.FreeTensor(outTemp);
#else
    if (ORIG_DTYPE_Y == DT_INT4) {
      yOutOffset = yOutOffset >> 1;
    }
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yN, yLen, tilingData_.dim1, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
#endif
    outQueueY_.FreeTensor(outLocal);
  }

  __aicore__ inline void Compute(int64_t nRow, int64_t dataCount) {
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<T> sLocal = inQueueScale_.DeQue<T>();
    LocalTensor<int8_t> outLocal = outQueueY_.AllocTensor<int8_t>();
    LocalTensor<T> oLocal;
    if (tilingData_.hasOffset != 0) {
      oLocal = inQueueOffset_.DeQue<T>();
    }
    for (int64_t i = 0; i < nRow; ++i) {
      Mul(xLocal[i * tilingData_.baseLen], xLocal[i * tilingData_.baseLen], sLocal, dataCount);
      PipeBarrier<PIPE_V>();
      if (tilingData_.sqrtMode != 0) {
        Mul(xLocal[i * tilingData_.baseLen], xLocal[i * tilingData_.baseLen], sLocal, dataCount);
        PipeBarrier<PIPE_V>();
      }
      if (tilingData_.hasOffset != 0) {
        Add(xLocal[i * tilingData_.baseLen], xLocal[i * tilingData_.baseLen], oLocal, dataCount);
        PipeBarrier<PIPE_V>();
      }
      this->CastOut(tilingData_, dataCount, i * tilingData_.baseLen, i * tilingData_.baseLen, outLocal, xLocal);
    }
    inQueueScale_.EnQue(sLocal);
    if (tilingData_.hasOffset != 0) {
      inQueueOffset_.EnQue(oLocal);
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue<int8_t>(outLocal);
  }

  __aicore__ inline void FreeScaleAndOffset() {
  LocalTensor<T> sLocal = inQueueScale_.DeQue<T>();
    inQueueScale_.FreeTensor(sLocal);
    if (tilingData_.hasOffset) {
      LocalTensor<T> oLocal = inQueueOffset_.DeQue<T>();
      inQueueOffset_.FreeTensor(oLocal);
    }
  }

private:
  constexpr static int32_t bufferNum_ = 1;
  TPipe pipe_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
  TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
  GlobalTensor<T> xGm_;
  GlobalTensor<T> scaleGm_;
  GlobalTensor<T> offsetGm_;
  GlobalTensor<int8_t> yGm_;
#if defined(__DAV_M200__)
  TQue<QuePosition::VECOUT, bufferNum_> outTempQueue_;
#endif
  AscendQuantV2TilingData tilingData_;
  int32_t blockIdx_ = 0;
  int64_t gmXOffset_ = 0;
  int64_t gmSOffset_ = 0;
  int64_t blockN_ = 1;
  int64_t blockLen_ = 1;
};


template <typename T>
class AscendQuantV2PerTensorFP32 : public AscendQuantV2Base<T> {
public:
  __aicore__ inline AscendQuantV2PerTensorFP32(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
                              const AscendQuantV2TilingData* tilingData) {
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));

    this->ParseTilingData(tilingData, tilingData_);
    this->ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockLen_);

    // calc n size to alloc queue
    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_.baseN * tilingData_.baseLen * sizeof(T));
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_.baseN * tilingData_.baseLen);
#if defined(__DAV_M200__)
    pipe_.InitBuffer(outTempQueue_, bufferNum_, this->BLOCK_SIZE);
#endif
  }

  __aicore__ inline void Process() {
    if (blockIdx_ >= tilingData_.numCore || blockN_ == 0) {
      return;
    }
    if (tilingData_.blockAxis == 0) {
      gmXOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.dim1;
    } else if (tilingData_.blockAxis == 1) {
      gmXOffset_ = blockIdx_ * tilingData_.blockFactor;
    }

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_.baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_.baseLen;
    for (int64_t i = 0; i < lenLoopNum; ++i) {
      CopyXAndCompute(tilingData_.baseLen, gmXOffset_ + i * tilingData_.baseLen);
    }
    if (lenLoopTail != 0) {
      CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_.baseLen);
    }
  }

private:
  __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset) {
    int64_t nLoopNum = blockN_ / tilingData_.baseN;
    int64_t nLoopTail = blockN_ % tilingData_.baseN;
    int64_t xoffset = offset;
    for (int64_t nIdx = 0; nIdx < nLoopNum; ++nIdx) {
      xoffset = offset + nIdx * tilingData_.baseN * tilingData_.dim1;
      CopyInX(tilingData_.baseN, dataCount, xoffset);
      Compute(tilingData_.baseN, dataCount);
      CopyOutY(tilingData_.baseN, dataCount, xoffset);
    }
    if (nLoopTail != 0) {
      xoffset = offset + nLoopNum * tilingData_.baseN * tilingData_.dim1;
      CopyInX(nLoopTail, dataCount, xoffset);
      Compute(nLoopTail, dataCount);
      CopyOutY(nLoopTail, dataCount, xoffset);
    }
  }

  __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset) {
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
#if defined(__DAV_M200__)
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = (xLen * sizeof(T) + this->BLOCK_SIZE - 1) / this->BLOCK_SIZE;
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    for (int64_t nIdx = 0; nIdx < xN; ++nIdx) {
      DataCopy(xLocal[nIdx * tilingData_.baseLen], xGm_[xInOffset + nIdx * tilingData_.dim1], copyParams);
    }
#else
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim1, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);
#endif
    inQueueX_.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset) {
    LocalTensor<int8_t> outLocal = outQueueY_.DeQue<int8_t>();
#if defined(__DAV_M200__)
    LocalTensor<int8_t> outTemp = outTempQueue_.AllocTensor<int8_t>();
    this->MoveOutY(outLocal, yGm_, outTemp, tilingData_, yN, yLen, yOutOffset);
    outTempQueue_.FreeTensor(outTemp);
#else
    if (ORIG_DTYPE_Y == DT_INT4) {
      yOutOffset = yOutOffset >> 1;
    }
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yN, yLen, tilingData_.dim1, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
#endif
    outQueueY_.FreeTensor(outLocal);
  }

  __aicore__ inline void Compute(int64_t nRow, int64_t dataCount) {
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    scale_ = scaleGm_.GetValue(0);
    LocalTensor<int8_t> outLocal = outQueueY_.AllocTensor<int8_t>();
    LocalTensor<T> oLocal;
    if (tilingData_.hasOffset != 0) {
      offset_ = offsetGm_.GetValue(0);
    }
    for (int64_t i = 0; i < nRow; ++i) {
      Muls(xLocal[i * tilingData_.baseLen], xLocal[i * tilingData_.baseLen], scale_, dataCount);
      PipeBarrier<PIPE_V>();
      if (tilingData_.sqrtMode != 0) {
        Muls(xLocal[i * tilingData_.baseLen], xLocal[i * tilingData_.baseLen], scale_, dataCount);
        PipeBarrier<PIPE_V>();
      }
      if (tilingData_.hasOffset != 0) {
        Adds(xLocal[i * tilingData_.baseLen], xLocal[i * tilingData_.baseLen], offset_, dataCount);
        PipeBarrier<PIPE_V>();
      }
      this->CastOut(tilingData_, dataCount, i * tilingData_.baseLen, i * tilingData_.baseLen, outLocal, xLocal);
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue<int8_t>(outLocal);
  }

private:
  constexpr static int32_t bufferNum_ = 1;
  TPipe pipe_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
  TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
  GlobalTensor<T> xGm_, scaleGm_, offsetGm_;
  GlobalTensor<int8_t> yGm_;
#if defined(__DAV_M200__)
  TQue<QuePosition::VECOUT, bufferNum_> outTempQueue_;
#endif
  AscendQuantV2TilingData tilingData_;
  float scale_ = 0;
  float offset_ = 0;
  int32_t blockIdx_ = 0;
  int64_t gmXOffset_ = 0;
  int64_t blockN_ = 1;
  int64_t blockLen_ = 1;
};


template <typename T>
class AscendQuantV2PerHead : public AscendQuantV2Base<T> {
 public:
  __aicore__ inline AscendQuantV2PerHead(){};

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
                              const AscendQuantV2TilingData* tilingData) {
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));

    this->ParseTilingData(tilingData, tilingData_);
    ParsePerHeadCoreBlocks();

    auto xDataLen = tilingData_.baseN * tilingData_.baseLen * sizeof(T);
    auto yDataLen = tilingData_.baseN * tilingData_.baseLen * sizeof(uint8_t);
    auto paramDataLen = this->CeilAlign(tilingData_.baseN * sizeof(T), this->BLOCK_SIZE);

    pipe_.InitBuffer(inQueueX_, bufferNum_, xDataLen);
    pipe_.InitBuffer(outQueueY_, bufferNum_, yDataLen);
    pipe_.InitBuffer(inQueueScale_, bufferNum_, paramDataLen);
    pipe_.InitBuffer(bufX_, tilingData_.baseLen * sizeof(float));
    pipe_.InitBuffer(bufScale_, this->CeilAlign(tilingData_.baseN * sizeof(float), this->BLOCK_SIZE));
    if (tilingData_.hasOffset) {
      pipe_.InitBuffer(inQueueOffset_, bufferNum_, paramDataLen);
      pipe_.InitBuffer(bufOffset_, this->CeilAlign(tilingData_.baseN * sizeof(float), this->BLOCK_SIZE));
    }
  }

  __aicore__ inline void Process() {
    if (blockIdx_ >= tilingData_.numCore || blockN_ == 0) {
      return;
    }
    CalcOffset();
    if (tilingData_.hasOffset != 0) {
      if (tilingData_.sqrtMode != 0) {
        ProcessLoop<true, true>();
      } else {
        ProcessLoop<false, true>();
      }
    } else {
      if (tilingData_.sqrtMode != 0) {
        ProcessLoop<true, false>();
      } else {
        ProcessLoop<false, false>();
      }
    }
  }

 private:
  __aicore__ inline void ParsePerHeadCoreBlocks() {
    if (tilingData_.blockAxis == 0) {
      // blockFactor is in [1, S]
      if (blockIdx_ == tilingData_.numCore - 1) {
        blockS_ = tilingData_.blockTailFactor;
      } else {
        blockS_ = tilingData_.blockFactor;
      }
      blockN_ = tilingData_.dim1;
      blockLen_ = tilingData_.dim2;
    } else if (tilingData_.blockAxis == 1) {
      // blockFactor is in [1, N], blockUnion is in [1, N / blockFactor]
      if (blockIdx_ % tilingData_.blockUnion == tilingData_.blockUnion - 1) {
        blockN_ = tilingData_.blockTailFactor;
      } else {
        blockN_ = tilingData_.blockFactor;
      }
      blockLen_ = tilingData_.dim2;
    } else {
      // blockFactor is in [1, D]
      blockN_ = 1;
      if (blockIdx_ % tilingData_.blockUnion == tilingData_.blockUnion - 1) {
        blockLen_ = tilingData_.blockTailFactor;
      } else {
        blockLen_ = tilingData_.blockFactor;
      }
    }
  }

  __aicore__ inline void CalcOffset() {
    if (tilingData_.blockAxis == 0) {
      // only split axis 0
      gmXOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.dim1 * tilingData_.dim2;
      gmSOffset_ = 0;
    } else if (tilingData_.blockAxis == 1) {
      // only split axis 1, blockUnion means factor per block on split axis
      gmXOffset_ = blockIdx_ / tilingData_.blockUnion * tilingData_.dim1 * tilingData_.dim2 +
                   blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor * tilingData_.dim2;
      gmSOffset_ = blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor;
    } else {
      gmXOffset_ = (blockIdx_ / tilingData_.blockUnion * tilingData_.dim2 +
                    blockIdx_ % tilingData_.blockUnion * tilingData_.blockFactor);
      gmSOffset_ = blockIdx_ / tilingData_.blockUnion;
    }
  }

  template <bool SQRT_MODE, bool HAS_OFFSET>
  __aicore__ inline void ProcessLoop() {
    for (int64_t i = 0; i < blockS_; ++i) {
      ProcessParamLoop<SQRT_MODE, HAS_OFFSET>();
      gmXOffset_ += tilingData_.dim1 * tilingData_.dim2;
    }
  }

  template <bool SQRT_MODE, bool HAS_OFFSET>
  __aicore__ inline void ProcessParamLoop() {
    // main loop with column, for scale and offset only need copy once
    auto nLoopLen = tilingData_.baseN;
    auto nLoopNum = blockN_ / nLoopLen;
    auto nLoopTail = blockN_ % nLoopLen;

    // scale allows start from begin on each core
    int64_t baseSOffset = gmSOffset_;
    int64_t baseXOffset = gmXOffset_;

    // param loop
    for (int64_t i = 0; i < nLoopNum; ++i) {
      ProcessParamOneLoop<SQRT_MODE, HAS_OFFSET>(nLoopLen, baseSOffset, baseXOffset);
      baseXOffset += nLoopLen * tilingData_.dim2;
      baseSOffset += tilingData_.baseN;
    }
    if (nLoopTail != 0) {
      ProcessParamOneLoop<SQRT_MODE, HAS_OFFSET>(nLoopTail, baseSOffset, baseXOffset);
    }
  }

  template <bool SQRT_MODE, bool HAS_OFFSET>
  __aicore__ inline void ProcessParamOneLoop(int64_t nLoopLen, int64_t baseSOffset, int64_t baseXOffset) {
    // copy in scale
    CopyInParam(inQueueScale_, scaleGm_, nLoopLen, baseSOffset % tilingData_.dim1);
    auto scaleLocal = inQueueScale_.DeQue<T>();
    auto bufScale = bufScale_.Get<float>();
    ProbeCast(bufScale, scaleLocal, nLoopLen);
    ProbePipeS();
    if constexpr (HAS_OFFSET) {
      CopyInParam(inQueueOffset_, offsetGm_, nLoopLen, baseSOffset % tilingData_.dim1);
      auto offsetLocal = inQueueOffset_.DeQue<T>();
      auto bufOffset = bufOffset_.Get<float>();
      ProbeCast(bufOffset, offsetLocal, nLoopLen);
      ProbePipeS();
      ProcessInputLoop<SQRT_MODE, HAS_OFFSET>(nLoopLen, baseXOffset, bufScale, bufOffset);
      inQueueOffset_.FreeTensor(offsetLocal);
    } else {
      LocalTensor<float> bufOffset;
      ProcessInputLoop<SQRT_MODE, HAS_OFFSET>(nLoopLen, baseXOffset, bufScale, bufOffset);
    }

    inQueueScale_.FreeTensor(scaleLocal);
  }

  template <bool SQRT_MODE, bool HAS_OFFSET>
  __aicore__ inline void ProcessInputLoop(int64_t nLoopLen, int64_t baseXOffset,
                                          LocalTensor<float> &scaleLocal, LocalTensor<float> &offsetLocal) {
    auto loopLen = tilingData_.baseLen;
    auto loopNum = blockLen_ / loopLen;
    auto loopTail = blockLen_ % loopLen;
    for (auto i = 0; i < loopNum; ++i) {
      ProcessInputOneLoop<SQRT_MODE, HAS_OFFSET>(nLoopLen, loopLen, baseXOffset, scaleLocal, offsetLocal);
      baseXOffset += loopLen;
    }
    if (loopTail != 0) {
      ProcessInputOneLoop<SQRT_MODE, HAS_OFFSET>(nLoopLen, loopTail, baseXOffset, scaleLocal, offsetLocal);
    }
  }

  template <bool SQRT_MODE, bool HAS_OFFSET>
  __aicore__ inline void ProcessInputOneLoop(int64_t nLoopLen, int64_t loopLen, int64_t baseXOffset,
                                             LocalTensor<float> &scaleLocal,
                                             LocalTensor<float> &offsetLocal) {
    CopyInX(nLoopLen, loopLen, baseXOffset);
    Compute<SQRT_MODE, HAS_OFFSET>(nLoopLen, loopLen, scaleLocal, offsetLocal);
    CopyOutY(nLoopLen, loopLen, baseXOffset);
  }

  __aicore__ inline void CopyInParam(TQue<QuePosition::VECIN, 1> &inQueue,
                                     GlobalTensor<T> &inGm,
                                     int64_t paramLen, int64_t paramOffset) {
    auto paramLocal = inQueue.AllocTensor<T>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(paramLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(paramLocal, inGm[paramOffset], copyParams, {false, 0, 0, 0});
    inQueue.EnQue(paramLocal);
  }

  __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset) {
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();

    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim2, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset) {
    LocalTensor<int8_t> outLocal = outQueueY_.DeQue<int8_t>();
    if constexpr (ORIG_DTYPE_Y == DT_INT4) {
      yOutOffset = yOutOffset >> 1;
    }
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yN, yLen, tilingData_.dim2, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
  }

  __aicore__ inline void PipeSV() {
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);
  }

  __aicore__ inline void ProbeCast(LocalTensor<float> &buf, const LocalTensor<T> &localTensor, int64_t dataCount) {
    if constexpr (IsSameType<T, float>::value) {
      buf = localTensor;
    } else {
      Cast(buf, localTensor, RoundMode::CAST_NONE, dataCount);
    }
  }

  __aicore__ inline void ProbePipeS() {
    if constexpr (IsSameType<T, float>::value) {
      // if no need cast, Scalar wait MTE2
      event_t eventMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
      SetFlag<HardEvent::MTE2_S>(eventMS);
      WaitFlag<HardEvent::MTE2_S>(eventMS);
    } else {
      // if need cast, Scalar wait Vector
      event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      SetFlag<HardEvent::V_S>(eventVS);
      WaitFlag<HardEvent::V_S>(eventVS);
    }
  }

  template <bool SQRT_MODE, bool HAS_OFFSET>
  __aicore__ inline void Compute(int64_t nRow, int64_t dataCount,
                                 LocalTensor<float> &scaleLocal,
                                 LocalTensor<float> &offsetLocal) {
    auto inLocal = inQueueX_.DeQue<T>();
    auto outLocal = outQueueY_.AllocTensor<int8_t>();
    auto bufX = bufX_.Get<float>();
    int64_t offset = 0;
    for (int64_t i = 0; i < nRow; ++i) {
      auto scaleTmp = scaleLocal.GetValue(i);
      PipeSV();
      ProbeCast(bufX, inLocal[offset], dataCount);
      PipeBarrier<PIPE_V>();
      Muls(bufX, bufX, scaleTmp, dataCount);
      PipeBarrier<PIPE_V>();
      if constexpr (SQRT_MODE) {
        Muls(bufX, bufX, scaleTmp, dataCount);
        PipeBarrier<PIPE_V>();
      }
      if constexpr (HAS_OFFSET) {
        auto offsetTmp = offsetLocal.GetValue(i);
        PipeSV();
        Adds(bufX, bufX, offsetTmp, dataCount);
        PipeBarrier<PIPE_V>();
      }
      this->CastOut(tilingData_, dataCount, offset, 0, outLocal, bufX);
      offset += tilingData_.baseLen;
    }
    inQueueX_.FreeTensor(inLocal);
    outQueueY_.EnQue<int8_t>(outLocal);
  }

 private:
  constexpr static int32_t bufferNum_ = 1;
  TPipe pipe_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
  TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
  TBuf<TPosition::VECCALC> bufX_, bufScale_, bufOffset_;
  GlobalTensor<T> xGm_;
  GlobalTensor<T> scaleGm_;
  GlobalTensor<T> offsetGm_;
  GlobalTensor<int8_t> yGm_;
  AscendQuantV2TilingData tilingData_;
  int32_t blockIdx_ = 0;
  int64_t gmXOffset_ = 0;
  int64_t gmSOffset_ = 0;
  int64_t blockS_ = 1;
  int64_t blockN_ = 1;
  int64_t blockLen_ = 1;
};

} // namespace
#endif
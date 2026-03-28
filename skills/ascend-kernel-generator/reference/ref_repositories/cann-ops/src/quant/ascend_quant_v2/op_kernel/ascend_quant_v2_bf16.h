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
 * \file ascend_quant_v2_bf16.h
 * \brief
 */

#ifndef ASCEND_QUANT_V2_BF16_H
#define ASCEND_QUANT_V2_BF16_H

#include "ascend_quant_v2.h"

namespace AscendQuantV2 {
using namespace AscendC;
template <typename T>
class AscendQuantV2PerChannnelBF16 : public AscendQuantV2Base<T> {
public:
  __aicore__ inline AscendQuantV2PerChannnelBF16(){};
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
    pipe_.InitBuffer(castBufX_, tilingData_.baseLen * sizeof(float));
    pipe_.InitBuffer(castBufScale_, tilingData_.baseLen * sizeof(float));
    if (tilingData_.hasOffset) {
      pipe_.InitBuffer(inQueueOffset_, bufferNum_, tilingData_.baseLen * sizeof(T));
      pipe_.InitBuffer(castBufOffset_, tilingData_.baseLen * sizeof(float));
    }
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_.baseN * tilingData_.baseLen);
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
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;

    LocalTensor<T> sLocal = inQueueScale_.AllocTensor<T>();
    LocalTensor<float> castSLocal = castBufScale_.Get<float>();
    DataCopyPad(sLocal, scaleGm_[sInOffset], copyParams, {false, 0, 0, 0});
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
    Cast(castSLocal, sLocal, RoundMode::CAST_NONE, sLen);
    PipeBarrier<PIPE_V>();
    inQueueScale_.EnQue(sLocal);

    if (tilingData_.hasOffset != 0) {
      LocalTensor<T> oLocal = inQueueOffset_.AllocTensor<T>();
      LocalTensor<float> castOLocal = castBufOffset_.Get<float>();
      DataCopyPad(oLocal, offsetGm_[sInOffset], copyParams, {false, 0, 0, 0});
      event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(eventID);
      WaitFlag<HardEvent::MTE2_V>(eventID);
      Cast(castOLocal, oLocal, RoundMode::CAST_NONE, sLen);
      PipeBarrier<PIPE_V>();
      inQueueOffset_.EnQue(oLocal);
    }
  }

  __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset) {
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    this->GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim1, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, {false, 0, 0, 0});
    inQueueX_.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset) {
    if (ORIG_DTYPE_Y == DT_INT4) {
      yOutOffset = yOutOffset >> 1;
    }
    LocalTensor<int8_t> outLocal = outQueueY_.DeQue<int8_t>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yN, yLen, tilingData_.dim1, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
  }

  __aicore__ inline void Compute(int64_t nRow, int64_t dataCount) {
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<float> castXLocal = castBufX_.Get<float>();
    LocalTensor<float> sLocal = castBufScale_.Get<float>();
    LocalTensor<int8_t> outLocal = outQueueY_.AllocTensor<int8_t>();
    LocalTensor<float> oLocal;
    if (tilingData_.hasOffset != 0) {
      oLocal = castBufOffset_.Get<float>();
    }
    for (int64_t i = 0; i < nRow; ++i) {
      Cast(castXLocal, xLocal[i * tilingData_.baseLen], RoundMode::CAST_NONE, dataCount);
      PipeBarrier<PIPE_V>();
      Mul(castXLocal, castXLocal, sLocal, dataCount);
      PipeBarrier<PIPE_V>();
      if (tilingData_.sqrtMode != 0) {
        Mul(castXLocal, castXLocal, sLocal, dataCount);
        PipeBarrier<PIPE_V>();
      }
      if (tilingData_.hasOffset != 0) {
        Add(castXLocal, castXLocal, oLocal, dataCount);
        PipeBarrier<PIPE_V>();
      }
      this->CastOut(tilingData_, dataCount, i * tilingData_.baseLen, 0, outLocal, castXLocal);
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
  TBuf<TPosition::VECCALC> castBufX_;
  TBuf<TPosition::VECCALC> castBufScale_;
  TBuf<TPosition::VECCALC> castBufOffset_;
  TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
  GlobalTensor<T> xGm_;
  GlobalTensor<T> scaleGm_;
  GlobalTensor<T> offsetGm_;
  GlobalTensor<int8_t> yGm_;
  AscendQuantV2TilingData tilingData_;
  int32_t blockIdx_ = 0;
  int64_t gmXOffset_ = 0;
  int64_t gmSOffset_ = 0;
  int64_t blockN_ = 1;
  int64_t blockLen_ = 1;
};


template <typename T>
class AscendQuantV2PerTensorBF16 : public AscendQuantV2Base<T> {
public:
  __aicore__ inline AscendQuantV2PerTensorBF16(){};
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
    // scale only need one block
    pipe_.InitBuffer(inQueueScale_, bufferNum_, blockElemBF16_ * sizeof(T));
    pipe_.InitBuffer(castBufX_, tilingData_.baseLen * sizeof(float));
    if (tilingData_.hasOffset) {
      // offset only need one block
      pipe_.InitBuffer(inQueueOffset_, bufferNum_, blockElemBF16_ * sizeof(T));
    }
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_.baseN * tilingData_.baseLen);
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
      CopyInScaleAndOffset();
      CopyXAndCompute(tilingData_.baseLen, gmXOffset_ + i * tilingData_.baseLen);
    }
    if (lenLoopTail != 0) {
      CopyInScaleAndOffset();
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

  __aicore__ inline void CopyInScaleAndOffset() {
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<T> sLocal = inQueueScale_.AllocTensor<T>();
    DataCopyPad(sLocal, scaleGm_, copyParams, padParams);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    scale_ = ToFloat(sLocal.GetValue(0));
    PipeBarrier<PIPE_V>();
    inQueueScale_.FreeTensor(sLocal);

    if (tilingData_.hasOffset != 0) {
      LocalTensor<T> oLocal = inQueueOffset_.AllocTensor<T>();
      DataCopyPad(oLocal, offsetGm_, copyParams, padParams);
      SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

      offset_ = ToFloat(oLocal.GetValue(0));
      PipeBarrier<PIPE_V>();
      inQueueOffset_.FreeTensor(oLocal);
    }
  }

  __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset) {
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    this->GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim1, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, {false, 0, 0, 0});
    inQueueX_.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset) {
    if (ORIG_DTYPE_Y == DT_INT4) {
      yOutOffset = yOutOffset >> 1;
    }
    LocalTensor<int8_t> outLocal = outQueueY_.DeQue<int8_t>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yN, yLen, tilingData_.dim1, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
  }

  __aicore__ inline void Compute(int64_t nRow, int64_t dataCount) {
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<float> castXLocal = castBufX_.Get<float>();
    LocalTensor<int8_t> outLocal = outQueueY_.AllocTensor<int8_t>();

    for (int64_t i = 0; i < nRow; ++i) {
      Cast(castXLocal, xLocal[i * tilingData_.baseLen], RoundMode::CAST_NONE, dataCount);
      PipeBarrier<PIPE_V>();
      Muls(castXLocal, castXLocal, scale_, dataCount);
      PipeBarrier<PIPE_V>();
      if (tilingData_.sqrtMode != 0) {
        Muls(castXLocal, castXLocal, scale_, dataCount);
        PipeBarrier<PIPE_V>();
      }
      if (tilingData_.hasOffset != 0) {
        Adds(castXLocal, castXLocal, offset_, dataCount);
        PipeBarrier<PIPE_V>();
      }
      this->CastOut(tilingData_, dataCount, i * tilingData_.baseLen, 0, outLocal, castXLocal);
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue<int8_t>(outLocal);
  }

private:
  constexpr static int32_t bufferNum_ = 1;
  constexpr static int32_t blockElemBF16_ = 16;
  TPipe pipe_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
  TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
  TBuf<TPosition::VECCALC> castBufX_;
  TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
  GlobalTensor<T> xGm_;
  GlobalTensor<T> scaleGm_;
  GlobalTensor<T> offsetGm_;
  GlobalTensor<int8_t> yGm_;
  AscendQuantV2TilingData tilingData_;
  float scale_ = 0;
  float offset_ = 0;
  int32_t blockIdx_ = 0;
  int64_t gmXOffset_ = 0;
  int64_t blockN_ = 1;
  int64_t blockLen_ = 1;
};
} // namespace
#endif
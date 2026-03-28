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
 * \file avg_pool3d_common.h
 * \brief
 */

#ifndef AVG_POOL3D_COMMON_H_
#define AVG_POOL3D_COMMON_H_

#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline int64_t Max(int64_t a, int64_t b) {
  return a > b ? a : b;
}

__aicore__ inline int64_t Min(int64_t a, int64_t b) {
  return a < b ? a : b;
}

struct ShapeAvgPool3d {
  int64_t N, C, D, H, W;
  int64_t strideN, strideC, strideD, strideH, strideW;

  __aicore__ inline ShapeAvgPool3d() {}

  __aicore__ inline ShapeAvgPool3d(int64_t N, int64_t C, int64_t D, int64_t H, int64_t W)
    : N(N), C(C), D(D), H(H), W(W),
      strideN(C * D * H * W), strideC(D * H * W), strideD(H * W), strideH(W), strideW(1) {}
};

struct PoolParameter {
  int64_t kernelD;
  int64_t kernelH;
  int64_t kernelW;
  int64_t strideD;
  int64_t strideH;
  int64_t strideW;
  int64_t padD;
  int64_t padH;
  int64_t padW;
  int64_t divisorOverride;
  int64_t countIncludePad;

  __aicore__ inline PoolParameter() {}

  __aicore__ inline PoolParameter(
    int64_t kernelD, int64_t kernelH, int64_t kernelW, int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t padD, int64_t padH, int64_t padW, int64_t divisorOverride, int64_t countIncludePad)
      : kernelD(kernelD), kernelH(kernelH), kernelW(kernelW),
        strideD(strideD), strideH(strideH), strideW(strideW),
        padD(padD), padH(padH), padW(padW),
        divisorOverride(divisorOverride), countIncludePad(countIncludePad) {}
};

struct PoolWindow {
  int64_t start;
  int64_t end;
  int64_t poolSize;

  __aicore__ inline PoolWindow() {}

  __aicore__ inline void Compute(int64_t idx, int64_t inputSize, int64_t kernelSize, int64_t stride,
                                 int64_t padding, int64_t countIncludePad) {
    start = idx * stride - padding;
    end = Min(start + kernelSize, inputSize + padding);

    int64_t includePadPoolSize = end - start;

    start = Max(start, 0);
    end = Min(end, inputSize);

    poolSize = countIncludePad ? includePadPoolSize : end - start;
  }
};

struct KernelInfoBuffer {
  TBuf<TPosition::VECCALC> startIndexBuf;
  TBuf<TPosition::VECCALC> endIndexBuf;
  TBuf<TPosition::VECCALC> poolSizeBuf;

  LocalTensor<int64_t> startIndexLocal;
  LocalTensor<int64_t> endIndexLocal;
  LocalTensor<int64_t> poolSizeLocal;

  int32_t maxBufferLen;
  int32_t bufferLen{0};
  int64_t startIndex{-1};

  __aicore__ inline KernelInfoBuffer() {}

  __aicore__ inline void Init(TPipe* pipe, int64_t bufLen) {
    maxBufferLen = bufLen;
    pipe->InitBuffer(startIndexBuf, bufLen * sizeof(int64_t));
    pipe->InitBuffer(endIndexBuf, bufLen * sizeof(int64_t));
    pipe->InitBuffer(poolSizeBuf, bufLen * sizeof(int64_t));

    startIndexLocal = startIndexBuf.Get<int64_t>();
    endIndexLocal = endIndexBuf.Get<int64_t>();
    poolSizeLocal = poolSizeBuf.Get<int64_t>();
  }

  __aicore__ inline void SetValue(int32_t bufIdx, const PoolWindow& window) {
    startIndexLocal.SetValue(bufIdx, window.start);
    endIndexLocal.SetValue(bufIdx, window.end);
    poolSizeLocal.SetValue(bufIdx, window.poolSize);
  }

  __aicore__ inline void GetValue(int32_t idx, PoolWindow& window) {
    int32_t bufIdx = idx - startIndex;
    window.start = startIndexLocal.GetValue(bufIdx);
    window.end = endIndexLocal.GetValue(bufIdx);
    window.poolSize = poolSizeLocal.GetValue(bufIdx);
  }
};

struct Index {
  PoolWindow D;
  PoolWindow H;
  PoolWindow W;

  __aicore__ inline Index() {}
};

struct IndexBuffer {
  KernelInfoBuffer D;
  KernelInfoBuffer H;
  KernelInfoBuffer W;

  ShapeAvgPool3d outputShape;
  ShapeAvgPool3d inputShape;
  PoolParameter pool;

  __aicore__ inline IndexBuffer() {}

  __aicore__ inline void Init(TPipe* pipe, int64_t bufLen) {
    D.Init(pipe, bufLen);
    H.Init(pipe, bufLen);
    W.Init(pipe, bufLen);
  }

  __aicore__ inline void SetComputeParameter(const ShapeAvgPool3d& outShape, const ShapeAvgPool3d& inShape,
                                             const PoolParameter& poolParam) {
    outputShape = outShape;
    inputShape = inShape;
    pool = poolParam;
  }

  __aicore__ inline void ComputeIndex(int64_t idx, KernelInfoBuffer& kernelInfo, int64_t inputSize,
                                      int64_t outputSize, int64_t kernelSize, int64_t stride, int64_t padding) {
    kernelInfo.startIndex = idx;
    kernelInfo.bufferLen = outputSize - idx < kernelInfo.maxBufferLen ? outputSize - idx : kernelInfo.maxBufferLen;
    for (int32_t i = 0; i < kernelInfo.bufferLen; ++i) {
      PoolWindow window;
      window.Compute(idx + i, inputSize, kernelSize, stride, padding, pool.countIncludePad);
      kernelInfo.SetValue(i, window);
    }
  }

  __aicore__ inline void GetIndex(int64_t idx, Index& index) {
    idx = idx % outputShape.strideC;
    int64_t od = idx / outputShape.strideD;
    int64_t oh = idx % outputShape.strideD / outputShape.strideH;
    int64_t ow = idx % outputShape.strideH / outputShape.strideW;

    if (od < D.startIndex || od >= D.startIndex + D.bufferLen) [[unlikely]] {
      ComputeIndex(od, D, inputShape.D, outputShape.D, pool.kernelD, pool.strideD, pool.padD);
    }

    if (oh < H.startIndex || oh >= H.startIndex + H.bufferLen) [[unlikely]] {
      ComputeIndex(oh, H, inputShape.H, outputShape.H, pool.kernelH, pool.strideH, pool.padH);
    }

    if (ow < W.startIndex || ow >= W.startIndex + W.bufferLen) [[unlikely]] {
      ComputeIndex(ow, W, inputShape.W, outputShape.W, pool.kernelW, pool.strideW, pool.padW);
    }

    D.GetValue(od, index.D);
    H.GetValue(oh, index.H);
    W.GetValue(ow, index.W);
  }

  __aicore__ inline void GetIndexWithoutCheck(int64_t idx, Index& index) {
    idx = idx % outputShape.strideC;
    int64_t od = idx / outputShape.strideD;
    int64_t oh = idx % outputShape.strideD / outputShape.strideH;
    int64_t ow = idx % outputShape.strideH / outputShape.strideW;

    D.GetValue(od, index.D);
    H.GetValue(oh, index.H);
    W.GetValue(ow, index.W);
  }

  __aicore__ inline void GetWIndex(int64_t idx, Index& index) {
    int64_t ow = idx % outputShape.strideH / outputShape.strideW;
    W.GetValue(ow, index.W);
  }
};

__aicore__ inline void SToMTE2Sync() {
  event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
  SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
  WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}

__aicore__ inline void MTE2ToSSync() {
  event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
  WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

__aicore__ inline void SToMTE3Sync() {
  event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
  SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

__aicore__ inline void MTE3ToSSync() {
  event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
  SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
  WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

__aicore__ inline void SToVSync() {
  event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  SetFlag<HardEvent::S_V>(eventIDSToV);
  WaitFlag<HardEvent::S_V>(eventIDSToV);
}

#endif // AVG_POOL3D_COMMON_H_

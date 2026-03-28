/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * \file adaptive_avg_pool3d_common.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_COMMON_H_
#define ADAPTIVE_AVG_POOL3D_COMMON_H_

#include "kernel_operator.h"
#include "kernel_log.h"

using namespace AscendC;

struct AdaAvgPool3DShape {
  int64_t d;
  int64_t h;
  int64_t w;
  int64_t nstride;
  int64_t dstride;
  int64_t hstride;
  int64_t wstride;

  __aicore__ inline AdaAvgPool3DShape() {}

  __aicore__ inline AdaAvgPool3DShape(int64_t d, int64_t h, int64_t w)
    : d(d), h(h), w(w), nstride(d * h * w), dstride(h * w), hstride(w), wstride(1) {}
};

struct Index {
  int64_t dstart;
  int64_t dend;
  int64_t hstart;
  int64_t hend;
  int64_t wstart;
  int64_t wend;

   __aicore__ inline Index() {}

  __aicore__ inline Index(int64_t dstart, int64_t dend, int64_t hstart, int64_t hend, int64_t wstart, int64_t wend)
    : dstart(dstart), dend(dend), hstart(hstart), hend(hend), wstart(wstart), wend(wend) {}
};

struct IndexBuffer {
  TBuf<TPosition::VECCALC> startDIndexBuf;
  TBuf<TPosition::VECCALC> endDIndexBuf;
  TBuf<TPosition::VECCALC> startHIndexBuf;
  TBuf<TPosition::VECCALC> endHIndexBuf;
  TBuf<TPosition::VECCALC> startWIndexBuf;
  TBuf<TPosition::VECCALC> endWIndexBuf;

  __aicore__ inline IndexBuffer() {}
};

__aicore__ inline int64_t StartIndex(int64_t idx, int64_t osize, int64_t isize) {
  ASSERT_MSG(osize != 0, "Division by zero error!");
  return (idx / osize) * isize + ((idx % osize) * isize) / osize;
}

__aicore__ inline int64_t EndIndex(int64_t idx, int64_t osize, int64_t isize) {
  ASSERT_MSG(osize != 0, "Division by zero error!");
  return 1 + ((idx + 1) * isize - 1) / osize;
}

__aicore__ inline int64_t StartIndexToOffset(const Index& index, const AdaAvgPool3DShape& shape) {
  return index.dstart * shape.dstride + index.hstart * shape.hstride + index.wstart * shape.wstride;
}

__aicore__ inline void OutputOffsetToInputIndex(
    int64_t offset, const AdaAvgPool3DShape& outputShape, const AdaAvgPool3DShape& inputShape, Index& index) {
  int64_t od = offset % outputShape.nstride / outputShape.dstride;
  int64_t oh = offset % outputShape.dstride / outputShape.hstride;
  int64_t ow = offset % outputShape.hstride / outputShape.wstride;

  index.dstart = StartIndex(od, outputShape.d, inputShape.d);
  index.dend = EndIndex(od, outputShape.d, inputShape.d);
  index.hstart = StartIndex(oh, outputShape.h, inputShape.h);
  index.hend = EndIndex(oh, outputShape.h, inputShape.h);
  index.wstart = StartIndex(ow, outputShape.w, inputShape.w);
  index.wend = EndIndex(ow, outputShape.w, inputShape.w);
}

__aicore__ inline void CalculateIndex(IndexBuffer& indexBuf, AdaAvgPool3DShape& inputShape, AdaAvgPool3DShape& outputShape,
                                      int64_t offset, int64_t len) {
  LocalTensor<int64_t> startDIndexLocal = indexBuf.startDIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endDIndexLocal = indexBuf.endDIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> startHIndexLocal = indexBuf.startHIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endHIndexLocal = indexBuf.endHIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> startWIndexLocal = indexBuf.startWIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endWIndexLocal = indexBuf.endWIndexBuf.Get<int64_t>();

  Index index;
  for (int64_t i = offset, j = 0; i < offset + len; ++i, ++j) {
    OutputOffsetToInputIndex(i, outputShape, inputShape, index);
    startDIndexLocal.SetValue(j, index.dstart);
    endDIndexLocal.SetValue(j, index.dend);
    startHIndexLocal.SetValue(j, index.hstart);
    endHIndexLocal.SetValue(j, index.hend);
    startWIndexLocal.SetValue(j, index.wstart);
    endWIndexLocal.SetValue(j, index.wend);
  }
}

__aicore__ inline void GetIndexFromBuffer(IndexBuffer& indexBuf, int64_t start, int64_t end, Index& index) {
  LocalTensor<int64_t> startDIndexLocal = indexBuf.startDIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endDIndexLocal = indexBuf.endDIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> startHIndexLocal = indexBuf.startHIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endHIndexLocal = indexBuf.endHIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> startWIndexLocal = indexBuf.startWIndexBuf.Get<int64_t>();
  LocalTensor<int64_t> endWIndexLocal = indexBuf.endWIndexBuf.Get<int64_t>();

  index.dstart = startDIndexLocal.GetValue(start);
  index.dend = endDIndexLocal.GetValue(end);
  index.hstart = startHIndexLocal.GetValue(start);
  index.hend = endHIndexLocal.GetValue(end);
  index.wstart = startWIndexLocal.GetValue(start);
  index.wend = endWIndexLocal.GetValue(end);
}

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

__aicore__ inline void MTE3ToVSync() {
  event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
  SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
  WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

#endif // ADAPTIVE_AVG_POOL3D_COMMON_H_

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
 * \file ge_glu_v2_fp16_stride_310p.h
 * \brief
 */
#ifndef GeGluV2_HALF_STRIDE_310P_H
#define GeGluV2_HALF_STRIDE_310P_H

#include "ge_glu_v2_base_310p.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Fp16Stride310P : public GeGluV2Base310P<T> {
public:
  __aicore__ inline GeGluV2Fp16Stride310P(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gelu,
                              GM_ADDR workspace, const GeGluV2TilingData *tilingData);
  __aicore__ inline void Process();

private:
  __aicore__ inline void CopyIn(const int64_t& index, const int64_t& blockCount);
  __aicore__ inline void ComputeGelu(const int64_t& length);
  __aicore__ inline void CopyOutGelu(const int64_t& index, const int64_t& length, const int64_t& group);
  __aicore__ inline void ComputeMul(const int64_t& length);
  __aicore__ inline void CopyOutMul(const int64_t& index, const int64_t& length, const int64_t& group);
  __aicore__ inline void ProcessPerCore();
  __aicore__ inline void ProcessLastCore();

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> m_inQueueX;
  TQue<QuePosition::VECOUT, BUFFER_NUM> m_outQueue;

  TBuf<QuePosition::VECCALC> m_castTmpBuf;
  TBuf<QuePosition::VECCALC> m_tmpBuf;
};

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR gelu,
                                                      GM_ADDR workspace, const GeGluV2TilingData *tilingData) {
  this->BaseInit(x, y, gelu, tilingData);
  this->BaseInit310P(workspace);

  pipe.InitBuffer(m_inQueueX, BUFFER_NUM, CHUNK_NUM * BUFFER_SIZE * sizeof(T));
  pipe.InitBuffer(m_outQueue, 1, CHUNK_NUM * BUFFER_SIZE * sizeof(T));
  pipe.InitBuffer(m_castTmpBuf, CHUNK_NUM * BUFFER_SIZE * sizeof(float));
  pipe.InitBuffer(m_tmpBuf, CHUNK_NUM * BUFFER_SIZE * sizeof(float));
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::Process() {
  if (this->blockIdx >= this->m_tilingData.realCoreNum) {
    return;
  }

  if (this->m_isLastCore) { // process last core
    ProcessLastCore();
  } else {
    ProcessPerCore();
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::ProcessPerCore() {
  // process core
  for (int64_t idx = 0; idx < this->m_tilingData.loopNum; idx++) {
    CopyIn(idx, this->m_tilingData.group);
    ComputeGelu(this->group_ub_num);
    CopyOutGelu(idx, this->group_ub_num, this->m_tilingData.group);
    ComputeMul(this->group_ub_num);
    CopyOutMul(idx, this->group_ub_num, this->m_tilingData.group);
  }

  if (this->m_tilingData.nLastTailGroup > 0) {
    CopyIn(this->m_tilingData.loopNum, this->m_tilingData.nLastTailGroup);
    ComputeGelu(this->nlast_tail_ub_num);
    CopyOutGelu(this->m_tilingData.loopNum, this->nlast_tail_ub_num, this->m_tilingData.nLastTailGroup);
    ComputeMul(this->nlast_tail_ub_num);
    CopyOutMul(this->m_tilingData.loopNum, this->nlast_tail_ub_num, this->m_tilingData.nLastTailGroup);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::ProcessLastCore() {
  for (int64_t idx = 0; idx < this->m_tilingData.tailLoopNum; idx++) {
    CopyIn(idx, this->m_tilingData.group);
    ComputeGelu(this->group_ub_num);
    CopyOutGelu(idx, this->group_ub_num, this->m_tilingData.group);
    ComputeMul(this->group_ub_num);
    CopyOutMul(idx, this->group_ub_num, this->m_tilingData.group);
  }
  if (this->m_tilingData.lastTailGroup > 0) {
    CopyIn(this->m_tilingData.tailLoopNum, this->m_tilingData.lastTailGroup);
    ComputeGelu(this->last_tail_ub_num);
    CopyOutGelu(this->m_tilingData.tailLoopNum, this->last_tail_ub_num, this->m_tilingData.lastTailGroup);
    ComputeMul(this->last_tail_ub_num);
    CopyOutMul(this->m_tilingData.tailLoopNum, this->last_tail_ub_num, this->m_tilingData.lastTailGroup);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::CopyIn(const int64_t& index, const int64_t& blockCount) {
  LocalTensor<T> xLocal = m_inQueueX.AllocTensor<T>();
  LocalTensor<T> x1Local = xLocal;
  LocalTensor<T> x2Local = xLocal[BUFFER_SIZE];
  this->CopyInX(index, blockCount, x2Local, x1Local);

  m_inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::ComputeGelu(const int64_t& length) {
  LocalTensor<T> xLocal = m_inQueueX.DeQue<T>();
  LocalTensor<float> castLocal = m_castTmpBuf.Get<float>();
  LocalTensor<float> x1Fp32 = castLocal[BUFFER_SIZE];
  Cast(x1Fp32, xLocal, RoundMode::CAST_NONE, length);

  if (this->m_tilingData.approximate == 1) {
    this->ComputeGeluBase(x1Fp32, castLocal, length);
  } else {
    LocalTensor<float> x = m_tmpBuf.Get<float>();
    LocalTensor<float> xPow = x[BUFFER_SIZE];
    this->ComputeGeluErf(x1Fp32, castLocal, x, xPow, length);
  }

  LocalTensor<T> outLocal = m_outQueue.AllocTensor<T>();
  Cast(outLocal, castLocal, RoundMode::CAST_NONE, length);
  m_inQueueX.EnQue(xLocal);
  m_outQueue.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::ComputeMul(const int64_t& length) {
  LocalTensor<T> xLocal = m_inQueueX.DeQue<T>();
  LocalTensor<T> x2Local = xLocal[BUFFER_SIZE];
  LocalTensor<T> outLocal = m_outQueue.DeQue<T>();
  LocalTensor<T> yLocal = outLocal[BUFFER_SIZE];
  PipeBarrier<PIPE_V>();
  Mul(yLocal, outLocal, x2Local, length);
  m_outQueue.EnQue(outLocal);

  m_inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::CopyOutGelu(const int64_t& index, const int64_t& length,
                                                             const int64_t& group) {
  LocalTensor<T> outLocal = m_outQueue.DeQue<T>();
  this->CopyOutBase(index, length, group, outLocal, this->yGeluGm);
  m_outQueue.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16Stride310P<T>::CopyOutMul(const int64_t& index, const int64_t& length,
                                                            const int64_t& group) {
  LocalTensor<T> outLocal = m_outQueue.DeQue<T>();
  LocalTensor<T> yLocal = outLocal[BUFFER_SIZE];
  this->CopyOutBase(index, length, group, yLocal, this->yMulGm);
  m_outQueue.FreeTensor(outLocal);
}
}  // namespace GeGluV2
#endif  // GeGluV2_HALF_STRIDE_310P_H
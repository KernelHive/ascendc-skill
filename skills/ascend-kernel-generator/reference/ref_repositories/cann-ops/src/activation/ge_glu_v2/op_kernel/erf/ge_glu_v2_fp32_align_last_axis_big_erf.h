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
 * \file ge_glu_v2_fp32_align_last_axis_big_erf.h
 * \brief
 */

#ifndef GeGluV2_Float32_LAST_BIG_ERF_H
#define GeGluV2_Float32_LAST_BIG_ERF_H

#include "../ge_glu_v2_base.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Fp32AlignLastAxisBigErf : public GeGluV2Base<T> {
 public:
  __aicore__ inline GeGluV2Fp32AlignLastAxisBigErf(){};
  __aicore__ inline void Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData);
  __aicore__ inline void Process();

  constexpr static int32_t bufferNum = 2;
  constexpr static int64_t bufferSize = 6144;

 private:
  __aicore__ inline void CopyInX(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
  __aicore__ inline void ComputeGelu(const int64_t& ub_num);
  __aicore__ inline void CopyOutGelu(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
  __aicore__ inline void ComputeMul(const int64_t& ub_num);
  __aicore__ inline void CopyOutMul(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
  __aicore__ inline void ProcessPerCore();
  __aicore__ inline void ProcessLastCore();

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, bufferNum> inQueueX1;
  TQue<QuePosition::VECIN, bufferNum> inQueueX2;
  TQue<QuePosition::VECOUT, bufferNum> outQueueGelu;
  TQue<QuePosition::VECOUT, bufferNum> outQueueMul;

  TBuf<QuePosition::VECCALC> resultTempBuf1;
  TBuf<QuePosition::VECCALC> resultTempBuf2;
};

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::Init(
  GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData) {
  this->BaseInit(x, y, gelu, tilingData, true);
  pipe.InitBuffer(inQueueX1, bufferNum, bufferSize * sizeof(T));
  pipe.InitBuffer(inQueueX2, bufferNum, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueGelu, 1, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueMul, 1, bufferSize * sizeof(T));

  pipe.InitBuffer(resultTempBuf1, bufferSize * sizeof(float));
  pipe.InitBuffer(resultTempBuf2, bufferSize * sizeof(float));
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::Process() {
  if (this->blockIdx >= this->m_tilingData.realCoreNum) {
    return;
  }

  ProcessPerCore();
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::ProcessPerCore() {
  // process core
  int64_t actualLoopNum =
    (this->m_tilingData.loopNum + this->m_tilingData.realCoreNum - 1) / this->m_tilingData.realCoreNum;
  int64_t tailLoopIdx = this->m_tilingData.loopNum % this->m_tilingData.realCoreNum;

  for (int64_t idx = 0; idx < actualLoopNum; idx++) {
    if (tailLoopIdx !=0 && idx == actualLoopNum - 1) {
      if (this->blockIdx >= tailLoopIdx) {
        return;
      }
    }
    int64_t z = this->m_tilingData.realCoreNum * idx + this->blockIdx;
    int64_t idx_x = 0;
    int64_t idx_y = 0;
    int64_t length = 0;
    if (this->m_tilingData.tailLoopNum != 0) {
      idx_x = z / (this->m_tilingData.group + 1);
      idx_y = z % (this->m_tilingData.group + 1);
      length = idx_y == this->m_tilingData.group ? this->m_tilingData.tailLoopNum : this->m_tilingData.splitSize;
    } else {
      idx_x = z / this->m_tilingData.group;
      idx_y = z % this->m_tilingData.group;
      length = this->m_tilingData.splitSize;
    }

    CopyInX(idx_x, idx_y, length);
    ComputeGelu(length);
    CopyOutGelu(idx_x, idx_y, length);
    ComputeMul(length);
    CopyOutMul(idx_x, idx_y, length);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf <T>::CopyInX(
  const int64_t& idx_x, const int64_t& idx_y, const int64_t& length) {
  LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
  LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
  this->CopyInXAlignLastBig(idx_x, idx_y, length, ubX1, ubX2);

  inQueueX1.EnQue(ubX1);
  inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::ComputeGelu(const int64_t& ub_num) {
  LocalTensor<T> ubx2_fp32 = inQueueX2.DeQue<T>();

  // after cast to fp32 , input buffer release, to use as tmp buffer wihle do geluv2 compute.
  LocalTensor<T> gelu_out = outQueueGelu.AllocTensor<T>();
  LocalTensor<float> tmpBuf1 = resultTempBuf1.Get<T>();
  LocalTensor<float> tmpBuf2 = resultTempBuf2.Get<T>();
  this->ComputeGeluErf(ubx2_fp32, gelu_out, tmpBuf1, tmpBuf2, ub_num);

  inQueueX2.FreeTensor(ubx2_fp32);
  outQueueGelu.EnQue(gelu_out);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::ComputeMul(const int64_t& ub_num) {
  LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
  LocalTensor<T> gelu_out = outQueueGelu.DeQue<T>();
  LocalTensor<T> mul_out = outQueueMul.AllocTensor<T>();
  PipeBarrier<PIPE_V>();
  Mul(mul_out, gelu_out, ubX1, ub_num);
  outQueueMul.EnQue(mul_out);

  outQueueGelu.FreeTensor(gelu_out);
  inQueueX1.FreeTensor(ubX1);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::CopyOutGelu(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length) {
  LocalTensor<T> outLocalGelu = outQueueGelu.DeQue<T>();
  this->CopyOutGeluBaseLastBig(idx_x, idx_y, length, outLocalGelu);
  outQueueGelu.EnQue(outLocalGelu);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigErf<T>::CopyOutMul(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length) {
  LocalTensor<T> outLocalMul = outQueueMul.DeQue<T>();
  this->CopyOutMulBaseLastBig(idx_x, idx_y, length, outLocalMul);
  outQueueMul.FreeTensor(outLocalMul);
}
}  // namespace GeGluV2
#endif  // GeGluV2_Float32_LAST_BIG_ERF_H
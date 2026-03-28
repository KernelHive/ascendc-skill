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
 * \file ge_glu_v2_base.h
 * \brief
 */
 
#ifndef GeGluV2_BASE_H
#define GeGluV2_BASE_H

#include "kernel_operator.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Base {
 public:
  __aicore__ inline GeGluV2Base(){};
  constexpr static int32_t BLOCK_BYTES = 32;
  constexpr static float negativeOne = -1.0;
  constexpr static float scalarOne = 1.0;
  constexpr static float scalarZero = 0.0;
  constexpr static float beta = 0.044715;
  constexpr static float alpha = 1.5957691;
  constexpr static uint8_t ODD = 2;
  constexpr static uint8_t EVEN = 1;

  constexpr static float ERF_PARAM1 = -0.3512339572e-8;
  constexpr static float ERF_PARAM2 = 0.2645266170e-6;
  constexpr static float ERF_PARAM3 = -0.7929488134e-5;
  constexpr static float ERF_PARAM4 = 0.1106123840e-3;
  constexpr static float ERF_PARAM5 = 0.6518995814e-4;
  constexpr static float ERF_PARAM6 = -0.7266616915e-1;
  constexpr static float ERF_PARAM7 = -0.1595769883e1;
  constexpr static float ERF_THRESHOLD = 5.75;

 protected:
  __aicore__ inline void ParseTilingData(const GeGluV2TilingData* tilingData, GeGluV2TilingData& m_tilingData);
  __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR y, GM_ADDR gelu, const GeGluV2TilingData* tilingData, 
                                  bool isLastBig=false, bool isVreudce=false);
  __aicore__ inline void CopyInXAlign(
    const int64_t& index, const int64_t& blockCount, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2);
  __aicore__ inline void CopyOutMulBase(
    const int64_t& index, const int64_t& ub_num, const int64_t& group, LocalTensor<T>& outLocal);
  __aicore__ inline void CopyOutGeluBase(
    const int64_t& index, const int64_t& ub_num, const int64_t& group, LocalTensor<T>& outLocal);
  __aicore__ inline void ComputeGeluBase(
    LocalTensor<float>& ubx2_fp32, LocalTensor<float>& tmpBuf2, const int64_t& ub_num);
  __aicore__ inline void ComputeGeluErf(
    LocalTensor<float>& ubx2_fp32, LocalTensor<float>& computeOut, LocalTensor<float>& x1,
    LocalTensor<float>& xPow, const int64_t& length);
  __aicore__ inline void ComputeFp16VReduce(
    LocalTensor<half>& ubX1, LocalTensor<half>& ubX2, LocalTensor<half>& ubX, const int64_t& ub_num);
  __aicore__ inline void ComputeFp32VReduce(
    LocalTensor<float>& ubX1, LocalTensor<float>& ubX2, LocalTensor<float>& ubX, const int64_t& ub_num);
  __aicore__ inline void CopyInXAlignLastBig(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2);
  __aicore__ inline void CopyOutMulBaseLastBig(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length, LocalTensor<T>& outLocal);
  __aicore__ inline void CopyOutGeluBaseLastBig(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length, LocalTensor<T>& outLocal);
  __aicore__ inline void CopyInXVreduce(const int64_t& index, const int64_t& blockCount, LocalTensor<T>& ubX);
  __aicore__ inline void CopyOutGeluVreduce(const int64_t& index, const int64_t& ub_num, LocalTensor<T>& outLocal);
  __aicore__ inline void CopyOutMulVreduce(const int64_t& index, const int64_t& ub_num, LocalTensor<T>& outLocal);

  protected:
    GlobalTensor<T> xGm, yGeluGm, yMulGm;
    int32_t blockIdx = 0;
    int64_t gmXOffset = 0;
    int64_t gmDYOffset = 0;
    int64_t group_ub_num = 0;
    int64_t nlast_tail_ub_num = 0;
    int64_t last_tail_ub_num = 0;
    int64_t one_process_out_stride = 0;
    int64_t one_process_in_stride = 0;
    int64_t x1_stride = 0;
    int64_t x2_stride = 0;

    uint8_t vreduce_srcPattern_x1 = EVEN;
    uint8_t vreduce_srcPattern_x2 = ODD;

    bool isLastCore;
 
    // tiling params
    GeGluV2TilingData m_tilingData;
};

template <typename T>
__aicore__ inline void GeGluV2Base<T>::BaseInit(
  GM_ADDR x, GM_ADDR y, GM_ADDR gelu, const GeGluV2TilingData* tilingData, bool isLastBig, bool isVreduce) {
  blockIdx = GetBlockIdx();
  this->ParseTilingData(tilingData, m_tilingData);

  gmXOffset = blockIdx * m_tilingData.numPerCore * m_tilingData.splitSize * 2;
  gmDYOffset = blockIdx * m_tilingData.numPerCore * m_tilingData.splitSize;
  xGm.SetGlobalBuffer((__gm__ T*)x);
  yMulGm.SetGlobalBuffer((__gm__ T*)y);
  yGeluGm.SetGlobalBuffer((__gm__ T*)gelu);

  if (m_tilingData.splitSize % m_tilingData.blockSize == 0 || isVreduce) {
    group_ub_num = m_tilingData.group * m_tilingData.splitSize;
    nlast_tail_ub_num  = m_tilingData.nLastTailGroup * this->m_tilingData.splitSize;
    last_tail_ub_num  = m_tilingData.lastTailGroup * this->m_tilingData.splitSize;
  } else {
    int64_t splitSizeAlign =
        (m_tilingData.splitSize + m_tilingData.blockSize - 1) / m_tilingData.blockSize * m_tilingData.blockSize;
    group_ub_num = m_tilingData.group * splitSizeAlign;
    nlast_tail_ub_num = m_tilingData.nLastTailGroup * splitSizeAlign;
    last_tail_ub_num = m_tilingData.lastTailGroup * splitSizeAlign;
  }

  one_process_in_stride = isLastBig ? m_tilingData.ny * 2 : m_tilingData.group * m_tilingData.splitSize * 2;
  one_process_out_stride = isLastBig ? m_tilingData.ny : m_tilingData.group * m_tilingData.splitSize;

  if (m_tilingData.activateLeft == 1) {
    x1_stride = isLastBig ? m_tilingData.ny : m_tilingData.splitSize;
    vreduce_srcPattern_x1 = ODD;
    vreduce_srcPattern_x2 = EVEN;
  } else {
    x2_stride = isLastBig ? m_tilingData.ny : m_tilingData.splitSize;
  }

  isLastCore = (this->blockIdx == this->m_tilingData.realCoreNum - 1) &&
               (this->m_tilingData.tailLoopNum != 0 || this->m_tilingData.lastTailGroup != 0);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyInXAlign(
  const int64_t& index, const int64_t& group, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2) {
  DataCopyParams intriParams;
  intriParams.blockCount = group;
  intriParams.dstStride = 0;
  if (m_tilingData.splitSize % m_tilingData.blockSize == 0) {
    // align case
    intriParams.blockLen = m_tilingData.splitSize / m_tilingData.blockSize;
    intriParams.srcStride = m_tilingData.splitSize / m_tilingData.blockSize;
    DataCopy(ubX1, xGm[gmXOffset + index * one_process_in_stride + x1_stride], intriParams);
    DataCopy(ubX2, xGm[gmXOffset + index * one_process_in_stride + x2_stride], intriParams);
  } else {
    // not align case
    intriParams.blockLen = m_tilingData.splitSize * sizeof(T);
    intriParams.srcStride = m_tilingData.splitSize * sizeof(T);
    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = true;
    intriPadParams.leftPadding = 0;
    intriPadParams.rightPadding = BLOCK_BYTES / sizeof(T) - m_tilingData.splitSize % m_tilingData.blockSize;
    intriPadParams.paddingValue = 1;
    DataCopyPad(ubX1,xGm[gmXOffset + index * one_process_in_stride + x1_stride], intriParams, intriPadParams);
    DataCopyPad(ubX2,xGm[gmXOffset + index * one_process_in_stride + x2_stride], intriParams, intriPadParams);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyOutMulBase(
  const int64_t& index, const int64_t& ub_num, const int64_t& group, LocalTensor<T>& outLocal) {
  if (m_tilingData.splitSize % m_tilingData.blockSize == 0) {
    DataCopy(yMulGm[gmDYOffset + index * one_process_out_stride], outLocal, ub_num);
  } else {
    DataCopyParams intriParams;
    intriParams.blockCount = group;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = m_tilingData.splitSize * sizeof(T);
    DataCopyPad(yMulGm[gmDYOffset + index * one_process_out_stride],outLocal, intriParams);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyOutGeluBase(
    const int64_t& index, const int64_t& ub_num, const int64_t& group, LocalTensor<T>& outLocal) {
  if (m_tilingData.splitSize % m_tilingData.blockSize == 0) {
    DataCopy(yGeluGm[gmDYOffset + index * one_process_out_stride], outLocal, ub_num);
  } else {
    DataCopyParams intriParams;
    intriParams.blockCount = group;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = m_tilingData.splitSize * sizeof(T);
    DataCopyPad(yGeluGm[gmDYOffset + index * one_process_out_stride],outLocal, intriParams);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyInXAlignLastBig(
  const int64_t& idx_x, const int64_t& idx_y, const int64_t& length, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2) {
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);
    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = false;

    int64_t offset_a = idx_x * one_process_in_stride + x1_stride + idx_y * m_tilingData.splitSize;
    int64_t offset_b = idx_x * one_process_in_stride + x2_stride + idx_y * m_tilingData.splitSize;
    DataCopyPad(ubX1, xGm[offset_a], intriParams, intriPadParams);
    DataCopyPad(ubX2, xGm[offset_b], intriParams, intriPadParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyOutMulBaseLastBig(
  const int64_t& idx_x, const int64_t& idx_y, const int64_t& length, LocalTensor<T>& outLocal) {
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);

    int64_t offset = idx_x * one_process_out_stride + idx_y * m_tilingData.splitSize;
    DataCopyPad(yMulGm[offset], outLocal, intriParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyOutGeluBaseLastBig(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length, LocalTensor<T>& outLocal) {
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);

    int64_t offset = idx_x * one_process_out_stride + idx_y * m_tilingData.splitSize;
    DataCopyPad(yGeluGm[offset], outLocal, intriParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::ComputeGeluBase(
  LocalTensor<float>& ubx2_fp32, LocalTensor<float>& tmpBuf, const int64_t& ub_num) {
  // compute (2 * np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))
  Mul(tmpBuf, ubx2_fp32, ubx2_fp32, ub_num);
  Mul(tmpBuf, ubx2_fp32, tmpBuf, ub_num);
  Muls(tmpBuf, tmpBuf, beta, ub_num);
  Add(tmpBuf, ubx2_fp32, tmpBuf, ub_num);
  Muls(tmpBuf, tmpBuf, alpha, ub_num);  // get tan paramter tmpBuf

  // compute x * 0.5 * (1 + tanh(tmpBuf))
  Muls(tmpBuf, tmpBuf, negativeOne, ub_num);
  Exp(tmpBuf, tmpBuf, ub_num);
  Adds(tmpBuf, tmpBuf, scalarOne, ub_num);
  Div(tmpBuf, ubx2_fp32, tmpBuf, ub_num);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::ComputeGeluErf(
  LocalTensor<float>& ubx2_fp32, LocalTensor<float>& computeOut, LocalTensor<float>& x1,
  LocalTensor<float>& xPow, const int64_t& length) {
  //res = x/(1+exp(((((((a1*x^2+a2)*x^2+a3)*x^2+a4)*x^2+a5)*x^2+a6)*x^2+a7)*x))
  Mins(x1, ubx2_fp32, ERF_THRESHOLD, length);

  Mul(xPow, x1, x1, length);
  Muls(computeOut, xPow, ERF_PARAM1, length);

  Adds(computeOut, computeOut, ERF_PARAM2, length);
  Mul(computeOut, computeOut, xPow, length);

  Adds(computeOut, computeOut, ERF_PARAM3, length);
  Mul(computeOut, computeOut, xPow, length);

  Adds(computeOut, computeOut, ERF_PARAM4, length);
  Mul(computeOut, computeOut, xPow, length);

  Adds(computeOut, computeOut, ERF_PARAM5, length);
  Mul(computeOut, computeOut, xPow, length);

  Adds(computeOut, computeOut, ERF_PARAM6, length);
  Mul(computeOut, computeOut, xPow, length);

  Adds(computeOut, computeOut, ERF_PARAM7, length);
  Mul(computeOut, computeOut, x1, length);

  Exp(computeOut, computeOut, length);

  Adds(computeOut, computeOut, 1.0f, length);
  Div(computeOut, ubx2_fp32, computeOut, length);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyInXVreduce(
  const int64_t& index, const int64_t& blockCount, LocalTensor<T>& ubX) {
    int64_t one_process_total_num = blockCount * m_tilingData.splitSize * 2;
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = one_process_total_num * sizeof(T);
    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = false;
    DataCopyPad(ubX, xGm[gmXOffset + index * one_process_in_stride], intriParams, intriPadParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::ComputeFp16VReduce(
  LocalTensor<half>& ubX1, LocalTensor<half>& ubX2, LocalTensor<half>& ubX, const int64_t& ub_num) {
  // do computeVreduce
  uint64_t rsvdCnt = 0;
  uint64_t total_vreduce_num = ub_num * 2;
  uint16_t repeat = (total_vreduce_num + 127) / 128; 
  GatherMask(ubX1, ubX, vreduce_srcPattern_x1, false, 0, {1, repeat, 8, 0}, rsvdCnt);
  GatherMask(ubX2, ubX, vreduce_srcPattern_x2, false, 0, {1, repeat, 8, 0}, rsvdCnt);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::ComputeFp32VReduce(
  LocalTensor<float>& ubX1, LocalTensor<float>& ubX2, LocalTensor<float>& ubX, const int64_t& ub_num) {
  // do computeVreduce
  uint64_t rsvdCnt = 0;
  uint64_t total_vreduce_num = ub_num * 2;
  uint16_t repeat = (total_vreduce_num + 63) / 64; 
  GatherMask(ubX1, ubX, vreduce_srcPattern_x1, false, 0, {1, repeat, 8, 0}, rsvdCnt);
  GatherMask(ubX2, ubX, vreduce_srcPattern_x2, false, 0, {1, repeat, 8, 0}, rsvdCnt);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyOutMulVreduce(
  const int64_t& index, const int64_t& ub_num, LocalTensor<T>& outLocal) {
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = ub_num * sizeof(T);
    DataCopyPad(yMulGm[gmDYOffset + index * one_process_out_stride], outLocal, intriParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::CopyOutGeluVreduce(
  const int64_t& index, const int64_t& ub_num, LocalTensor<T>& outLocal) {
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = ub_num * sizeof(T);
    DataCopyPad(yGeluGm[gmDYOffset + index * one_process_out_stride], outLocal, intriParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base<T>::ParseTilingData(const GeGluV2TilingData* tilingData,
                                                       GeGluV2TilingData& m_tilingData) {
  m_tilingData.group = tilingData->group;
  m_tilingData.loopNum = tilingData->loopNum;
  m_tilingData.tailLoopNum = tilingData->tailLoopNum;
  m_tilingData.nLastTailGroup = tilingData->nLastTailGroup;
  m_tilingData.lastTailGroup = tilingData->lastTailGroup;
  m_tilingData.splitSize = tilingData->splitSize;
  m_tilingData.realCoreNum = tilingData->realCoreNum;
  m_tilingData.numPerCore = tilingData->numPerCore;
  m_tilingData.blockSize = tilingData->blockSize;
  m_tilingData.tilingKey = tilingData->tilingKey;
  m_tilingData.activateLeft = tilingData->activateLeft;
  m_tilingData.ny = tilingData->ny;
  m_tilingData.approximate = tilingData->approximate;
}
}  // namespace GeGluV2
#endif  // GeGluV2_BASE_H

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
 * \file ge_glu_v3_base_310p.h
 * \brief
 */
 
#ifndef GeGluV3_BASE_310P_H
#define GeGluV3_BASE_310P_H

#include "ge_glu_v3_base.h"

namespace GeGluV3 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t BUFFER_SIZE = 8192;
constexpr int32_t CHUNK_NUM = 2;
constexpr int32_t MASK_SIZE = 2;
constexpr int64_t MAX_UINT8 = 255;
constexpr int64_t NUM_ONE_BLOCK_INT = 8;
constexpr int64_t BYTE_ONE_BLOCK = 32;
constexpr float ERF_PARAM1 = -0.3512339572e-8;
constexpr float ERF_PARAM2 = 0.2645266170e-6;
constexpr float ERF_PARAM3 = -0.7929488134e-5;
constexpr float ERF_PARAM4 = 0.1106123840e-3;
constexpr float ERF_PARAM5 = 0.6518995814e-4;
constexpr float ERF_PARAM6 = -0.7266616915e-1;
constexpr float ERF_PARAM7 = -0.1595769883e1;
constexpr float ERF_THRESHOLD = 5.75;

template <typename T>
class GeGluV3Base310P : public GeGluV3Base<T> {
public:
  __aicore__ inline GeGluV3Base310P(){};

protected:
  __aicore__ inline void BaseInit310P(GM_ADDR workspace);
  __aicore__ inline void CopyInX(const int64_t& index, const int64_t& blockCount,
                                 LocalTensor<T>& ubX1, LocalTensor<T>& ubX2);
  __aicore__ inline void ComputeGeluBaseErf(LocalTensor<float>& ubx2_fp32, LocalTensor<float>& computeOut,
                                            LocalTensor<float>& x1, LocalTensor<float>& x_pow, const int64_t& length);
  __aicore__ inline void CopyOutBase(const int64_t& index, const int64_t& length, const int64_t& group,
                                     LocalTensor<T>& outLocal, GlobalTensor<T>& outGm);
  __aicore__ inline void CopyInXAlignLastBigWithoutPad(const int64_t& idxX, const int64_t& idxY,
                                                       const int64_t& blockLen, LocalTensor<T>& ubX1,
                                                       LocalTensor<T>& ubX2, const int64_t& delta);
  __aicore__ inline void CopyOutGeluBaseLastBigWithoutPad(const int64_t& idxX, const int64_t& idxY,
                                                          const int64_t& blockLen, LocalTensor<T>& outLocal,
                                                          const int64_t& delta);
  __aicore__ inline void CopyOutMulBaseLastBigWithoutPad(const int64_t& idxX, const int64_t& idxY,
                                                         const int64_t& blockLen, LocalTensor<T>& outLocal,
                                                         const int64_t& delta);
  template <typename U>
  __aicore__ inline void MemSetZero(GlobalTensor<U> gmTensor, int64_t size);
protected:
  uint64_t m_rightPadNum;
  int64_t m_splitSizeAlign;
  int64_t m_blockCount;
  uint64_t m_duplicateMask;
  int64_t m_duplicateOffset;
  bool m_isSplitSizeAlign;
  bool m_isLastCore;
private:
  GlobalTensor<int32_t> m_syncGlobal;
  TQue<QuePosition::VECIN, 1> m_workQueue;
  TPipe m_pipe;
};

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::BaseInit310P(GM_ADDR workspace) {
  if (this->blockIdx >= this->m_tilingData.realCoreNum) {
    return;
  }
  m_isLastCore = (this->blockIdx == this->m_tilingData.realCoreNum - 1) &&
                 (this->m_tilingData.tailLoopNum != 0 || this->m_tilingData.lastTailGroup != 0);
  m_isSplitSizeAlign = (this->m_tilingData.splitSize % this->m_tilingData.blockSize) == 0;
  if (!m_isSplitSizeAlign) {
    int64_t gmSizePerCore = this->m_tilingData.numPerCore * this->m_tilingData.splitSize;
    if (m_isLastCore) {
      gmSizePerCore = this->m_tilingData.tailLoopNum * this->m_tilingData.group * this->m_tilingData.splitSize +
                      this->m_tilingData.lastTailGroup * this->m_tilingData.splitSize;
    }
    m_syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace, this->m_tilingData.realCoreNum * NUM_ONE_BLOCK_INT);
    MemSetZero<int32_t>(m_syncGlobal, this->m_tilingData.realCoreNum * NUM_ONE_BLOCK_INT);
    // set workspace for sync
    m_pipe.InitBuffer(m_workQueue, 1, this->m_tilingData.realCoreNum * BYTE_ONE_BLOCK);
    MemSetZero<T>(this->yMulGm[this->gmDYOffset], gmSizePerCore);
    MemSetZero<T>(this->yGeluGm[this->gmDYOffset], gmSizePerCore);
    LocalTensor<int32_t> workLocal = m_workQueue.AllocTensor<int32_t>();
    SyncAll(m_syncGlobal, workLocal);
    m_workQueue.FreeTensor(workLocal);
    m_splitSizeAlign = (this->m_tilingData.splitSize + this->m_tilingData.blockSize - 1) /
      this->m_tilingData.blockSize * this->m_tilingData.blockSize;
    m_rightPadNum = m_splitSizeAlign - this->m_tilingData.splitSize;
    m_duplicateMask = ((1 << m_rightPadNum) - 1) << (this->m_tilingData.blockSize - m_rightPadNum);
    m_blockCount = m_splitSizeAlign / this->m_tilingData.blockSize;
    m_duplicateOffset = m_splitSizeAlign - this->m_tilingData.blockSize;
  }
}

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::CopyInX(const int64_t& index, const int64_t& group,
                                                        LocalTensor<T>& ubX1, LocalTensor<T>& ubX2) {
  DataCopyParams intriParams;
  intriParams.blockCount = group;
  intriParams.dstStride = 0;
  uint64_t curLoopSrcAddr = this->gmXOffset + index * this->one_process_in_stride;
  if (m_isSplitSizeAlign) {
    // align case
    intriParams.blockLen = this->m_tilingData.splitSize / this->m_tilingData.blockSize;
    intriParams.srcStride = this->m_tilingData.splitSize / this->m_tilingData.blockSize;
    DataCopy(ubX1, this->xGm[curLoopSrcAddr + this->x1_stride], intriParams);
    DataCopy(ubX2, this->xGm[curLoopSrcAddr + this->x2_stride], intriParams);
  } else {
    // not align case
    for (uint64_t idx = 0; idx < group; ++idx) {
      uint64_t dstAddr = idx * m_splitSizeAlign;
      uint64_t srcAddrOffset = idx * this->m_tilingData.splitSize * CHUNK_NUM;
      DataCopy(ubX1[dstAddr], this->xGm[curLoopSrcAddr + this->x1_stride + srcAddrOffset], m_splitSizeAlign);
      DataCopy(ubX2[dstAddr], this->xGm[curLoopSrcAddr + this->x2_stride + srcAddrOffset], m_splitSizeAlign);
    }
  }
}

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::CopyOutBase(const int64_t& index, const int64_t& length,
                                                       const int64_t& group, LocalTensor<T>& outLocal,
                                                       GlobalTensor<T>& outGm) {
  uint64_t curLoopDstAddr = this->gmDYOffset + index * this->one_process_out_stride;
  if (m_isSplitSizeAlign) {
    DataCopy(outGm[curLoopDstAddr], outLocal, length);
  } else {
    T value = 0.0;
    uint64_t mask[MASK_SIZE] = {m_duplicateMask, 0};
    uint8_t repeatTimes = static_cast<uint8_t>(group);
    uint8_t dstRepeatStride = static_cast<uint8_t>(m_blockCount);
    PipeBarrier<PIPE_V>();
    if (group <= MAX_UINT8 && m_blockCount < MAX_UINT8) {
      Duplicate(outLocal[m_duplicateOffset], value, mask, repeatTimes, 1, dstRepeatStride);
    } else if (group > MAX_UINT8) {
      repeatTimes = MAX_UINT8;
      int64_t duplicateLoop = (group + MAX_UINT8 - 1) / MAX_UINT8;
      uint8_t tail = group % MAX_UINT8;
      tail = (tail == 0) ? MAX_UINT8 : tail;
      for (int64_t i = 0; i < duplicateLoop; ++i) {
        repeatTimes = (i == duplicateLoop - 1) ? tail : repeatTimes;
        Duplicate(outLocal[i * MAX_UINT8 * m_splitSizeAlign + m_duplicateOffset],
                  value, mask, repeatTimes, 1, dstRepeatStride);
      }
    } else {
      for (int64_t i = 0; i < group; ++i) {
        Duplicate(outLocal[i * m_splitSizeAlign + m_duplicateOffset], value, mask, 1, 1, 0);
      }
    }
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);

    SetAtomicAdd<T>();
    for (uint64_t idx = 0; idx < group; ++idx) {
      DataCopy(outGm[curLoopDstAddr + idx * this->m_tilingData.splitSize], outLocal[idx * m_splitSizeAlign],
               m_splitSizeAlign);
    }
    SetAtomicNone();
  }
}

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::CopyInXAlignLastBigWithoutPad(const int64_t& idxX, const int64_t& idxY,
                                                                         const int64_t& blockLen, LocalTensor<T>& ubX1,
                                                                         LocalTensor<T>& ubX2, const int64_t& delta) {
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen;

    int64_t offset = idxX * this->one_process_in_stride + idxY * this->m_tilingData.splitSize - delta;
    int64_t offsetA = offset + this->x1_stride;
    int64_t offsetB = offset + this->x2_stride;

    DataCopy(ubX1, this->xGm[offsetA], copyParams);
    DataCopy(ubX2, this->xGm[offsetB], copyParams);
}

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::CopyOutMulBaseLastBigWithoutPad(
  const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& outLocal,
  const int64_t& delta) {
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen;

    int64_t offset = idxX * this->one_process_out_stride + idxY * this->m_tilingData.splitSize - delta;
    DataCopy(this->yMulGm[offset], outLocal, copyParams);
}

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::CopyOutGeluBaseLastBigWithoutPad(
    const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& outLocal,
    const int64_t& delta) {
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen;

    int64_t offset = idxX * this->one_process_out_stride + idxY * this->m_tilingData.splitSize - delta;
    DataCopy(this->yGeluGm[offset], outLocal, copyParams);
}

template <typename T>
__aicore__ inline void GeGluV3Base310P<T>::ComputeGeluBaseErf(
  LocalTensor<float>& ubx2_fp32, LocalTensor<float>& computeOut, LocalTensor<float>& x1,
  LocalTensor<float>& xPow, const int64_t& length) {
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
template <typename U>
__aicore__ inline void GeGluV3Base310P<T>::MemSetZero(GlobalTensor<U> gmTensor, int64_t size)
{
    if (g_coreType == AIC) {
        return;
    }
    int64_t int16Size = (size * sizeof(U) + sizeof(int16_t) - 1) / sizeof(int16_t);
    LocalTensor<int16_t> popBuffer;
    bool ret = PopStackBuffer<int16_t, TPosition::LCM>(popBuffer);
    uint32_t maxBurstSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(int16_t);
    uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    uint32_t round = int16Size / popSize;
    uint32_t tail = int16Size % popSize;
    uint32_t roundSize = round != 0 ? popSize : 0;
    DuplicateImpl<int16_t>((__ubuf__ int16_t *)popBuffer.GetPhyAddr(), 0, popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    uint32_t comOffset = 0;
    // compute the main block
    for (int index = 0; index < round; ++index) {
        DataCopyUB2GMImpl((__gm__ int16_t *)gmTensor.GetPhyAddr() + comOffset,
            (__ubuf__ int16_t *)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((roundSize * sizeof(int16_t) + ONE_BLK_SIZE - 1) / (ONE_BLK_SIZE)), 0, 0});
        comOffset += roundSize;
    }
    // compute the tail block
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyUB2GMImpl((__gm__ int16_t *)gmTensor.GetPhyAddr() + comOffset,
            (__ubuf__ int16_t *)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((tail * sizeof(int16_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE), 0, 0});
    }
}

}  // namespace GeGluV3
#endif  // GeGluV3_BASE_310P_H

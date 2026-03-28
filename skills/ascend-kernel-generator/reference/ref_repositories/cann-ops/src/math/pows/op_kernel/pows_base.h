/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file pows_base.h
 * \brief
 */
 
#ifndef Pows_BASE_H
#define Pows_BASE_H

#include "kernel_operator.h"

namespace Pows {
using namespace AscendC;
 
template <typename T>
class PowsBase {
 public:
  __aicore__ inline PowsBase(){};
  constexpr static float scalarOne = 1.0;

  constexpr static float SQRT_EXP_UPPER = 0.500000001;
  constexpr static float SQRT_EXP_LOWER = 0.499999999;
  constexpr static float SQUARE_EXP_UPPER = 2.000000001;
  constexpr static float SQUARE_EXP_LOWER = 1.999999999;
  constexpr static float CUBE_EXP_UPPER = 3.000000001;
  constexpr static float CUBE_EXP_LOWER = 2.999999999;
  constexpr static float NEGTIVE_SQRT_EXP_UPPER = -0.499999999;
  constexpr static float NEGTIVE_SQRT_EXP_LOWER = -0.500000001;
  constexpr static float NEGTIVE_ONE_EXP_UPPER = -0.999999999;
  constexpr static float NEGTIVE_ONE_EXP_LOWER = -1.000000001;
  constexpr static float NEGTIVE_SQUARE_EXP_UPPER = -1.999999999;
  constexpr static float NEGTIVE_SQUARE_EXP_LOWER = -2.000000001;

 protected:
  __aicore__ inline void ParseTilingData(const PowsTilingData* tilingData, PowsTilingData& m_tilingData);
  __aicore__ inline void BaseInit(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const PowsTilingData* tilingData);
  __aicore__ inline void CopyInXBase(const int64_t& index, const int64_t& dataLength, LocalTensor<T>& ubX1);
  __aicore__ inline void ComputePowsBase(LocalTensor<float>& ubx2_fp32, LocalTensor<float>& result, const int64_t& dataLength);
  __aicore__ inline void CopyOutBase(const int64_t& index, const int64_t& dataLength, LocalTensor<T>& outLocal);

  __aicore__ inline void CopyInXDataCopy(const int64_t& index, const int64_t& dataLength, LocalTensor<T>& ubX1);
  __aicore__ inline void CopyInXDataCopyPad(const int64_t& index, const int64_t& dataLength, LocalTensor<T>& ubX1);
  __aicore__ inline void CopyOutDataCopy(const int64_t& index, const int64_t& dataLength, LocalTensor<T>& outLocal);
  __aicore__ inline void CopyOutDataCopyPad(const int64_t& index, const int64_t& dataLength, LocalTensor<T>& outLocal);

  protected:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    int32_t blockIdx = 0;
    int64_t gmOffset = 0;
    bool isLastCore;
    int64_t processStride = 0;
    float scalar = 0;
 
    // tiling params
    PowsTilingData m_tilingData;
};

template <typename T>
__aicore__ inline void PowsBase<T>::BaseInit(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const PowsTilingData* tilingData) {
  blockIdx = GetBlockIdx();
  this->ParseTilingData(tilingData, m_tilingData);

  gmOffset = blockIdx * m_tilingData.numPerCore;
  x1Gm.SetGlobalBuffer((__gm__ T*)x1);
  x2Gm.SetGlobalBuffer((__gm__ T*)x2);
  yGm.SetGlobalBuffer((__gm__ T*)y);

  processStride = m_tilingData.dataLength;
  isLastCore = (this->blockIdx == this->m_tilingData.realCoreNum - 1) &&
    (this->m_tilingData.tailCoreLoopNum != 0 || this->m_tilingData.tailCoreTailLength != 0);
}

template <typename T>
__aicore__ inline void PowsBase<T>::CopyInXBase(
  const int64_t& index, const int64_t& dataLength, LocalTensor<T>& ubX1) {
#if __CCE_AICORE__ == 220
  CopyInXDataCopyPad(index, dataLength, ubX1);
#else
  CopyInXDataCopy(index, dataLength, ubX1);
#endif
}

template <typename T>
__aicore__ inline void PowsBase<T>::CopyInXDataCopy(
  const int64_t& index, const int64_t& dataLength, LocalTensor<T>& ubX1) {
  int64_t dataLengthAlign = (dataLength + m_tilingData.blockSize -1) / m_tilingData.blockSize * m_tilingData.blockSize;
  DataCopy(ubX1, x1Gm[gmOffset + index * processStride], dataLengthAlign);
  event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T>
__aicore__ inline void PowsBase<T>::CopyInXDataCopyPad(
  const int64_t& index, const int64_t& dataLength, LocalTensor<T>& ubX1) {
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = dataLength * sizeof(T);
    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = false;
    DataCopyPad(ubX1, x1Gm[gmOffset + index * processStride], intriParams, intriPadParams);
}

template <typename T>
__aicore__ inline void PowsBase<T>::ComputePowsBase(
  LocalTensor<float>& ubx2_fp32, LocalTensor<float>& result, const int64_t& dataLength) {
  if (scalar >= NEGTIVE_SQUARE_EXP_LOWER && scalar <= NEGTIVE_SQUARE_EXP_UPPER) {
    // 1.0/(x*x)
    AscendC::PipeBarrier<PIPE_V>();
    Mul(result, ubx2_fp32, ubx2_fp32, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(ubx2_fp32, scalarOne, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
    Div(result, ubx2_fp32, result,dataLength);
    AscendC::PipeBarrier<PIPE_V>();
  } else if (scalar >= NEGTIVE_ONE_EXP_LOWER && scalar <= NEGTIVE_ONE_EXP_UPPER) {
    // 1.0/x
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(result, scalarOne, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
    Div(result, result, ubx2_fp32, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
  } else if (scalar >= SQRT_EXP_LOWER && scalar <= SQRT_EXP_UPPER) {
    // sqrt(x)
    AscendC::PipeBarrier<PIPE_V>();
    Sqrt(result, ubx2_fp32, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
  } else if (scalar >= NEGTIVE_SQRT_EXP_LOWER && scalar <= NEGTIVE_SQRT_EXP_UPPER) {
    // 1.0/sqrt(x)
    AscendC::PipeBarrier<PIPE_V>();
    Sqrt(result, ubx2_fp32, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(ubx2_fp32, scalarOne, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
    Div(result, ubx2_fp32, result,dataLength);
    AscendC::PipeBarrier<PIPE_V>();
  } else if (scalar >= SQUARE_EXP_LOWER && scalar <= SQUARE_EXP_UPPER) {
    // x*x
    AscendC::PipeBarrier<PIPE_V>();
    Mul(result, ubx2_fp32, ubx2_fp32, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
  } else if (scalar >= CUBE_EXP_LOWER && scalar <= CUBE_EXP_UPPER) {
    // x*x*x
    AscendC::PipeBarrier<PIPE_V>();
    Mul(result, ubx2_fp32, ubx2_fp32, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(result, ubx2_fp32, result, dataLength);
    AscendC::PipeBarrier<PIPE_V>();
  }
}

template <typename T>
__aicore__ inline void PowsBase<T>::CopyOutBase(
  const int64_t& index, const int64_t& dataLength, LocalTensor<T>& outLocal) {
  event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
  
#if __CCE_AICORE__ == 220
  CopyOutDataCopyPad(index, dataLength, outLocal);
#else
  CopyOutDataCopy(index, dataLength, outLocal);
#endif
}

template <typename T>
__aicore__ inline void PowsBase<T>::CopyOutDataCopy(
  const int64_t& index, const int64_t& dataLength, LocalTensor<T>& outLocal) {
  int64_t dataLengthAlign = (dataLength + m_tilingData.blockSize -1) / m_tilingData.blockSize * m_tilingData.blockSize;
  DataCopy(yGm[gmOffset + index * processStride], outLocal, dataLengthAlign);
}

template <typename T>
__aicore__ inline void PowsBase<T>::CopyOutDataCopyPad(
  const int64_t& index, const int64_t& dataLength, LocalTensor<T>& outLocal) {
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.dstStride = 0;
  intriParams.srcStride = 0;
  intriParams.blockLen = dataLength * sizeof(T);
  DataCopyPad(yGm[gmOffset + index * processStride], outLocal, intriParams);
}

template <typename T>
__aicore__ inline void PowsBase<T>::ParseTilingData(const PowsTilingData* tilingData, PowsTilingData& m_tilingData) {
  m_tilingData.mainCoreLoopNum = tilingData->mainCoreLoopNum;
  m_tilingData.mainCoreTailLength = tilingData->mainCoreTailLength;
  m_tilingData.tailCoreLoopNum = tilingData->tailCoreLoopNum;
  m_tilingData.tailCoreTailLength = tilingData->tailCoreTailLength;
  m_tilingData.realCoreNum = tilingData->realCoreNum;
  m_tilingData.numPerCore = tilingData->numPerCore;
  m_tilingData.tilingKey = tilingData->tilingKey;
  m_tilingData.dataLength = tilingData->dataLength;
  m_tilingData.bufSize = tilingData->bufSize;
  m_tilingData.blockSize = tilingData->blockSize;
}
}  // namespace Pows
#endif  // Pows_BASE_H

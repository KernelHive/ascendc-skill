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
 * @file angle_v2.h
 */
#ifndef _ANGLE_V2_H_
#define _ANGLE_V2_H_

#include "angle_v2_base.h"

namespace AngleV2N {
using namespace AscendC;

template <typename yType>
class AngleV2 : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
    blockLen = this->tileLength / dataPerBlock;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + this->offset, this->blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(yType));
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));

    pipe.InitBuffer(maskBuf1, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(piBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(nanBuf, this->tileLength * sizeof(yType));
  }

  __aicore__ inline void Process() {
    BufferGet();
    // loop count need to be doubled, due to double buffer
    for (int32_t i = 0; i < this->tileNum; i++) {
      int64_t coreOffset = i * static_cast<int64_t>(this->tileLength);
      CopyIn(coreOffset);
      Compute();
      CopyOut(coreOffset);
    }

    if (this->lastTileLength > 0) {
      int64_t coreOffset = static_cast<int64_t>(this->blockLength - this->lastTileLength);
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      blockLen = this->lastTileLength / dataPerBlock;
      CopyIn(coreOffset);
      Compute();
      CopyOut(coreOffset);
    }
  }

 private:
  __aicore__ inline void BufferGet() {
    zeroTensor = zeroBuf.Get<yType>();
    piTensor = piBuf.Get<yType>();
    nanTensor = nanBuf.Get<yType>();
    mask1 = maskBuf1.Get<uint8_t>();

    Duplicate(zeroTensor, static_cast<yType>(0.0), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
    Duplicate(piTensor, static_cast<yType>(constData.CONST_PI), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
    Duplicate(nanTensor, static_cast<yType>(NAN), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
  }

  __aicore__ inline void CopyIn(int64_t coreOffset) {
    // alloc tensor from queue memory
    LocalTensor<yType> xLocal = inQueue.AllocTensor<yType>();
    // copy progress_th tile from global tensor to local tensor
    DataCopy(xLocal, xGm[coreOffset], {1, blockLen, 0, 0});
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    // enque input tensors to VECIN queue
    inQueue.EnQue(xLocal);
  }

  __aicore__ inline void Compute() {
    // deque input tensors from VECIN queue
    LocalTensor<yType> input = inQueue.DeQue<yType>();
    LocalTensor<yType> result = outQueue.AllocTensor<yType>();

    // result = if input >= 0 then 0 else pi
    Compare(mask1, input, zeroTensor, CMPMODE::GE, this->mask, repeatTimes, this->repeatParams);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, zeroTensor, piTensor, this->mask, repeatTimes);
    PipeBarrier<PIPE_V>();

    // select nan
    Compare(mask1, input, input, CMPMODE::EQ, this->mask, repeatTimes, this->repeatParams);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, result, nanTensor, this->mask, repeatTimes);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    // enque the output tensor to VECOUT queue
    outQueue.EnQue<yType>(result);
    // free input tensors for reuse
    inQueue.FreeTensor(input);
  }

  __aicore__ inline void CopyOut(int64_t coreOffset) {
    // deque output tensor from VECOUT queue
    LocalTensor<yType> result = outQueue.DeQue<yType>();
    // copy progress_th tile from local tensor to global tensor
    DataCopy(yGm[coreOffset], result, {1, blockLen, 0, 0});
    // free output tensor for reuse
    outQueue.FreeTensor(result);
  }

 private:
  TPipe pipe;
  ConstData constData;
  uint8_t repeatTimes = 1;
  GlobalTensor<yType> xGm;
  GlobalTensor<yType> yGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<TPosition::VECCALC> maskBuf1;
  TBuf<TPosition::VECCALC> piBuf;
  TBuf<TPosition::VECCALC> nanBuf;
  TBuf<TPosition::VECCALC> zeroBuf;

  LocalTensor<yType> zeroTensor;
  LocalTensor<yType> piTensor;
  LocalTensor<yType> nanTensor;
  LocalTensor<uint8_t> mask1;
  int32_t dataPerBlock = 32 / sizeof(yType);
  uint16_t blockLen = 1;
};
} // AngleV2N
#endif  // _ANGLE_V2_H_

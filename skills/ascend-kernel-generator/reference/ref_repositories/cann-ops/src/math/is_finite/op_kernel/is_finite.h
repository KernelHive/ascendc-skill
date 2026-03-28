/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file is_finite.h
 * \brief
 */
#ifndef IS_FINITE_H
#define IS_FINITE_H

#include "kernel_operator.h"

namespace IsFiniteNs {
using namespace AscendC;

template <typename T, auto MASK>
class IsFinite {
 public:
  __aicore__ inline IsFinite(){};
  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR y,
                              GM_ADDR workspace,
                              const IsFiniteTilingData* tilingData);
  __aicore__ inline void Process();

  constexpr static uint8_t BUFFER_NUM = 2;
  constexpr static uint8_t DATA_BLOCK = 32;
  constexpr static uint8_t FLOAT_INTERVAL_TYPE = 2;
  constexpr static int16_t FLOAT_SHL_NUM = 16384;   // 01000000 00000000
 private:
  __aicore__ inline void CopyIn(uint64_t offset, int32_t calCount);
  __aicore__ inline void CopyOut(uint64_t offset, int32_t calCount);
  __aicore__ inline void ParseTilingData(const IsFiniteTilingData* tilingData);
  
  __aicore__ inline void Compute(int32_t computeCount);
  template <typename T1, typename T2>
  __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    return (a + b - 1) / b;
  };
  template <typename T1, typename T2>
  __aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
  };

 private:
  TPipe pipe;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

  TBuf<> cacheTensorBuff;
  GlobalTensor<int16_t> inputGM;
  GlobalTensor<uint8_t> outputGM;
  LocalTensor<int16_t> cacheTensor;

  uint8_t selectInterval = 0;
  uint32_t coreNum = 0;
  uint64_t tailCoreNum = 0;
  uint64_t perCoreDataCount = 0;
  uint64_t blockOffset = 0;
  uint32_t blockIdx = 0;
  uint32_t maxDataCount = 0;
  uint32_t actualMaxDataCount = 0;
  uint32_t usableUbSize = 0;
};

template <typename T, auto MASK>
__aicore__ inline void IsFinite<T, MASK>::Init(
                              GM_ADDR x,
                              GM_ADDR y,
                              GM_ADDR workspace,
                              const IsFiniteTilingData* tilingData) {

    inputGM.SetGlobalBuffer((__gm__ int16_t*)x);
    outputGM.SetGlobalBuffer((__gm__ uint8_t*)y);

    ParseTilingData(tilingData);

    maxDataCount = usableUbSize / sizeof(int16_t) / DATA_BLOCK * DATA_BLOCK;
    actualMaxDataCount = maxDataCount / selectInterval;

    pipe.InitBuffer(inputQueue, BUFFER_NUM, maxDataCount * sizeof(int16_t));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, actualMaxDataCount * sizeof(uint8_t));
    pipe.InitBuffer(cacheTensorBuff, maxDataCount * sizeof(int16_t));
}


template <typename T, auto MASK>
__aicore__ inline void IsFinite<T, MASK>::ParseTilingData(
    const IsFiniteTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    coreNum = tilingData->needCoreNum;
    usableUbSize = tilingData->usableUbSize;
    perCoreDataCount = tilingData->perCoreDataCount;
    tailCoreNum = tilingData->tailDataCoreNum;

    selectInterval = sizeof(T) / sizeof(int16_t);

    if (tailCoreNum == 0) {
      blockOffset = perCoreDataCount * blockIdx;
    } else {
      if ((blockIdx + 1) <= tailCoreNum) {
        perCoreDataCount += DATA_BLOCK;
        blockOffset = perCoreDataCount * blockIdx;
      } else {
        blockOffset = ((perCoreDataCount + DATA_BLOCK) * tailCoreNum) + (perCoreDataCount * (blockIdx - tailCoreNum));
      }
    }

    if (blockIdx == coreNum - 1) {
      perCoreDataCount = tilingData->lastCoreDataCount;
    }

    blockOffset *= selectInterval;
    perCoreDataCount *= selectInterval;
}


template <typename T, auto MASK>
__aicore__ inline void IsFinite<T, MASK>::CopyIn(const uint64_t offset, const int32_t calCount) {
  LocalTensor<int16_t> dataLocal = inputQueue.AllocTensor<int16_t>();
#if __CCE_AICORE__ == 220
  DataCopyParams copyParams;
  copyParams.blockCount = 1;
  copyParams.blockLen = calCount * sizeof(uint16_t);
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  DataCopyPad(dataLocal, inputGM[offset], copyParams, {false, 0, 0, 0});
#else
  DataCopy(dataLocal, inputGM[offset], calCount);
#endif
  inputQueue.EnQue(dataLocal);
}

template <typename T, auto MASK>
__aicore__ inline void IsFinite<T, MASK>::CopyOut(const uint64_t offset, const int32_t calCount) {
  LocalTensor<uint8_t> dstLocal = outputQueue.DeQue<uint8_t>();
#if __CCE_AICORE__ == 220
  DataCopyParams copyParams;
  copyParams.blockCount = 1;
  copyParams.blockLen = calCount * sizeof(uint8_t);
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  DataCopyPad(outputGM[offset], dstLocal, copyParams);
#else
  DataCopy(outputGM[offset], dstLocal, calCount);
#endif
  outputQueue.FreeTensor(dstLocal);
}

template <typename T, auto MASK>
__aicore__ inline void IsFinite<T, MASK>::Compute(const int32_t calCount) {
  LocalTensor<int16_t> srcTensor = inputQueue.DeQue<int16_t>();
  LocalTensor<uint8_t> dstTensor = outputQueue.AllocTensor<uint8_t>();
  cacheTensor = cacheTensorBuff.Get<int16_t>();

  Duplicate(cacheTensor, MASK, calCount);
  And(srcTensor, srcTensor, cacheTensor, calCount);
  Sub(srcTensor, srcTensor, cacheTensor, calCount);
  Maxs(srcTensor, srcTensor, (int16_t)-1, calCount);
  Muls(srcTensor, srcTensor, (int16_t)-1, calCount);
  
  uint32_t actualCalCount = calCount / selectInterval;
  if (selectInterval == FLOAT_INTERVAL_TYPE) {
    Muls(srcTensor, srcTensor, FLOAT_SHL_NUM, calCount);
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<float>(), RoundMode::CAST_NONE, actualCalCount);
    Mins(srcTensor, srcTensor, (int16_t)1, actualCalCount);
  }

  Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_CEIL, actualCalCount);

  inputQueue.FreeTensor(srcTensor);
  outputQueue.EnQue(dstTensor);
}

template <typename T, auto MASK>
__aicore__ inline void IsFinite<T, MASK>::Process() {
  uint32_t loopCount = perCoreDataCount / maxDataCount;
  uint32_t tailDataCount = perCoreDataCount % maxDataCount;
  uint64_t actualInOffset = blockOffset;
  uint64_t actualOutOffset = blockOffset / selectInterval;
  uint32_t actualOutCount = maxDataCount / selectInterval;

  for (uint32_t i = 0; i < loopCount; i++) {
    CopyIn(actualInOffset, maxDataCount);
    Compute(maxDataCount);
    CopyOut(actualOutOffset, actualOutCount);
    actualOutOffset += actualOutCount;
    actualInOffset += maxDataCount;
  }

  if (tailDataCount > 0) {
    uint32_t dataBlock = DATA_BLOCK / sizeof(int16_t);
    uint32_t dataCount = tailDataCount;
    actualOutCount = dataCount / selectInterval;
    
#if __CCE_AICORE__ == 220
    CopyIn(actualInOffset, tailDataCount);
    Compute(dataCount);
    CopyOut(actualOutOffset, actualOutCount);
#else
    dataCount = CeilAlign(tailDataCount , dataBlock);
    actualOutCount = dataCount / selectInterval;
    actualOutCount = CeilAlign(actualOutCount, DATA_BLOCK);
    CopyIn(actualInOffset, dataCount);
    Compute(dataCount);
    CopyOut(actualOutOffset, actualOutCount);
#endif
  }
}
} // namespace IsFiniteNs
#endif  // IS_FINITE_H
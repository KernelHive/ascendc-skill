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
 * @file angle_v2_u8.h
 */
#ifndef _ANGLE_V2_U8_H_
#define _ANGLE_V2_U8_H_

#include "angle_v2_base.h"

namespace AngleV2N {
using namespace AscendC;

template <typename yType>
class AngleV2U8 : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2U8() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
    blockLen = this->tileLength / dataPerBlock;

    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));
  }

  __aicore__ inline void Process() {
    LocalTensor<yType> zeroTensor = outQueue.AllocTensor<yType>();
    Duplicate(zeroTensor, static_cast<yType>(0.0), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);

    // loop count need to be doubled, due to double buffer
    for (int64_t i = 0; i < static_cast<int64_t>(this->tileNum); i++) {
      int64_t coreOffset = i * static_cast<int64_t>(this->tileLength);
      DataCopy(yGm[coreOffset], zeroTensor, {1, blockLen, 0, 0});
    }

    if (this->lastTileLength > 0) {
      int64_t coreOffset = static_cast<int64_t>(this->blockLength) - static_cast<int64_t>(this->lastTileLength);
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      blockLen = this->lastTileLength / dataPerBlock;
      DataCopy(yGm[coreOffset], zeroTensor, {1, blockLen, 0, 0});
    }
  }

 private:
  TPipe pipe;
  GlobalTensor<yType> yGm;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  uint8_t repeatTimes = 1;
  int32_t dataPerBlock = BLOCK_BYTES / sizeof(yType);
  uint16_t blockLen = 1;
};
} // AngleV2N
#endif  // _ANGLE_V2_U8_H_

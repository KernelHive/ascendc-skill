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
 * @file angle_v2_base.h
 */
#ifndef _ANGLE_V2_BASE_H_
#define _ANGLE_V2_BASE_H_

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AngleV2N {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t COEFFICENT = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t DATA_PER_REPEAT_B32 = 64;
constexpr uint8_t REPEAT_STRIDE_EIGHT = 8;
constexpr uint8_t REPEAT_STRIDE_FOUR = 4;

struct ConstData {
  const float CONST_PI = M_PI;
  const float CONST_PI_BY_TWO = CONST_PI / 2;
  const float CONST_PI_BY_FOUR = CONST_PI / 4;
  const float CONST_PI_BY_THREE_QUARTERS = CONST_PI * 3 / 4;
  const float CONST_PI_BY_EIGHT = CONST_PI / 8;
  const float CONST_TAN_PI_BY_EIGHT = 0.4142135623730950;

  const float CONST_NEG_PI = -CONST_PI;
  const float CONST_NEG_PI_BY_TWO = -CONST_PI_BY_TWO;
  const float CONST_NEG_PI_BY_FOUR = -CONST_PI_BY_FOUR;
  const float CONST_NEG_PI_BY_THREE_QUARTERS = -CONST_PI_BY_THREE_QUARTERS;
};

template <typename yType>
class AngleV2Base {
 public:
  __aicore__ inline AngleV2Base() {
  }
  __aicore__ inline void BaseMemberDataInit(const AngleV2TilingData* __restrict tilingData) {
    totalLength = tilingData->totalLength;
    alignNum = tilingData->alignNum;
    formerNum = tilingData->formerNum;
    formerLength = tilingData->formerLength;
    tailNum = tilingData->tailNum;
    tailLength = tilingData->tailLength;
    totalLengthAligned = tilingData->totalLengthAligned;
    tileLength = tilingData->tileLength;
    mask = tilingData->dataPerRepeat;
    
    if (GetBlockIdx() < formerNum) {
      blockLength = static_cast<uint64_t>(formerLength);
      offset = static_cast<int64_t>(blockLength) * GetBlockIdx();
    } else {
      blockLength = static_cast<uint64_t>(tailLength);
      offset = static_cast<int64_t>(formerLength) * static_cast<int64_t>(formerNum) +
               static_cast<int64_t>(tailLength) * (GetBlockIdx() - static_cast<int64_t>(formerNum));
    }
    CalTileLength();
    tileNum = blockLength / tileLength;
    lastTileLength = blockLength % tileLength;
  }

  __aicore__ inline void CalTileLength() {
#if (__CCE_AICORE__ >= 200)
    // calculate tileLength
    if (blockLength <= tileLength) {
        tileLength = (blockLength + mask - 1) / mask * mask;
    }
#else
    tileLength = mask;
#endif
  }

  template<typename T>
  __aicore__ inline void DoSelect(LocalTensor<T> &dstLocal, LocalTensor<uint8_t> &selMask,
                                  LocalTensor<T> &src0Local, LocalTensor<T> &src1Local, uint64_t mask,
                                  uint8_t repeatTimes) {
#if (__CCE_AICORE__ >= 200)
    // Select for select mode 2
    Select(dstLocal, selMask, src0Local, src1Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeatTimes, repeatParams);
#else
    // Select for select mode 0
    Select(dstLocal, selMask, src0Local, src1Local, SELMODE::VSEL_CMPMASK_SPR, mask, repeatTimes, repeatParams);
#endif
  }

 protected:
  uint32_t totalLength;
  uint32_t alignNum = 0;
  uint32_t totalLengthAligned = 0;
  uint32_t formerNum = 0;
  uint32_t formerLength = 0;
  uint32_t tailNum = 0;
  uint32_t tailLength = 0;
  uint64_t blockLength = 0;
  int64_t offset = 0;
  uint32_t tileNum = 0;
  uint32_t lastTileLength = 0;
  uint32_t tileLength = DATA_PER_REPEAT_B32;
  uint64_t mask = DATA_PER_REPEAT_B32;
  BinaryRepeatParams repeatParams = {1, 1, 1, REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT};
  UnaryRepeatParams CastDownParams= {1, 1, REPEAT_STRIDE_FOUR, REPEAT_STRIDE_EIGHT};
  UnaryRepeatParams CastKeepParams= {1, 1, REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT};
  UnaryRepeatParams CastHighParams= {1, 1, REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_FOUR};
  uint16_t dupDstBlockStride = 1;
  uint8_t dupDstRepeatStride = REPEAT_STRIDE_EIGHT;
};
} // AngleV2N
#endif  // _ANGLE_V2_BASE_H_

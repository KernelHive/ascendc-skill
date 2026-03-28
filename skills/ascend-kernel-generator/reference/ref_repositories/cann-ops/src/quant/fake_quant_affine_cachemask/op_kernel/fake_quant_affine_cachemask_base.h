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
 * \file fake_quant_affine_cachemask_base.h
 * \brief
 */

#ifndef _FAKE_QUANT_AFFINE_CACHEMASK_BASE_H_
#define _FAKE_QUANT_AFFINE_CACHEMASK_BASE_H_

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace FakeQuantAffineCachemaskN {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename yType>
class FakeQuantAffineCachemaskBase {
    public:
        __aicore__ inline FakeQuantAffineCachemaskBase() {}
        __aicore__ inline void BaseMemberDataInit(const FakeQuantAffineCachemaskTilingData* tilingData) {
            loopNum = tilingData->loopNum;
            remainNum = tilingData->remainNum;
            calcLength = tilingData->calcLength;
            headNum = tilingData->headNum;
            totalLengthAligned = tilingData->totalLengthAligned;
            tileLength = tilingData->tileLength;
            quantMin = static_cast<int32_t>(tilingData->quantMin);
            quantMax = static_cast<int32_t>(tilingData->quantMax);
            mask = tilingData->dataPerRepeat;
            offset = 0;
            scaleOffset = 0;
            circleNum = loopNum;

            if (remainNum == 0) {
                blockLength = totalLengthAligned * loopNum;
                offset = calcLength * loopNum * GetBlockIdx();
                scaleOffset = loopNum * GetBlockIdx();
            } else {
                if (GetBlockIdx() < remainNum) {
                  blockLength = totalLengthAligned * (loopNum + 1);
                  for (uint32_t i = 0; i < GetBlockIdx(); i++) {
                    offset += calcLength * (loopNum + 1);
                    scaleOffset = scaleOffset + (loopNum + 1);
                  }
                  circleNum = loopNum + 1;
                } else {
                  blockLength = totalLengthAligned * loopNum;
                  offset = calcLength * (remainNum + GetBlockIdx() * loopNum);
                  scaleOffset = remainNum + GetBlockIdx() * loopNum;
                }
            }
            tileNum = totalLengthAligned / tileLength;
            lastTileLength = totalLengthAligned % tileLength;
            lastActulTileLength = calcLength % tileLength;
        }
  
    template<typename T>
    __aicore__ inline void CommonBufferGet(
      TBuf<QuePosition::VECCALC> &infBuf, TBuf<QuePosition::VECCALC> &zeroBuf, TBuf<QuePosition::VECCALC> &oneBuf, TBuf<QuePosition::VECCALC> &quantMinQueueBuf, TBuf<QuePosition::VECCALC> &quantMaxQueueBuf,
      LocalTensor<T> &infTensor, LocalTensor<T> &zeroTensor, LocalTensor<T> &oneTensor, LocalTensor<T> &quantMinTensor, LocalTensor<T> &quantMaxTensor, uint32_t coreLength) {
      quantMinTensor = quantMinQueueBuf.Get<T>();
      quantMaxTensor = quantMaxQueueBuf.Get<T>();
      zeroTensor = zeroBuf.Get<T>();
      oneTensor = oneBuf.Get<T>();
      infTensor = infBuf.Get<T>();

      Duplicate(quantMinTensor, static_cast<T>(quantMin), coreLength);
      Duplicate(quantMaxTensor, static_cast<T>(quantMax), coreLength);
      Duplicate(oneTensor, static_cast<T>(1.0f), coreLength);
      Duplicate(zeroTensor, static_cast<T>(0.0f), coreLength);
      Duplicate(infTensor, static_cast<T>(0x80000000), coreLength);
    }

    template<typename T>
    __aicore__ inline void CommonCopyIn(TQue<QuePosition::VECIN, BUFFER_NUM> &inQueueData, GlobalTensor<T> &xGm,
                                        uint32_t calcOffset, uint32_t coreLength) {
      // alloc tensor from queue memory
      LocalTensor<T> xLocal = inQueueData.AllocTensor<T>();

      // copy progress_th tile from global tensor to local tensor
      DataCopyExtParams copyParams{1, static_cast<uint32_t>(coreLength * sizeof(T)), 0, 0, 0}; 
      DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
      DataCopyPad(xLocal, xGm[calcOffset], copyParams, padParams);

      // enque input tensors to VECIN queue
      inQueueData.EnQue(xLocal);
    }

    template<typename T>
    __aicore__ inline void CommonCopyOut(TQue<QuePosition::VECOUT, BUFFER_NUM> &outQueueOut, TQue<QuePosition::VECOUT, BUFFER_NUM>&outQueueMask,
      GlobalTensor<T> &yGm, GlobalTensor<uint8_t> &maskGm, uint32_t calcOffset, uint32_t coreLength) {
      // deque output tensor from VECOUT queue
      LocalTensor<T> yLocal = outQueueOut.DeQue<T>();
      LocalTensor<uint8_t> maskLocal = outQueueMask.DeQue<uint8_t>();
      DataCopyExtParams outCopyOutParams;
      outCopyOutParams.blockCount = 1;
      outCopyOutParams.blockLen = coreLength * sizeof(T);
      outCopyOutParams.dstStride = 0;
      outCopyOutParams.srcStride = 0;
      outCopyOutParams.rsv = 0;
      DataCopyExtParams maskCopyOutParams;
      maskCopyOutParams.blockCount = 1;
      maskCopyOutParams.blockLen = coreLength * sizeof(uint8_t);
      maskCopyOutParams.dstStride = 0;
      maskCopyOutParams.srcStride = 0;
      maskCopyOutParams.rsv = 0;

      DataCopyPad(yGm[calcOffset], yLocal, outCopyOutParams);
      DataCopyPad(maskGm[calcOffset], maskLocal, maskCopyOutParams);

      // free output tensor for reuse
      outQueueOut.FreeTensor(yLocal);
      outQueueMask.FreeTensor(maskLocal);
    }

    protected:
        uint32_t headNum, calcLength, loopNum, remainNum, circleNum, tileNum, totalLengthAligned, tileLength,
          blockLength, offset, scaleOffset, lastTileLength, lastActulTileLength;
        int32_t quantMin, quantMax;
        uint64_t mask;
};
} // FakeQuantAffineCachemaskN
#endif  // _FAKE_QUANT_AFFINE_CACHEMASK_BASE_H_

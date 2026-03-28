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
 * @file expand_v2.cpp
 */
#include "kernel_operator.h"

constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t UB_BLOCK_SIZE = 32; // 32B对齐

class KernelExpand {
public:
    __aicore__ inline KernelExpand() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t expandSize, uint64_t blockLength, uint64_t tileNum, uint64_t tileLength, uint64_t miniTileLength)
    {
        uint64_t repeatCount = expandSize / AscendC::GetBlockNum();
        uint64_t delta = expandSize % AscendC::GetBlockNum();
        int bigCore = 0; // 1表示多做1次的核，0是expandSize / AscendC::GetBlockNum()这种核
        if (AscendC::GetBlockIdx() < delta) {
            repeatCount += 1;
            bigCore = 1;
        }
        this->blockLength = blockLength;
        this->repeatCount = repeatCount;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->miniTileLength = miniTileLength;

        this->dummyLength = 0;
        const uint64_t ubBlockLength = UB_BLOCK_SIZE / sizeof(DTYPE_X);
        if (this->miniTileLength % ubBlockLength) {
            this->dummyLength = (this->miniTileLength / ubBlockLength + 1) * ubBlockLength - this->miniTileLength;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->blockLength);
        if (bigCore) {
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx() * repeatCount, this->blockLength * repeatCount);
        } else {
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y +  this->blockLength * (repeatCount + 1) * delta + this->blockLength * (AscendC::GetBlockIdx() - delta) * repeatCount, this->blockLength * repeatCount);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < this->repeatCount; i++) {
            const int64_t loopCount = this->tileNum;
            const int64_t delta = i * (loopCount + 1);
            for (uint64_t j = 0; j < loopCount; j++) {
                CopyIn(j, this->tileLength);
                Compute(this->tileLength);
                CopyOut(delta + j, this->tileLength);
            }
            if (this->miniTileLength) {
                CopyIn(loopCount, this->miniTileLength);
                Compute(this->miniTileLength);
                CopyOut(delta + loopCount, this->miniTileLength);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, uint64_t length)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        if (length % (UB_BLOCK_SIZE / sizeof(DTYPE_X))) {
            uint16_t copyBytes = length * sizeof(DTYPE_X);
            AscendC::DataCopyParams dataCopyParams = {1, copyBytes, 0, 0};
            AscendC::DataCopyPadParams padParams = {true, 0, this->dummyLength, 0};
            AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], dataCopyParams, padParams);
        } else {
            AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint64_t length)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        if (length == this->tileLength) {
            AscendC::DataCopy(yLocal, xLocal, length);
        } else {
            AscendC::DataCopy(yLocal, xLocal, length + this->dummyLength);
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress, uint64_t length)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        if (length % (UB_BLOCK_SIZE / sizeof(DTYPE_Y))) {
            //AscendC::SetAtomicAdd<DTYPE_Y>();
            AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, length + this->dummyLength);
            //AscendC::SetAtomicNone();
        } else {
            AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, length);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> shapeGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t miniTileLength;
    uint64_t loopCount;
    uint64_t repeatCount; // 重复次数
    uint8_t dummyLength;
};

extern "C" __global__ __aicore__ void expand_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelExpand op;
    op.Init(x, y, tiling_data.expandSize, tiling_data.blockLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.miniTileLength);
    op.Process();
}
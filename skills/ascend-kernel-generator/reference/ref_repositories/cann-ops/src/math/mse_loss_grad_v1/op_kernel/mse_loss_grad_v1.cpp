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
 * @file mse_loss_grad_v1.cpp
 */

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMseLossGrad {
public:
    __aicore__ inline KernelMseLossGrad() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y,
                                uint32_t remain_start, uint32_t mode, uint32_t totalLength, 
                                uint32_t blockLength, uint32_t tileNum, uint32_t tileLength,
                                uint32_t lastTileLength, uint32_t tilingKey) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->compute_mode = static_cast<int>(mode);
        this->totalLength = static_cast<int32_t>(totalLength);
        this->totalLength_f32 = static_cast<float>(this->totalLength);
        this->tiling_mode = static_cast<int>(tilingKey);
        this->remain_start = remain_start;

        this->blockLength = blockLength;
        this->tileNum =
            tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = tileLength / BUFFER_NUM;
        this->lastTileLength = lastTileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predict + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)label + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Y*)dout + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        pipe.InitBuffer(inQueueIN, BUFFER_NUM, this->tileLength * 3 * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = static_cast<int32_t>(this->tileNum) * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> inLocal = inQueueIN.AllocTensor<DTYPE_Y>();
        if (this->tiling_mode == 1) {
            AscendC::DataCopy(inLocal[0], xGm[0], this->lastTileLength);
            AscendC::DataCopy(inLocal[this->tileLength], yGm[0], this->lastTileLength);
            AscendC::DataCopy(inLocal[2 * (this->tileLength)], zGm[0], this->lastTileLength);
            for (int32_t index = remain_start; index < this->lastTileLength; index++) {
                inLocal.SetValue(index, xGm.GetValue(index));
                inLocal.SetValue(index + this->tileLength, yGm.GetValue(index));
                inLocal.SetValue(index + this->tileLength + this->tileLength, zGm.GetValue(index));
            }
        } 
        else if (this->tiling_mode == 2) {
            if (progress == this->tileNum - 1 && this->lastTileLength != 0) {
                AscendC::DataCopy(inLocal[0], xGm[progress * this->tileLength], this->lastTileLength);
                AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength], this->lastTileLength);
                AscendC::DataCopy(inLocal[2 * (this->tileLength)], zGm[progress * this->tileLength], this->lastTileLength);
                for (int32_t index = remain_start; index < this->lastTileLength; index++) {
                    inLocal.SetValue(index, xGm.GetValue(index + progress * this->tileLength));
                    inLocal.SetValue(index + this->tileLength, yGm.GetValue(index + progress * this->tileLength));
                    inLocal.SetValue(index + this->tileLength + this->tileLength, zGm.GetValue(index + progress * this->tileLength));
                }
            } 
            else {
                AscendC::DataCopy(inLocal[0], xGm[progress * this->tileLength], this->tileLength);
                AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength], this->tileLength);
                AscendC::DataCopy(inLocal[2 * (this->tileLength)], zGm[progress * this->tileLength], this->tileLength);
            }
        }
        inQueueIN.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> inLocal = inQueueIN.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> xLocal = inLocal;
        AscendC::LocalTensor<DTYPE_Y> yLocal = inLocal[this->tileLength];
        AscendC::LocalTensor<DTYPE_Y> zLocal = inLocal[2 * (this->tileLength)];
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueOUT.AllocTensor<DTYPE_Y>();

        AscendC::Sub(outLocal, xLocal, yLocal, this->tileLength);
        AscendC::Muls(outLocal, outLocal, (DTYPE_Y)2, this->tileLength);

        // 如果reduction是mean
        if (this->compute_mode == 1) {
            DTYPE_Y len = static_cast<DTYPE_Y>(this->totalLength_f32);
            AscendC::Duplicate(yLocal, (DTYPE_Y)0, this->tileLength);
            AscendC::Adds(yLocal, yLocal, len, this->tileLength);
            AscendC::Div(outLocal, outLocal, yLocal, this->tileLength);
        } 
        AscendC::Mul(outLocal, outLocal, zLocal, this->tileLength);
        outQueueOUT.EnQue<DTYPE_Y>(outLocal);
        inQueueIN.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueOUT.DeQue<DTYPE_Y>();

        if (this->tiling_mode == 1) {
            AscendC::DataCopy(outGm[0], outLocal, this->lastTileLength);
            for (int32_t index = remain_start; index < this->lastTileLength; index++) {
                outGm.SetValue(index, outLocal.GetValue(index));
            }
        } 
        else if (this->tiling_mode == 2) {
            if (progress == this->tileNum - 1 && this->lastTileLength != 0) {
                AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->lastTileLength);
                for (int32_t index = remain_start; index < this->lastTileLength; index++) {
                    outGm.SetValue(index + progress * this->tileLength, outLocal.GetValue(index));
                }
            } else {
                AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
            }
        }
        outQueueOUT.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueIN;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
    AscendC::GlobalTensor<DTYPE_Y> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Y> zGm;
    AscendC::GlobalTensor<DTYPE_Y> outGm;
    int compute_mode;
    int tiling_mode;
    int32_t totalLength;
    float totalLength_f32;
    uint32_t remain_start;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};


extern "C" __global__ __aicore__ void mse_loss_grad_v1(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLossGrad op;
    uint32_t tilingKey = 1;
    if (TILING_KEY_IS(1)) {
        tilingKey = 1;
    } else if (TILING_KEY_IS(2)) {
        tilingKey = 2;
    }

    op.Init(predict, label, dout, y, tiling_data.remain_start, tiling_data.mode, 
            tiling_data.totalLength, tiling_data.blockLength,
            tiling_data.tileNum, tiling_data.tileLength,
            tiling_data.lastTileLength, tilingKey);
    op.Process();
}
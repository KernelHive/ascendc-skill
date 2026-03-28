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
 * @file scatter_max.cpp
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename T_VAR, typename T_INDICES> class ScatterMaxGrad {
public:
    __aicore__ inline ScatterMaxGrad() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                uint32_t lastdim, uint32_t totalLength, uint32_t ALIGN_NUM, 
                                uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->lastdim = lastdim;
        this->totalLength = totalLength;
        this->ALIGN_NUM = ALIGN_NUM;

        auto startPointer = core_size * AscendC::GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T_VAR*)var, totalLength);
        yGm.SetGlobalBuffer((__gm__ T_INDICES*)indices + startPointer, bufferlength);
        dGm.SetGlobalBuffer((__gm__ T_VAR*)updates + startPointer * this->lastdim, bufferlength * this->lastdim);

        this->tileNum = this->lastdim / this->tileLength + (this->lastdim % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T_VAR));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(inQueueD, BUFFER_NUM, this->tileLength * sizeof(T_VAR));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T_VAR));
    }
    __aicore__ inline void Process() {
        for(int32_t j = 0; j < this->blockLength; j++) {
            T_INDICES p = yGm.GetValue(j);
            int32_t loopCount = this->tileNum;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyIn(p, j, i, this->tileLength, 0);
                Compute(i, this->tileLength);
                CopyOut(p, i, this->tileLength);
            }
            auto padding = (this->lastdim % this->ALIGN_NUM ? this->ALIGN_NUM - this->lastdim % this->ALIGN_NUM : 0);
            auto length = this->lastdim + padding;
            length = length - this->tileLength * (loopCount - 1);
            CopyIn(p, j, loopCount - 1, length, padding);
            Compute(loopCount - 1, length);
            CopyOut(p, loopCount - 1, length);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t p, int32_t j, int32_t progress, uint32_t length, uint32_t padding) {
        AscendC::LocalTensor<T_VAR> dLocal = inQueueD.AllocTensor<T_VAR>();
        AscendC::DataCopy(dLocal, dGm[j * this->lastdim + progress * this->tileLength], length);
        for(int i=length-padding;i<length;i++){
            if constexpr (std::is_same_v<T_VAR, half> || std::is_same_v<T_VAR, float>) {
                dLocal.SetValue(i, -3.40282346638528859811704183484516925e+38F);
            }
            if constexpr (std::is_same_v<T_VAR, int32_t>) {
                dLocal.SetValue(i, -2147483647-1);
            }
            if constexpr (std::is_same_v<T_VAR, int8_t>) {
                dLocal.SetValue(i, -128);
            }
        }
        
        inQueueD.EnQue(dLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<T_VAR> dLocal = inQueueD.DeQue<T_VAR>();
        AscendC::LocalTensor<T_VAR> zLocal = outQueueZ.AllocTensor<T_VAR>();

        AscendC::DataCopy(zLocal, dLocal, length);

        outQueueZ.EnQue<T_VAR>(zLocal);
        inQueueD.FreeTensor(dLocal);
    }
    __aicore__ inline void CopyOut(int32_t p, int32_t progress, uint32_t length) {
        AscendC::LocalTensor<T_VAR> zLocal = outQueueZ.DeQue<T_VAR>();

        AscendC::SetAtomicMax<T_VAR>();
        AscendC::DataCopy(xGm[p * this->lastdim + progress * this->tileLength], zLocal, length);
        AscendC::SetAtomicNone();

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueD;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T_VAR> xGm;
    AscendC::GlobalTensor<T_INDICES> yGm;
    AscendC::GlobalTensor<T_VAR> dGm;
    uint32_t lastdim;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t ALIGN_NUM;
};

extern "C" __global__ __aicore__ void scatter_max(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    ScatterMaxGrad<DTYPE_VAR, DTYPE_INDICES> op;
    op.Init(var, indices, updates, tiling_data.lastdim, 
            tiling_data.totalLength, tiling_data.ALIGN_NUM,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}
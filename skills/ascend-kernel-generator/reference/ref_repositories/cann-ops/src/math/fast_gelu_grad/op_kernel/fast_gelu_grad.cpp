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
 * @file fast_gelu_grad.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelFastGeluGrad {
public:
    __aicore__ inline KernelFastGeluGrad() {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z,
                                uint32_t totalLength, uint32_t ALIGN_NUM,
                                uint32_t block_size, uint32_t core_size,
                                uint32_t core_remain) {
        this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * AscendC::GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY*)dy + startPointer, bufferlength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_DY));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(signbitBuffer, this->tileLength * sizeof(DTYPE_X));
        this->signbit = signbitBuffer.Get<DTYPE_X>();
        if constexpr (std::is_same_v<DTYPE_X, float>) {
            Duplicate(signbit.ReinterpretCast<uint32_t>(), uint32_t(2147483648u), this->tileLength);
        }
        else {
            Duplicate(signbit.ReinterpretCast<uint16_t>(), uint16_t(32768u), this->tileLength);
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueDY.AllocTensor<DTYPE_DY>();
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        AscendC::DataCopy(dyLocal, dyGm[progress * this->tileLength], length);
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);

        inQueueDY.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuffer.Get<DTYPE_X>();

        DTYPE_X c2 = 1.702, c3 = 1.0;

        AscendC::Muls(xLocal, xLocal, c2, length);     // xLocal = 1.702x
        if constexpr (std::is_same_v<DTYPE_X, float>) { // tmp = 1.702|x|
            AscendC::Or(tmp.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), signbit.ReinterpretCast<uint16_t>(), length * 2);
        }
        else {
            AscendC::Or(tmp.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), signbit.ReinterpretCast<uint16_t>(), length);  
        }
        AscendC::Add(zLocal, xLocal, tmp, length);     // 1.702(x-|x|)
        AscendC::Exp(zLocal, zLocal, length);          // e^(1.702(x-|x|))
        AscendC::Exp(tmp, tmp, length);                // e^(-1.702|x|)
        AscendC::Mul(xLocal, xLocal, tmp, length);     // 1.702xe^(-1.702|x|)
        AscendC::Add(xLocal, xLocal, tmp, length);     // e^(-1.702|x|) + 1.702xe^(-1.702|x|)
        AscendC::Add(zLocal, xLocal, zLocal, length);  // e^(-1.702|x|) + 1.702xe^(-1.702|x|) + e^(1.702(x-|x|))
        AscendC::Adds(tmp, tmp, c3, length);           // e^(-1.702|x|) + 1
        AscendC::Mul(tmp, tmp, tmp, length);           // (e^(-1.702|x|) + 1)^2
        AscendC::Div(zLocal, zLocal, tmp, length);
        AscendC::Mul(zLocal, zLocal, dyLocal, length);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueDY.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer, signbitBuffer;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDY, inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_DY> dyGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    AscendC::LocalTensor<DTYPE_X> signbit;
};

extern "C" __global__ __aicore__ void fast_gelu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelFastGeluGrad op;
    op.Init(dy, x, z, tiling_data.totalLength, 
            tiling_data.ALIGN_NUM, tiling_data.block_size,
            tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}
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
 * \file mse_loss_grad_v2_310p.h
 * \brief
 */
#ifndef _MSE_LOSS_GRAD_V2_310P_H
#define _MSE_LOSS_GRAD_V2_310P_H
#pragma once
#include <cstdint>
#include "mse_loss_grad_v2_base.h"
using namespace AscendC;

template <typename inType>
class KernelMseLossGrad310P:public KernelMseLossGradBase<inType> {
public:
    __aicore__ inline KernelMseLossGrad310P() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y,
                                float cof, uint64_t totalLength, uint64_t tileNum,
                                uint64_t blockLength, uint64_t padLength, uint64_t usedDb) {
        this->cof = cof;
        this->totalLength = static_cast<int32_t>(totalLength);
        this->blockLength = blockLength;
        uint32_t blockNum = GetBlockNum();
        this->gmSize = blockLength;
        this->tileNum = tileNum;
        this->bufferNum = 1;
        if (static_cast<int32_t>(usedDb) == 1) {
            // 2 means using double buffer
            this->bufferNum = 2;
            this->tileNum = this->tileNum * this->bufferNum;
        }
        this->tileLength = this->blockLength / this->tileNum;
        if (static_cast<int32_t>(GetBlockIdx()) == static_cast<int32_t>(blockNum) - 1
            && static_cast<int32_t>(padLength) != 0) {
            this->tileNum = 1;
            this->bufferNum = 1;
            this->tileLength = padLength;
            this->gmSize = padLength;
            this->tileLengthAlign = this->CeilAlign(this->tileLength);
            this->tileLength = this->tileLengthAlign;
            this->tileLengthPtr = this->tileLength;
            this->tailLen = padLength;
            if (this->tailLen != 0) {
                // 32 means the alignment size of memory access
                this->tailStart = 32 * static_cast<uint32_t>(this->tailLen / 32);
            }
        } else {
            this->tileLengthAlign = this->CeilAlign(this->tileLength);
            this->tileNum = this->blockLength / this->tileLengthAlign;
            this->tileLength = this->tileLengthAlign;
            this->tileLengthPtr = this->tileLength;
            this->tailLen = this->blockLength - this->tileNum * this->tileLength;
            if (this->tailLen != 0) {
                // 32 means the alignment size of memory access
                this->tailStart = 32 * static_cast<uint32_t>(this->tailLen / 32);
                ++this->tileNum;
            }
        }

        uint32_t gmOffset = this->blockLength * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ inType*)predict + gmOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ inType*)label + gmOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ inType*)dout + gmOffset, this->blockLength);
        outGm.SetGlobalBuffer((__gm__ inType*)y + gmOffset, this->blockLength);

        uint32_t queueOffset = this->tileLengthAlign * sizeof(inType);
        // 3 means there are 3 inputs
        pipe.InitBuffer(inQueueIN, this->bufferNum, queueOffset * 3);
        pipe.InitBuffer(outQueueOUT, this->bufferNum, queueOffset);

        if constexpr(IsSameType<inType, half>::value) {
            uint32_t calcOffset = this->tileLengthAlign * sizeof(float);
            pipe.InitBuffer(resultTmpBuf, calcOffset);
            // 3 means there are 3 inputs
            pipe.InitBuffer(calcValueLocal, calcOffset * 3);
        }
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<inType> inLocal = inQueueIN.AllocTensor<inType>();
        uint32_t offset = progress * this->tileLengthPtr;
        DataCopy<inType>(inLocal[0], xGm[offset], this->tileLengthAlign);
        DataCopy<inType>(inLocal[this->tileLengthAlign], yGm[offset], this->tileLengthAlign);
        // 2 means two tileLengthAlign offsets are needed
        DataCopy<inType>(inLocal[2 * this->tileLengthAlign], zGm[offset], this->tileLengthAlign);
        inQueueIN.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<inType> inLocal = inQueueIN.DeQue<inType>();
        if constexpr(IsSameType<inType, half>::value) {
            // convert bf16 to fp32
            LocalTensor <float> calcValueLocalFP32 = calcValueLocal.Get<float>();
            // 3 means there are 3 inputs
            Cast(calcValueLocalFP32, inLocal, RoundMode::CAST_NONE, this->tileLengthAlign * 3);
            LocalTensor <float> xLocal = calcValueLocalFP32;
            LocalTensor <float> yLocal = calcValueLocalFP32[this->tileLengthAlign];
            LocalTensor <float> zLocal = calcValueLocalFP32[2 * (this->tileLengthAlign)];
            LocalTensor <float> outLoclFp32 = resultTmpBuf.Get<float>();
            Sub(outLoclFp32, xLocal, yLocal, this->tileLengthPtr);
            Muls(outLoclFp32, outLoclFp32, static_cast<float>(this->cof), this->tileLengthPtr);
            Mul(outLoclFp32, outLoclFp32, zLocal, this->tileLengthPtr);
            // convert fp32 to bf16
            LocalTensor<half> outLocal = outQueueOUT.AllocTensor<half>();
            // 3 means there are 3 inputs
            Cast(outLocal, outLoclFp32, RoundMode::CAST_NONE, this->tileLengthAlign * 3);
            outQueueOUT.EnQue(outLocal);
        } else {
            LocalTensor <inType> xLocal = inLocal;
            LocalTensor <inType> yLocal = inLocal[this->tileLengthAlign];
            LocalTensor <inType> zLocal = inLocal[2 * (this->tileLengthAlign)];
            LocalTensor <inType> outLocal = outQueueOUT.AllocTensor<inType>();
            Sub(outLocal, xLocal, yLocal, this->tileLengthPtr);
            Muls(outLocal, outLocal, static_cast <inType>(this->cof), this->tileLengthPtr);
            Mul(outLocal, outLocal, zLocal, this->tileLengthPtr);
            outQueueOUT.EnQue(outLocal);
        }
        inQueueIN.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor <inType> outLocal = outQueueOUT.DeQue <inType>();
        uint32_t offset = progress * this->tileLengthPtr;
        if (progress == this->tileNum - 1 && this->tailLen != 0) {
            DataCopy(this->outGm[offset], outLocal, this->tileLengthPtr); // 先搬运对齐的32个元素
            for(uint32_t index = this->tailStart; index < this->tailLen; index ++) {
                this->outGm.SetValue(index + offset, outLocal.GetValue(index));
            }
        } else {
            DataCopy(this->outGm[offset], outLocal, this->tileLengthPtr);
        }
        outQueueOUT.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    GlobalTensor<inType> xGm;
    GlobalTensor<inType> yGm;
    GlobalTensor<inType> zGm;
    GlobalTensor<inType> outGm;
    TBuf<TPosition::VECCALC> resultTmpBuf;
    TBuf<TPosition::VECCALC> calcValueLocal;
    uint32_t tailStart= 0;
    uint32_t tailLen = 0;
    TQue<QuePosition::VECIN, 1, &conf> inQueueIN;
    TQue<QuePosition::VECOUT, 1, &conf> outQueueOUT;
};
#endif // _MSE_LOSS_GRAD_V2_310P_H

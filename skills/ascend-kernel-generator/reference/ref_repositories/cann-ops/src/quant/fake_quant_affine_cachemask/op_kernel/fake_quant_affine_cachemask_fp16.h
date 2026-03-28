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
 * \file fake_quant_affine_cachemask_fp16.h
 * \brief
 */

#ifndef _FAKE_QUANT_AFFINE_CACHEMASK_FP16_H_
#define _FAKE_QUANT_AFFINE_CACHEMASK_FP16_H_
#include "fake_quant_affine_cachemask_base.h"

namespace FakeQuantAffineCachemaskN {
using namespace AscendC;
using namespace std;

template <typename yType>
class FakeQuantAffineCachemaskFp16 : public FakeQuantAffineCachemaskBase<yType> {
    public:
        __aicore__ inline FakeQuantAffineCachemaskFp16() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR zero_point, GM_ADDR y, GM_ADDR mask,
        const FakeQuantAffineCachemaskTilingData* tilingData) {
            this->BaseMemberDataInit(tilingData);

            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + this->offset, this->blockLength);
            scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(scale) + this->scaleOffset, this->circleNum);
            zeroGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(zero_point) + this->scaleOffset, this->circleNum);
            yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);
            maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask) + this->offset, this->blockLength);

            // pipe alloc memory to queue, the unit is Bytes
            pipe.InitBuffer(inQueueData, BUFFER_NUM, this->tileLength * sizeof(yType));
            pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileLength * sizeof(yType));
            pipe.InitBuffer(outQueueMask, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(calcBuf, BUFFER_NUM * this->tileLength * sizeof(float));
            pipe.InitBuffer(calcHf16Buf, BUFFER_NUM * this->tileLength * sizeof(yType));
            pipe.InitBuffer(calcInt32Buf, BUFFER_NUM * this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(maskBuf, BUFFER_NUM * this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(selectBuf, BUFFER_NUM * this->tileLength * sizeof(yType));
            
            pipe.InitBuffer(quantMinQueueBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(quantMaxQueueBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(oneBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(infBuf, this->tileLength * sizeof(yType));
        }

        __aicore__ inline void Process() {
            this->CommonBufferGet(infBuf, zeroBuf, oneBuf, quantMinQueueBuf, quantMaxQueueBuf,
                infTensor, zeroTensor, oneTensor, quantMinTensor, quantMaxTensor, this->tileLength);
            for (uint32_t i = 0; i < this->circleNum; i++) {
                scaleValue = static_cast<float>(scaleGm.GetValue(i));
                zeroPointValue = zeroGm.GetValue(i);

                uint32_t dataOffset = i * this->calcLength;
                for (uint32_t j = 0; j < this->tileNum; j++) {
                    uint32_t calcOffset = dataOffset + j * this->tileLength;
                    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
                    this->CommonCopyIn(inQueueData, xGm, calcOffset, this->tileLength);
                    Compute(this->tileLength);
                    this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->tileLength);
                }

                if (this->lastTileLength > 0) {
                    uint32_t calcOffset = dataOffset + this->tileNum * this->tileLength;
                    repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
                    this->CommonCopyIn(inQueueData, xGm, calcOffset, this->lastActulTileLength);
                    Compute(this->tileLength);
                    this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->lastActulTileLength);
                }
            }
        }

    private:
        __aicore__ inline void Compute(uint32_t calCount) {
            LocalTensor<yType> xLocal = inQueueData.DeQue<yType>();
            LocalTensor<yType> yLocal = outQueueOut.AllocTensor<yType>();
            LocalTensor<uint8_t> maskLocal = outQueueMask.AllocTensor<uint8_t>();

            curTemp = calcBuf.Get<float>();
            curHf16Temp = calcHf16Buf.Get<yType>();
            curInt32Temp = calcInt32Buf.Get<int32_t>();
            selectTemp = selectBuf.Get<yType>();
            maskTemp = maskBuf.Get<uint8_t>();

            // tmp = x / scale + zero_point
            Cast(curTemp, xLocal, RoundMode::CAST_NONE, calCount);
            Muls(curTemp, curTemp, static_cast<float>(1.0f / scaleValue), calCount);
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
            Cast(curTemp, curInt32Temp, RoundMode::CAST_NONE, calCount);
            Adds(curTemp, curTemp, static_cast<float>(zeroPointValue), calCount);
            Cast(curHf16Temp, curTemp, RoundMode::CAST_RINT, calCount);


            // maskTemp = (round(tmp) >= quant_min) & (round(tmp) <= quant_max)
            Compare(maskTemp, curHf16Temp, quantMinTensor, CMPMODE::GE, calCount);
            BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
            Select(selectTemp, maskTemp, oneTensor, zeroTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
            Compare(maskTemp, curHf16Temp, quantMaxTensor, CMPMODE::LE, calCount);
            Select(curHf16Temp, maskTemp, oneTensor, zeroTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
            Mul(curHf16Temp, selectTemp, curHf16Temp, calCount);
            Cast(maskLocal, curHf16Temp, RoundMode::CAST_RINT, calCount);

            // output = (Mins(Maxs(tmp, quant_min), quant_max) - zero_point) * scale
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
            Maxs(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMin), calCount);
            Mins(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMax), calCount);
            Cast(curTemp, curInt32Temp, RoundMode::CAST_ROUND, calCount);
            Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount);
            Select(curTemp, maskTemp, curTemp, static_cast<float>(this->quantMin), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->mask, repeatTimes, repeatParams);
            Compare(maskTemp, xLocal, infTensor, CMPMODE::NE, calCount);
            Select(curTemp, maskTemp, curTemp, static_cast<float>(this->quantMin), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->mask, repeatTimes, repeatParams);
            Adds(curTemp, curTemp, static_cast<float>(-1 * zeroPointValue), calCount);
            Muls(curTemp, curTemp, static_cast<float>(scaleValue), calCount);
            Cast(yLocal, curTemp, RoundMode::CAST_RINT, calCount);

            outQueueOut.EnQue<yType>(yLocal);
            outQueueMask.EnQue<uint8_t>(maskLocal);
            inQueueData.FreeTensor(xLocal);
        }

    private:
        TPipe pipe;

        int32_t zeroPointValue = 1;
        float scaleValue = 1.0;
        uint8_t repeatTimes = 0;

        GlobalTensor<yType> xGm, scaleGm, yGm;
        GlobalTensor<int32_t> zeroGm;
        GlobalTensor<uint8_t> maskGm;

        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut, outQueueMask;
        TBuf<QuePosition::VECCALC> calcBuf, calcHf16Buf, calcInt32Buf, maskBuf, infBuf, zeroBuf, oneBuf, selectBuf,
            quantMinQueueBuf, quantMaxQueueBuf;

        LocalTensor<yType> curHf16Temp, infTensor, zeroTensor, oneTensor, quantMinTensor, quantMaxTensor, selectTemp;
        LocalTensor<float> curTemp;
        LocalTensor<int32_t> curInt32Temp;
        LocalTensor<uint8_t> maskTemp;
};
} // FakeQuantAffineCachemaskN
#endif // _FAKE_QUANT_AFFINE_CACHEMASK_FP16_H_
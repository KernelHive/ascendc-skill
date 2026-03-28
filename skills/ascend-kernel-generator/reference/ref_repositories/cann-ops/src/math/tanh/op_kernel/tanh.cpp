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
 * @file tanh.cpp
 */
#include "kernel_operator.h"
#define GENERAL_OP_IMPL(templateClass, ...)                                     \
    do                                                                          \
    {                                                                           \
        GET_TILING_DATA(tiling_data, tiling);                                   \
        templateClass<__VA_ARGS__> op;                                          \
        op.Init(x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, \
                tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,     \
                tiling_data.tileDataNum, tiling_data.smallTailDataNum,          \
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum);          \
        op.Process();                                                           \
    } while (0)

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
class KernelTanh
{
    using T = TYPE_X;

public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if constexpr (IsExistBigCore)
        {
            if (coreNum < tailBlockNum)
            {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = finalBigTileNum;
                this->tailDataNum = bigTailDataNum;
            }
            else
            {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = finalSmallTileNum;
                this->tailDataNum = smallTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe.InitBuffer(tmpBuf, this->tileDataNum * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();

        if constexpr (!std::is_same_v<T, float>)
        {
            AscendC::Cast(tmp, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Tanh(tmp, tmp, this->processDataNum);
            AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }
        else
        {
            AscendC::Tanh(yLocal, xLocal, this->processDataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(1))
    {
        GENERAL_OP_IMPL(KernelTanh, DTYPE_X, DTYPE_Y, true);
    }
    else if (TILING_KEY_IS(0))
    {
        GENERAL_OP_IMPL(KernelTanh, DTYPE_X, DTYPE_Y, false);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void tanh_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y,
             uint8_t *workspace, uint8_t *tiling)
{
    tanh<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
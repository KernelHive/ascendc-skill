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
 * @file logical_not.cpp
 */
#include "kernel_operator.h"
#define GENERAL_OP_IMPL(templateClass, ...)                                       \
    do                                                                            \
    {                                                                             \
        GET_TILING_DATA(tiling_data, tiling);                                     \
        templateClass<__VA_ARGS__> op;                                            \
        op.Init(x, y, tiling_data.smallCoreDataNum,                               \
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,           \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,          \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, \
                tiling_data.tailBlockNum);                                        \
        op.Process();                                                             \
    } while (0)
// tensor num for each queue
constexpr int32_t BUFFER_NUM = 2;
constexpr half ONE = 1.0f;
constexpr half NEGATIVE_ONE = -1.0f;

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
class KernelLogicalNot
{
public:
    __aicore__ inline KernelLogicalNot() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum,
                                uint32_t smallCoreLoopNum, uint32_t ubPartDataNum,
                                uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
                                uint32_t tailBlockNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore)
        {
            if (coreNum < tailBlockNum)
            {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            }
            else
            {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
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
        AscendC::DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<half> tmp1Local = tmp1.Get<half>();
        AscendC::Cast(tmp1Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Abs(tmp1Local, tmp1Local, this->processDataNum);
        AscendC::Mins(tmp1Local, tmp1Local, ONE, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, NEGATIVE_ONE, this->processDataNum);
        AscendC::Abs(tmp1Local, tmp1Local, this->processDataNum);
        AscendC::Cast(yLocal, tmp1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void logical_not(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(1))
    {
        GENERAL_OP_IMPL(KernelLogicalNot, int8_t, int8_t, true);
    }
    else if (TILING_KEY_IS(0))
    {
        GENERAL_OP_IMPL(KernelLogicalNot, int8_t, int8_t, false);
    }
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void logical_not_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y,
                    uint8_t *workspace, uint8_t *tiling)
{
    logical_not<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif

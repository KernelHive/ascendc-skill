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
 * @file cast.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int16_t CONST_128 = 128;
constexpr int16_t CONST_NE128 = -128;
constexpr int16_t CONST_255 = 255;
constexpr int16_t CONST_1 = 1;
constexpr half HALF_ONE = 1.0;
template <typename TYPE_X, typename TYPE_Y>
class BaseKernelCast
{
public:
    __aicore__ inline BaseKernelCast() {}

protected:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

protected:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast0TBuf : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    无临时变量
    half -> float
    half -> int32       (TRUNC)
    half -> bool        (Abs)
    half -> int16       (TRUNC)
    float -> half
    float -> bfloat16   (RINT)
    float -> int32      (TRUNC)
    float -> int64      (TRUNC)
    float -> int16      (TRUNC)
    int32 -> float
    int32 -> int64
    int32 -> int16
    int8 -> half
    uint8 -> half
    bool -> half
    int64 -> float      (ROUND)
    int64 -> int32
    bfloat16 -> float
    bfloat16 -> int32   (TRUNC)
    int16 -> float
    int16 -> half
    */
public:
    __aicore__ inline KernelCast0TBuf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int32_t>) ||
                      (std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int16_t>) ||
                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int32_t>) ||
                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int16_t>) ||
                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int64_t>) ||
                      (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, int32_t>))
        {
            Cast(yLocal, xLocal, RoundMode::CAST_TRUNC, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, float>))
        {
            Cast(yLocal, xLocal, RoundMode::CAST_ROUND, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, bool>))
        {
            Abs(xLocal, xLocal, this->processDataNum);
            Mins(xLocal, xLocal, HALF_ONE, this->processDataNum);
            Cast(yLocal, xLocal, RoundMode::CAST_CEIL, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, float>) ||
                           (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, half>) ||
                           (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, int16_t>) ||
                           (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, int64_t>) ||
                           (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, float>) ||
                           (std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, half>) ||
                           (std::is_same_v<DTYPE_X, uint8_t> && std::is_same_v<DTYPE_Y, half>) ||
                           (std::is_same_v<DTYPE_X, bool> && std::is_same_v<DTYPE_Y, half>) ||
                           (std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, int32_t>) ||
                           (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, float>) ||
                           (std::is_same_v<DTYPE_X, int16_t> && std::is_same_v<DTYPE_Y, half>) ||
                           (std::is_same_v<DTYPE_X, int16_t> && std::is_same_v<DTYPE_Y, float>))
        {
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, bfloat16_t>))
        {
            Cast(yLocal, xLocal, RoundMode::CAST_RINT, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast1TBuf4B : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    1个4Bytes的临时变量
    half -> float ->(RINT) bfloat16
    int32 -> float ->(RINT) bfloat16
    int32 -> float -> half
    int64 ->(ROUND) float -> half
    int64 ->(ROUND) float ->(RINT) bfloat16
    int64 -> int32 -> int16
    bfloat16 -> float -> half
    int16 -> float ->(ROUND) int32
    int16 -> float ->(ROUND) int64
    */
public:
    __aicore__ inline KernelCast1TBuf4B() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        this->pipe->InitBuffer(tmp4Bytes1, this->tileDataNum * sizeof(float));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, bfloat16_t>) ||
                      (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, bfloat16_t>) ||
                      (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, half>) ||
                      (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, half>))
        {
            LocalTensor<float> tmp1 = tmp4Bytes1.Get<float>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, half>)
            {
                Cast(yLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr (std::is_same_v<DTYPE_Y, bfloat16_t>)
            {
                Cast(yLocal, tmp1, RoundMode::CAST_RINT, this->processDataNum);
            }
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, half>) ||
                           (std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, bfloat16_t>))
        {
            LocalTensor<float> tmp1 = tmp4Bytes1.Get<float>();
            Cast(tmp1, xLocal, RoundMode::CAST_ROUND, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, half>)
            {
                Cast(yLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr (std::is_same_v<DTYPE_Y, bfloat16_t>)
            {
                Cast(yLocal, tmp1, RoundMode::CAST_RINT, this->processDataNum);
            }
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, int16_t>))
        {
            LocalTensor<int32_t> tmp1 = tmp4Bytes1.Get<int32_t>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int16_t> && std::is_same_v<DTYPE_Y, int32_t>) ||
                           (std::is_same_v<DTYPE_X, int16_t> && std::is_same_v<DTYPE_Y, int64_t>))
        {
            LocalTensor<float> tmp1 = tmp4Bytes1.Get<float>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp1, RoundMode::CAST_ROUND, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }

private:
    TBuf<QuePosition::VECCALC> tmp4Bytes1;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast2TBuf2B : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    2个2Bytes的临时变量
    half -> int8
    half -> uint8
    float -> bool
    int32 -> bool
    int16 -> int8
    int16 -> uint8
    */
public:
    __aicore__ inline KernelCast2TBuf2B() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        this->pipe->InitBuffer(tmp2Bytes1, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp2Bytes2, this->tileDataNum * sizeof(half));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int8_t>) ||
                      (std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, uint8_t>))
        {
            LocalTensor<int16_t> tmp1 = tmp2Bytes1.Get<int16_t>();
            LocalTensor<int16_t> tmp2 = tmp2Bytes2.Get<int16_t>();
            Cast(tmp1, xLocal, RoundMode::CAST_TRUNC, this->processDataNum);
            Duplicate(tmp2, CONST_255, this->processDataNum);
            And(tmp1, tmp1, tmp2, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, int8_t>)
            {
                Adds(tmp1, tmp1, CONST_128, this->processDataNum);
                And(tmp1, tmp1, tmp2, this->processDataNum);
                Adds(tmp1, tmp1, CONST_NE128, this->processDataNum);
            }
            Cast(xLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, bool>))
        {
            LocalTensor<int16_t> tmp1 = tmp2Bytes1.Get<int16_t>();
            LocalTensor<half> tmp2 = tmp2Bytes2.Get<half>();
            Abs(xLocal, xLocal, this->processDataNum);
            Cast(tmp1, xLocal, RoundMode::CAST_CEIL, this->processDataNum);
            Mins(tmp1, tmp1, CONST_1, this->processDataNum);
            Cast(tmp2, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp2, RoundMode::CAST_NONE, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, bool>))
        {
            LocalTensor<int16_t> tmp1 = tmp2Bytes1.Get<int16_t>();
            LocalTensor<half> tmp2 = tmp2Bytes2.Get<half>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(tmp2, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            Abs(tmp2, tmp2, this->processDataNum);
            Mins(tmp2, tmp2, HALF_ONE, this->processDataNum);
            Cast(yLocal, tmp2, RoundMode::CAST_NONE, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int16_t> && std::is_same_v<DTYPE_Y, int8_t>) ||
                           (std::is_same_v<DTYPE_X, int16_t> && std::is_same_v<DTYPE_Y, uint8_t>))
        {
            LocalTensor<int16_t> tmp1 = tmp2Bytes1.Get<int16_t>();
            LocalTensor<half> tmp2 = tmp2Bytes2.Get<half>();
            Duplicate(tmp1, CONST_255, this->processDataNum);
            And(xLocal, xLocal, tmp1, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, int8_t>)
            {
                Adds(xLocal, xLocal, CONST_128, this->processDataNum);
                And(xLocal, xLocal, tmp1, this->processDataNum);
                Adds(xLocal, xLocal, CONST_NE128, this->processDataNum);
            }
            Cast(tmp2, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp2, RoundMode::CAST_NONE, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }

private:
    TBuf<QuePosition::VECCALC> tmp2Bytes1;
    TBuf<QuePosition::VECCALC> tmp2Bytes2;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast3TBuf2B : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    3个2Bytes的临时变量
    float -> int8
    float -> uint8
    int32 -> int8
    int32 -> uint8
    */
public:
    __aicore__ inline KernelCast3TBuf2B() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        this->pipe->InitBuffer(tmp2Bytes1, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp2Bytes2, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp2Bytes3, this->tileDataNum * sizeof(half));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int8_t>) ||
                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, uint8_t>) ||
                      (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, int8_t>) ||
                      (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, uint8_t>))
        {
            LocalTensor<int16_t> tmp1 = tmp2Bytes1.Get<int16_t>();
            LocalTensor<int16_t> tmp2 = tmp2Bytes2.Get<int16_t>();
            LocalTensor<half> tmp3 = tmp2Bytes3.Get<half>();
            if constexpr (std::is_same_v<DTYPE_X, float>)
            {
                Cast(tmp1, xLocal, RoundMode::CAST_TRUNC, this->processDataNum);
            }
            else if constexpr (std::is_same_v<DTYPE_X, int32_t>)
            {
                Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            Duplicate(tmp2, CONST_255, this->processDataNum);
            And(tmp1, tmp1, tmp2, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, int8_t>)
            {
                Adds(tmp1, tmp1, CONST_128, this->processDataNum);
                And(tmp1, tmp1, tmp2, this->processDataNum);
                Adds(tmp1, tmp1, CONST_NE128, this->processDataNum);
            }
            Cast(tmp3, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp3, RoundMode::CAST_NONE, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }

private:
    TBuf<QuePosition::VECCALC> tmp2Bytes1;
    TBuf<QuePosition::VECCALC> tmp2Bytes2;
    TBuf<QuePosition::VECCALC> tmp2Bytes3;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast1TBuf2B : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    1个2Bytes的临时变量
    int8/uint8/bool -> float
    int8/uint8/bool -> int32
    int8/uint8 -> int16
    int8 -> bool
    */
public:
    __aicore__ inline KernelCast1TBuf2B() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        this->pipe->InitBuffer(tmp2Bytes1, this->tileDataNum * sizeof(half));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, float>) ||
                      (std::is_same_v<DTYPE_X, uint8_t> && std::is_same_v<DTYPE_Y, float>) ||
                      (std::is_same_v<DTYPE_X, bool> && std::is_same_v<DTYPE_Y, float>) ||
                      (std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, int32_t>) ||
                      (std::is_same_v<DTYPE_X, uint8_t> && std::is_same_v<DTYPE_Y, int32_t>) ||
                      (std::is_same_v<DTYPE_X, bool> && std::is_same_v<DTYPE_Y, int32_t>) ||
                      (std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, int16_t>) ||
                      (std::is_same_v<DTYPE_X, uint8_t> && std::is_same_v<DTYPE_Y, int16_t>) ||
                      (std::is_same_v<DTYPE_X, bool> && std::is_same_v<DTYPE_Y, uint8_t>))
        {
            LocalTensor<half> tmp1 = tmp2Bytes1.Get<half>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, int32_t> || std::is_same_v<DTYPE_Y, int16_t> || std::is_same_v<DTYPE_Y, uint8_t>)
            {
                Cast(yLocal, tmp1, RoundMode::CAST_TRUNC, this->processDataNum);
            }
            else
            {
                Cast(yLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            }
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, bool>))
        {
            LocalTensor<half> tmp1 = tmp2Bytes1.Get<half>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Abs(tmp1, tmp1, this->processDataNum);
            Mins(tmp1, tmp1, HALF_ONE, this->processDataNum);
            Cast(yLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }

private:
    TBuf<QuePosition::VECCALC> tmp2Bytes1;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast1TBuf2B1TBuf4B : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    1个2Bytes,1个4Bytes的临时变量
    int8/uint8/bool -> half ->(TRUNC) int32 -> int64
    int8/uint8/bool -> half -> float -> bfloat16
    int64 -> bool
    bfloat16 -> bool
    */
public:
    __aicore__ inline KernelCast1TBuf2B1TBuf4B() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        this->pipe->InitBuffer(tmp2Bytes1, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp4Bytes1, this->tileDataNum * sizeof(float));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, int64_t>) ||
                      (std::is_same_v<DTYPE_X, uint8_t> && std::is_same_v<DTYPE_Y, int64_t>) ||
                      (std::is_same_v<DTYPE_X, bool> && std::is_same_v<DTYPE_Y, int64_t>))
        {
            LocalTensor<half> tmp1 = tmp2Bytes1.Get<half>();
            LocalTensor<int32_t> tmp2 = tmp4Bytes1.Get<int32_t>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(tmp2, tmp1, RoundMode::CAST_TRUNC, this->processDataNum);
            Cast(yLocal, tmp2, RoundMode::CAST_NONE, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int8_t> && std::is_same_v<DTYPE_Y, bfloat16_t>) ||
                           (std::is_same_v<DTYPE_X, uint8_t> && std::is_same_v<DTYPE_Y, bfloat16_t>) ||
                           (std::is_same_v<DTYPE_X, bool> && std::is_same_v<DTYPE_Y, bfloat16_t>))
        {
            LocalTensor<half> tmp1 = tmp2Bytes1.Get<half>();
            LocalTensor<float> tmp2 = tmp4Bytes1.Get<float>();
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(tmp2, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp2, RoundMode::CAST_RINT, this->processDataNum);
        }
        else if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, bool>) ||
                           (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, bool>))
        {
            LocalTensor<half> tmp1 = tmp2Bytes1.Get<half>();
            LocalTensor<float> tmp2 = tmp4Bytes1.Get<float>();
            if constexpr (std::is_same_v<DTYPE_X, int64_t>)
            {
                Cast(tmp2, xLocal, RoundMode::CAST_ROUND, this->processDataNum);
            }
            else if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>)
            {
                Cast(tmp2, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            Cast(tmp1, tmp2, RoundMode::CAST_CEIL, this->processDataNum);
            Abs(tmp1, tmp1, this->processDataNum);
            Mins(tmp1, tmp1, HALF_ONE, this->processDataNum);
            Cast(yLocal, tmp1, RoundMode::CAST_CEIL, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }

private:
    TBuf<QuePosition::VECCALC> tmp2Bytes1;
    TBuf<QuePosition::VECCALC> tmp4Bytes1;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelCast3TBuf2B1TBuf4B : public BaseKernelCast<TYPE_X, TYPE_Y>
{
    /*
    3个2Bytes,1个4Bytes的临时变量
    int64 -> int8
    int64 -> uint8
    bfloat16 -> int8
    bfloat16 -> uint8
    */
public:
    __aicore__ inline KernelCast3TBuf2B1TBuf4B() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        this->yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        BufferInit();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            this->CopyIn(i);
            Compute(i);
            this->CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        this->CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void BufferInit()
    {
        this->pipe->InitBuffer(this->inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe->InitBuffer(this->outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        this->pipe->InitBuffer(tmp2Bytes1, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp2Bytes2, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp2Bytes3, this->tileDataNum * sizeof(half));
        this->pipe->InitBuffer(tmp4Bytes1, this->tileDataNum * sizeof(float));
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
        if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, int8_t>) ||
                      (std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, uint8_t>) ||
                      (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, int8_t>) ||
                      (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, uint8_t>))
        {
            LocalTensor<int16_t> tmp1 = tmp2Bytes1.Get<int16_t>();
            LocalTensor<int16_t> tmp2 = tmp2Bytes2.Get<int16_t>();
            LocalTensor<half> tmp3 = tmp2Bytes3.Get<half>();
            LocalTensor<int32_t> tmp4 = tmp4Bytes1.Get<int32_t>();
            if constexpr (std::is_same_v<DTYPE_X, int64_t>)
            {
                Cast(tmp4, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>)
            {
                Cast(tmp4, xLocal, RoundMode::CAST_TRUNC, this->processDataNum);
            }
            Cast(tmp1, tmp4, RoundMode::CAST_NONE, this->processDataNum);
            Duplicate(tmp2, CONST_255, this->processDataNum);
            And(tmp1, tmp1, tmp2, this->processDataNum);
            if constexpr (std::is_same_v<DTYPE_Y, int8_t>)
            {
                Adds(tmp1, tmp1, CONST_128, this->processDataNum);
                And(tmp1, tmp1, tmp2, this->processDataNum);
                Adds(tmp1, tmp1, CONST_NE128, this->processDataNum);
            }
            Cast(tmp3, tmp1, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, tmp3, RoundMode::CAST_NONE, this->processDataNum);
        }
        this->outQueueY.template EnQue<TYPE_Y>(yLocal);
        this->inQueueX.template FreeTensor(xLocal);
    }

private:
    TBuf<QuePosition::VECCALC> tmp2Bytes1;
    TBuf<QuePosition::VECCALC> tmp2Bytes2;
    TBuf<QuePosition::VECCALC> tmp2Bytes3;
    TBuf<QuePosition::VECCALC> tmp4Bytes1;
};

class KernelCastTQueBind
{
    /*
    使用TQueBind直接传输8bit的数据类型
    bool -> int8/uint8
    int8 -> uint8
    uint8 -> int8
    */
public:
    __aicore__ inline KernelCastTQueBind() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, TPipe *pipeIn)
    {
        pipe = pipeIn;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ uint8_t *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y + globalBufferIndex, this->coreDataNum);
        pipe->InitBuffer(queBind, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            auto bindLocal = queBind.AllocTensor<uint8_t>();
            DataCopy(bindLocal, xGm[i * this->tileDataNum], this->processDataNum);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<uint8_t>();
            DataCopy(yGm[i * this->tileDataNum], bindLocal, this->processDataNum);
            queBind.FreeTensor(bindLocal);
        }
        this->processDataNum = this->tailDataNum;
        auto bindLocal = queBind.AllocTensor<uint8_t>();
        DataCopy(bindLocal, xGm[(loopCount - 1) * this->tileDataNum], this->processDataNum);
        queBind.EnQue(bindLocal);
        bindLocal = queBind.DeQue<uint8_t>();
        DataCopy(yGm[(loopCount - 1) * this->tileDataNum], bindLocal, this->processDataNum);
        queBind.FreeTensor(bindLocal);
    }

private:
    TPipe *pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
    GlobalTensor<uint8_t> xGm;
    GlobalTensor<uint8_t> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void cast(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1))
    {
        if constexpr (std::is_same_v<DTYPE_X, bool>)
        {
            KernelCast0TBuf<int8_t, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else if constexpr (std::is_same_v<DTYPE_Y, bool>)
        {
            KernelCast0TBuf<DTYPE_X, int8_t> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else
        {
            KernelCast0TBuf<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(2))
    {
        KernelCast1TBuf4B<DTYPE_X, DTYPE_Y> op;
        op.Init(x, y, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                tiling_data.tailBlockNum, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(3))
    {
        if constexpr (std::is_same_v<DTYPE_Y, bool>)
        {
            KernelCast2TBuf2B<DTYPE_X, int8_t> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else
        {
            KernelCast2TBuf2B<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(4))
    {
        KernelCast3TBuf2B<DTYPE_X, DTYPE_Y> op;
        op.Init(x, y, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                tiling_data.tailBlockNum, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(5))
    {
        if constexpr (std::is_same_v<DTYPE_X, bool>)
        {
            KernelCast1TBuf2B<int8_t, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else if constexpr (std::is_same_v<DTYPE_Y, bool>)
        {
            KernelCast1TBuf2B<DTYPE_X, int8_t> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else
        {
            KernelCast1TBuf2B<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(6))
    {
        if constexpr (std::is_same_v<DTYPE_X, bool>)
        {
            KernelCast1TBuf2B1TBuf4B<int8_t, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else if constexpr (std::is_same_v<DTYPE_Y, bool>)
        {
            KernelCast1TBuf2B1TBuf4B<DTYPE_X, int8_t> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
        else
        {
            KernelCast1TBuf2B1TBuf4B<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, tiling_data.smallCoreDataNum,
                    tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                    tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                    tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                    tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    }
    else if (TILING_KEY_IS(7))
    {
        KernelCast3TBuf2B1TBuf4B<DTYPE_X, DTYPE_Y> op;
        op.Init(x, y, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                tiling_data.tailBlockNum, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(8))
    {
        KernelCastTQueBind op;
        op.Init(x, y, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                tiling_data.tailBlockNum, &pipe);
        op.Process();
    }
}
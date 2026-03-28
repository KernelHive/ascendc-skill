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
 * @file less.cpp
 */
#include <type_traits>
#include "kernel_operator.h"

#define GENERAL_OP_IMPL(templateClass,...)                                                           \
   do{                                                                                               \
         templateClass<__VA_ARGS__>op;                                                               \
         op.Init(x1, x2, y, tiling_data.smallCoreDataNum,                                            \
                 tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,                             \
                 tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,                            \
                 tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,                   \
                 tiling_data.tailBlockNum, tiling_data.bigprocessDataNumComputes,                    \
                 tiling_data.smallprocessDataNumComputes, tiling_data.tailbigprocessDataNumComputes, \
                 tiling_data.tailsmallprocessDataNumComputes );                                      \
         op.Process();                                                                               \
 }while(0)   

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;
constexpr float SCALAR_MIN_FP32 = 1.1754943508222875e-38;
constexpr float SCALAR_MUL_FP32 = 4611686018427387904;
constexpr float SCALAR_MUL1_FP32 = 4;
constexpr float SCALAR_ZERO_FP32 = 0;
constexpr float SCALAR_MIN_FP16 = 0.00000005960464477539063F;
constexpr float SCALAR_MUL_FP16 = 4096; 

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y, bool IsExistBigCore> 
class KernelLess 
{
    using T = TYPE_X1;
public:
    __aicore__ inline KernelLess() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, 
                                uint32_t bigCoreLoopNum, uint32_t smallCoreLoopNum, 
                                uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, 
                                uint32_t bigCoreTailDataNum, uint32_t tailBlockNum, 
                                uint32_t bigprocessDataNumComputes, uint32_t smallprocessDataNumComputes, 
                                uint32_t tailbigprocessDataNumComputes, uint32_t tailsmallprocessDataNumComputes)  
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
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
                this->processDataNumComputes = bigprocessDataNumComputes;
                this->tailprocessDataNumComputes = tailbigprocessDataNumComputes;
            }
            else
            { 
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                this->processDataNumComputes = smallprocessDataNumComputes;
                this->tailprocessDataNumComputes = tailsmallprocessDataNumComputes;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            this->processDataNumComputes = smallprocessDataNumComputes;
            this->tailprocessDataNumComputes = tailsmallprocessDataNumComputes;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1 *)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2 *)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->ubPartDataNum  * sizeof(TYPE_X1) + 256);
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->ubPartDataNum  * sizeof(TYPE_X2) + 256);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum  * sizeof(int8_t) + 256);

        if (TILING_KEY_IS(0)) 
        {
            pipe.InitBuffer(tmp1, this->ubPartDataNum  * sizeof(half));
            pipe.InitBuffer(tmp2, this->ubPartDataNum  * sizeof(half));
        } 
        else if (TILING_KEY_IS(1)) 
        {   
            if constexpr (std::is_same_v<T, int64_t>)
            {
                pipe.InitBuffer(tmp1, this->ubPartDataNum  * sizeof(half));
                pipe.InitBuffer(tmp3, this->ubPartDataNum  * sizeof(float));
                pipe.InitBuffer(tmp4, this->ubPartDataNum  * sizeof(float));
            }
            else
            {
                pipe.InitBuffer(tmp1, this->ubPartDataNum  * sizeof(half) + 256);
                pipe.InitBuffer(tmp2, this->ubPartDataNum  * sizeof(float) + 256);
                pipe.InitBuffer(tmp3, this->ubPartDataNum  * sizeof(float) + 256);
            }
        }
    }

     __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i);
            ProcessCompute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->processDataNumComputes = this->tailprocessDataNumComputes;
        CopyIn(loopCount-1);
        ProcessCompute(loopCount-1);
        CopyOut(loopCount-1);
    }

    __aicore__ inline void ProcessCompute(int32_t progress)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) 
        {
            ComputeInt8(progress);
        } 
        else if constexpr (std::is_same_v<T, float16_t>) 
        {
            ComputeFp16(progress);
        }
        else if constexpr (std::is_same_v<T, float32_t>) 
        {
            ComputeFp(progress);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            ComputeInt(progress);
        }   
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            ComputeInt64(progress);
        }
        else 
        {
            ComputeBf16(progress);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) 
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.AllocTensor<TYPE_X2>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->ubPartDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void ComputeInt8(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        AscendC::Cast(p1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Sub(p1, p2, p1, this->processDataNum);
        AscendC::Mins(p1, p1, (half)SCALAR_MIN_FP16, this->processDataNum);
        AscendC::Maxs(p1, p1, (half)SCALAR_ZERO_FP32, this->processDataNum);
        AscendC::Muls(p1, p1, (half)SCALAR_MUL_FP16, this->processDataNum);
        AscendC::Muls(p1, p1, (half)SCALAR_MUL_FP16, this->processDataNum);
        AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void ComputeFp16(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::LT, this->processDataNumComputes);
        AscendC::Duplicate<TYPE_X2>(x1Local, (TYPE_X2)1, this->processDataNumComputes);
        AscendC::Select(x1Local, yLocal, x1Local, (TYPE_X2)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNumComputes);
        AscendC::Cast(yLocal, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void ComputeFp(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::LT, this->processDataNumComputes);
        AscendC::Duplicate<TYPE_X2>(x1Local, (TYPE_X2)1, this->processDataNumComputes);
        AscendC::Select(x1Local, yLocal, x1Local, (TYPE_X2)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNumComputes);
        auto p1 = tmp1.Get<half>();
        AscendC::Cast(p1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void ComputeInt(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        auto p1 = tmp1.Get<half>();
        AscendC::Max(x2Local, x1Local, x2Local, this->processDataNumComputes);
        AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::EQ, this->processDataNumComputes);
        AscendC::Duplicate<half>(p1, (half)0, this->processDataNumComputes);
        AscendC::Select(p1, yLocal, p1, (half)1, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNumComputes);
        AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<int8_t>(yLocal);
    }

   __aicore__ inline void ComputeInt64(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        auto p1 = tmp3.Get<int>();
        auto p2 = tmp4.Get<int>();
        AscendC::Cast(p1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        auto y1 = tmp3.Get<float>();
        auto y2 = tmp4.Get<float>();
        AscendC::Cast(y1, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(y2, p2, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Sub(y1, y2, y1, this->processDataNum);
        AscendC::Mins(y1, y1, (float)SCALAR_MIN_FP32, this->processDataNum);
        AscendC::Maxs(y1, y1, (float)SCALAR_ZERO_FP32, this->processDataNum);
        AscendC::Muls(y1, y1, (float)SCALAR_MUL_FP32, this->processDataNum);
        AscendC::Muls(y1, y1, (float)SCALAR_MUL_FP32, this->processDataNum);
        AscendC::Muls(y1, y1, (float)SCALAR_MUL1_FP32, this->processDataNum);
        auto p3 = tmp1.Get<half>();
        AscendC::Cast(p3, y1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(yLocal, p3, AscendC::RoundMode::CAST_NONE, this->processDataNum);  
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<int8_t>(yLocal);
    }
    
    __aicore__ inline void ComputeBf16(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        auto p1 = tmp1.Get<half>();
        auto y1 = tmp2.Get<float>();
        auto y2 = tmp3.Get<float>();
        AscendC::Cast(y1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        AscendC::Cast(y2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        AscendC::Compare(yLocal, y1, y2, AscendC::CMPMODE::LT, this->processDataNumComputes);
        AscendC::Duplicate<float>(y1, (float)1, this->processDataNumComputes);
        AscendC::Select(y2, yLocal, y1, (float)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNumComputes);
        AscendC::Cast(p1, y2, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) 
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4;
    AscendC::GlobalTensor<TYPE_X1> x1Gm;
    AscendC::GlobalTensor<TYPE_X2> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t processDataNumComputes;
    uint32_t tailprocessDataNumComputes;
};

extern "C" __global__ __aicore__ void less(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    GET_TILING_DATA(tiling_data, tiling);
    if(tiling_data.isTailBlock == 1)
    {
        GENERAL_OP_IMPL(KernelLess, DTYPE_X1, DTYPE_X2, DTYPE_Y, true);
    }
    else 
    {
        GENERAL_OP_IMPL(KernelLess, DTYPE_X1, DTYPE_X2, DTYPE_Y, false);
    }
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void less_do(uint32_t blockDim, void *l2ctrl, void *stream,
             uint8_t *x1, uint8_t *x2, uint8_t *y,
             uint8_t *workspace, uint8_t *tiling) {
    less<<<blockDim, l2ctrl, stream>>>(x1, x2, y, workspace, tiling);
}
#endif

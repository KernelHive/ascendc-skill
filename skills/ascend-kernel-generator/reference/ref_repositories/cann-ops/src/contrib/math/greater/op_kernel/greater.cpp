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
 * @file greater_kernel.cpp
 * Function : z = x1 > x2
 */
#include "kernel_operator.h"

#define GENERAL_OP_IMPL(templateClass,...)                                          \
  do{                                                                               \
      GET_TILING_DATA(tiling_data, tiling);                                        \
      templateClass<__VA_ARGS__>op;                                                 \
      op.Init(x1, x2, y, tiling_data.smallCoreDataNum,                                \
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);                                         \
      op.Process();                                                                \
  }while(0)

// tensor num for each queue
constexpr int32_t BUFFER_NUM = 2;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;
constexpr float MAX_F16 = 0.0f;
constexpr float MAX_F32 = 0.0f;

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y, bool IsExistBigCore> 
class KernelGreater {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelGreater() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t smallCoreDataNum,
                                uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum, 
                                uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, 
                                uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum, 
                                uint64_t tailBlockNum) 
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
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
          
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X1));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X2));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        if(std::is_same_v<TYPE_X1, float>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half));
        } else if(std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half));
            pipe.InitBuffer(calc_buf_2, this->ubPartDataNum * sizeof(half));
        } else if(std::is_same_v<TYPE_X1, int32_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half));
            pipe.InitBuffer(calc_buf_2, this->ubPartDataNum * sizeof(float));
            pipe.InitBuffer(calc_buf_3, this->ubPartDataNum * sizeof(float));
        } else if(std::is_same_v<TYPE_X1, int64_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(float));
            pipe.InitBuffer(calc_buf_2, this->ubPartDataNum * sizeof(float));
            pipe.InitBuffer(calc_buf_3, this->ubPartDataNum * sizeof(half));
        }
    }
    
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (int32_t i = 0; i < loopCount-1; i++) 
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
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
    
    __aicore__ inline void Compute(int32_t progress)
    {
        if constexpr (std::is_same_v<TYPE_X1, half> || std::is_same_v<TYPE_X1, bfloat16_t>)
        {
            AscendC::LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            AscendC::LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
            AscendC::Sub(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Maxs(x1Local, x1Local, (half)0.0, this->processDataNum);
            AscendC::Maxs(x1Local, x1Local, (half)0.0, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, (half)MAX_MUL_FP16, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, (half)MAX_MUL_FP16, this->processDataNum);
            AscendC::Mins(x1Local, x1Local, (half)1.0, this->processDataNum);
            AscendC::Cast(yLocal, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        else if constexpr (std::is_same_v<TYPE_X1, float>)
        {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
            AscendC::LocalTensor<half> y_compute = calc_buf_1.Get<half>();
            AscendC::Sub(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Maxs(x1Local, x1Local, (float)MIN_ACCURACY_FP32, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, (float)MAX_MUL_1_FP32, this->processDataNum);
            AscendC::Mins(x1Local, x1Local, (float)1.0f, this->processDataNum);
            Cast(y_compute, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, y_compute, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        else if constexpr (std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>)
        {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
            AscendC::LocalTensor<half> x1_fp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<half> x2_fp16 = calc_buf_2.Get<half>();
            AscendC::Cast(x1_fp16, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(x2_fp16, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Sub(x1_fp16, x1_fp16, x2_fp16, this->processDataNum);
            AscendC::Mins(x1_fp16, x1_fp16, (half)POSITIVE_ONE_FP32, this->processDataNum);
            AscendC::Maxs(x1_fp16, x1_fp16, (half)MAX_F16, this->processDataNum);
            Cast(yLocal, x1_fp16, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        else if constexpr (std::is_same_v<TYPE_X1, int32_t>)
        {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
            AscendC::LocalTensor<half> y_fp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<float> x1 = calc_buf_2.Get<float>();
            AscendC::LocalTensor<float> x2 = calc_buf_3.Get<float>();
            Cast(x1, x1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            Cast(x2, x2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            AscendC::Sub(x1, x1, x2, this->processDataNum);
            AscendC::Maxs(x1, x1, (float)MIN_ACCURACY_FP32, this->processDataNum);
            AscendC::Muls(x1, x1, (float)MAX_MUL_1_FP32, this->processDataNum);
            AscendC::Mins(x1, x1, (float)1.0f, this->processDataNum);
            Cast(y_fp16, x1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, y_fp16, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if(std::is_same_v<TYPE_X1, int64_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
            AscendC::LocalTensor<float> y_compute = calc_buf_1.Get<float>();
            AscendC::LocalTensor<float> y_compute1 = calc_buf_2.Get<float>();
            AscendC::LocalTensor<half> y_fp16 = calc_buf_3.Get<half>();
            Cast(y_compute, x1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            Cast(y_compute1, x2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            AscendC::Sub(y_compute, y_compute, y_compute1, this->processDataNum);
            AscendC::Mins(y_compute, y_compute, (float)MIN_ACCURACY_FP32, this->processDataNum);
            AscendC::Maxs(y_compute, y_compute, (float)MAX_F32, this->processDataNum);
            AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->processDataNum);
            AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->processDataNum);
            AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_2_FP32, this->processDataNum);
            Cast(y_fp16, y_compute, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, y_fp16, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
{
    AscendC::LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
    AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_3;
    AscendC::GlobalTensor<TYPE_X1> x1Gm;
    AscendC::GlobalTensor<TYPE_X2> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

extern "C" __global__ __aicore__ void greater(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if(TILING_KEY_IS(1))
    {
        GENERAL_OP_IMPL(KernelGreater, DTYPE_X1, DTYPE_X2, DTYPE_Y, true);
    }
    else if(TILING_KEY_IS(0))
    {
        GENERAL_OP_IMPL(KernelGreater, DTYPE_X1, DTYPE_X2, DTYPE_Y, false);
    }
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void greater_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x1, uint8_t* x2, uint8_t* y,
    uint8_t* workspace, uint8_t* tiling)
{
    greater<<<blockDim, l2ctrl, stream>>>(x1, x2, y, workspace, tiling);
}
#endif
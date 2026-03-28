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
 * @file exp.cpp
 */
 
#include "kernel_operator.h"
using namespace AscendC;
#define GENERAL_OP_IMPL(templateClass,...)                                          \
  do{                                                                               \
      GET_TILING_DATA(tiling_data, tiling);                                         \
      templateClass<__VA_ARGS__>op;                                                 \
      op.Init(x, y, tiling_data.smallCoreDataNum,                                   \
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum, tiling_data.base, tiling_data.scale, tiling_data.shift);                                          \
      op.Process();                                                                 \
  }while(0)
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X, typename TYPE_Y ,bool IsExistBigCore> class KernelExp {
public:
    __aicore__ inline KernelExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,  uint64_t smallCoreDataNum,
                                uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum, 
                                uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, 
                                uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum, 
                                uint64_t tailBlockNum, float base, float scale, float shift) 
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
        this->base = base;
        this->shift = shift;
        this->scale = scale;
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) 
        {
          pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(float));
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
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        if constexpr ( std::is_same_v< DTYPE_X, float32_t> || std::is_same_v<TYPE_X, float16_t>)
        {   
            Muls(yLocal, xLocal, (DTYPE_X)this->scale, this->processDataNum);
            Adds(yLocal, yLocal, (DTYPE_X)this->shift, this->processDataNum);
            Muls(yLocal, yLocal, (DTYPE_X)this->base, this->processDataNum);
            Exp(yLocal, yLocal, this->processDataNum);
        } 
        else 
        {
            LocalTensor<float> xLocalFp32 = tmp1.Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(xLocalFp32, xLocalFp32, this->scale, this->processDataNum);
            Adds(xLocalFp32, xLocalFp32, this->shift, this->processDataNum);
            Muls(xLocalFp32, xLocalFp32, this->base, this->processDataNum);
            Exp(xLocalFp32, xLocalFp32, this->processDataNum);
            Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    float base;
    float shift;
    float scale;
};
extern "C" __global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if(TILING_KEY_IS(1))
    {
      GENERAL_OP_IMPL(KernelExp,DTYPE_X, DTYPE_Y, true);
    }
     else if(TILING_KEY_IS(0))
    {
      GENERAL_OP_IMPL(KernelExp,DTYPE_X, DTYPE_Y, false);
    }
}
#ifndef ASCENDC_CPU_DEBUG
void reciprocal_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y,
    uint8_t* workspace, uint8_t* tiling)
{
    reciprocal<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
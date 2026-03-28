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
 * @file add_custom.cpp
 */

#include "kernel_operator.h"
#define GENERAL_OP_IMPL(templateClass,...)                                          \
  do{                                                                               \
      GET_TILING_DATA(tiling_data, tiling);                                         \
      templateClass<__VA_ARGS__>op;                                                 \
      op.Init(x, y, z, tiling_data.smallCoreDataNum,                                \
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);                                          \
      op.Process();                                                                 \
  }while(0)
// tensor num for each queue
constexpr int32_t BUFFER_NUM = 2;
constexpr int16_t Shift = 8;
template<typename TYPE_X, typename TYPE_Y, typename TYPE_Z,bool IsExistBigCore> class KernelAdd {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t smallCoreDataNum,
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
          
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, this->coreDataNum);;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Z));
        if constexpr (std::is_same_v<DTYPE_X, int8_t>) 
        {
          pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(half));
          pipe.InitBuffer(tmp2, this->ubPartDataNum * sizeof(half));
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
      AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
      AscendC::DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
      AscendC::DataCopy(yLocal, yGm[progress * this->ubPartDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
      inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
      AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();
      if constexpr (std::is_same_v<DTYPE_X, int8_t>) 
      {
        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Add(p2, p1, p2, this->processDataNum);
        AscendC::Cast(p1.ReinterpretCast<int16_t>(), p2, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        AscendC::ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), Shift, this->processDataNum); 
        AscendC::ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), Shift, this->processDataNum);
        AscendC::Cast(p2, p1.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(zLocal, p2, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      }
      else
      {
        AscendC::Add(zLocal, xLocal, yLocal, this->processDataNum);
      }
      
      outQueueZ.EnQue<TYPE_Z>(zLocal);
      inQueueX.FreeTensor(xLocal);
      inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();  
      AscendC::DataCopy(zGm[progress * this->ubPartDataNum], zLocal, this->processDataNum);
      outQueueZ.FreeTensor(zLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    if(TILING_KEY_IS(1))
    {
      GENERAL_OP_IMPL(KernelAdd,DTYPE_X, DTYPE_Y, DTYPE_Z,true);
    }
     else if(TILING_KEY_IS(0))
    {
      GENERAL_OP_IMPL(KernelAdd,DTYPE_X, DTYPE_Y, DTYPE_Z,false);
    }
}
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z,
    uint8_t* workspace, uint8_t* tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif

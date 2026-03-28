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
 * @file lerp.cpp
 */

#include "kernel_operator.h"
#define GENERAL_OP_IMPL(templateClass,...)                                          \
  do{                                                                               \
      GET_TILING_DATA(tiling_data, tiling);                                         \
      templateClass<__VA_ARGS__>op;                                                 \
      op.Init(start, end, weight, y, tiling_data.smallCoreDataNum,                  \
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            \
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   \
                tiling_data.tailBlockNum);                                          \
      op.Process();                                                                 \
  }while(0)

// tensor num for each queue
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_START, typename TYPE_END, typename TYPE_WEIGHT, typename TYPE_Y, bool IsExistBigCore> 
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, 
                                uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, 
                                uint64_t bigCoreLoopNum, uint64_t smallCoreLoopNum, 
                                uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum, 
                                uint64_t bigCoreTailDataNum, uint64_t tailBlockNum) 
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
          
        startGm.SetGlobalBuffer((__gm__ TYPE_START*)start + globalBufferIndex, this->coreDataNum);
        endGm.SetGlobalBuffer((__gm__ TYPE_END*)end + globalBufferIndex, this->coreDataNum);
        weightGm.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weight + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        
        pipe.InitBuffer(inQueueStart, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_START));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_END));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
        
        if constexpr (!std::is_same_v<TYPE_START, float>) 
        {
          pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(float));
          pipe.InitBuffer(tmp2, this->ubPartDataNum * sizeof(float));
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
      AscendC::LocalTensor<TYPE_START> startLocal = inQueueStart.AllocTensor<TYPE_START>();
      AscendC::LocalTensor<TYPE_END> endLocal = inQueueEnd.AllocTensor<TYPE_END>();
      AscendC::LocalTensor<TYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<TYPE_WEIGHT>();
      
      AscendC::DataCopy(startLocal, startGm[progress * this->ubPartDataNum], this->processDataNum);
      AscendC::DataCopy(endLocal, endGm[progress * this->ubPartDataNum], this->processDataNum);
      AscendC::DataCopy(weightLocal, weightGm[progress * this->ubPartDataNum], this->processDataNum);
      
      inQueueStart.EnQue(startLocal);
      inQueueEnd.EnQue(endLocal);
      inQueueWeight.EnQue(weightLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_START> startLocal = inQueueStart.DeQue<TYPE_START>();
      AscendC::LocalTensor<TYPE_END> endLocal = inQueueEnd.DeQue<TYPE_END>();
      AscendC::LocalTensor<TYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<TYPE_WEIGHT>();
      AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
      
      if constexpr (std::is_same_v<TYPE_START, float>) 
      {
        // For float: y = start + weight * (end - start)
        AscendC::Sub(yLocal, endLocal, startLocal, this->processDataNum);
        AscendC::Mul(yLocal, weightLocal, yLocal, this->processDataNum);
        AscendC::Add(yLocal, yLocal, startLocal, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_START, half> || std::is_same_v<TYPE_START, bfloat16_t>) 
      {  
        // FP16 high precision path
        auto fstart = tmp1.Get<float>();
        auto fend = tmp2.Get<float>();
        
        // Convert to float with no rounding
        AscendC::Cast(fstart, startLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(fend, endLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        
        // High precision computation: fy = start + weight * (end - start)
        AscendC::Sub(fend, fend, fstart, this->processDataNum);
        AscendC::Cast(fstart, weightLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(fend, fstart, fend, this->processDataNum);
        AscendC::Cast(fstart, startLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Add(fend, fend, fstart, this->processDataNum);
        
        // Convert back to FP16 with rounding to nearest even
        AscendC::Cast(yLocal, fend, AscendC::RoundMode::CAST_RINT, this->processDataNum);
      }
      outQueueY.EnQue<TYPE_Y>(yLocal);
      inQueueStart.FreeTensor(startLocal);
      inQueueEnd.FreeTensor(endLocal);
      inQueueWeight.FreeTensor(weightLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();  
      AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
      outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueStart;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueEnd;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueWeight;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::GlobalTensor<TYPE_START> startGm;
    AscendC::GlobalTensor<TYPE_END> endGm;
    AscendC::GlobalTensor<TYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

extern "C" __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if(TILING_KEY_IS(1))
    {
      GENERAL_OP_IMPL(KernelLerp, DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y, true);
    }
    else if(TILING_KEY_IS(0))
    {
      GENERAL_OP_IMPL(KernelLerp, DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y, false);
    }
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void lerp_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* start, uint8_t* end, uint8_t* weight, uint8_t* y,
    uint8_t* workspace, uint8_t* tiling)
{
    lerp<<<blockDim, l2ctrl, stream>>>(start, end, weight, y, workspace, tiling);
}
#endif

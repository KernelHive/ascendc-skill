/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "kernel_operator.h"
 #define GENERAL_OP_IMPL(templateClass,...)                                          \
   do{                                                                               \
         GET_TILING_DATA(tiling_data,tiling);                                        \
         templateClass<__VA_ARGS__>op;                                               \
         op.Init(x1, x2, y,tiling_data.smallCoreDataNum,                             \
                 tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             \
                 tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,                \
                 tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,       \
                 tiling_data.tailBlockNum);                                              \
         op.Process();                                                                   \
 }while(0)           
 
 
 #include <type_traits>
 constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
 template <bool IsExistBigCore> 
 class KernelLogicalOr {
 public:
       __aicore__ inline KernelLogicalOr() {}
       __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                     uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, 
                                     uint32_t bigCoreLoopNum, uint32_t smallCoreLoopNum, 
                                     uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, 
                                     uint32_t bigCoreTailDataNum, uint32_t tailBlockNum)
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
         
        x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1 + globalBufferIndex,this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2 + globalBufferIndex,this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y + globalBufferIndex,this->coreDataNum);
          
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(half));
        pipe.InitBuffer(tmp2, this->ubPartDataNum * sizeof(half));  
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
        AscendC::LocalTensor<int8_t> x1Local = inQueueX1.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> x2Local = inQueueX2.AllocTensor<int8_t>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->ubPartDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->ubPartDataNum], this->processDataNum);
 
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
     __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> x1Local = inQueueX1.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> x2Local = inQueueX2.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
 
        auto p1=tmp1.Get<half>();
        auto p2=tmp2.Get<half>();
        AscendC::Cast(p1,x1Local,AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2,x2Local,AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Max(p1, p1, p2, this->processDataNum);
         
        AscendC::Cast(yLocal,p1,AscendC::RoundMode::CAST_NONE,this->processDataNum);
         
        outQueueY.EnQue<int8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>(); 
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }
  
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1,inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1,tmp2;
    AscendC::GlobalTensor<int8_t> x1Gm;
    AscendC::GlobalTensor<int8_t> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void logical_or(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if(TILING_KEY_IS(1))
    {
        GENERAL_OP_IMPL(KernelLogicalOr,true);
    }
    else if(TILING_KEY_IS(0))
    {
        GENERAL_OP_IMPL(KernelLogicalOr,false);
    }
}
 
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void logical_or_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x1, uint8_t *x2, uint8_t *y,
                    uint8_t *workspace, uint8_t *tiling)
{
    logical_or<<<blockDim, l2ctrl, stream>>>(x1, x2, y, workspace, tiling);
}
#endif
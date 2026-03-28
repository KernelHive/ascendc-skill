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
 * @file muls.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename TYPE_X>
class KernelMuls
{
public:
    __aicore__ inline KernelMuls() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t ubPartDataNum,
                                uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
                                uint32_t smallCoreLoopNum, uint32_t bigCoreLoopNum,
                                uint32_t tailBlockNum,uint32_t IsExistBigCore)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        this->IsExistBigCore = IsExistBigCore;
        if (1==IsExistBigCore) 
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
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)y + globalBufferIndex, this->coreDataNum);
        valueGm.SetGlobalBuffer((__gm__ float *)value);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        this->value = valueGm.GetValue(0);
        if constexpr(std::is_same_v<TYPE_X, float16_t>)
        {
            m_value=(half)this->value;
        }
        //tmp1用于临时存储数据用，方便类型的转换，用于转换成float
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t> || 
                          std::is_same_v<TYPE_X, int16_t>|| 
                         std::is_same_v<TYPE_X, int32_t> || 
                          std::is_same_v<TYPE_X, int64_t>){
            pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(float32_t));
        }
    }
    __aicore__ inline void Process()
    {
        //在process侧实现分流，实现对复数类型和常规数据类型的处理
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
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t> || 
                          std::is_same_v<TYPE_X, int16_t>|| 
                         std::is_same_v<TYPE_X, int32_t> || 
                          std::is_same_v<TYPE_X, int64_t>){
            LocalTensor<float32_t> p1 = tmp1.Get<float32_t>();
            if constexpr (std::is_same_v<TYPE_X, int64_t>)
            {
               Cast(p1, xLocal, RoundMode::CAST_RINT, this->processDataNum);
            } 
            else
	        {
               Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            Muls(p1, p1,this->value , this->processDataNum);
            Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
        }
        else if constexpr(std::is_same_v<TYPE_X, float16_t>)
        {
            Muls(yLocal, xLocal,this->m_value , this->processDataNum);
        }
        else if constexpr(std::is_same_v<TYPE_X, float32_t>)
        {
            Muls(yLocal, xLocal,this->value , this->processDataNum);
        }
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1;
    TBuf<QuePosition::VECCALC> tmp2;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<float> valueGm;
    GlobalTensor<TYPE_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t IsExistBigCore;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float value;
    TYPE_X m_value;
};

//在这里编写一个适配complex64类型的算子
template <typename TYPE_X>
class KernelMulsComplex64
{
public:
    __aicore__ inline KernelMulsComplex64() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t ubPartDataNum,
        uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
        uint32_t smallCoreLoopNum, uint32_t bigCoreLoopNum,
        uint32_t tailBlockNum,uint32_t IsExistBigCore)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        this->IsExistBigCore = IsExistBigCore;
        if (1==IsExistBigCore) 
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
        xGm.SetGlobalBuffer((__gm__ float *)x + globalBufferIndex * BUFFER_NUM, this->coreDataNum * BUFFER_NUM); // 1 complex = 2 float
        yGm.SetGlobalBuffer((__gm__ float *)y + globalBufferIndex * BUFFER_NUM, this->coreDataNum * BUFFER_NUM);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * BUFFER_NUM * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * BUFFER_NUM * sizeof(float));
        valueGm.SetGlobalBuffer((__gm__ float *)value);
        this->value = valueGm.GetValue(0);
    }
    __aicore__ inline void Process()
    {
        //在process侧实现分流，实现对复数类型和常规数据类型的处理
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
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->ubPartDataNum * BUFFER_NUM], this->processDataNum * BUFFER_NUM);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Muls(yLocal, xLocal, this->value, this->processDataNum* BUFFER_NUM);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->ubPartDataNum * BUFFER_NUM], yLocal, this->processDataNum * BUFFER_NUM);
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm;
    GlobalTensor<float> valueGm;
    GlobalTensor<float> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t IsExistBigCore;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float value;
};
extern "C" __global__ __aicore__ void muls( GM_ADDR x,
      GM_ADDR value,
      GM_ADDR y, 
      GM_ADDR workspace, 
      GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(0)){
        KernelMuls<bfloat16_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    } else if(TILING_KEY_IS(1)){
        KernelMuls<float16_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    } else if(TILING_KEY_IS(2)){
        KernelMuls<float32_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    } else if(TILING_KEY_IS(3)){
        KernelMuls<int16_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    } else if(TILING_KEY_IS(4)){
        KernelMuls<int32_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    }
    else if(TILING_KEY_IS(5)){
        KernelMuls<int64_t> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    }
    else if(TILING_KEY_IS(6)){
        KernelMulsComplex64<float> op;
        op.Init(x,value, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreLoopNum, tiling_data.bigCoreLoopNum,
            tiling_data.tailBlockNum,tiling_data.IsExistBigCore);
        op.Process();
    }
}
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void muls_do(uint32_t blockDim,
    void *l2ctrl,
    void *stream,
    uint8_t *x,
    uint8_t *value,
    uint8_t *y,
    uint8_t *workspace,
    uint8_t *tiling)
{
    muls<<<blockDim, l2ctrl, stream>>>(x,value, y, workspace, tiling);
}
#endif

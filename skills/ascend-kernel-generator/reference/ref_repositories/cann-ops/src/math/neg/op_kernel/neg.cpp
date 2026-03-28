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
 * @file neg.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T, bool IsExistBigCore>
class KernelNeg
{
public:
    __aicore__ inline KernelNeg() {}
    __aicore__ inline void Init(
        GM_ADDR src_gm, 
        GM_ADDR dst_gm,
        uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum, 
        uint32_t smallCoreLoopNum, uint32_t ubPartDataNum, 
        uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum, 
        uint32_t tailBlockNum,
        TPipe* pipeIn
        )
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
        src_global.SetGlobalBuffer((__gm__ T *)src_gm + globalBufferIndex, this->coreDataNum);
        dst_global.SetGlobalBuffer((__gm__ T *)dst_gm + globalBufferIndex, this->coreDataNum);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        pipe->InitBuffer(QueueTmp, this->ubPartDataNum * sizeof(half));
        pipe->InitBuffer(QueueTmp2, this->ubPartDataNum * sizeof(half));
        pipe->InitBuffer(QueueTmp1, this->ubPartDataNum * sizeof(float));
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
    __aicore__ inline void CopyIn(uint32_t process)
    {
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        DataCopy(srcLocal, src_global[process * this->ubPartDataNum], this->processDataNum);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(uint32_t process)
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        if constexpr (std::is_same_v<T, int32_t>){
            Duplicate(dstLocal, T(-1), this->processDataNum);
            Mul(dstLocal, srcLocal, dstLocal, this->processDataNum);
        }
        else if constexpr (std::is_same_v<T, int8_t>){
            LocalTensor<half> tmp = QueueTmp.Get<half>();
            Cast(tmp, srcLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(tmp, tmp, half(-1), this->processDataNum);
            //移位操作实现溢出处理
            LocalTensor<int16_t> tmp2 = QueueTmp2.Get<int16_t>();
            Cast(tmp2, tmp, RoundMode::CAST_RINT, this->processDataNum); // float16 -> int16
            // 处理溢出 (模拟 int8 计算的行为)
            ShiftLeft(tmp2, tmp2, int16_t(8), this->processDataNum);
            ShiftRight(tmp2, tmp2, int16_t(8), this->processDataNum);
            // 转回 half
            Cast(tmp, tmp2, RoundMode::CAST_NONE, this->processDataNum);
            // 转回int8
            Cast(dstLocal, tmp, RoundMode::CAST_NONE, this->processDataNum);
            QueueTmp2.FreeTensor(tmp2);
            QueueTmp.FreeTensor(tmp);
        }
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            Muls(dstLocal, srcLocal, T(-1), this->processDataNum);
        }
        //Muls不支持bfloat16类型
        else{
            LocalTensor<float> tmp1 = QueueTmp.Get<float>();
            Cast(tmp1, srcLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(tmp1, tmp1, float(-1), this->processDataNum);
            Cast(dstLocal, tmp1, RoundMode::CAST_RINT, this->processDataNum);
            QueueTmp.FreeTensor(tmp1);
        }       
        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);   
    }

    __aicore__ inline void CopyOut(uint32_t process)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dst_global[process * this->ubPartDataNum], dstLocal, this->processDataNum);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> QueueTmp, QueueTmp2, QueueTmp1;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void neg(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{   
    if(TILING_KEY_IS(1))
    {
        GET_TILING_DATA(tiling_data, tiling);
        TPipe pipe;
        KernelNeg<DTYPE_X, true> op;
        op.Init(x, y,
                tiling_data.smallCoreDataNum,                                
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   
                tiling_data.tailBlockNum,
                &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(0))
    {
        GET_TILING_DATA(tiling_data, tiling);
        TPipe pipe;
        KernelNeg<DTYPE_X, false> op;
        op.Init(x, y,
                tiling_data.smallCoreDataNum,                                
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   
                tiling_data.tailBlockNum,
                &pipe);
        op.Process();
    }
}
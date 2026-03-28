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
 * @file abs.cpp
 */
#include "abs_common.h"
#include "abs_int32.h"
#include "abs_int64.h"
#include "abs_complex64.h"
#include "abs_float32.h"

constexpr int ABS_TILING_0 = 1; // 直接改变符号 :float16, bfp16, float32
constexpr int ABS_TILING_1 = 2; //int32
constexpr int ABS_TILING_2 = 3; //int64
constexpr int ABS_TILING_3 = 4; //complex64
constexpr int ABS_TILING_4 = 5; //float32

class KernelAbs {
public:
    __aicore__ inline KernelAbs() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t bigDataCoreNum, uint32_t smallBlockLength, uint32_t bigBlockLength,
                                uint32_t smallTileNum, uint32_t smallTileLength, uint32_t smallLasttileLength,
                                uint32_t bigTileNum, uint32_t bigTileLength, uint32_t bigLasttileLength,
                                uint32_t dataWidth)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
              
        if(AscendC::GetBlockIdx() >= bigDataCoreNum)
        {
            xGm.SetGlobalBuffer((__gm__ half*)x + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength* this->times);
            yGm.SetGlobalBuffer((__gm__ half*)y + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength * this->times);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, smallTileLength * sizeof(half));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, smallTileLength * sizeof(half));
            this->tileNum = smallTileNum;
            this->tileLength = smallTileLength;
            this->lasttileLength = smallLasttileLength;         
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ half*)x + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            yGm.SetGlobalBuffer((__gm__ half*)y + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bigTileLength * sizeof(half));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, bigTileLength * sizeof(half));
            this->tileNum = bigTileNum;
            this->tileLength = bigTileLength;
            this->lasttileLength = bigLasttileLength;  
        }  
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->calcLength = this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            if(i == loopCount -1)
            {
                this->calcLength = this->lasttileLength;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        AscendC::Abs(yLocal, xLocal, (this->calcLength));

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC>  tempBuf;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::LocalTensor<half> tempLocal;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t calcLength;
    uint32_t lasttileLength;
    uint32_t times;
    uint32_t repeat;
};

extern "C" __global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) //ABS_TILING_0
    {
        KernelAbs op;
        op.Init(x, y, 
                tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength,
                tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, 
                tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength,
                tiling_data.dataWidth);
        op.Process();         
    }
    else if (TILING_KEY_IS(ABS_TILING_1)) 
    {
        KernelAbs_int32 op1;
        op1.Init(x, y, 
                tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength,
                tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, 
                tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength,
                tiling_data.dataWidth);
        op1.Process_int32();           
    }
    else if (TILING_KEY_IS(ABS_TILING_2)) 
    {
        KernelAbs_int64 op2;
        op2.Init(x, y, 
                tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength,
                tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, 
                tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength,
                tiling_data.dataWidth);
        op2.Process_int64();           
    }    
    
    else if (TILING_KEY_IS(ABS_TILING_3)) 
    {
        KernelAbs_complex64 op3;
        op3.Init(x, y, 
                tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength,
                tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, 
                tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength,
                tiling_data.dataWidth);
        op3.Process();           
    }   
    else if (TILING_KEY_IS(ABS_TILING_4)) 
    {
        KernelAbs_float op4;
        op4.Init(x, y, 
                tiling_data.bigDataCoreNum, tiling_data.smallBlockLength, tiling_data.bigBlockLength,
                tiling_data.smallTileNum, tiling_data.smallTileLength, tiling_data.smallLasttileLength, 
                tiling_data.bigTileNum, tiling_data.bigTileLength, tiling_data.bigLasttileLength,
                tiling_data.dataWidth);
        op4.Process_float32();           
    }     
}
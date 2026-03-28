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
 * @file abs_float32.h
 */
#ifndef ABS_FLOAT32_H 
#define ABS_FLOAT32_H

#include "abs_common.h"

class KernelAbs_float {
public:
    __aicore__ inline KernelAbs_float() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t bigDataCoreNum, uint32_t smallBlockLength, uint32_t bigBlockLength,
                                uint32_t smallTileNum, uint32_t smallTileLength, uint32_t smallLasttileLength,
                                uint32_t bigTileNum, uint32_t bigTileLength, uint32_t bigLasttileLength,
                                uint32_t dataWidth)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        if(AscendC::GetBlockIdx() >= bigDataCoreNum)
        {
            xGm.SetGlobalBuffer((__gm__ float*)x + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength* this->times);
            yGm.SetGlobalBuffer((__gm__ float*)y + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength * this->times);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, smallTileLength * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, smallTileLength * sizeof(float));
            this->tileNum = smallTileNum;
            this->tileLength = smallTileLength;
            this->lasttileLength = smallLasttileLength;         
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ float*)x + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            yGm.SetGlobalBuffer((__gm__ float*)y + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bigTileLength * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, bigTileLength * sizeof(float));
            this->tileNum = bigTileNum;
            this->tileLength = bigTileLength;
            this->lasttileLength = bigLasttileLength;  
        }  
    }
    __aicore__ inline void Process_float32()
    {
        int32_t loopCount = this->tileNum;
        this->calcLength = this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            if(i == loopCount -1){
                this->calcLength = this->lasttileLength;
            }
            CopyIn_f32(i);
            Compute_f32(i);
            CopyOut_f32(i);
        }
    }

private:
    __aicore__ inline void CopyIn_f32(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute_f32(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        AscendC::Abs(yLocal, xLocal, (this->calcLength));

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut_f32(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC>  tempBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::LocalTensor<float> tempLocal;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t calcLength;
    uint32_t lasttileLength;
    uint32_t times;
    uint32_t repeat;
};

#endif  // ABS_FLOAT32_H
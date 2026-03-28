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
 * @file abs_int32.h
 */
#ifndef ABS_INT32_H 
#define ABS_INT32_H

#include "abs_common.h"

class KernelAbs_int32 {
public:
    __aicore__ inline KernelAbs_int32() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t bigDataCoreNum, uint32_t smallBlockLength, uint32_t bigBlockLength,
                                uint32_t smallTileNum, uint32_t smallTileLength, uint32_t smallLasttileLength,
                                uint32_t bigTileNum, uint32_t bigTileLength, uint32_t bigLasttileLength,
                                uint32_t dataWidth)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        if(AscendC::GetBlockIdx() >= bigDataCoreNum)
        {
            xGm.SetGlobalBuffer((__gm__ int32_t*)x + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength* this->times);
            yGm.SetGlobalBuffer((__gm__ int32_t*)y + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength * this->times);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, smallTileLength * sizeof(int32_t));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, smallTileLength * sizeof(int32_t));
            this->tileNum = smallTileNum;
            this->tileLength = smallTileLength;
            this->lasttileLength = smallLasttileLength;         
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ int32_t*)x + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            yGm.SetGlobalBuffer((__gm__ int32_t*)y + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bigTileLength * sizeof(int32_t));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, bigTileLength * sizeof(int32_t));
            this->tileNum = bigTileNum;
            this->tileLength = bigTileLength;
            this->lasttileLength = bigLasttileLength;  
        }  

        pipe.InitBuffer(tempBuf, (this->tileLength * sizeof(int32_t)));
        this->tempLocal = tempBuf.Get<int32_t>();
    }
    __aicore__ inline void Process_int32()
    {
        int32_t loopCount = this->tileNum;
        this->calcLength = this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            if(i == loopCount -1){
                this->calcLength = this->lasttileLength;
            }
            CopyIn_int32(i);
            Compute_int32(i);
            CopyOut_int32(i);
        }
    }

private:
    __aicore__ inline void CopyIn_int32(int32_t progress)
    {
        AscendC::LocalTensor<int32_t> xLocal = inQueueX.AllocTensor<int32_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute_int32(int32_t progress)
    {
        AscendC::LocalTensor<int32_t> xLocal = inQueueX.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> yLocal = outQueueY.AllocTensor<int32_t>();
        
        int32_t scalar = 32;
        AscendC::ShiftRight(tempLocal, xLocal, scalar, (this->calcLength));
        AscendC::Or(yLocal.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), tempLocal.ReinterpretCast<uint16_t>(), (this->calcLength)*2);
        AscendC::And(xLocal.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), tempLocal.ReinterpretCast<uint16_t>(), (this->calcLength)*2);
        AscendC::Not(xLocal.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), (this->calcLength)*2);
        AscendC::And(yLocal.ReinterpretCast<uint16_t>(), yLocal.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), (this->calcLength)*2);
        AscendC::Sub(yLocal.ReinterpretCast<int32_t>(), yLocal.ReinterpretCast<int32_t>(), tempLocal.ReinterpretCast<int32_t>(), (this->calcLength));

        outQueueY.EnQue<int32_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut_int32(int32_t progress)
    {
        AscendC::LocalTensor<int32_t> yLocal = outQueueY.DeQue<int32_t>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC>  tempBuf;
    AscendC::GlobalTensor<int32_t> xGm;
    AscendC::GlobalTensor<int32_t> yGm;
    AscendC::LocalTensor<int32_t> tempLocal;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t calcLength;
    uint32_t lasttileLength;
    uint32_t times;
    uint32_t repeat;
};

#endif  // ABS_INT32_H
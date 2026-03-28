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
 * @file abs_int64.h
 */
#ifndef ABS_INT64_H 
#define ABS_INT64_H

#include "abs_common.h"

class KernelAbs_int64 {
public:
    __aicore__ inline KernelAbs_int64() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t bigDataCoreNum, uint32_t smallBlockLength, uint32_t bigBlockLength,
                                uint32_t smallTileNum, uint32_t smallTileLength, uint32_t smallLasttileLength,
                                uint32_t bigTileNum, uint32_t bigTileLength, uint32_t bigLasttileLength,
                                uint32_t dataWidth)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        if(AscendC::GetBlockIdx() >= bigDataCoreNum)
        {
            xGm.SetGlobalBuffer((__gm__ int64_t*)x + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength* this->times);
            yGm.SetGlobalBuffer((__gm__ int64_t*)y + (bigDataCoreNum * bigBlockLength) + (smallBlockLength * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength * this->times);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, smallTileLength * sizeof(int64_t));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, smallTileLength * sizeof(int64_t));
            this->tileNum = smallTileNum;
            this->tileLength = smallTileLength;
            this->lasttileLength = smallLasttileLength;         
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ int64_t*)x + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            yGm.SetGlobalBuffer((__gm__ int64_t*)y + bigBlockLength * AscendC::GetBlockIdx(), bigBlockLength);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bigTileLength * sizeof(int64_t));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, bigTileLength  * sizeof(int64_t));
            this->tileNum = bigTileNum;
            this->tileLength = bigTileLength;
            this->lasttileLength = bigLasttileLength;  
        }  
    }
    __aicore__ inline void Process_int64()
    {
        int32_t loopCount = this->tileNum;
        this->calcLength = this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            if(i == loopCount -1)
            {
                this->calcLength = this->lasttileLength;
            }
            CopyIn_int64(i);
            Compute_int64(i);
            CopyOut_int64(i);
        }
    }

private:
    __aicore__ inline void CopyIn_int64(int32_t progress)
    {
        AscendC::LocalTensor<int64_t> xLocal = inQueueX.AllocTensor<int64_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute_int64(int32_t progress)
    {
        AscendC::LocalTensor<int64_t> xLocal = inQueueX.DeQue<int64_t>();
        AscendC::LocalTensor<int64_t> yLocal = outQueueY.AllocTensor<int64_t>();
        
        int64_t temp, sign;
        for (int32_t i=0; i<(this->calcLength); i++) {
            temp = xLocal.GetValue(i);
            sign = temp >> 63;
            temp = (temp ^ sign) - sign;
            yLocal.SetValue(i, temp);
        }

        outQueueY.EnQue<int64_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut_int64(int32_t progress)
    {
        AscendC::LocalTensor<int64_t> yLocal = outQueueY.DeQue<int64_t>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<int64_t> xGm;
    AscendC::GlobalTensor<int64_t> yGm;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t calcLength;
    uint32_t lasttileLength;
    uint32_t times;
    uint32_t repeat;
};

#endif  // ABS_INT64_H
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
 * @file abs_complex64.h
 */
#ifndef ABS_COMPLEX64_H 
#define ABS_COMPLEX64_H

#include "abs_common.h"

class KernelAbs_complex64 {
public:
    __aicore__ inline KernelAbs_complex64() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t bigDataCoreNum, uint32_t smallBlockLength, uint32_t bigBlockLength,
                                uint32_t smallTileNum, uint32_t smallTileLength, uint32_t smallLasttileLength,
                                uint32_t bigTileNum, uint32_t bigTileLength, uint32_t bigLasttileLength,
                                uint32_t dataWidth)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        if(AscendC::GetBlockIdx() >= bigDataCoreNum)
        {
            xGm.SetGlobalBuffer((__gm__ float*)x + (bigDataCoreNum * bigBlockLength * 2) + (smallBlockLength *2 * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength* this->times);
            yGm.SetGlobalBuffer((__gm__ float*)y + (bigDataCoreNum * bigBlockLength * 2) + (smallBlockLength * 2 * (AscendC::GetBlockIdx() - bigDataCoreNum)), smallBlockLength * this->times);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, smallTileLength * 2 * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, smallTileLength * 2 * sizeof(float));
            this->tileNum = smallTileNum;
            this->tileLength = smallTileLength;
            this->lasttileLength = smallLasttileLength;         
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ float*)x + bigBlockLength * 2 * AscendC::GetBlockIdx(), bigBlockLength* 2);
            yGm.SetGlobalBuffer((__gm__ float*)y + bigBlockLength * 2 * AscendC::GetBlockIdx(), bigBlockLength* 2);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bigTileLength * 2 * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, bigTileLength * 2 * sizeof(float));
            this->tileNum = bigTileNum;
            this->tileLength = bigTileLength;
            this->lasttileLength = bigLasttileLength;  
        }  
      
        pipe.InitBuffer(realBuf, (this->tileLength * 2 * sizeof(float)));
        pipe.InitBuffer(imagBuf, (this->tileLength * 2 * sizeof(float)));
        pipe.InitBuffer(srcOffsetBuf, (this->tileLength * 2 * sizeof(uint32_t)));
        this->realLocal = realBuf.Get<float>();
        this->imagLocal = imagBuf.Get<float>();
        this->srcOffsetLocal = srcOffsetBuf.Get<uint32_t>();
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->calcLength = this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            if(i == loopCount -1){
                this->calcLength = this->lasttileLength;
            }
            CopyIn_complex64(i);
            Compute_complex64(i);
            CopyOut_complex64(i);
        }
    }

private:
    __aicore__ inline void CopyIn_complex64(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength * 2], this->calcLength * 2);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute_complex64(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        for (int32_t j = 0; j < (this->calcLength); j++) {
            srcOffsetLocal.SetValue(j, j*8);
        }

        AscendC::Gather(realLocal, xLocal, srcOffsetLocal, 0, (this->calcLength));
        AscendC::Gather(imagLocal, xLocal, srcOffsetLocal, 4, (this->calcLength));

        AscendC::Mul(realLocal, realLocal, realLocal, (this->calcLength));
        AscendC::Mul(imagLocal, imagLocal, imagLocal, (this->calcLength));
        AscendC::Add(realLocal, realLocal, imagLocal, (this->calcLength));
        AscendC::Sqrt(realLocal, realLocal, (this->calcLength));
        AscendC:Duplicate(yLocal, (float)(0.0), (this->calcLength) * 2);
        for (int32_t j = 0; j < (this->calcLength); j++) {
            yLocal.SetValue(j*2, realLocal.GetValue(j));
        }  
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut_complex64(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength * 2], yLocal, this->calcLength * 2);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC>  realBuf, imagBuf, srcOffsetBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::LocalTensor<float> realLocal, imagLocal;
    AscendC::LocalTensor<uint32_t> srcOffsetLocal;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t calcLength;
    uint32_t lasttileLength;
    uint32_t times;
    uint32_t repeat;
};

#endif  // ABS_COMPLEX64_H
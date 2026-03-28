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
 * @file sub_frameworklaunch.cpp
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t formerNum, uint32_t tailNum,
                                uint32_t formerLength, uint32_t tailLength, uint32_t alignNum)
    {
        if (AscendC::GetBlockIdx() < formerNum) {
            this->tileLength = formerLength;
            xGm.SetGlobalBuffer((__gm__ half *)x + formerLength * AscendC::GetBlockIdx(), formerLength);
            yGm.SetGlobalBuffer((__gm__ half *)y + formerLength * AscendC::GetBlockIdx(), formerLength);
            zGm.SetGlobalBuffer((__gm__ half *)z + formerLength * AscendC::GetBlockIdx(), formerLength);
        } else {
            this->tileLength = tailLength;
            xGm.SetGlobalBuffer((__gm__ half *)x + formerLength * formerNum + tailLength * (AscendC::GetBlockIdx() - formerNum),
                                tailLength);
            yGm.SetGlobalBuffer((__gm__ half *)y + formerLength * formerNum + tailLength * (AscendC::GetBlockIdx() - formerNum),
                                tailLength);
            zGm.SetGlobalBuffer((__gm__ half *)z + formerLength * formerNum + tailLength * (AscendC::GetBlockIdx() - formerNum),
                                tailLength);
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm, this->tileLength);
        AscendC::DataCopy(yLocal, yGm, this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm, zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_frameworklaunch(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSub op;
    op.Init(x, y, z, tilingData.formerNum, tilingData.tailNum, tilingData.formerLength, tilingData.tailLength,
            tilingData.alignNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
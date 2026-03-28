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
 * @file mul_sigmoid_mul_add_custom.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;

namespace AscendC {
template <typename T>
class MulSigmoidMulAddCustom {
public:
    __aicore__ inline MulSigmoidMulAddCustom(){}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR mulScalar1, GM_ADDR mulScalar2, GM_ADDR addScalar3, GM_ADDR output, const MulSigmoidMulAddCustomTilingData &tiling)
    {
        this->mulScalar1 = *((__gm__ T*)mulScalar1);
        this->mulScalar2 = *((__gm__ T*)mulScalar2);
        this->addScalar3 = *((__gm__ T*)addScalar3);

        this->blockIdx = AscendC::GetBlockIdx();

        this->completeTileNum = tiling.completeTileNum;
        this->partTileNum = tiling.partTileNum;
        this->completeTileLen = tiling.completeTileLen;
        this->partTileLen = tiling.partTileLen;
        this->totalTileNum = tiling.totalTileNum;

        /* 前面block分配tile小，后面block分配tile多 */
        if (blockIdx < tiling.frontBlockNum) {
            this->startTileIdx = blockIdx * tiling.tileNumInFrontBlock;
            this->tileNumInBlock = tiling.tileNumInFrontBlock;
        } else {
            this->startTileIdx = (tiling.frontBlockNum * tiling.tileNumInFrontBlock) + (blockIdx - tiling.frontBlockNum) * tiling.tileNumInLatterBlock;
            this->tileNumInBlock = tiling.tileNumInLatterBlock;
        }
        this->endTileIdx = this->startTileIdx + this->tileNumInBlock - 1;

        /* 不是最后一个Tile */
        this->startPosInBlock = this->startTileIdx * tiling.completeTileLen;
        if (this->endTileIdx < tiling.completeTileNum) {                
            this->blockLen = this->tileNumInBlock * tiling.completeTileLen;
        } else {
            this->blockLen = (this->tileNumInBlock - 1) * tiling.completeTileLen + tiling.partTileLen;
        }

        /* SetGlobalBuffer接口中，bufferSize单位为sizeof(T)，不需要再乘以sizeof(T) */
        inputGlobal.SetGlobalBuffer((__gm__ T*)input + this->startPosInBlock, this->blockLen);
        outputGlobal.SetGlobalBuffer((__gm__ T*)output + this->startPosInBlock * this->out_put_len, this->blockLen * this->out_put_len);

        /*  分配内存块的个数。double buffer功能通过该参数开启：num设置为1，表示不开启double buffer；num设置为2，表示开启double buffer。*/
        pipe.InitBuffer(inQueueInput, 2, this->completeTileLen * sizeof(T));
        pipe.InitBuffer(outQueueOutput, 2, this->completeTileLen * sizeof(T) * this->out_put_len);

        /* T_Buf，每次使用2块buf，2PP */
        pipe.InitBuffer(calcDataBuf, 2 * 2 * this->completeTileLen * sizeof(T));
    }

     __aicore__ inline void Process()
    {     
        for (int32_t tileIdx = this->startTileIdx; tileIdx <= this->endTileIdx; tileIdx++) {
            int32_t tileLen = (tileIdx < this->completeTileNum) ? this->completeTileLen : this->partTileLen;
            CopyIn(tileIdx, tileLen);
            Compute(tileIdx, tileLen);
            CopyOut(tileIdx, tileLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t tileIdx, int32_t tileLen)
    {
        /* 输入数据1拷贝，DataCopy中calCout单位为sizeof(T)，所以配置calCout时，不需要乘以sizeof(T) */
        LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        DataCopy(inputLocal, inputGlobal[tileIdx * tileLen], tileLen);
        inQueueInput.EnQue(inputLocal);
    }

    __aicore__ inline void CopyOut(int32_t tileIdx, int32_t tileLen)
    {
        LocalTensor<T> outputLocal = outQueueOutput.DeQue<T>();
        DataCopy(outputGlobal[tileIdx * tileLen], outputLocal, tileLen * this->out_put_len);
        outQueueOutput.FreeTensor(outputLocal);
    }

    __aicore__ inline void Compute(int32_t tileIdx, int32_t tileLen)
    {       
        /* 输入输出 */
        LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T> outputLocal = outQueueOutput.AllocTensor<T>();

        /* 申请2块BUFFER循环使用 */
        LocalTensor<T> calBuffer1 = calcDataBuf.Get<T>(tileLen);
        LocalTensor<T> calBuffer2 = calcDataBuf.GetWithOffset<T>(tileLen, tileLen);

        /* Step1:Mul计算,输出到buffer1 */        
        AscendC::Muls(calBuffer1, inputLocal, this->mulScalar1, tileLen);

        /* Step2:sigmoid计算,输出到buffer2 */       
        AscendC::Sigmoid(calBuffer2, calBuffer1, tileLen);

        /* Step3:Mul计算,输出到buffer1 */
        AscendC::Muls(calBuffer1, calBuffer2, this->mulScalar2, tileLen);

        /* Step4:add计算,输出到buffer1 */
        AscendC::Adds(outputLocal, calBuffer1, this->addScalar3, tileLen);
        inQueueInput.FreeTensor(inputLocal);
        outQueueOutput.EnQue<T>(outputLocal);
    }

private:
    TPipe pipe;

    /* 队列的深度表示该队列可以连续进行入队/出队的次数，在代码运行时，对同一个队列有n次连续的EnQue（中间没有DeQue），那么该队列的
    深度就需要设置为n。注意，这里的队列深度和double buffer无关，队列机制用于实现流水线并行，double buffer在此基础上进一步提高流水线的利用率。即使队列
    的深度为1，仍可以开启double buffer。队列的深度设置为1时，编译器对这种场景做了特殊优化，性能通常更好，推荐设置为1。 */
    TQue<QuePosition::VECIN, 1> inQueueInput;
    TQue<QuePosition::VECOUT, 1> outQueueOutput;

    TBuf<QuePosition::VECCALC> calcDataBuf;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    uint32_t blockIdx;

    uint32_t completeTileNum;
    uint32_t partTileNum;
    uint32_t completeTileLen;
    uint32_t partTileLen;
    uint32_t totalTileNum;

    uint32_t startTileIdx;
    uint32_t tileNumInBlock;
    uint32_t endTileIdx;

    uint32_t startPosInBlock;
    uint32_t blockLen;

    T mulScalar1;
    T mulScalar2;
    T addScalar3;

    /* 测试输出，将中间步骤数据打印出来 */
    uint32_t out_put_len = 1;
};
}


extern "C" __global__ __aicore__ 
void mul_sigmoid_mul_add_custom(GM_ADDR input, GM_ADDR mulScalar1, GM_ADDR mulScalar2, GM_ADDR addScalar3, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    MulSigmoidMulAddCustom<float16_t> op;
    op.Init(input, mulScalar1, mulScalar2, addScalar3, output, tiling_data);
    op.Process();
}

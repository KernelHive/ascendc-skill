/*
* @file global_avg_pool.cpp
*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This program is free software; you can redistribute it and/or modify
* it under the terms of the Apache License Version 2.0.
* You may not use this file except in compliance with the License.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
class KernelGlobalAvgPool
{
public:
    __aicore__ inline KernelGlobalAvgPool() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t outDim,
                                int32_t dimLength, uint32_t tileNum, uint32_t blockLength,
                                uint32_t tileLength, uint32_t lasttileLength,
                                uint32_t workLength, uint32_t actLastLen,
                                uint32_t typeKey, TPipe* pipeIn, uint32_t stride)
    {
        pipe = pipeIn;
        this->typeKey = typeKey;

        if(typeKey == 3){ // 处理一般特征图的情况
            this->outDim = outDim;
            this->dimLength = dimLength;
            this->blockLength = blockLength;
            this->stride = stride;
            this->tileLength = tileLength;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, this->tileLength);
            pipe->InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
            pipe->InitBuffer(calcBuf1, this->tileLength * sizeof(DTYPE_X));
        }else{
            this->outDim = outDim;
            this->dimLength = dimLength;
            this->tileNum = tileNum;
            this->blockLength = blockLength;
            this->tileLength = tileLength / BUFFER_NUM;
            this->lasttileLength = lasttileLength / BUFFER_NUM;
            this->px = (__gm__ DTYPE_X *)x;
            this->actLastLen = actLastLen;
            if(typeKey == 0){ // 处理输入等于输出的情况
                xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->blockLength);
                yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, this->blockLength);
                pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
                pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
            }else{ // 处理一个ub可放下所有数据的情况
                yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, this->outDim);
                pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
                pipe->InitBuffer(calcBuf1, workLength * sizeof(DTYPE_X));
                pipe->InitBuffer(calcBuf2, sizeof(DTYPE_X));
            }
        }
    }

    __aicore__ inline void MainCopy()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1) {
                CopyIn(i, lasttileLength);
                EqealCompute(i, lasttileLength);
                CopyOut(i, lasttileLength);
            } else {
                CopyIn(i, tileLength);
                EqealCompute(i, tileLength);
                CopyOut(i, tileLength);
            }
        }
    }

    __aicore__ inline void MainProcess()
    {
        for(int32_t i = 0; i < outDim; i++){
            sum = 0;
            xGm.SetGlobalBuffer(px + i * dimLength, dimLength);
            SubProcess();
            float t = static_cast<float>(sum);
            t = t / dimLength;
            sum = static_cast<DTYPE_X>(t);
            yGm.SetValue(i, sum);
        }
    }

    __aicore__ inline void SubProcess() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1) {
                CopyIn(i, lasttileLength);
                Compute(i, actLastLen);
            } else {
                CopyIn(i, tileLength);
                Compute(i, tileLength);
            }
        }
    }

    __aicore__ inline void FastCompute(){
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal[0], xGm[0], blockLength);
        inQueueX.EnQue(xLocal);

        xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_X> tmp1 = calcBuf1.Get<DTYPE_X>();
        DTYPE_X scalar = static_cast<DTYPE_X>(dimLength);
        WholeReduceSum(yLocal, xLocal, dimLength, outDim, 1, 1, stride);
        Duplicate(tmp1, scalar, tileLength);
        Div(yLocal, yLocal, tmp1, tileLength);
        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);

        yLocal = outQueueY.DeQue<DTYPE_X>();
        DataCopy(yGm[0], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> tmp1 = calcBuf1.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmp2 = calcBuf2.Get<DTYPE_X>();
        float t1, t2;
        if(typeKey == 1){
            for(int32_t i = 0; i < length; i++){
                t1 = static_cast<float>(xLocal.GetValue(i));
                t2 = static_cast<float>(sum);
                t1 += t2;
                sum = static_cast<DTYPE_X>(t1);
            }
        }else{
            ReduceSum(tmp2, xLocal, tmp1, length);
            t1 = static_cast<float>(tmp2.GetValue(0));
            t2 = static_cast<float>(sum);
            t1 += t2;
            sum = static_cast<DTYPE_X>(t1);
        }

        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void EqealCompute(int32_t progress, uint32_t length)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();

        DTYPE_X scalar = 0;
        Adds(yLocal, xLocal, scalar, length);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<DTYPE_X> yLocal = outQueueY.DeQue<DTYPE_X>();
        DataCopy(yGm[progress * tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf1, calcBuf2;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    DTYPE_X sum;
    uint32_t outDim;
    int32_t dimLength;
    uint32_t tileNum, blockLength;
    uint32_t tileLength;
    uint32_t lasttileLength;
    uint32_t actLastLen;
    uint32_t typeKey, stride;
    __gm__ DTYPE_X *px;
};
extern "C" __global__ __aicore__ void global_avg_pool(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGlobalAvgPool op;

    TPipe pipe;
    op.Init(x, y, tiling_data.outDim, tiling_data.dimLength, tiling_data.tileNum, tiling_data.blockLength,
            tiling_data.tileLength, tiling_data.lasttileLength, tiling_data.workLength,
            tiling_data.actLastLen, tiling_data.typeKey, &pipe, tiling_data.stride);

    if(tiling_data.typeKey == 3){
        op.FastCompute();
    }else{
        if(tiling_data.typeKey == 0){
            op.MainCopy();
        }else{
            op.MainProcess();
        }
    }
}
#ifndef __CCE_KT_TEST__
void global_avg_pool_do(uint32_t blockDim, void* l2ctrl, void* stream,
             uint8_t* x, uint8_t* y,
             uint8_t* workspace, uint8_t* tiling) {
    global_avg_pool<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
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
 * @file tril.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t minNum = 1;

constexpr int keyOne = 1;
constexpr int keyTwo = 2;
constexpr int keyThree = 3;
constexpr int keyFour = 4;

constexpr int computeBatchSize = 256;

struct IntegrateParam{
uint32_t totalLengthAligned;
int32_t matrixNum;
int32_t matrixSize;
int32_t rowLength;
int32_t columnLength;
int32_t diagVal;
int32_t loopCnt;
uint32_t fullTileLength;
uint32_t lastTileLength;
int32_t fullCnt;
int32_t lastCnt;
uint32_t alignNum;
uint32_t typeSize;
};

class KernelTril {
public:
    __aicore__ inline KernelTril() {}
    //only pass the length this one is assigned to
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, IntegrateParam& paramList, uint32_t key)
    {
        this->matrixNum = paramList.matrixNum;
        this->matrixSize = paramList.matrixSize;
        this->rowLength = paramList.rowLength;
        this->columnLength = paramList.columnLength;
        this->diagVal = paramList.diagVal;
        this->fullCnt = paramList.fullCnt;
        this->lastCnt = paramList.lastCnt;
        if(paramList.columnLength==0){
            paramList.columnLength = minNum;
        }
        this->fullRowInc = paramList.fullTileLength / paramList.columnLength;
        this->initLength = 1;
        // The result would not be the expected
        if(paramList.typeSize==0){
            paramList.typeSize = sizeof(float);
        }

        this->typeSize = paramList.typeSize;

        this->key=key;

        uint64_t gmBuffer=paramList.totalLengthAligned;
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, gmBuffer);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X*)y, gmBuffer);

        this->loopCnt = paramList.loopCnt;
        this->fullTileLength = paramList.fullTileLength;
        this->lastTileLength = paramList.lastTileLength;

        uint32_t singleBuffer = paramList.fullTileLength;
        if(singleBuffer < paramList.lastTileLength){
            singleBuffer = paramList.lastTileLength;
        }
        if(key==keyThree || key==keyFour){
            pipe.InitBuffer(inQueueX, BUFFER_NUM, singleBuffer * this->typeSize);
            pipe.InitBuffer(outQueueY, BUFFER_NUM, singleBuffer * this->typeSize);
        }else{
            pipe.InitBuffer(queBind, BUFFER_NUM, singleBuffer * this->typeSize);
        }
    }
    
    __aicore__ inline void Process()
    {
        if(this->key==keyOne){
            NaivePath();
        }else if(this->key==keyTwo){
            SheerDup();
        }else if(this->key==keyThree){
            SheerZero();
        }else if(key==keyFour){
            FastPath();
        }
    }
    
private:
    __aicore__ inline void SheerDup()
    {
        uint32_t GmOffset=0;
        for (int i = 0; i < this->loopCnt-1; i++, GmOffset+=this->fullTileLength) {
            auto bindLocal = queBind.AllocTensor<DTYPE_X>();
            DataCopy(bindLocal, xGm[GmOffset], this->fullTileLength);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_X>();
            DataCopy(yGm[GmOffset], bindLocal, this->fullTileLength);
            queBind.FreeTensor(bindLocal);
        }
        auto bindLocal = queBind.AllocTensor<DTYPE_X>();
        DataCopy(bindLocal, xGm[GmOffset], this->lastTileLength);
        queBind.EnQue(bindLocal);
        bindLocal = queBind.DeQue<DTYPE_X>();
        DataCopy(yGm[GmOffset], bindLocal, this->lastTileLength);
        queBind.FreeTensor(bindLocal);
    }

    __aicore__ inline void SheerZero(){
        uint32_t GmOffset=0;
        for (int i = 0; i < this->loopCnt-1; i++, GmOffset+=this->fullTileLength) {
            CopyIn(GmOffset,this->fullTileLength);
            AllZero(this->fullTileLength);
            CopyOut(GmOffset,this->fullTileLength);
        }
        CopyIn(GmOffset,this->lastTileLength);
        AllZero(this->lastTileLength);
        CopyOut(GmOffset,this->lastTileLength);
    }

    __aicore__ inline void NaivePath(){
        int32_t cnt=0;
        for(int32_t i=0;i<this->matrixNum;i++){
            for(int32_t j=0;j<this->rowLength;j++){
                int32_t k=0;
                while(k<this->columnLength && k-j<=this->diagVal){
                    DTYPE_X curr=xGm.GetValue(cnt);
                    yGm.SetValue(cnt,curr);
                    k++;
                    cnt++;
                }
                while(k<this->columnLength){
                    yGm.SetValue(cnt,(DTYPE_X)0);
                    k++;
                    cnt++; 
                }
            }
        }
    }

    __aicore__ inline void FastPath(){
        uint32_t GmOffset=0;
        int32_t init_row = 0;
        for(int num=0;num<this->matrixNum;num++){
            uint32_t calLength=this->initLength;
            if(this->diagVal<=0){
                init_row = -diagVal;
            }
            for (int32_t i = 0; i < this->loopCnt-1; i++) {
                CopyIn(GmOffset,this->fullTileLength);
                Compute(this->fullCnt, calLength, init_row);
                CopyOut(GmOffset,this->fullTileLength);
                if(init_row>0){
                    init_row-=this->fullRowInc;
                    if(init_row<0){
                        calLength-=init_row;
                    }
                }else{
                    calLength+=this->fullRowInc;
                }
                GmOffset+=this->fullTileLength;
            }
            CopyIn(GmOffset,this->lastTileLength);
            Compute(this->lastCnt, calLength, init_row);
            CopyOut(GmOffset,this->lastTileLength);
            GmOffset+=this->lastTileLength;
        }
    }

    __aicore__ inline void CopyIn(uint32_t GmOffset, uint32_t tileLength){
        auto xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[GmOffset], tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t GmOffset, uint32_t tileLength){
        auto yLocal=outQueueY.DeQue<DTYPE_X>();
        DataCopy(yGm[GmOffset], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(int32_t cnt, uint32_t initLength, int32_t adjust){
        auto xLocal = inQueueX.DeQue<DTYPE_X>();
        auto yLocal = outQueueY.AllocTensor<DTYPE_X>();
        uint32_t localOffset=0;
        uint32_t currLength=initLength;
        DTYPE_X scalarZero=0;
        for(int32_t i=0;i<cnt;i++){
            Sub(yLocal[localOffset],xLocal[localOffset],xLocal[localOffset],this->columnLength);
            if(i>=adjust){
                Adds(yLocal[localOffset],xLocal[localOffset],scalarZero,currLength);
                currLength++;
            }
            localOffset+=this->columnLength;
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void AllZero(uint32_t tileLength){
        auto xLocal = inQueueX.DeQue<DTYPE_X>();
        auto yLocal = outQueueY.AllocTensor<DTYPE_X>();
        Sub(yLocal,xLocal,xLocal,tileLength);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;
    //queue for simple duplication
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind; // Use TQueBind to replace QueI，QueO if needed
    
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> yGm;

    int32_t matrixNum;
    int32_t matrixSize;
    int32_t rowLength;
    int32_t columnLength;
    int32_t diagVal;
    int32_t fullCnt;
    int32_t lastCnt;

    int32_t loopCnt;
    uint32_t fullTileLength;
    uint32_t lastTileLength;
    uint32_t fullRowInc;
    uint32_t initLength;

    uint32_t typeSize;
    uint32_t alignNum;
    uint32_t key;
};  

extern "C" __global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelTril op;
    IntegrateParam paramList = {
        .totalLengthAligned=tiling_data.totalLengthAligned, 
        .matrixNum=tiling_data.matrixNum, 
        .matrixSize=tiling_data.matrixSize, 
        .rowLength=tiling_data.rowLength, 
        .columnLength=tiling_data.columnLength,
        .diagVal=tiling_data.diagVal,
        .loopCnt=tiling_data.loopCnt, 
        .fullTileLength=tiling_data.fullTileLength, 
        .lastTileLength=tiling_data.lastTileLength,
        .fullCnt=tiling_data.fullCnt, 
        .lastCnt=tiling_data.lastCnt,
        .alignNum=tiling_data.alignNum,
        .typeSize=tiling_data.typeSize
    };
    if(TILING_KEY_IS(1)){
        op.Init(x, y, paramList, 1);
    }else if(TILING_KEY_IS(2)){
        op.Init(x, y, paramList, 2);
    }else if(TILING_KEY_IS(3)){
        op.Init(x, y, paramList, 3);
    }else if(TILING_KEY_IS(4)){
        op.Init(x, y, paramList, 4);
    }
    op.Process();
}
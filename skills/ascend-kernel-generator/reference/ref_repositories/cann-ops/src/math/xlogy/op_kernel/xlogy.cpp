/*
* 版权所有 (c) 华为技术有限公司 2024
*/
// Copyright 2024 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t inputVarNum = 2;
constexpr int32_t maxDimNum = 64;

class TileVar{
public:    
    uint32_t CoreDataNum; 
    uint32_t finalTileNum; 
    uint32_t tileDataNum; 
    uint32_t TailDataNum;
    uint32_t x1_length;
    uint32_t x2_length; 
    int64_t numshapes; 
    int64_t ss[inputVarNum * maxDimNum];
    int64_t sf[maxDimNum];
};

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class Kernelxlogy_Broadcast {
    using T = TYPE_X1;
public:
    __aicore__ inline Kernelxlogy_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TileVar* tilevar) {                               
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->coreDataNum = tilevar->CoreDataNum;
        this->tileNum = tilevar->finalTileNum;
        this->tileDataNum = tilevar->tileDataNum;
        this->tailDataNum = tilevar->TailDataNum;

        for (int i = 0; i < maxDimNum * inputVarNum; ++i) {
            ((int64_t *)this->shape)[i] = tilevar->ss[i];
        }
        this->numshapes = tilevar->numshapes;
        for(int i = 0; i < maxDimNum; ++i) {
            ((int64_t *)this->shapefull)[i] = tilevar->sf[i];
        }
        this->total_length = 1;
        for(int i = 0; i < this->numshapes; ++i) {
            this->total_length *= ((int64_t *)this->shapefull)[i];
        }
        this->x1_Length = tilevar->x1_length;
        this->x2_Length = tilevar->x2_length;
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2*)x2, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->coreDataNum);
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X2));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(QueueTmp1, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(QueueTmp2, this->tileDataNum * sizeof(int8_t));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();
        
        if(this->x1_Length < this->total_length) {
            BroadCX1(x1Local, progress * this->tileDataNum, this->processDataNum);
        } else {            
            DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        }
        if(this->x2_Length < this->total_length) {
            BroadCX2(x2Local, progress * this->tileDataNum, this->processDataNum);
        } else {
            DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        }
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void BroadCX1(LocalTensor<DTYPE_X1> &dst, uint32_t offset, uint32_t length) {
        if(this->x1_Length == 1) {
            DTYPE_X1 tmp = x1Gm.GetValue(0);
            Duplicate(dst, tmp, length);
            return;
        }
        for(uint32_t i = 0; i < length; i++) {
            int istart = i + offset;
            int idxtmp = GetPos(istart, 0);
            DTYPE_X1 tmp = x1Gm.GetValue(idxtmp);
            dst.SetValue(i, tmp);
        }
    }
    __aicore__ inline void BroadCX2(LocalTensor<DTYPE_X1> &dst, uint32_t offset, uint32_t length) {
        if(this->x2_Length == 1) {
            DTYPE_X1 tmp = x2Gm.GetValue(0);
            Duplicate(dst, tmp, length);
            return;
        }
        for(uint32_t i = 0; i < length; i++) {
            int istart = i + offset;
            int idxtmp = GetPos(istart, 1);
            DTYPE_X1 tmp = x2Gm.GetValue(idxtmp);
            dst.SetValue(i, tmp);
        }
    }
    __aicore__ inline int GetPos(int istart, int inputNo)  {
        int idxtmp = 0;
        for(int k = 1; k <= this->numshapes; k++) {
            int kpos = 0;
            int krange = 1;
            if(k < this->numshapes) {
                for(int m = k + 1; m <= this->numshapes; m++) {
                    krange *= shapefull[m - 1];
                }
                kpos = istart / krange;
                istart = istart % krange;
            } else {
                krange = shapefull[k - 1];
                kpos = istart % krange;
            }
            int krangeB = 1;
            if(shapefull[k - 1] == shape[inputNo][k - 1]) {
                if(k < this->numshapes) {
                    for(int m = k + 1; m <= this->numshapes; m++) {
                        krangeB *= shape[1][m - 1];
                    }
                    idxtmp += kpos * krangeB;
                }  else {
                    idxtmp += kpos;
                }
            }
        }
        return idxtmp;
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X1> tmp1 = QueueTmp1.Get<DTYPE_X1>();
        LocalTensor<int8_t> cmp1 = QueueTmp2.Get<int8_t>();

        Duplicate(tmp1, DTYPE_X1(0), this->processDataNum);
        Compare(cmp1, x1Local, tmp1, CMPMODE::EQ, this->processDataNum);        
        Select(tmp1, cmp1, tmp1, DTYPE_X1(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        Ln(yLocal, x2Local, this->processDataNum);
        Mul(yLocal, yLocal, x1Local, this->processDataNum);
        Mul(yLocal, yLocal, tmp1, this->processDataNum);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueTmp1, QueueTmp2;

    GlobalTensor<DTYPE_X1> x1Gm;
    GlobalTensor<DTYPE_X2> x2Gm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t x1_Length;
    uint32_t x2_Length;
    uint32_t total_length;
    int64_t shape[2][64];
    int64_t numshapes;
    int64_t shapefull[64];
};

extern "C" __global__ __aicore__ void xlogy(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    Kernelxlogy_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
    TileVar tilevar; 
    tilevar.CoreDataNum = tiling_data.CoreDataNum;
    tilevar.finalTileNum = tiling_data.finalTileNum;
    tilevar.tileDataNum = tiling_data.tileDataNum;
    tilevar.TailDataNum = tiling_data.TailDataNum;
    tilevar.x1_length = tiling_data.x1_length;
    tilevar.x2_length =  tiling_data.x2_length;  
    tilevar.numshapes =  tiling_data.numshapes;  
    for(int32_t i = 0; i < inputVarNum * maxDimNum; i++) {
        tilevar.ss[i] = tiling_data.shape[i];  
    }
    for(int32_t i = 0; i < maxDimNum; i++) {
        tilevar.sf[i] = tiling_data.shapefull[i];  
    }
    op.Init(x1, x2, y, &tilevar);  
    op.Process();
}
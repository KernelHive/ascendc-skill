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
 * @file motion_compensation.cpp
 */
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelMotionCompensation
{
public:
    __aicore__ inline KernelMotionCompensation() {}
    __aicore__ inline void Init(GM_ADDR points, GM_ADDR timestamp, GM_ADDR out,
                                TPipe *pipeIn, int64_t N, int64_t ndim, float tMax,
                                float f, float theta, float r_sin_theta, 
                                float trans[3], float qRel[4], int32_t doRotation, float d_sign, int32_t tMaxLow32,
                                int64_t smallCoreDataNum, int64_t bigCoreDataNum, int64_t finalBigTileNum,
                                int64_t finalSmallTileNum, int64_t tileDataNum, int64_t smallTailDataNum,
                                int64_t bigTailDataNum, int64_t tailBlockNum,
                                int64_t smallCoreDataNum2, int64_t bigCoreDataNum2, int64_t finalBigTileNum2,
                                int64_t finalSmallTileNum2, int64_t tileDataNum2, int64_t smallTailDataNum2,
                                int64_t bigTailDataNum2, int64_t tailBlockNum2
                               )
    {
        uint32_t coreNum = GetBlockIdx();
        globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        globalBufferIndex2 = bigCoreDataNum2 * GetBlockIdx();
        this->tileDataNum2 = tileDataNum2;
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        if (coreNum < tailBlockNum2)
        {
            this->coreDataNum2 = bigCoreDataNum2;
            this->tileNum2 = finalBigTileNum2;
            this->tailDataNum2 = bigTailDataNum2;
        }
        else
        {
            this->coreDataNum2 = smallCoreDataNum2;
            this->tileNum2 = finalSmallTileNum2;
            this->tailDataNum2 = smallTailDataNum2;
            globalBufferIndex2 -= (bigCoreDataNum2 - smallCoreDataNum2) * (GetBlockIdx() - tailBlockNum2);
        }

        this->pipe = pipeIn;
        pointsGm.SetGlobalBuffer((__gm__ float *)points);
        timestampGm.SetGlobalBuffer((__gm__ int32_t *)timestamp);
        outGm.SetGlobalBuffer((__gm__ float *)out);

        this->tMax = tMax;
        this->tMaxLow32 = tMaxLow32;
        this->f = f;
        this->trans[0] = trans[0];
        this->trans[1] = trans[1];
        this->trans[2] = trans[2];
        this->qRel[0] = qRel[0];
        this->qRel[1] = qRel[1];
        this->qRel[2] = qRel[2];
        this->qRel[3] = qRel[3];
        this->theta = theta;
        this->r_sin_theta = r_sin_theta;
        this->d_sign = d_sign;
        this->doRotation = doRotation;
        this->N = N;
        this->ndim = ndim;

        pipe->InitBuffer(inQueueP, BUFFER_NUM, this->tileDataNum * 3 * 4);
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * 3 * 4);
        pipe->InitBuffer(inQueueTs, BUFFER_NUM, this->tileDataNum * 8);

        pipe->InitBuffer(tmp1Buf, this->tileDataNum * 4);
        pipe->InitBuffer(tmp2Buf, this->tileDataNum * 4);
        pipe->InitBuffer(tmp3Buf, this->tileDataNum * 4);
        pipe->InitBuffer(tmp4Buf, this->tileDataNum * 4);
        pipe->InitBuffer(tmp5Buf, this->tileDataNum * 4);
        pipe->InitBuffer(tmp6Buf, this->tileDataNum * 4);
    }

    __aicore__ inline void Process(){
        this->processDataNum = this->tileDataNum;
        for(int i = 0; i < this->tileNum; i++){
            if (i == this->tileNum - 1)
            {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

        pipe->Reset();
        pipe->InitBuffer(inQueueOther, BUFFER_NUM, this->tileDataNum2 * 4);
        pipe->InitBuffer(outQueueOther, BUFFER_NUM, this->tileDataNum2 * 4);
        this->processDataNum2 = this->tileDataNum2;
        for(int i = 0; i < this->tileNum2; i++){
            if (i == this->tileNum2 - 1)
            {
                this->processDataNum2 = this->tailDataNum2;
            }
            OnlyCopy(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int progress)
    {
        LocalTensor<float> pLocal = inQueueP.AllocTensor<float>();
        LocalTensor<int32_t> tsLocal = inQueueTs.AllocTensor<int32_t>();
        
        DataCopy(pLocal, pointsGm[globalBufferIndex + progress * this->tileDataNum], this->processDataNum);
        DataCopy(pLocal[processDataNum], pointsGm[N + globalBufferIndex + progress * this->tileDataNum], this->processDataNum);
        DataCopy(pLocal[processDataNum + processDataNum], pointsGm[N + N + globalBufferIndex + progress * this->tileDataNum], this->processDataNum);
        DataCopy(tsLocal, timestampGm[(globalBufferIndex + progress * this->tileDataNum) * 2], this->processDataNum * 2);
        
        inQueueP.EnQue(pLocal);
        inQueueTs.EnQue(tsLocal);
    }

    __aicore__ inline void CopyOut(int progress)
    {
        LocalTensor<float> cLocal = outQueue.DeQue<float>();
        int offset = globalBufferIndex + progress * this->tileDataNum;
        if(progress == tileNum - 1 && offset + this->processDataNum >= N){
            for(int i = 0; offset + i < N; i++){
                outGm.SetValue(offset + i, cLocal.GetValue(i));
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(outGm[offset + i]);
                outGm.SetValue(N + offset + i, cLocal.GetValue(processDataNum + i));
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(outGm[N + offset + i]);
                outGm.SetValue(N + N + offset + i, cLocal.GetValue(processDataNum + processDataNum + i));
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(outGm[N + N + offset + i]);
            }
        }else{
            DataCopy(outGm[offset], cLocal, this->processDataNum);
            DataCopy(outGm[N + offset], cLocal[processDataNum], this->processDataNum);
            DataCopy(outGm[N + N + offset], cLocal[processDataNum + processDataNum], this->processDataNum);
        }
        outQueue.FreeTensor(cLocal);
    }

    __aicore__ inline void OnlyCopy(int progress)
    {
        int offset = N + N + N + globalBufferIndex2;
        auto inLocal = inQueueOther.AllocTensor<float>();
        DataCopy(inLocal, pointsGm[offset + progress * this->tileDataNum2], this->processDataNum2);
        inQueueOther.EnQue(inLocal);
        auto outLocal = outQueueOther.AllocTensor<float>();
        inLocal = inQueueOther.DeQue<float>();
        DataCopy(outLocal, inLocal, this->processDataNum2);
        inQueueOther.FreeTensor(inLocal);
        outQueueOther.EnQue(outLocal);
        outLocal = outQueueOther.DeQue<float>();
        DataCopy(outGm[offset + progress * this->tileDataNum2], outLocal, this->processDataNum2);
        outQueueOther.FreeTensor(outLocal);
    }

    __aicore__ inline void Compute(int progress)
    {
        LocalTensor<float> pLocal = inQueueP.DeQue<float>();
        LocalTensor<int32_t> tsLocal = inQueueTs.DeQue<int32_t>();
        LocalTensor<float> pxLocal = pLocal[0];
        LocalTensor<float> pyLocal = pLocal[processDataNum];
        LocalTensor<float> pzLocal = pLocal[processDataNum + processDataNum];
        
        LocalTensor<float> cLocal = outQueue.AllocTensor<float>();
        
        LocalTensor<float> cxLocal = cLocal[0];
        LocalTensor<float> cyLocal = cLocal[processDataNum];
        LocalTensor<float> czLocal = cLocal[processDataNum + processDataNum];
        LocalTensor<int32_t> offsetLocal = tmp1Buf.Get<int32_t>();
        LocalTensor<int32_t> high32Local = tmp2Buf.Get<int32_t>();
        LocalTensor<int32_t> low32Local = tmp3Buf.Get<int32_t>();
        LocalTensor<float> tmpLocal = tmp4Buf.Get<float>();
        LocalTensor<float> low32LocalFp = tmp5Buf.Get<float>();
        auto resLocal = offsetLocal.template ReinterpretCast<uint8_t>();
        CreateVecIndex(offsetLocal, 0, this->processDataNum);
        Muls(offsetLocal, offsetLocal, 8, this->processDataNum);
        Gather(low32Local, tsLocal, offsetLocal.template ReinterpretCast<uint32_t>(), (uint32_t)0, this->processDataNum);
        Duplicate(high32Local, tMaxLow32, this->processDataNum);
        Sub(low32Local, high32Local, low32Local, this->processDataNum);
        
        Cast(low32LocalFp, low32Local, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<float> tLocal = low32LocalFp[0];
        Muls(tLocal, tLocal, f, this->processDataNum);

        if(doRotation & 1){
            LocalTensor<float> c0Local = high32Local.template ReinterpretCast<float>();
            LocalTensor<float> c1Local = low32Local.template ReinterpretCast<float>();

            Muls(tmpLocal, tLocal, theta, this->processDataNum);
            Duplicate(c0Local, theta, this->processDataNum);
            Sub(c0Local, c0Local, tmpLocal, this->processDataNum);
            Sin(c0Local, c0Local, this->processDataNum);
            Muls(c0Local, c0Local, r_sin_theta, this->processDataNum);
            Sin(c1Local, tmpLocal, this->processDataNum);
            Muls(c1Local, c1Local, (r_sin_theta * d_sign), this->processDataNum);

            auto qiWLocal = offsetLocal.template ReinterpretCast<float>();
            auto qiXLocal = c0Local[0];
            auto qiYLocal = tsLocal.template ReinterpretCast<float>();
            auto qiZLocal = tsLocal[this->processDataNum].template ReinterpretCast<float>();

            Muls(qiWLocal, c1Local, qRel[0], this->processDataNum);
            Add(qiWLocal, qiWLocal, c0Local, this->processDataNum);
            Muls(qiXLocal, c1Local, qRel[1], this->processDataNum);
            Muls(qiYLocal, c1Local, qRel[2], this->processDataNum);
            Muls(qiZLocal, c1Local, qRel[3], this->processDataNum);

            auto invLocal = low32Local.template ReinterpretCast<float>();

            Mul(invLocal, qiWLocal, qiWLocal, this->processDataNum);
            Mul(tmpLocal, qiXLocal, qiXLocal, this->processDataNum);
            Add(invLocal, invLocal, tmpLocal, this->processDataNum);
            Mul(tmpLocal, qiYLocal, qiYLocal, this->processDataNum);
            Add(invLocal, invLocal, tmpLocal, this->processDataNum);
            Mul(tmpLocal, qiZLocal, qiZLocal, this->processDataNum);
            Add(invLocal, invLocal, tmpLocal, this->processDataNum);
            Duplicate(tmpLocal, 2.0f, this->processDataNum);
            Div(invLocal, tmpLocal, invLocal, this->processDataNum);

            auto tmp2Local = tmp6Buf.Get<float>();
            {
                Mul(tmp2Local, qiYLocal, qiYLocal, this->processDataNum);
                Mul(tmpLocal, qiZLocal, qiZLocal, this->processDataNum);
                Add(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(cxLocal, tmp2Local, pxLocal, this->processDataNum);

                Mul(tmp2Local, qiXLocal, qiYLocal, this->processDataNum);
                Mul(tmpLocal, qiWLocal, qiZLocal, this->processDataNum);
                Sub(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(tmp2Local, tmp2Local, pyLocal, this->processDataNum);
                Sub(cxLocal, tmp2Local, cxLocal, this->processDataNum);

                Mul(tmp2Local, qiXLocal, qiZLocal, this->processDataNum);
                Mul(tmpLocal, qiWLocal, qiYLocal, this->processDataNum);
                Add(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(tmp2Local, tmp2Local, pzLocal, this->processDataNum);
                Add(cxLocal, tmp2Local, cxLocal, this->processDataNum);
                Mul(cxLocal, cxLocal, invLocal, this->processDataNum);
                Add(cxLocal, pxLocal, cxLocal, this->processDataNum);
            }
            
            {
                Mul(tmp2Local, qiXLocal, qiYLocal, this->processDataNum);
                Mul(tmpLocal, qiWLocal, qiZLocal, this->processDataNum);
                Add(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(cyLocal, tmp2Local, pxLocal, this->processDataNum);

                Mul(tmp2Local, qiXLocal, qiXLocal, this->processDataNum);
                Mul(tmpLocal, qiZLocal, qiZLocal, this->processDataNum);
                Add(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(tmp2Local, tmp2Local, pyLocal, this->processDataNum);
                Sub(cyLocal, cyLocal, tmp2Local, this->processDataNum);

                Mul(tmp2Local, qiYLocal, qiZLocal, this->processDataNum);
                Mul(tmpLocal, qiWLocal, qiXLocal, this->processDataNum);
                Sub(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(tmp2Local, tmp2Local, pzLocal, this->processDataNum);
                Add(cyLocal, tmp2Local, cyLocal, this->processDataNum);
                Mul(cyLocal, cyLocal, invLocal, this->processDataNum);
                Add(cyLocal, pyLocal, cyLocal, this->processDataNum);
            }

            {
                Mul(tmp2Local, qiXLocal, qiZLocal, this->processDataNum);
                Mul(tmpLocal, qiWLocal, qiYLocal, this->processDataNum);
                Sub(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(czLocal, tmp2Local, pxLocal, this->processDataNum);

                Mul(tmp2Local, qiYLocal, qiZLocal, this->processDataNum);
                Mul(tmpLocal, qiWLocal, qiXLocal, this->processDataNum);
                Add(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(tmp2Local, tmp2Local, pyLocal, this->processDataNum);
                Add(czLocal, tmp2Local, czLocal, this->processDataNum);

                Mul(tmp2Local, qiXLocal, qiXLocal, this->processDataNum);
                Mul(tmpLocal, qiYLocal, qiYLocal, this->processDataNum);
                Add(tmp2Local, tmp2Local, tmpLocal, this->processDataNum);
                Mul(tmp2Local, tmp2Local, pzLocal, this->processDataNum);
                Sub(czLocal, czLocal, tmp2Local, this->processDataNum);
                Mul(czLocal, czLocal, invLocal, this->processDataNum);
                Add(czLocal, pzLocal, czLocal, this->processDataNum);
            }

            Muls(tmpLocal, tLocal, trans[0], this->processDataNum);
            Add(cxLocal, cxLocal, tmpLocal, this->processDataNum);
            
            Muls(tmpLocal, tLocal, trans[1], this->processDataNum);
            Add(cyLocal, cyLocal, tmpLocal, this->processDataNum);

            Muls(tmpLocal, tLocal, trans[2], this->processDataNum);
            Add(czLocal, czLocal, tmpLocal, this->processDataNum);
        }else{
            Muls(tmpLocal, tLocal, trans[0], this->processDataNum);
            Add(cxLocal, pxLocal, tmpLocal, this->processDataNum);

            Muls(tmpLocal, tLocal, trans[1], this->processDataNum);
            Add(cyLocal, pyLocal, tmpLocal, this->processDataNum);

            Muls(tmpLocal, tLocal, trans[2], this->processDataNum);
            Add(czLocal, pzLocal, tmpLocal, this->processDataNum);
        }
        Compare(resLocal, pxLocal, pxLocal, CMPMODE::EQ, this->processDataNum);
        Select(cxLocal, resLocal, cxLocal, pxLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Select(cyLocal, resLocal, cyLocal, pyLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Select(czLocal, resLocal, czLocal, pzLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

        outQueue.EnQue<float>(cLocal);
        inQueueP.FreeTensor(pLocal);
        inQueueTs.FreeTensor(tsLocal);
    }

private:
    TPipe *pipe;
    GlobalTensor<float>    pointsGm;
    GlobalTensor<int32_t> timestampGm;
    GlobalTensor<float>    outGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueP, inQueueTs, inQueueOther;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue, outQueueOther;
    TBuf<QuePosition::VECCALC> tmp1Buf, tmp2Buf, tmp3Buf, tmp4Buf, tmp5Buf, tmp6Buf;
    int64_t N, ndim;
    int64_t coreDataNum, tileNum, tileDataNum, tailDataNum, processDataNum, globalBufferIndex;
    int64_t coreDataNum2, tileNum2, tileDataNum2, tailDataNum2, processDataNum2, globalBufferIndex2;
    float tMax;
    float f, theta, r_sin_theta;
    float trans[3];
    float qRel[4];
    int32_t doRotation, tMaxLow32;
    float d_sign;
};

 
extern "C" __global__ __aicore__ void motion_compensation(GM_ADDR points, GM_ADDR timestamp, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipeIn;
    KernelMotionCompensation op;
    op.Init(points, timestamp, out, &pipeIn,
    tiling_data.N, tiling_data.ndim, tiling_data.tMax, tiling_data.f, tiling_data.theta, tiling_data.r_sin_theta,
    tiling_data.trans, tiling_data.qRel, tiling_data.doRotation, tiling_data.d_sign, tiling_data.tMaxLow32,
    tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
    tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
    tiling_data.bigTailDataNum, tiling_data.tailBlockNum,
    tiling_data.smallCoreDataNum2, tiling_data.bigCoreDataNum2, tiling_data.finalBigTileNum2, 
    tiling_data.finalSmallTileNum2, tiling_data.tileDataNum2, tiling_data.smallTailDataNum2,
    tiling_data.bigTailDataNum2, tiling_data.tailBlockNum2);
    op.Process();
}
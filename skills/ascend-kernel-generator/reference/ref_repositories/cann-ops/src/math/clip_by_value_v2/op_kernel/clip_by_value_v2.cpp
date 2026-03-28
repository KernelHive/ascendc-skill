/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include <type_traits>
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint64_t BUFFER_NUM = 2;
template<typename TYPE_X, typename TYPE_CLIP_VALUE_MIN, typename TYPE_CLIP_VALUE_MAX, typename TYPE_Y, bool IsExistBigCore> 
class KernelClipByValueV2 {
public:
    __aicore__ inline KernelClipByValueV2() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, 
        uint64_t smallCoreDataNum,
        uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum, 
        uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, 
        uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum, 
        uint64_t tailBlockNum,
        TPipe* pipeIn
    ) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore) 
        {
            if (coreNum < tailBlockNum) 
            { 
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            }
            else 
            { 
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe = pipeIn;
        pipe->InitBuffer(Q_x, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe->InitBuffer(Q_min, this->ubPartDataNum * sizeof(TYPE_CLIP_VALUE_MIN));
        pipe->InitBuffer(Q_max, this->ubPartDataNum * sizeof(TYPE_CLIP_VALUE_MAX));
        Gm_clip_value_min.SetGlobalBuffer((__gm__ TYPE_CLIP_VALUE_MIN*)clip_value_min, 1);
        Gm_clip_value_max.SetGlobalBuffer((__gm__ TYPE_CLIP_VALUE_MAX*)clip_value_max, 1);
        pipe->InitBuffer(Q_y, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
        this->clip_value_min = Gm_clip_value_min.GetValue(0);
        this->clip_value_max = Gm_clip_value_max.GetValue(0);
    }
    __aicore__ inline void Process() {
        uint64_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (uint64_t i = 0; i < loopCount-1; i++) 
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }
private:
    __aicore__ inline void CopyIn(uint64_t progress) {
        AscendC::LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        AscendC::DataCopy(x, Gm_x[progress * this->ubPartDataNum], this->processDataNum);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(uint64_t progress) {
        AscendC::LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<TYPE_CLIP_VALUE_MIN> min = Q_min.Get<TYPE_CLIP_VALUE_MIN>();
        AscendC::LocalTensor<TYPE_CLIP_VALUE_MAX> max = Q_max.Get<TYPE_CLIP_VALUE_MAX>();
        AscendC::Duplicate(min, this->clip_value_min, this->processDataNum);
        AscendC::Duplicate(max, this->clip_value_max, this->processDataNum);
        AscendC::Max(y, x, min, this->processDataNum);
        AscendC::Min(y, y, max, this->processDataNum);
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(uint64_t progress) {
        AscendC::LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        AscendC::DataCopy(Gm_y[progress * this->ubPartDataNum], y, this->processDataNum);
        Q_y.FreeTensor(y);
    }
private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_x;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> Q_min;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> Q_max;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;
    AscendC::GlobalTensor<TYPE_X> Gm_x;
    AscendC::GlobalTensor<TYPE_CLIP_VALUE_MIN> Gm_clip_value_min;
    AscendC::GlobalTensor<TYPE_CLIP_VALUE_MAX> Gm_clip_value_max;
    AscendC::GlobalTensor<TYPE_Y> Gm_y;
    TYPE_CLIP_VALUE_MIN clip_value_min;
    TYPE_CLIP_VALUE_MAX clip_value_max;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template<bool IsExistBigCore> 
class KernelClipByValueV2<bfloat16_t,bfloat16_t,bfloat16_t,bfloat16_t,IsExistBigCore> {
public:
    __aicore__ inline KernelClipByValueV2() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, 
        uint64_t smallCoreDataNum,
        uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum, 
        uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, 
        uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum, 
        uint64_t tailBlockNum,
        TPipe* pipeIn
        ) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore) 
        {
            if (coreNum < tailBlockNum) 
            { 
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            }
            else 
            { 
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        Gm_x.SetGlobalBuffer((__gm__ bfloat16_t*)x + globalBufferIndex, this->coreDataNum);
        Gm_y.SetGlobalBuffer((__gm__ bfloat16_t*)y + globalBufferIndex, this->coreDataNum);

        pipe = pipeIn;

        pipe->InitBuffer(Q_x_fp32, this->ubPartDataNum * sizeof(float));
        pipe->InitBuffer(Q_min_fp32, this->ubPartDataNum * sizeof(float));
        pipe->InitBuffer(Q_max_fp32, this->ubPartDataNum * sizeof(float));
        pipe->InitBuffer(Q_y_fp32, this->ubPartDataNum * sizeof(float));
        pipe->InitBuffer(Q_x_bf16, BUFFER_NUM, this->ubPartDataNum * sizeof(bfloat16_t));
        pipe->InitBuffer(Q_y_bf16, BUFFER_NUM, this->ubPartDataNum * sizeof(bfloat16_t));
        Gm_clip_value_min.SetGlobalBuffer((__gm__ bfloat16_t*)clip_value_min, 1);
        Gm_clip_value_max.SetGlobalBuffer((__gm__ bfloat16_t*)clip_value_max, 1);
        
        this->clip_value_min = AscendC::ToFloat(Gm_clip_value_min.GetValue(0));
        this->clip_value_max = AscendC::ToFloat(Gm_clip_value_max.GetValue(0));
    }
    __aicore__ inline void Process() {
        uint64_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (uint64_t i = 0; i < loopCount-1; i++) 
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }
private:
    __aicore__ inline void CopyIn(uint64_t progress) {
        AscendC::LocalTensor<bfloat16_t> x_bf16 = Q_x_bf16.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(x_bf16, Gm_x[progress * this->ubPartDataNum], this->processDataNum);
        Q_x_bf16.EnQue(x_bf16);
    }
    __aicore__ inline void Compute(uint64_t progress) {
        AscendC::LocalTensor<bfloat16_t> x_bf16 = Q_x_bf16.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> y_bf16 = Q_y_bf16.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<float> x_fp32 = Q_x_fp32.Get<float>();
        AscendC::LocalTensor<float> y_fp32 = Q_y_fp32.Get<float>();
        AscendC::Cast(x_fp32,x_bf16,AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::LocalTensor<float> min = Q_min_fp32.Get<float>();
        AscendC::LocalTensor<float> max = Q_max_fp32.Get<float>();
        AscendC::Duplicate(min, this->clip_value_min, this->processDataNum);
        AscendC::Duplicate(max, this->clip_value_max, this->processDataNum);
        AscendC::Max(y_fp32, x_fp32, min, this->processDataNum);
        AscendC::Min(y_fp32, y_fp32, max, this->processDataNum);
        AscendC::Cast(y_bf16, y_fp32, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        Q_x_bf16.FreeTensor(x_bf16);
        Q_y_bf16.EnQue<bfloat16_t>(y_bf16);
    }
    __aicore__ inline void CopyOut(uint64_t progress) {
        AscendC::LocalTensor<bfloat16_t> y_bf16 = Q_y_bf16.DeQue<bfloat16_t>();
        AscendC::DataCopy(Gm_y[progress * this->ubPartDataNum], y_bf16, this->processDataNum);
        Q_y_bf16.FreeTensor(y_bf16);
    }
private:
    AscendC::TPipe* pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> Q_x_fp32;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> Q_min_fp32;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> Q_max_fp32;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> Q_y_fp32;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_x_bf16;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y_bf16;
    AscendC::GlobalTensor<bfloat16_t> Gm_x;
    AscendC::GlobalTensor<bfloat16_t> Gm_clip_value_min;
    AscendC::GlobalTensor<bfloat16_t> Gm_clip_value_max;
    AscendC::GlobalTensor<bfloat16_t> Gm_y;
    float clip_value_min;
    float clip_value_max;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

extern "C" __global__ __aicore__ void clip_by_value_v2(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if(TILING_KEY_IS(1))
    {
        GET_TILING_DATA(tiling_data, tiling);
        TPipe pipe;
        KernelClipByValueV2<DTYPE_X, DTYPE_CLIP_VALUE_MIN, DTYPE_CLIP_VALUE_MAX, DTYPE_Y, true> op;
        op.Init(x, clip_value_min, clip_value_max, y,
                tiling_data.smallCoreDataNum,                                
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   
                tiling_data.tailBlockNum,
                &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(0))
    {
        GET_TILING_DATA(tiling_data, tiling);
        TPipe pipe;
        KernelClipByValueV2<DTYPE_X, DTYPE_CLIP_VALUE_MIN, DTYPE_CLIP_VALUE_MAX, DTYPE_Y, false> op;
        op.Init(x, clip_value_min, clip_value_max, y,
                tiling_data.smallCoreDataNum,                                
                tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,             
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,            
                tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum,   
                tiling_data.tailBlockNum,
                &pipe);
        op.Process();
    }
}
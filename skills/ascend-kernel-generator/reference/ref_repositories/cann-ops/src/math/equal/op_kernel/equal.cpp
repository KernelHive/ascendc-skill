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
 * @file equal.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr int32_t EQ_VALUE = 1006648320;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;
constexpr uint32_t BLOCK_SIZE = 32;


template <typename TYPE_X>
class KernelEqual
{
public:
    __aicore__ inline KernelEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum,uint32_t bigprocessDataNum_computes, 
                                uint32_t smallprocessDataNum_computes, uint32_t tailbigprocessDataNum_computes, 
                                uint32_t tailsmallprocessDataNum_computes)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        uint32_t increment=0;
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
            this->processDataNum_computes=bigprocessDataNum_computes;
            this->tailprocessDataNum_computes=tailbigprocessDataNum_computes;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            this->processDataNum_computes=smallprocessDataNum_computes;
            this->tailprocessDataNum_computes=tailsmallprocessDataNum_computes;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X *)x1+ globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X *)x2+ globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y+ globalBufferIndex, this->coreDataNum);
        if constexpr (std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, uint32_t> || std::is_same_v<DTYPE_X1, float>)
        {
            increment=256;
        }

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X)+increment);
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X)+increment);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(int8_t)+increment);
        if constexpr (std::is_same_v<DTYPE_X1, half> || std::is_same_v<DTYPE_X1, bfloat16_t>)
        {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half));
        }
        else if constexpr (std::is_same_v<DTYPE_X1, float>)
        {
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half)+increment);
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int8_t> || std::is_same_v<DTYPE_X1, uint8_t>)
        {
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(half));
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, uint32_t>)
        {
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half)+increment);
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount-1; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->processDataNum_computes=this->tailprocessDataNum_computes;

        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> x1Local = inQueueX1.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_X> x2Local = inQueueX2.AllocTensor<TYPE_X>();
        DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        if constexpr (std::is_same_v<DTYPE_X1, half> || std::is_same_v<DTYPE_X1, bfloat16_t>)
        {
            LocalTensor<half> x1_local = inQueueX1.DeQue<half>();
            LocalTensor<half> x2_local = inQueueX2.DeQue<half>();
            LocalTensor<int8_t> y_local = outQueueY.AllocTensor<int8_t>();
            LocalTensor<half> y_compute = tmp1.Get<half>();
            // Step 1: 计算差值 diff = x1 - x2
            Sub(y_compute, x1_local, x2_local, this->processDataNum);

            // Step 2: 取绝对值 abs_diff = |diff|
            Abs(y_compute, y_compute, this->processDataNum);

            // Step 3: 误差容差处理，将小于误差值的差值设置为 0
            Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->processDataNum);

            // Step 4: 将所有非零值设置为 1
            Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->processDataNum);
            Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->processDataNum);

            // Step 5: 最终结果：将所有非零值设置为 1，零值保持为 0
            Duplicate(x1_local, (half)POSITIVE_ONE_FP32, this->processDataNum);
            Sub(y_compute, x1_local, y_compute, this->processDataNum);

            // 将结果转换为 FP16 类型并保存到输出张量
            Cast(y_local, y_compute, RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(y_local);
            inQueueX1.FreeTensor(x1_local);
            inQueueX2.FreeTensor(x2_local);
        }
        else if constexpr (std::is_same_v<DTYPE_X1, float>)
        {
            LocalTensor<float> x1_local = inQueueX1.DeQue<float>();
            LocalTensor<float> x2_local = inQueueX2.DeQue<float>();
            LocalTensor<int8_t> y_local = outQueueY.AllocTensor<int8_t>();
            auto p1 = tmp2.Get<half>();

            AscendC::Compare(y_local, x1_local, x2_local, AscendC::CMPMODE::EQ, this->processDataNum_computes);
            AscendC::Duplicate<float>(x1_local, (float)1, this->processDataNum_computes);
            AscendC::Select(x1_local, y_local, x1_local, (float)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum_computes);
            AscendC::Cast(p1, x1_local, AscendC::RoundMode::CAST_NONE, this->processDataNum_computes);
            AscendC::Cast(y_local, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum_computes);

            outQueueY.EnQue<int8_t>(y_local);
            inQueueX1.FreeTensor(x1_local);
            inQueueX2.FreeTensor(x2_local);
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int8_t> || std::is_same_v<DTYPE_X1, uint8_t>)
        {
            LocalTensor<int8_t> x1_local = inQueueX1.DeQue<int8_t>();
            LocalTensor<int8_t> x2_local = inQueueX2.DeQue<int8_t>();
            LocalTensor<int8_t> y_local = outQueueY.AllocTensor<int8_t>();
            // Step 1: 初始化中间张量
            LocalTensor<half> x1_local_fp16 = tmp2.Get<half>();
            LocalTensor<half> x2_local_fp16 = tmp3.Get<half>();
            LocalTensor<half> y_local_fp16 = tmp4.Get<half>();

            // Step 2: 将 int8 转换为 FP16
            Cast(x1_local_fp16, x1_local, RoundMode::CAST_NONE, this->processDataNum);
            Cast(x2_local_fp16, x2_local, RoundMode::CAST_NONE, this->processDataNum);

            // Step 3: 计算差值 diff = x1 - x2
            Sub(y_local_fp16, x1_local_fp16, x2_local_fp16, this->processDataNum);
            // Step 4: 取绝对值 abs_diff = |diff|
            Abs(y_local_fp16, y_local_fp16, this->processDataNum);

            // Step 5: 将所有非零值设置为 1
            Mins(y_local_fp16, y_local_fp16, (half)POSITIVE_ONE_FP32, this->processDataNum);

            // Step 6: 布尔值反转，将零值设为 1，非零值设为 0
            Duplicate(x1_local_fp16, (half)POSITIVE_ONE_FP32, this->processDataNum);
            Sub(y_local_fp16, x1_local_fp16, y_local_fp16, this->processDataNum);

            // Step 7: 将结果转换为 int8 并保存到输出张量
            Cast(y_local, y_local_fp16, RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<int8_t>(y_local);
            inQueueX1.FreeTensor(x1_local);
            inQueueX2.FreeTensor(x2_local);
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int32_t> || std::is_same_v<DTYPE_X1, uint32_t>)
        {
            LocalTensor<int32_t> x1_local = inQueueX1.DeQue<int32_t>();
            LocalTensor<int32_t> x2_local = inQueueX2.DeQue<int32_t>();
            LocalTensor<int8_t> y_local = outQueueY.AllocTensor<int8_t>();

            LocalTensor<half> y_fp16 = tmp2.Get<half>();

            AscendC::Compare(y_local, x1_local, x2_local, AscendC::CMPMODE::EQ, this->processDataNum_computes);
            AscendC::Duplicate<half>(y_fp16, (half)1, this->processDataNum_computes);
            AscendC::Select(y_fp16, y_local, y_fp16, (half)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum_computes);
            AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->processDataNum_computes);
            
            inQueueX1.FreeTensor(x1_local);
            inQueueX2.FreeTensor(x2_local);
            outQueueY.EnQue<int8_t>(y_local);
        }
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4;

    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<TYPE_X> x2Gm;
    GlobalTensor<int8_t> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum_computes;
    uint32_t tailprocessDataNum_computes;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelEqual<DTYPE_X1> op;
    
    op.Init(x1, x2, y, tiling_data.smallCoreDataNum, 
        tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
        tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
        tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
        tiling_data.tailBlockNum,
        tiling_data.bigprocessDataNum_computes, tiling_data.smallprocessDataNum_computes, 
        tiling_data.tailbigprocessDataNum_computes, tiling_data.tailsmallprocessDataNum_computes);
    op.Process();
}

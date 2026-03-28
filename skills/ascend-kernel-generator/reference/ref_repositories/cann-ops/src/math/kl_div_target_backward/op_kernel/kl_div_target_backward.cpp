/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t INPUT_VAR_NUM = 3;
constexpr int32_t MAX_DIM_NUM = 8;
constexpr uint32_t ALIGNED_UB_SIZE = 256;

template<typename T>
__aicore__ inline T RoundUp(T a, T b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
class KernelKlDivTargetBackward {
public:
    __aicore__ inline KernelKlDivTargetBackward() {}
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR target, GM_ADDR gradTarget,
                                int64_t smallCoreDataNum,
                                int64_t bigCoreDataNum, int64_t finalBigTileNum, 
                                int64_t finalSmallTileNum, int64_t tileDataNum, 
                                int64_t smallTailDataNum, int64_t bigTailDataNum, 
                                int64_t tailBlockNum, uint32_t reduction, uint32_t logTarget,
                                int64_t inputNum) 
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        int64_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        // 单次搬运数据个数 tileDataNum
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) { 
            // 每个大核要处理的总输入数据量 bigCoreDataNum*sizeof(T)
            this->coreDataNum = bigCoreDataNum;
            // 每个大核要搬运几次 finalBigTileNum 次
            this->tileNum = finalBigTileNum;
            // 每个大核最后一次搬运数据量 bigTailDataNum
            this->tailDataNum = bigTailDataNum;
        }
        else { 
            // 每个小核要处理的总输入数据量 smallCoreDataNum*sizeof(T)
            this->coreDataNum = smallCoreDataNum;
            // 每个小核要搬运几次 finalSmallTileNum 次
            this->tileNum = finalSmallTileNum;
            // 每个小核最后一次搬运数据量 smallTailDataNum
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->globalBufferIndex = globalBufferIndex;
        this->reduction = reduction;
        this->logTarget = logTarget;
        this->inputNum = inputNum;
        gradOutputGm.SetGlobalBuffer((__gm__ T*)gradOutput);
        selfGm.SetGlobalBuffer((__gm__ T*)self);
        targetGm.SetGlobalBuffer((__gm__ T*)target);
        gradTargetGm.SetGlobalBuffer((__gm__ T*)gradTarget);
        pipe.InitBuffer(inQueueGradOuput, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueSelf, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueTarget, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueGradTarget, BUFFER_NUM, this->tileDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(gradOutputBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(selfBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(targetBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(gradTargetBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBuf, BUFFER_NUM * this->tileDataNum * sizeof(float));
        } else {
            pipe.InitBuffer(tmpBuf, BUFFER_NUM * this->tileDataNum * sizeof(T));
        }
    }
    __aicore__ inline void Process()
    {
        int64_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int64_t i = 0; i < loopCount; i++) {
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
        LocalTensor<T> gradOutputLocal = inQueueGradOuput.AllocTensor<T>();
        LocalTensor<T> selfLocal = inQueueSelf.AllocTensor<T>();
        LocalTensor<T> targetLocal = inQueueTarget.AllocTensor<T>();
        
        int64_t offset = globalBufferIndex + progress * this->tileDataNum;
        DataCopy(gradOutputLocal, gradOutputGm[offset], this->processDataNum);
        DataCopy(selfLocal, selfGm[offset], this->processDataNum);
        DataCopy(targetLocal, targetGm[offset], this->processDataNum);
        
        inQueueGradOuput.EnQue(gradOutputLocal);
        inQueueSelf.EnQue(selfLocal);
        inQueueTarget.EnQue(targetLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> gradOutputLocal = inQueueGradOuput.DeQue<T>();
        LocalTensor<T> selfLocal = inQueueSelf.DeQue<T>();
        LocalTensor<T> targetLocal = inQueueTarget.DeQue<T>();
        LocalTensor<T> gradTargetLocal = outQueueGradTarget.AllocTensor<T>();
        LocalTensor<uint8_t> maskLocal = tmpBuf.Get<uint8_t>();
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            T scalarOne = 1;
            T scalarZero = 0;
            LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
            if (logTarget) {
                Adds(gradTargetLocal, targetLocal, scalarOne, this->processDataNum);
                Sub(gradTargetLocal, gradTargetLocal, selfLocal, this->processDataNum);
                Exp(tmpLocal, targetLocal, this->processDataNum);
                Mul(gradTargetLocal, gradTargetLocal, tmpLocal, this->processDataNum);
                Mul(gradTargetLocal, gradOutputLocal, gradTargetLocal, this->processDataNum);
            } else {
                Ln(tmpLocal, targetLocal, this->processDataNum);
                Adds(gradTargetLocal, tmpLocal, scalarOne, this->processDataNum);
                Sub(gradTargetLocal, gradTargetLocal, selfLocal, this->processDataNum);
                Mul(gradTargetLocal, gradOutputLocal, gradTargetLocal, this->processDataNum);
                uint32_t calCount =
                    RoundUp(static_cast<uint32_t>(this->processDataNum * sizeof(T)), ALIGNED_UB_SIZE) / sizeof(T);
                CompareScalar(maskLocal, targetLocal, scalarZero, CMPMODE::NE, calCount);
                Select(gradTargetLocal, maskLocal, gradTargetLocal, static_cast<T>(0),
                    SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            }
            if (reduction == 1) {
                T elemTotalCof = static_cast<T>(1.0f / static_cast<float>(inputNum));
                Muls(gradTargetLocal, gradTargetLocal, elemTotalCof, this->processDataNum);
            }
        } else {
            float scalarOne = 1;
            float scalarZero = 0;
            LocalTensor<float> tmpLocalFp32 = tmpBuf.Get<float>();
            LocalTensor<float> gradOutputLocalFp32 = gradOutputBuf.Get<float>();
            LocalTensor<float> selfLocalFp32 = selfBuf.Get<float>();
            LocalTensor<float> targetLocalFp32 = targetBuf.Get<float>();
            LocalTensor<float> gradTargetLocalFp32 = gradTargetBuf.Get<float>();
            Cast(gradOutputLocalFp32, gradOutputLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(selfLocalFp32, selfLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(targetLocalFp32, targetLocal, RoundMode::CAST_NONE, this->processDataNum);
            if (logTarget) {
                Adds(gradTargetLocalFp32, targetLocalFp32, scalarOne, this->processDataNum);
                Sub(gradTargetLocalFp32, gradTargetLocalFp32, selfLocalFp32, this->processDataNum);
                Exp(tmpLocalFp32, targetLocalFp32, this->processDataNum);
                Mul(gradTargetLocalFp32, gradTargetLocalFp32, tmpLocalFp32, this->processDataNum);
                Mul(gradTargetLocalFp32, gradOutputLocalFp32, gradTargetLocalFp32, this->processDataNum);
            } else {
                Ln(tmpLocalFp32, targetLocalFp32, this->processDataNum);
                Adds(gradTargetLocalFp32, tmpLocalFp32, scalarOne, this->processDataNum);
                Sub(gradTargetLocalFp32, gradTargetLocalFp32, selfLocalFp32, this->processDataNum);
                Mul(gradTargetLocalFp32, gradOutputLocalFp32, gradTargetLocalFp32, this->processDataNum);
                uint32_t calCount =
                    RoundUp(static_cast<uint32_t>(this->processDataNum * sizeof(float)), ALIGNED_UB_SIZE) /
                    sizeof(float);
                CompareScalar(maskLocal, targetLocalFp32, scalarZero, CMPMODE::NE, calCount);
                Select(gradTargetLocalFp32, maskLocal, gradTargetLocalFp32, 0.0f,
                    SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            }
            if (reduction == 1) {
                float elemTotalCof = 1.0f / static_cast<float>(inputNum);
                Muls(gradTargetLocalFp32, gradTargetLocalFp32, elemTotalCof, this->processDataNum);
            }
            Cast(gradTargetLocal, gradTargetLocalFp32, RoundMode::CAST_RINT, this->processDataNum);
        }

        outQueueGradTarget.EnQue<T>(gradTargetLocal);
        inQueueGradOuput.FreeTensor(gradOutputLocal);
        inQueueSelf.FreeTensor(selfLocal);
        inQueueTarget.FreeTensor(targetLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      LocalTensor<T> gradTargetLocal = outQueueGradTarget.DeQue<T>();  
      DataCopy(gradTargetGm[globalBufferIndex + progress * this->tileDataNum], gradTargetLocal, this->processDataNum);
      outQueueGradTarget.FreeTensor(gradTargetLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradOuput, inQueueSelf, inQueueTarget;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradTarget;
    TBuf<QuePosition::VECCALC> tmpBuf, gradOutputBuf, selfBuf, targetBuf, gradTargetBuf;
    GlobalTensor<T> gradOutputGm, selfGm, targetGm, gradTargetGm;
    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    int64_t inputNum;
    int64_t globalBufferIndex;
    uint32_t reduction;
    uint32_t logTarget;
};

extern "C" __global__ __aicore__ void kl_div_target_backward(GM_ADDR grad_output, GM_ADDR self, GM_ADDR target,
    GM_ADDR grad_target, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    KernelKlDivTargetBackward<DTYPE_TARGET> op;
    op.Init(grad_output, self, target, grad_target, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum, tiling_data.reduction, tiling_data.logTarget,
            tiling_data.inputNum);
    op.Process();
    return;
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void kl_div_target_backward_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *grad_output,
    uint8_t *self, uint8_t *target, uint8_t *grad_target, uint8_t *workspace, uint8_t *tiling)
{
    kl_div_target_backward<<<blockDim, l2ctrl, stream>>>(grad_output, self, target, grad_target, workspace, tiling);
}
#endif
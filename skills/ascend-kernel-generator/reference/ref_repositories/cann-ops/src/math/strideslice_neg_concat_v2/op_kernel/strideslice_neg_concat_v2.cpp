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
 * @file stride_slice_neg_concat_v2.cpp
 */

#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
namespace AscendC {
template <typename T>
class StridesliceNegConcatV2 {
public:
    __aicore__ inline StridesliceNegConcatV2(){}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const StridesliceNegConcatV2Tiling &tiling_data)
    {
        this->blockLength = tiling_data.totalLength / AscendC::GetBlockNum();
        this->tileNumAverage = tiling_data.tileNumAverage;
        this->tileNumLast = tiling_data.tileNumLast;
        this->tileLength = tiling_data.tileLength;
        inputGlobal.SetGlobalBuffer((__gm__ T*)input + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGlobal.SetGlobalBuffer((__gm__ T*)output + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(mulVecBuf, BUFFER_NUM * this->tileLength * sizeof(T));
    }

     __aicore__ inline void Process()
    {
        // 构造[1;-1]向量
        LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>();
        T scalar1 = 1.0;
        T scalar2 = -1.0;
        uint32_t repeatNum = this->tileLength / 2;
        Duplicate(mulVecBufLocal, scalar1, repeatNum);
        Duplicate(mulVecBufLocal[repeatNum], scalar2, repeatNum);
        uint32_t loopCount = 0;
        if (GetBlockIdx() == GetBlockNum() - 1) {
	    	loopCount = this->tileNumLast;
    	} else {
    		loopCount = this->tileNumAverage;
    	} for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        DataCopy(inputLocal, inputGlobal[progress * this->tileLength], this->tileLength);
        inQueueInput.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>();
        LocalTensor<T> outputLocal = outQueueOutput.AllocTensor<T>();
        Mul(outputLocal, inputLocal, mulVecBufLocal, this->tileLength);
        inQueueInput.FreeTensor(inputLocal);
        outQueueOutput.EnQue<T>(outputLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> outputLocal = outQueueOutput.DeQue<T>();
        DataCopy(outputGlobal[progress * this->tileLength], outputLocal, this->tileLength);
        outQueueOutput.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;

    TBuf<QuePosition::VECCALC> mulVecBuf;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    uint32_t blockLength;
    uint32_t tileNumAverage;
    uint32_t tileNumLast;
    uint32_t tileLength;
};
}

extern "C" __global__ __aicore__ void strideslice_neg_concat_v2(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    StridesliceNegConcatV2<half> op;
    op.Init(input, output, tiling_data);
    op.Process();
}
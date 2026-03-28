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
 * @file heaviside.cpp
 */
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
using namespace AscendC;
class KernelHeaviside_broadcast {
public:
    __aicore__ inline KernelHeaviside_broadcast() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out, uint32_t size, uint16_t *mmInputDims, uint16_t *mmValuesDims,
                                uint16_t *mmOutputDims, uint8_t nOutputDims) {
        inputGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)input, size);
        valuesGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)values, size);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out, size);
        this->size = size;
        this->mmInputDims = mmInputDims;
        this->mmValuesDims = mmValuesDims;
        this->mmOutputDims = mmOutputDims;
        this->nOutputDims = nOutputDims;
    }
    __aicore__ inline uint32_t mapIndex(uint32_t outputIdx, uint16_t *inputShape) {
        uint32_t inputIdx = 0;      // 输入张量的线性下标
        uint32_t stride = 1;        // 输入张量的步长累积
        uint32_t remainingIdx = outputIdx;  // 剩余待分解的输出下标
        for (int dim = nOutputDims - 1; dim >= 0; --dim) {
            int coord = remainingIdx % mmOutputDims[dim];  // 当前维度坐标
            remainingIdx = remainingIdx / mmOutputDims[dim];
                // 仅当输入维度不为1时累加坐标贡献
            if (inputShape[dim] != 1) {
                inputIdx += coord * stride;
                stride *= inputShape[dim];
            }
        }
        return inputIdx;
    }
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < size; i++) {
            uint32_t inputIdx = mapIndex(i, mmInputDims);
            float currentInput = static_cast<float>(inputGm.GetValue(inputIdx));
            if (currentInput < 0.0f) {
                outGm.SetValue(i, 0.0f);
            } else if (currentInput > 0.0f) {
                outGm.SetValue(i, 1.0f);
            } else if (currentInput == 0.0f) {
                uint32_t valuesIdx = mapIndex(i, mmValuesDims);
                outGm.SetValue(i, (float)valuesGm.GetValue(valuesIdx));
            }
        }
    }

private:
    GlobalTensor<DTYPE_OUT> inputGm;
    GlobalTensor<DTYPE_OUT> valuesGm;
    GlobalTensor<DTYPE_OUT> outGm;
    uint32_t size;
    uint16_t *mmInputDims;
    uint16_t *mmValuesDims;
    uint16_t *mmOutputDims;
    uint8_t nOutputDims;
};
constexpr uint16_t BufferNum = 1;
template <bool flag, bool valuesCountIsOne>
class KernelHeaviside {
public:
    __aicore__ inline KernelHeaviside() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out, uint32_t smallSize, uint16_t incSize, uint16_t formerNum, TPipe *pipeIn) {
        pipe = pipeIn;
        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = size * GetBlockIdx() + formerNum * incSize;
        }
        inputGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input + beginIndex, size);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out + beginIndex, size);
        if constexpr (valuesCountIsOne) {
            valuesGm.SetGlobalBuffer((__gm__ DTYPE_VALUES *)values, size);
            this->scalar = valuesGm.GetValue(0);
        } else {
            valuesGm.SetGlobalBuffer((__gm__ DTYPE_VALUES *)values + beginIndex, size);
        }
        if constexpr (flag) {
            uint32_t spaceSize = size * sizeof(DTYPE_VALUES) / BufferNum;
            pipe->InitBuffer(inQueueInput, BufferNum, spaceSize);
            pipe->InitBuffer(inQueueValues, BufferNum, spaceSize);
            pipe->InitBuffer(selMaskQueue, BufferNum, size >> 3);
            pipe->InitBuffer(outQueueY, BufferNum, spaceSize);
        } else {
            pipe->InitBuffer(inQueueInput, BufferNum, 100 * 512 / BufferNum);
            pipe->InitBuffer(inQueueValues, BufferNum, 100 * 512 / BufferNum);
            pipe->InitBuffer(selMaskQueue, BufferNum, 100 * 512 / BufferNum / 8 / sizeof(DTYPE_VALUES));
            pipe->InitBuffer(outQueueY, BufferNum, 100 * 512 / BufferNum);
        }
    }
    __aicore__ inline void Process_iter(uint32_t beginIndex, uint32_t iterSize) {
        LocalTensor<DTYPE_INPUT> valuesLocal = inQueueValues.AllocTensor<DTYPE_VALUES>();
        if constexpr (valuesCountIsOne) {
            PipeBarrier<PIPE_V>();
            Duplicate(valuesLocal, scalar, iterSize);
        } else {
            DataCopy(valuesLocal, valuesGm[beginIndex], iterSize);
            inQueueValues.EnQue<DTYPE_VALUES>(valuesLocal);
            valuesLocal = inQueueValues.DeQue<DTYPE_VALUES>();
        }
        LocalTensor<DTYPE_INPUT> inputLocal = inQueueInput.AllocTensor<DTYPE_INPUT>();
        DataCopy(inputLocal, inputGm[beginIndex], iterSize);
        inQueueInput.EnQue<DTYPE_INPUT>(inputLocal);
        inputLocal = inQueueInput.DeQue<DTYPE_INPUT>();
        LocalTensor<DTYPE_INPUT> outLocal = outQueueY.AllocTensor<DTYPE_VALUES>();
        LocalTensor<uint8_t> selMaskLocal = selMaskQueue.AllocTensor<uint8_t>();
        PipeBarrier<PIPE_V>();
            // 等于0，从valuesLocal取,否则从1取
        CompareScalar(selMaskLocal, inputLocal, static_cast<DTYPE_VALUES>(0.0), CMPMODE::EQ, iterSize);
        PipeBarrier<PIPE_V>();
        Select(valuesLocal, selMaskLocal, valuesLocal, static_cast<DTYPE_VALUES>(1.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        PipeBarrier<PIPE_V>();
            // 大于等于0，从valuesLocal取，否则从0取
        CompareScalar(selMaskLocal, inputLocal, static_cast<DTYPE_VALUES>(0.0), CMPMODE::GE, iterSize);
        PipeBarrier<PIPE_V>();
        Select(outLocal, selMaskLocal, valuesLocal, static_cast<DTYPE_VALUES>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        inQueueInput.FreeTensor<DTYPE_INPUT>(inputLocal);
        inQueueValues.FreeTensor<DTYPE_INPUT>(valuesLocal);
        selMaskQueue.FreeTensor<uint8_t>(selMaskLocal);
        outQueueY.EnQue<DTYPE_VALUES>(outLocal);
        outLocal = outQueueY.DeQue<DTYPE_VALUES>();
        DataCopy(outGm[beginIndex], outLocal, iterSize);
        outQueueY.FreeTensor<DTYPE_INPUT>(outLocal);
    }
    __aicore__ inline void Process() {
        if constexpr (flag) {
            if constexpr (BufferNum == 2) {
                uint32_t n_elements_per_iter = size / BufferNum;
                Process_iter(0, n_elements_per_iter);
                Process_iter(n_elements_per_iter, n_elements_per_iter);
            } else {
                Process_iter(0, size);
            }
        } else {
            uint32_t n_elements_per_iter = 100 * 512 / sizeof(DTYPE_VALUES) / BufferNum;
            uint16_t repeatTimes = size / n_elements_per_iter;
            uint16_t remainSize = size % n_elements_per_iter;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                Process_iter(i * n_elements_per_iter, n_elements_per_iter);
            }
            Process_iter(repeatTimes * n_elements_per_iter, remainSize);
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_INPUT> inputGm;
    GlobalTensor<DTYPE_VALUES> valuesGm;
    GlobalTensor<DTYPE_OUT> outGm;
    TQue<QuePosition::VECIN, BufferNum> inQueueInput;
    TQue<QuePosition::VECIN, BufferNum> inQueueValues;
    TQue<QuePosition::VECOUT, BufferNum> outQueueY;
    TQue<QuePosition::VECCALC, BufferNum> selMaskQueue;
    uint32_t size;
    DTYPE_VALUES scalar;
};
class KernelHeaviside_smallData_and_scalar {
public:
    __aicore__ inline KernelHeaviside_smallData_and_scalar() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out, uint32_t smallSize, uint16_t incSize, uint16_t formerNum, TPipe *pipe) {
        GlobalTensor<DTYPE_INPUT> inputGm;
        GlobalTensor<DTYPE_VALUES> valuesGm;
        GlobalTensor<DTYPE_OUT> outGm;
        TBuf<QuePosition::VECCALC> inQueueInput;
        TBuf<QuePosition::VECCALC> outQueueY;
        TBuf<QuePosition::VECCALC> selMaskQueue0;
        TBuf<QuePosition::VECCALC> selMaskQueue1;
        uint32_t beginIndex = 0;
        uint32_t size;
        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            beginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            beginIndex = size * GetBlockIdx() + formerNum * incSize;
        }
        inputGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input + beginIndex, size);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out + beginIndex, size);
        valuesGm.SetGlobalBuffer((__gm__ DTYPE_VALUES *)values, 1);
        uint32_t spaceSize = size * sizeof(DTYPE_VALUES);
        uint32_t singleSize = size >> 1;
        pipe->InitBuffer(inQueueInput, spaceSize + 32);
        pipe->InitBuffer(outQueueY, spaceSize + 32);
        pipe->InitBuffer(selMaskQueue0, size + 32);
        pipe->InitBuffer(selMaskQueue1, size);
            // 初始化完成
        LocalTensor<DTYPE_INPUT> inputLocal = inQueueInput.Get<DTYPE_INPUT>();
        DataCopy(inputLocal, inputGm, singleSize);
        DTYPE_VALUES value = valuesGm.GetValue(0);
        LocalTensor<DTYPE_INPUT> outLocal = outQueueY.Get<DTYPE_VALUES>();
            // copy 1
        Duplicate(outLocal, static_cast<DTYPE_VALUES>(1.0), size);
        LocalTensor<uint8_t> selMaskLocal0 = selMaskQueue0.Get<uint8_t>();
        LocalTensor<uint8_t> selMaskLocal1 = selMaskQueue1.Get<uint8_t>();
        TQueSync<PIPE_MTE2, PIPE_V> sync1;
        sync1.SetFlag(0);
        sync1.WaitFlag(0);
            // 立马开始下一次搬运
        DataCopy(inputLocal[singleSize], inputGm[singleSize], singleSize);
            // 不等于0，从value取,否则从1取
        CompareScalar(selMaskLocal0, inputLocal, static_cast<DTYPE_VALUES>(0.0), CMPMODE::NE, singleSize);
        PipeBarrier<PIPE_V>();
        Select(outLocal, selMaskLocal0, outLocal, value, SELMODE::VSEL_TENSOR_SCALAR_MODE, singleSize);
            // 大于等于0，从outLocal取，否则从0取
        CompareScalar(selMaskLocal1, inputLocal, static_cast<DTYPE_VALUES>(0.0), CMPMODE::GE, singleSize);
        PipeBarrier<PIPE_V>();
        Select(outLocal, selMaskLocal1, outLocal, static_cast<DTYPE_VALUES>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, singleSize);
        TQueSync<PIPE_V, PIPE_MTE3> sync2;
        sync2.SetFlag(0);
        sync2.WaitFlag(0);
            // 第一次搬出
        DataCopy(outGm, outLocal, singleSize);
            // 第二次搬运成功，开始计算
        sync1.SetFlag(1);
        sync1.WaitFlag(1);
            // 不等于0，从value取,否则从1取
        CompareScalar(selMaskLocal0[singleSize], inputLocal[singleSize], static_cast<DTYPE_VALUES>(0.0), CMPMODE::NE, singleSize);
        PipeBarrier<PIPE_V>();
        Select(outLocal[singleSize], selMaskLocal0[singleSize], outLocal[singleSize], value, SELMODE::VSEL_TENSOR_SCALAR_MODE, singleSize);
            // 大于等于0，从outLocal取，否则从0取
        CompareScalar(selMaskLocal1[singleSize], inputLocal[singleSize], static_cast<DTYPE_VALUES>(0.0), CMPMODE::GE, singleSize);
        PipeBarrier<PIPE_V>();
        Select(outLocal[singleSize], selMaskLocal1[singleSize], outLocal[singleSize], static_cast<DTYPE_VALUES>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               singleSize);
        sync2.SetFlag(1);
        sync2.WaitFlag(1);
            // 第2次搬出
        DataCopy(outGm[singleSize], outLocal[singleSize], singleSize);
    }
};
extern "C" __global__ __aicore__ void heaviside(GM_ADDR input, GM_ADDR values, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelHeaviside<true, false> op;
        op.Init(input, values, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelHeaviside<false, false> op;
        op.Init(input, values, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelHeaviside_smallData_and_scalar op;
        op.Init(input, values, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
    } else if (TILING_KEY_IS(4)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelHeaviside<false, true> op;
        op.Init(input, values, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        GET_TILING_DATA_WITH_STRUCT(HeavisideTilingData_BroadCast, tiling_data, tiling);
        KernelHeaviside_broadcast op;
        op.Init(input, values, out, tiling_data.size, tiling_data.mmInputDims, tiling_data.mmValuesDims, tiling_data.mmOutputDims, tiling_data.nOutputDims);
        op.Process();
    }
}

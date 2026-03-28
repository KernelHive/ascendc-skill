/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_quant_base.h
 * \brief
 */

#ifndef GELU_QUANT_BASE_H
#define GELU_QUANT_BASE_H

#include "kernel_operator.h"

namespace GeluQuantALL {
using namespace AscendC;

constexpr float SCALAR_ONE = 1.0;
constexpr float BETA = 0.044715;
constexpr float ALPHA = -1.5957691;

constexpr float ERF_PARAM1 = -0.3512339572e-8;
constexpr float ERF_PARAM2 = 0.2645266170e-6;
constexpr float ERF_PARAM3 = -0.7929488134e-5;
constexpr float ERF_PARAM4 = 0.1106123840e-3;
constexpr float ERF_PARAM5 = 0.6518995814e-4;
constexpr float ERF_PARAM6 = -0.7266616915e-1;
constexpr float ERF_PARAM7 = -0.1595769883e1;
constexpr float ERF_MIN = 5.75;
constexpr float ERF_MAX = -13.15;
constexpr float MAX_INT8 = 127.0;

constexpr uint32_t APPROXIMATE_NONE = 0;
constexpr uint32_t APPROXIMATE_TANH = 1;

constexpr uint32_t BUFFER_NUM = 1;

constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t SCALAR_TENSOR = 1;
constexpr uint32_t NORMAL_TENSOR = 2;

constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr uint32_t FP16_BLOCK_NUM = 16;
constexpr uint32_t INT8_BLOCK_NUM = 32;
constexpr uint32_t FP32_VECTOR_CAPACITY_ONE_CYCLE = 64;
constexpr uint32_t MAX_ROWS_NUM = 1024;
class GeluQuantBase {
public:
    __aicore__ inline GeluQuantBase(){};

    __aicore__ inline void ParseTilingData(const GeluQuantTilingData &tilingData);
    __aicore__ inline void ComputeGeluErf(const LocalTensor<float> &castFp32, const LocalTensor<float> &tempRes,
        LocalTensor<float> &xSquared, const int32_t &calCount);

    __aicore__ inline void ComputeGeluTanh(const LocalTensor<float> &castFp32, const LocalTensor<float> &tempRes,
        const int32_t &calCount);
    __aicore__ inline void ComputeReduceMax(const LocalTensor<float> &tempRes, int32_t calCount, float &maxValue);

    // 变量区
    uint32_t usedCoreNum_;
    int64_t curCoreProcessNum_;
    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;
    int64_t endAxisLen_;

    uint32_t coexistentNodeNum_;
    uint32_t coexistentNodeElementNum_;

    uint32_t approximate_;
    uint32_t inputScaleType_;
    uint32_t inputOffsetType_;

    // optional
    float inputScaleScalar_;
    float inputOffsetScalar_;

    uint32_t blockIdx_;
};

__aicore__ inline void GeluQuantBase::ParseTilingData(const GeluQuantTilingData &tilingData)
{
    endAxisLen_ = tilingData.endAxisLen;
    usedCoreNum_ = tilingData.usedCoreNum;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    coexistentNodeNum_ = tilingData.coexistentNodeNum;
    coexistentNodeElementNum_ = tilingData.coexistentNodeElementNum;
    approximate_ = tilingData.approximate;
    inputScaleType_ = tilingData.inputScaleType;
    inputOffsetType_ = tilingData.inputOffsetType;
}

__aicore__ inline void GeluQuantBase::ComputeGeluErf(const LocalTensor<float> &castFp32,
    const LocalTensor<float> &tempRes, LocalTensor<float> &xSquared, const int32_t &calCount)
{
    // res = x/(1+exp(((((((a1*x^2+a2)*x^2+a3)*x^2+a4)*x^2+a5)*x^2+a6)*x^2+a7)*x))
    Maxs(castFp32, castFp32, ERF_MAX, calCount);
    PipeBarrier<PIPE_V>();
    Mins(tempRes, castFp32, ERF_MIN, calCount);
    PipeBarrier<PIPE_V>();

    Mul(xSquared, tempRes, tempRes, calCount);
    PipeBarrier<PIPE_V>();

    Muls(tempRes, xSquared, ERF_PARAM1, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM2, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM3, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM4, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM5, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM6, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM7, calCount);
    PipeBarrier<PIPE_V>();

    Mins(xSquared, castFp32, ERF_MIN, calCount);
    PipeBarrier<PIPE_V>();
    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();

    Exp(tempRes, tempRes, calCount);
    PipeBarrier<PIPE_V>();

    Adds(tempRes, tempRes, 1.0f, calCount);
    PipeBarrier<PIPE_V>();
    
    Div(tempRes, castFp32, tempRes, calCount);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GeluQuantBase::ComputeGeluTanh(const LocalTensor<float> &castFp32,
    const LocalTensor<float> &tempRes, const int32_t &calCount)
{
    Mul(tempRes, castFp32, castFp32, calCount); // x^2
    PipeBarrier<PIPE_V>();
    Mul(tempRes, castFp32, tempRes, calCount); // x^3
    PipeBarrier<PIPE_V>();
    Muls(tempRes, tempRes, BETA, calCount); // 0.044715 * x^3
    PipeBarrier<PIPE_V>();
    Add(tempRes, castFp32, tempRes, calCount); // x + 0.044715 * x^3
    PipeBarrier<PIPE_V>();
    Muls(tempRes, tempRes, ALPHA, calCount); // -sqrt(8/pi)(x + 0.044715 * x^3)
    PipeBarrier<PIPE_V>();
    Exp(tempRes, tempRes, calCount); // exp(-sqrt(8/pi)(x + 0.044715 * x^3))
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, SCALAR_ONE, calCount); // 1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3))
    PipeBarrier<PIPE_V>();
    Div(tempRes, castFp32, tempRes, calCount); // x / (1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3)))
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GeluQuantBase::ComputeReduceMax(const LocalTensor<float> &tempRes, int32_t calCount,
    float &maxValue)
{
    uint32_t vectorCycles = calCount / FP32_VECTOR_CAPACITY_ONE_CYCLE;
    uint32_t remainElements = calCount % FP32_VECTOR_CAPACITY_ONE_CYCLE;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.src0RepStride = FP32_BLOCK_NUM;
    repeatParams.src1RepStride = 0;

    if (vectorCycles > 0 && remainElements > 0) {
        Max(tempRes, tempRes, tempRes[vectorCycles * FP32_VECTOR_CAPACITY_ONE_CYCLE], remainElements, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }

    if (vectorCycles > 1) {
        Max(tempRes, tempRes[FP32_VECTOR_CAPACITY_ONE_CYCLE], tempRes, FP32_VECTOR_CAPACITY_ONE_CYCLE, vectorCycles - 1,
            repeatParams);
        PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(tempRes, tempRes, (vectorCycles == 0) ? remainElements : FP32_VECTOR_CAPACITY_ONE_CYCLE, 1, 1, 1,
        FP32_BLOCK_NUM, ReduceOrder::ORDER_ONLY_VALUE);

    event_t curEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, curEventID);
    wait_flag(PIPE_V, PIPE_S, curEventID);

    maxValue = static_cast<float>(tempRes.GetValue(0));
}
} // namespace GeluQuantALL

#endif // GELU_QUANT_BASE_H

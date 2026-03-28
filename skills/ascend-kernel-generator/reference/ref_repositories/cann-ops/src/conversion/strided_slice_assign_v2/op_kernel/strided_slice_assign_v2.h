/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strided_slice_assign_v2.h
 * \brief
 */
#ifndef STRIDED_SLICE_ASSIGN_V2_H
#define STRIDED_SLICE_ASSIGN_V2_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t UB_SIZE = 195584;  // (192 -1) * 1024
constexpr uint32_t MAX_DIM_NUM = 8;
constexpr uint64_t DILATION = 2;

template <typename T>
class KernelStridedSliceAssignV2 {
public:
    __aicore__ inline KernelStridedSliceAssignV2(TPipe *pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                GM_ADDR axes, GM_ADDR var_out, const StridedSliceAssignV2TilingData *tilingData)
    {
        usedCoreNum = GetBlockNum();
        ASSERT(usedCoreNum != 0 && "block dim can not be zero!");

        dimNum = tilingData->dimNum;

        varShape = tilingData->varDim;
        inputShape = tilingData->inputValueDim;
        beginList = tilingData->begin;
        stridesList = tilingData->strides;
        varCumShape = tilingData->varCumShape;
        inputCumShape = tilingData->inputCumShape;

        if constexpr (sizeof(T) == sizeof(int64_t)) {
            inputValueGmInt32.SetGlobalBuffer((__gm__ int32_t *)input_value);
            varOutGmInt32.SetGlobalBuffer((__gm__ int32_t *)var_out);
        } else {
            inputValueGm.SetGlobalBuffer((__gm__ T *)input_value);
            varOutGm.SetGlobalBuffer((__gm__ T *)var_out);
        }

        Ppipe->InitBuffer(valueBuf, UB_SIZE);

        inputLastDim = inputShape[dimNum - 1];
        ubFactor = UB_SIZE / sizeof(T);
        innerRepTimes = DivCeil(inputLastDim, ubFactor);
        tailNum = inputLastDim - (innerRepTimes - 1) * ubFactor;

        copyParams.blockCount = 1;
    }

    __aicore__ inline void Process()
    {
        uint64_t totalLoopNum = 1;
        for (size_t n = 0; n < dimNum - 1; n++) {
            totalLoopNum *= inputShape[n];
        }

        for (uint64_t totalIdx = GetBlockIdx(); totalIdx < totalLoopNum; totalIdx += usedCoreNum) {
            if (dimNum > 2) {
                CalcIdx(totalIdx);
            } else {
                inputIdx[0] = totalIdx;
            }
            int64_t inputOffset = CalcInputOffset();
            int64_t varOffset = CalcVarOffset();

            for (int64_t j = 0; j < innerRepTimes; j++) {
                uint64_t copyNum = (j == innerRepTimes - 1) ? tailNum : ubFactor;
                copyParams.blockLen = copyNum * sizeof(T);
                if constexpr (sizeof(T) == sizeof(int64_t)) {
                    LocalTensor<int32_t> valueLocal = valueBuf.Get<int32_t>();
                    DataCopyPad(valueLocal, inputValueGmInt32[(inputOffset + j * ubFactor) * DILATION], copyParams,
                                padParamsInt32);
                    event_t eventCopyOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                    set_flag(PIPE_MTE2, PIPE_MTE3, eventCopyOut);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, eventCopyOut);

                    DataCopyPad(varOutGmInt32[(varOffset + j * ubFactor) * DILATION], valueLocal, copyParams);
                    event_t eventCopyIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    set_flag(PIPE_MTE3, PIPE_MTE2, eventCopyIn);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, eventCopyIn);
                } else {
                    LocalTensor<T> valueLocal = valueBuf.Get<T>();
                    DataCopyPad(valueLocal, inputValueGm[inputOffset + j * ubFactor], copyParams, padParams);
                    event_t eventCopyOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                    set_flag(PIPE_MTE2, PIPE_MTE3, eventCopyOut);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, eventCopyOut);

                    DataCopyPad(varOutGm[varOffset + j * ubFactor], valueLocal, copyParams);
                    event_t eventCopyIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    set_flag(PIPE_MTE3, PIPE_MTE2, eventCopyIn);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, eventCopyIn);
                }
            }
        }
    }

protected:
    TPipe *Ppipe = nullptr;

    GlobalTensor<T> inputValueGm;
    GlobalTensor<T> varOutGm;
    GlobalTensor<int32_t> inputValueGmInt32;
    GlobalTensor<int32_t> varOutGmInt32;

    TBuf<TPosition::VECCALC> valueBuf;

    int64_t usedCoreNum;
    int64_t dimNum;
    const int64_t *varShape;
    const int64_t *inputShape;
    const int64_t *beginList;
    const int64_t *stridesList;
    const int64_t *varCumShape;
    const int64_t *inputCumShape;

    int64_t inputIdx[MAX_DIM_NUM];

    DataCopyPadExtParams<T> padParams;
    DataCopyPadExtParams<int32_t> padParamsInt32;
    DataCopyExtParams copyParams;

    int64_t inputLastDim;
    int64_t ubFactor;
    int64_t innerRepTimes;
    int64_t tailNum;
private:
    __aicore__ inline void CalcIdx(int64_t totalIdx)
    {
        inputIdx[dimNum - 3] = totalIdx / inputShape[dimNum - 2];
        inputIdx[dimNum - 2] = totalIdx % inputShape[dimNum - 2];
        for (uint64_t i = dimNum - 3; i > 0; i--) {
            inputIdx[i - 1] = inputIdx[i] / inputShape[i];
            inputIdx[i] = inputIdx[i] % inputShape[i];
        }
    }

    __aicore__ inline int64_t CalcInputOffset()
    {
        int64_t inputOffset = 0;
        for (size_t n = 0; n < dimNum - 1; n++) {
            inputOffset += inputIdx[n] * inputCumShape[n + 1];
        }
        return inputOffset;
    }

    __aicore__ inline int64_t CalcVarOffset()
    {
        int64_t varOffset = beginList[dimNum - 1];
        for (size_t n = 0; n < dimNum - 1; n++) {
            varOffset += (beginList[n] + inputIdx[n] * stridesList[n]) * varCumShape[n + 1];
        }
        return varOffset;
    }
};
#endif  // STRIDED_SLICE_ASSIGN_V2_H

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
 * \file grid_sampler2_d_grad_cast.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_GRAD_CAST_H_
#define GRID_SAMPLER_2D_GRAD_CAST_H_

#include "kernel_operator.h"

using namespace AscendC;

template <typename T, typename GridSamplerGradTilingData>
class GridSampler2DGradCast {
public:
    __aicore__ inline GridSampler2DGradCast(){};
    __aicore__ inline void Init(const GridSamplerGradTilingData &__restrict tilingData,
        GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1], TPipe *inputPipe);
    __aicore__ inline void CopyOut(const int32_t offset, const int32_t calCount);
    __aicore__ inline void CopyIn(const int64_t offset, const int32_t calCount);
    __aicore__ inline void Compute(const int32_t computeCount);
    __aicore__ inline void Process();

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return b == 0 ? a : (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b * b;
    };

private:
    TPipe *pipe;
    GlobalTensor<T> outputGm;
    GlobalTensor<float> inputGmT;

    TQue<QuePosition::VECIN, 1> dataInQueue;
    TQue<QuePosition::VECOUT, 1> dataOutQueue;

    uint32_t usedCoreNumCast = 0;
    uint32_t pNumPerCoreCast = 0;
    uint32_t tailPNumCast = 0;
    uint32_t castElement = 0;
    uint32_t blockIdx = 0;
};

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradCast<T, GridSamplerGradTilingData>::Init(
    const GridSamplerGradTilingData &__restrict tilingData, GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1],
    TPipe *inputPipe)
{
    usedCoreNumCast = tilingData.usedCoreNumCast;
    pNumPerCoreCast = tilingData.pNumPerCoreCast;
    tailPNumCast = tilingData.tailPNumCast;
    castElement = tilingData.castElement;
    blockIdx = GetBlockIdx();

    pipe = inputPipe;
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputTensors[DX_INPUT_INDEX]));
    inputGmT.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputTensors[WORKSPACE_INPUT_INDEX]));

    pipe->InitBuffer(dataInQueue, 1, castElement * sizeof(float));
    pipe->InitBuffer(dataOutQueue, 1, castElement * sizeof(T));
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradCast<T, GridSamplerGradTilingData>::CopyIn(
    const int64_t offset, const int32_t calCount)
{
    LocalTensor<float> dataLocal = dataInQueue.AllocTensor<float>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(calCount * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams = {true, 0, 0, 0};
    DataCopyPad(dataLocal, inputGmT[offset], copyParams, padParams);
    dataInQueue.EnQue(dataLocal);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradCast<T, GridSamplerGradTilingData>::CopyOut(
    const int32_t offset, const int32_t calCount)
{
    LocalTensor<T> dstLocal = dataOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[offset], dstLocal, copyParams);
    dataOutQueue.FreeTensor(dstLocal);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradCast<T, GridSamplerGradTilingData>::Process()
{
    uint32_t computePNum = 0;
    int64_t castGmOffset = 0;
    int32_t castOffset = 0;
    int32_t cycleOffset = 0;
    int64_t curGridPointIndex = 0;
    if (blockIdx < tailPNumCast) {
        computePNum = pNumPerCoreCast + 1;
        castOffset = blockIdx * computePNum;
    } else {
        computePNum = pNumPerCoreCast;
        castOffset = blockIdx * computePNum + tailPNumCast;
    }

    int32_t actualComputNum = castElement;
    int32_t copyTimes = CeilDiv(computePNum, castElement);
    for (int j = 0; j < copyTimes; j++) {
        if (j == copyTimes - 1) {
            actualComputNum = computePNum - (copyTimes - 1) * castElement;
        }
        cycleOffset = j * castElement;
        castGmOffset = cycleOffset + castOffset;
        CopyIn(castGmOffset, actualComputNum);
        Compute(actualComputNum);
        CopyOut(castGmOffset, actualComputNum);
    }
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradCast<T, GridSamplerGradTilingData>::Compute(const int32_t computeCount)
{
    LocalTensor<float> inputCoordinate = dataInQueue.DeQue<float>();
    LocalTensor<T> outputData = dataOutQueue.AllocTensor<T>();
    Cast(outputData, inputCoordinate, RoundMode::CAST_ROUND, computeCount);
    dataInQueue.FreeTensor(inputCoordinate);
    dataOutQueue.EnQue(outputData);
}

#endif  // GRID_SAMPLER_2D_GRAD_CAST_H_
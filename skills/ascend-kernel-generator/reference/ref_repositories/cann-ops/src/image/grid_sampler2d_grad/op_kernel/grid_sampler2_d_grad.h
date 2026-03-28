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
 * \file grid_sampler2_d_grad.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_GRAD_H_
#define GRID_SAMPLER_2D_GRAD_H_

#include "kernel_operator.h"

using namespace AscendC;

constexpr static int32_t INT_MAX = 2147483647;
constexpr static int32_t INT_MIN = -2147483648;
constexpr static int32_t INPUT_NUM = 3;
constexpr static int32_t OUTPUT_NUM = 2;
constexpr static int32_t BUFFER_NUM = 2;
constexpr static int32_t GRAD_INPUT_INDEX = 0;
constexpr static int32_t X_INPUT_INDEX = 1;
constexpr static int32_t GRID_INPUT_INDEX = 2;
constexpr static int32_t DX_INPUT_INDEX = 3;
constexpr static int32_t DGRID_INPUT_INDEX = 4;
constexpr static int32_t WORKSPACE_INPUT_INDEX = 5;
constexpr static int32_t X_GRAD_OUTPUT_INDEX = 0;
constexpr static int32_t GRID_GRAD_OUTPUT_INDEX = 1;
constexpr static int32_t GRID_GRAD_GM_INPUT_INDEX = 4;
constexpr static int32_t BUFFER_APPLY_NUM = 2;
constexpr static uint32_t BLOCK_BYTES = 32;
constexpr static uint32_t UINT8_BITS = 8;
constexpr static int32_t INPUT_GRAD_INDEX = 0;
constexpr static int32_t INPUT_X_INDEX = 1;
constexpr static int32_t INPUT_GRID_INDEX = 2;
constexpr static uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static uint32_t ALGIN_256_BYTES = 256;
constexpr static uint32_t CHANNEL_1024 = 1024;
constexpr static uint8_t REPEAT_STRIDE = 8;

template <typename T, typename GridSamplerGradTilingData>
class GridSampler2DGrad {
public:
    __aicore__ inline GridSampler2DGrad(){};
    __aicore__ inline void Init(
        const GridSamplerGradTilingData &__restrict tilingData, GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1]);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void InitBilinearLocalTensor();
    __aicore__ inline void InitNearestLocalTensor();
    __aicore__ inline void CopyOut(const int32_t offset, const int32_t calCount);
    __aicore__ inline void CopyIn(const int64_t offset, const int32_t calCount, const int32_t inputIndex);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(const int32_t computeCount, const int64_t curGridPointIndex);
    __aicore__ inline void ComputeWeight(LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2,
        LocalTensor<T> yCoorTensor1, LocalTensor<T> yCoorTensor2, const int32_t calCount);
    __aicore__ inline void ComputeIndex(LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> dstIndex2,
        LocalTensor<int32_t> yCoor, LocalTensor<int32_t> xCoor, const int32_t calCount);
    __aicore__ inline void ComputeSourceIndexSetGrad(
        LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, const int32_t calCount);
    __aicore__ inline void ComputeAfterTransposeGridGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> yCoor1,
        LocalTensor<T> yCoor2, LocalTensor<T> xCoor1, LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor,
        LocalTensor<T> selTensor, const int32_t coorIndex, const int32_t batchIdx);
    __aicore__ inline void ComputeAfterTransposeXGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> weight,
        const int32_t coorIndex, const int64_t ncOffset, LocalTensor<T> gOutLocalTensor);
    __aicore__ inline void ComputeNearestXGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> weight,
        const int32_t coorIndex, const int32_t cycle, const int64_t ncOffset, LocalTensor<T> gOutLocalTensor);
    __aicore__ inline void WithinBounds2d(
        LocalTensor<T> dst, LocalTensor<T> iyT, LocalTensor<T> ixT, LocalTensor<T> weight, const int32_t calCount);
    __aicore__ inline void DupValue();
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
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> dataInQueue[INPUT_NUM];
    TQue<QuePosition::VECOUT, BUFFER_NUM> dataOutQueue[OUTPUT_NUM];
    TBuf<TPosition::VECCALC> xCoordinateBuf;
    TBuf<TPosition::VECCALC> yCoordinateBuf;
    TBuf<TPosition::VECCALC> xGradInBuf;
    TBuf<TPosition::VECCALC> yGradInBuf;

    TBuf<TPosition::VECCALC> ixNwBuf;
    TBuf<TPosition::VECCALC> iyNwBuf;
    TBuf<TPosition::VECCALC> ixNeBuf;
    TBuf<TPosition::VECCALC> iyNeBuf;
    TBuf<TPosition::VECCALC> ixSwBuf;
    TBuf<TPosition::VECCALC> iySwBuf;
    TBuf<TPosition::VECCALC> ixSeBuf;
    TBuf<TPosition::VECCALC> iySeBuf;

    TBuf<TPosition::VECCALC> nwBuf;
    TBuf<TPosition::VECCALC> neBuf;
    TBuf<TPosition::VECCALC> swBuf;
    TBuf<TPosition::VECCALC> seBuf;

    TBuf<TPosition::VECCALC> ixNwIntBuf;
    TBuf<TPosition::VECCALC> iyNwIntBuf;

    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;

    TBuf<TPosition::VECCALC> ixNeIntBuf;
    TBuf<TPosition::VECCALC> iyNeIntBuf;
    TBuf<TPosition::VECCALC> ixSwIntBuf;
    TBuf<TPosition::VECCALC> iySwIntBuf;
    TBuf<TPosition::VECCALC> ixSeIntBuf;
    TBuf<TPosition::VECCALC> iySeIntBuf;

    TBuf<TPosition::VECCALC> mask1Buf;
    TBuf<TPosition::VECCALC> mask2Buf;

    TBuf<TPosition::VECCALC> dupOneBuf;
    TBuf<TPosition::VECCALC> selBuf1;
    TBuf<TPosition::VECCALC> selBuf2;
    TBuf<TPosition::VECCALC> selBuf3;
    TBuf<TPosition::VECCALC> selBuf4;

    TBuf<TPosition::VECCALC> computeIndexBuf;
    TBuf<TPosition::VECCALC> computeIndexBuf1;
    TBuf<TPosition::VECCALC> computeIndexBuf2;
    TBuf<TPosition::VECCALC> computeIndexBuf3;
    TBuf<TPosition::VECCALC> computeIndexBuf4;
    TBuf<TPosition::VECCALC> computeIndexBuf5;

    TBuf<TPosition::VECCALC> computeIndexBuf6;
    TBuf<TPosition::VECCALC> computeIndexBuf7;
    TBuf<TPosition::VECCALC> computeIndexBuf8;
    TBuf<TPosition::VECCALC> computeIndexBuf9;

    TBuf<TPosition::VECCALC> gixBuf;
    TBuf<TPosition::VECCALC> giyBuf;
    TBuf<TPosition::VECCALC> sumXBuf;
    TBuf<TPosition::VECCALC> sumYBuf;
    TBuf<TPosition::VECCALC> ixNearIntBuf;
    TBuf<TPosition::VECCALC> iyNearIntBuf;
    TBuf<TPosition::VECCALC> ixFloatBuf;
    TBuf<TPosition::VECCALC> iyFloatBuf;
    TBuf<TPosition::VECCALC> clipLimitBuf;
    GlobalTensor<T> inputGm[INPUT_NUM + OUTPUT_NUM];

    uint32_t batch = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    int32_t channel = 0;
    int32_t alignChannel = 0;
    int32_t height = 0;
    int32_t width = 0;
    T fheight = 0;
    T fwidth = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t interpolation = 0;  // 0:Bilinear, 1:Nearest
    uint32_t padding = 0;        // 0:Zeros, 1:Border
    uint32_t alignCorners = 0;   // 0:False, 1:True
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t outH = 0;
    uint32_t outW = 0;
    uint32_t perBlockCount = 0;
    uint32_t blockIdx = 0;
    uint32_t dataCount = 0;
    uint32_t batchOffset = 0;
    uint32_t baseOffset = 0;
    uint32_t alignBufferNum = 0;
    uint32_t xStrideC = 0;
    uint32_t dxStrideN = 0;
    uint32_t dxStrideC = 0;
    int32_t dxStrideH = 0;
    uint32_t dxStrideW = 0;
    uint32_t gradStrideC = 0;
    uint32_t gradStrideH = 0;
    uint32_t gradStrideW = 0;
    uint32_t maskSize = 0;
    uint32_t maskNum = 0;
    int32_t inputStrideH = 0;
    uint32_t inputStrideW = 0;
    uint32_t inputStrideN = 0;
    int64_t pointIndex = 0;
    int64_t baseGradGmOffset = 0;
    int64_t gradGmOffset = 0;
    int64_t baseGmOffset = 0;
    int32_t pointOffset = 0;
    int64_t xGmOffset = 0;
    int32_t ncOffset = 0;
    int32_t group = 0;
    T gix = static_cast<T>(0);
    T giy = static_cast<T>(0);

    LocalTensor<uint8_t> mask1Tensor;
    LocalTensor<uint8_t> mask2Tensor;
    LocalTensor<uint16_t> int8ToInt16Mask1;
    LocalTensor<uint16_t> int8ToInt16Mask2;
    LocalTensor<T> dupOneTensor;
    LocalTensor<T> selTensor1;
    LocalTensor<T> selTensor2;
    LocalTensor<T> selTensor3;
    LocalTensor<T> selTensor4;
    LocalTensor<T> tmp1Tensor;
    LocalTensor<T> tmp2Tensor;
    LocalTensor<int32_t> tmpIndex;

    LocalTensor<T> gixLocalTensor;
    LocalTensor<T> giyLocalTensor;
    LocalTensor<T> sumX;
    LocalTensor<T> sumY;
    LocalTensor<T> clipLimit;
};

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::Init(
    const GridSamplerGradTilingData &__restrict tilingData, GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1])
{
    batch = tilingData.batch;
    pNumPerCore = tilingData.pNumPerCore;
    tailPNum = tilingData.tailPNum;
    channel = tilingData.channel;
    height = tilingData.height;
    width = tilingData.width;
    fheight = static_cast<T>(height);
    fwidth = static_cast<T>(width);
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
    interpolation = tilingData.interpolation;
    padding = tilingData.padding;
    alignCorners = tilingData.alignCorners;
    group = tilingData.group;
    gridH = tilingData.gridH;
    gridW = tilingData.gridW;
    outW = gridW;
    outH = gridH;
    dataCount = gridH * gridW;
    maskSize = CeilAlign(CeilDiv(ubFactorElement, UINT8_BITS), BLOCK_BYTES);
    maskNum = maskSize / sizeof(uint8_t);
    xStrideC = width * height;
    dxStrideN = channel * width * height;
    dxStrideC = width * height;
    dxStrideH = width;
    dxStrideW = 1;

    gradStrideC = gridH * gridW;
    inputStrideH = width;
    inputStrideW = 1;
    inputStrideN = channel * width * height;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    alignChannel = CeilAlign(channel, perBlockCount);

    inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputTensors[GRAD_INPUT_INDEX]));
    inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputTensors[X_INPUT_INDEX]));
    inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputTensors[GRID_INPUT_INDEX]));
    inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputTensors[DX_INPUT_INDEX]));
    inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputTensors[DGRID_INPUT_INDEX]));
}

// init used buffer
template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::InitBuffer(TPipe *inputPipe)
{
    pipe = inputPipe;
    // Bilinear branch
    if (interpolation == 0) {
        pipe->InitBuffer(dataInQueue[0], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataInQueue[1], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataInQueue[GRID_INPUT_INDEX], BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));
        pipe->InitBuffer(dataOutQueue[0], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataOutQueue[1], BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));

        pipe->InitBuffer(xCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(yCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(xGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(yGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixNwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixNeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixNwIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNwIntBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(nwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(neBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(swBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(seBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(tmp1Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp2Buf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(ixNeIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNeIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSwIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySwIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSeIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySeIntBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(mask1Buf, maskSize);
        pipe->InitBuffer(mask2Buf, maskSize);

        pipe->InitBuffer(dupOneBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf1, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf2, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf3, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf4, ubFactorElement * sizeof(T));

        pipe->InitBuffer(computeIndexBuf1, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf2, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf3, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf4, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf5, ubFactorElement * sizeof(int32_t));

        pipe->InitBuffer(computeIndexBuf6, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf7, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf8, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf9, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(gixBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(giyBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumXBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumYBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(clipLimitBuf, ubFactorElement * sizeof(T));
    }
    // nearest branch
    if (interpolation == 1) {
        pipe->InitBuffer(dataInQueue[0], BUFFER_NUM, alignChannel * sizeof(T) * group);
        pipe->InitBuffer(dataInQueue[GRID_INPUT_INDEX], BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));
        pipe->InitBuffer(dataOutQueue[0], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataOutQueue[1], BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));
        pipe->InitBuffer(xCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(yCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(xGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(yGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixFloatBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyFloatBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(computeIndexBuf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf1, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(ixNearIntBuf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(iyNearIntBuf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(selBuf1, ubFactorElement * sizeof(T));
        pipe->InitBuffer(mask1Buf, maskSize);
        pipe->InitBuffer(mask2Buf, maskSize);

        pipe->InitBuffer(dupOneBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(clipLimitBuf, ubFactorElement * sizeof(T));
    }
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::InitBilinearLocalTensor()
{
    mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
    mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);
    dupOneTensor = dupOneBuf.Get<T>(ubFactorElement);
    tmpIndex = computeIndexBuf1.Get<int32_t>(ubFactorElement);
    selTensor1 = selBuf1.Get<T>(ubFactorElement);
    tmp1Tensor = tmp1Buf.Get<T>(ubFactorElement);
    tmp2Tensor = tmp2Buf.Get<T>(ubFactorElement);
    selTensor2 = selBuf2.Get<T>(ubFactorElement);
    selTensor3 = selBuf3.Get<T>(ubFactorElement);
    selTensor4 = selBuf4.Get<T>(ubFactorElement);
    sumX = sumXBuf.Get<T>(alignChannel);
    sumY = sumYBuf.Get<T>(alignChannel);
    clipLimit = clipLimitBuf.Get<T>(ubFactorElement);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::InitNearestLocalTensor()
{
    mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
    mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);
    dupOneTensor = dupOneBuf.Get<T>(ubFactorElement);
    tmpIndex = computeIndexBuf1.Get<int32_t>(ubFactorElement);
    selTensor1 = selBuf1.Get<T>(ubFactorElement);
    clipLimit = clipLimitBuf.Get<T>(ubFactorElement);
}

// grid_sampler_unnormalize_set_grad function
template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::ComputeSourceIndexSetGrad(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, const int32_t calCount)
{
    if (alignCorners == 1) {
        T val = static_cast<T>(size - 1) / 2;
        Duplicate<T>(dupTensor, val, calCount);
        Adds(dataTensor, dataTensor, static_cast<T>(1), calCount);
        Muls(dataTensor, dataTensor, static_cast<T>(0.5), calCount);
        Muls(dataTensor, dataTensor, static_cast<T>(size - 1), calCount);
    } else {
        T val = static_cast<T>(size) / 2;
        Duplicate<T>(dupTensor, val, calCount);
        Adds(dataTensor, dataTensor, static_cast<T>(1), calCount);
        Muls(dataTensor, dataTensor, static_cast<T>(size), calCount);
        Adds(dataTensor, dataTensor, static_cast<T>(-1), calCount);
        Muls(dataTensor, dataTensor, static_cast<T>(0.5), calCount);
    }
    int32_t newCalCount =
        ((calCount * FLOAT_BYTES - 1 + ALGIN_256_BYTES) / ALGIN_256_BYTES * ALGIN_256_BYTES) / FLOAT_BYTES;
    if (padding == 1) {
        CompareScalar(mask1Tensor, dataTensor, static_cast<T>(0), CMPMODE::GT, newCalCount);
        Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        CompareScalar(mask1Tensor, dataTensor, static_cast<T>(size - 1), CMPMODE::LT, newCalCount);
        Select(dataTensor,
            mask1Tensor,
            dataTensor,
            static_cast<T>(size - 1),
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            newCalCount);
        Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }
    // If the data is inf/-inf/nan, convert the data to -100.
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MAX - 1), CMPMODE::LE, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MIN), CMPMODE::GE, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    Compare(mask1Tensor, dataTensor, dataTensor, CMPMODE::EQ, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
}

// confirm coor is in range [0,h] or [0,w]
template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::WithinBounds2d(
    LocalTensor<T> dst, LocalTensor<T> iyT, LocalTensor<T> ixT, LocalTensor<T> weight, const int32_t calCount)
{
    int32_t newCalCount = ((calCount * FLOAT_BYTES - 1 + ALGIN_256_BYTES) / ALGIN_256_BYTES * ALGIN_256_BYTES) /
                          FLOAT_BYTES;  // align 256 bytes
    if (interpolation == 0) {
        CompareScalar(mask1Tensor, iyT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        CompareScalar(mask2Tensor, iyT, static_cast<T>(fheight), CMPMODE::LT, newCalCount);
        int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
        int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();

        // Read data according to int16
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(fwidth), CMPMODE::LT, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        Select(dst, int8ToInt16Mask2, dupOneTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        Select(weight, int8ToInt16Mask2, weight, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }
    if (interpolation == 1) {
        CompareScalar(mask1Tensor, iyT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        CompareScalar(mask2Tensor, iyT, static_cast<T>(fheight), CMPMODE::LT, newCalCount);
        int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
        int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();

        // Read data according to int16
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(fwidth), CMPMODE::LT, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        Select(weight, int8ToInt16Mask2, weight, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::ComputeIndex(LocalTensor<int32_t> dstIndex,
    LocalTensor<int32_t> dstIndex2, LocalTensor<int32_t> yCoor, LocalTensor<int32_t> xCoor, const int32_t calCount)
{
    if (interpolation == 0) {
        Mins(yCoor, yCoor, height - 1, calCount);
        Maxs(yCoor, yCoor, 0, calCount);
        Mins(xCoor, xCoor, width - 1, calCount);
        Maxs(xCoor, xCoor, 0, calCount);
        Muls(tmpIndex, yCoor, inputStrideH, calCount);
        Add(dstIndex, tmpIndex, xCoor, calCount);
        Muls(dstIndex, dstIndex, channel, calCount);

        Muls(tmpIndex, yCoor, dxStrideH, calCount);
        Add(dstIndex2, tmpIndex, xCoor, calCount);
        Muls(dstIndex2, dstIndex2, channel, calCount);
    }
    if (interpolation == 1) {
        Mins(yCoor, yCoor, height - 1, calCount);
        Maxs(yCoor, yCoor, 0, calCount);
        Mins(xCoor, xCoor, width - 1, calCount);
        Maxs(xCoor, xCoor, 0, calCount);
        Muls(tmpIndex, yCoor, inputStrideH, calCount);
        Add(dstIndex, tmpIndex, xCoor, calCount);
        Muls(dstIndex, dstIndex, channel, calCount);
    }
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::ComputeAfterTransposeGridGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> yCoor1, LocalTensor<T> yCoor2, LocalTensor<T> xCoor1,
    LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor, LocalTensor<T> selTensor, const int32_t coorIndex,
    const int32_t batchIdx)
{
    // Get the coordinates of the i-th point
    pointIndex = srcIndex.GetValue(coorIndex);
    xGmOffset = batchIdx * inputStrideN + pointIndex;
    T xVal = yCoor1.GetValue(coorIndex) - yCoor2.GetValue(coorIndex);
    T yVal = xCoor1.GetValue(coorIndex) - xCoor2.GetValue(coorIndex);
    T flag = selTensor.GetValue(coorIndex);

    LocalTensor<T> inputXLocalTensor = dataInQueue[1].AllocTensor<T>();
    DataCopyParams copyParams = {1, 0, 0, 0};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);
    padParams.rightPadding = alignChannel - channel;
    padParams.paddingValue = GetScalarBitcodeValue((T)0);
    DataCopyPad(inputXLocalTensor, inputGm[1][xGmOffset], copyParams, padParams);

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID1);
    WaitFlag<HardEvent::MTE2_V>(eventID1);
    Muls(giyLocalTensor, inputXLocalTensor, yVal, channel);
    Mul(giyLocalTensor, gOutLocalTensor, giyLocalTensor, channel);
    Muls(giyLocalTensor, giyLocalTensor, flag, channel);
    Muls(gixLocalTensor, inputXLocalTensor, xVal, channel);
    Mul(gixLocalTensor, gOutLocalTensor, gixLocalTensor, channel);
    Muls(gixLocalTensor, gixLocalTensor, flag, channel);
    Add(sumY, giyLocalTensor, sumY, channel);
    Add(sumX, gixLocalTensor, sumX, channel);
    dataInQueue[1].FreeTensor(inputXLocalTensor);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::ComputeAfterTransposeXGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coorIndex, const int64_t ncOffset,
    LocalTensor<T> gOutLocalTensor)
{
    T weigthVal = weight.GetValue(coorIndex);
    int64_t offset = ncOffset + srcIndex.GetValue(coorIndex);
    LocalTensor<T> localTensor = dataOutQueue[0].AllocTensor<T>();
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID1);
    WaitFlag<HardEvent::S_V>(eventID1);
    Muls(localTensor, gOutLocalTensor, weigthVal, channel);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    

    DataCopyParams copyParams{1, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);
    SetAtomicAdd<T>();
    DataCopyPad(inputGm[DX_INPUT_INDEX][offset], localTensor, copyParams);
    SetAtomicNone();
    dataOutQueue[0].FreeTensor(localTensor);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::ComputeNearestXGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coorIndex, const int32_t cycle,
    const int64_t ncOffset, LocalTensor<T> gOutLocalTensor)
{
    T weigthVal = weight.GetValue(coorIndex);
    int64_t offset = ncOffset + srcIndex.GetValue(coorIndex);
    LocalTensor<T> localTensor = dataOutQueue[0].AllocTensor<T>();

    Muls(localTensor, gOutLocalTensor[cycle * alignChannel], weigthVal, channel);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);

    DataCopyParams copyParams{1, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);
    SetAtomicAdd<T>();
    DataCopyPad(inputGm[DX_INPUT_INDEX][offset], localTensor, copyParams);
    SetAtomicNone();
    dataOutQueue[0].FreeTensor(localTensor);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::ComputeWeight(LocalTensor<T> dst,
    LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2, LocalTensor<T> yCoorTensor1, LocalTensor<T> yCoorTensor2,
    const int32_t calCount)
{
    Muls(tmp1Tensor, xCoorTensor1, static_cast<T>(-1), calCount);
    Add(tmp1Tensor, xCoorTensor2, tmp1Tensor, calCount);
    Muls(tmp2Tensor, yCoorTensor1, static_cast<T>(-1), calCount);
    Add(tmp2Tensor, yCoorTensor2, tmp2Tensor, calCount);
    Mul(dst, tmp1Tensor, tmp2Tensor, calCount);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::DupValue()
{
    Duplicate<T>(dupOneTensor, 1, ubFactorElement);
    if (interpolation == 0) {
        Duplicate<T>(sumX, 0, alignChannel);
        Duplicate<T>(sumY, 0, alignChannel);
    }
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::CopyIn(
    const int64_t offset, const int32_t calCount, const int32_t inputIndex)
{
    LocalTensor<T> dataLocal = dataInQueue[inputIndex].AllocTensor<T>();
    DataCopyParams copyParams = {1, 0, 0, 0};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    int32_t alignCalCount = CeilAlign(calCount, perBlockCount);
    copyParams.blockLen = calCount * sizeof(T);
    padParams.rightPadding = alignCalCount - calCount;
    padParams.paddingValue = GetScalarBitcodeValue((T)0);

    DataCopyPad(dataLocal, inputGm[inputIndex][offset], copyParams, padParams);
    dataInQueue[inputIndex].EnQue(dataLocal);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::CopyOut(
    const int32_t offset, const int32_t calCount)
{
    LocalTensor<T> dstLocal = dataOutQueue[1].DeQue<T>();
    DataCopyParams copyParams{1, 0, 0, 0};
    copyParams.blockLen = calCount * sizeof(T);
    DataCopyPad(inputGm[DGRID_INPUT_INDEX][offset], dstLocal, copyParams);
    dataOutQueue[1].FreeTensor(dstLocal);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::Compute(
    const int32_t computeCount, const int64_t curGridPointIndex)
{
    int64_t gridPointIndex = 0;
    int32_t gradStrideN = channel * outH * outW;
    int32_t gradStrideC = outH * outW;
    int32_t gradStrideH = outW;
    int32_t gradStrideW = 1;
    int32_t xStrideC = width * height;
    int64_t w = 0;
    int64_t h = 0;
    int64_t n = 0;
    int64_t ncBaseOffset = 0;
    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;
    bool reduceMode = false;
    uint8_t src0BlockStride = 1;
    uint16_t repeatTimes = CeilDiv(computeCount, ELE_NUM_PER_REPEAT);
    uint8_t src0RepeatStride = REPEAT_STRIDE;
    uint8_t src1RepeatStride = REPEAT_STRIDE;

    LocalTensor<T> xTensor = xCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);
    LocalTensor<T> yTensor = yCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);
    LocalTensor<T> xGradIn = xGradInBuf.Get<T>(ubFactorElement);
    LocalTensor<T> yGradIn = yGradInBuf.Get<T>(ubFactorElement);
    LocalTensor<T> inputCoordinate = dataInQueue[GRID_INPUT_INDEX].DeQue<T>();
    LocalTensor<T> dstLocal = dataOutQueue[1].AllocTensor<T>();

    DupValue();
    GatherMask(xTensor,
        inputCoordinate,
        xPattern,
        reduceMode,
        mask,
        {1, repeatTimes, src0RepeatStride, src1RepeatStride},
        rsvdCnt);
    GatherMask(yTensor,
        inputCoordinate,
        yPattern,
        reduceMode,
        mask,
        {1, repeatTimes, src0RepeatStride, src1RepeatStride},
        rsvdCnt);
    // gather ix
    ComputeSourceIndexSetGrad(xTensor, xGradIn, fwidth, computeCount / 2);

    // gather iy
    ComputeSourceIndexSetGrad(yTensor, yGradIn, fheight, computeCount / 2);
    if (interpolation == 0) {
        LocalTensor<T> ixNw = ixNwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iyNw = iyNwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ixNe = ixNeBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iyNe = iyNeBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ixSw = ixSwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iySw = iySwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ixSe = ixSeBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iySe = iySeBuf.Get<T>(ubFactorElement);

        LocalTensor<int32_t> ixNwInt = ixNwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iyNwInt = iyNwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> ixNeInt = ixNeIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iyNeInt = iyNeIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> ixSwInt = ixSwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iySwInt = iySwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> ixSeInt = ixSeIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iySeInt = iySeIntBuf.Get<int32_t>(ubFactorElement);

        LocalTensor<T> nw = nwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ne = neBuf.Get<T>(ubFactorElement);
        LocalTensor<T> sw = swBuf.Get<T>(ubFactorElement);
        LocalTensor<T> se = seBuf.Get<T>(ubFactorElement);

        LocalTensor<int32_t> nwIndex = computeIndexBuf2.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> neIndex = computeIndexBuf3.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> swIndex = computeIndexBuf4.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> seIndex = computeIndexBuf5.Get<int32_t>(ubFactorElement);

        LocalTensor<int32_t> nwIndex2 = computeIndexBuf6.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> neIndex2 = computeIndexBuf7.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> swIndex2 = computeIndexBuf8.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> seIndex2 = computeIndexBuf9.Get<int32_t>(ubFactorElement);

        gixLocalTensor = gixBuf.Get<T>(alignChannel);
        giyLocalTensor = giyBuf.Get<T>(alignChannel);

        Cast(ixNwInt, xTensor, RoundMode::CAST_FLOOR, computeCount / 2);
        Cast(iyNwInt, yTensor, RoundMode::CAST_FLOOR, computeCount / 2);
        Cast(iyNeInt, yTensor, RoundMode::CAST_FLOOR, computeCount / 2);
        Cast(ixSwInt, xTensor, RoundMode::CAST_FLOOR, computeCount / 2);
        Adds(ixNeInt, ixNwInt, static_cast<int32_t>(1), computeCount / 2);
        Adds(iySwInt, iyNwInt, static_cast<int32_t>(1), computeCount / 2);
        Adds(ixSeInt, ixNwInt, static_cast<int32_t>(1), computeCount / 2);
        Adds(iySeInt, iyNwInt, static_cast<int32_t>(1), computeCount / 2);
        // convert to float32
        Cast(ixNw, ixNwInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(iyNw, iyNwInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(ixNe, ixNeInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(iyNe, iyNeInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(ixSw, ixSwInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(iySw, iySwInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(ixSe, ixSeInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(iySe, iySeInt, RoundMode::CAST_NONE, computeCount / 2);
        // compute nw
        ComputeWeight(nw, xTensor, ixSe, yTensor, iySe, computeCount / 2);
        // compute ne
        ComputeWeight(ne, ixSw, xTensor, yTensor, iySw, computeCount / 2);
        // compute sw
        ComputeWeight(sw, xTensor, ixNe, iyNe, yTensor, computeCount / 2);
        // compute se
        ComputeWeight(se, ixNw, xTensor, iyNw, yTensor, computeCount / 2);
        WithinBounds2d(selTensor1, iyNw, ixNw, nw, computeCount / 2);
        WithinBounds2d(selTensor2, iyNe, ixNe, ne, computeCount / 2);
        WithinBounds2d(selTensor3, iySw, ixSw, sw, computeCount / 2);
        WithinBounds2d(selTensor4, iySe, ixSe, se, computeCount / 2);
        ComputeIndex(nwIndex, nwIndex2, iyNwInt, ixNwInt, computeCount / 2);
        ComputeIndex(neIndex, neIndex2, iyNeInt, ixNeInt, computeCount / 2);
        ComputeIndex(swIndex, swIndex2, iySwInt, ixSwInt, computeCount / 2);
        ComputeIndex(seIndex, seIndex2, iySeInt, ixSeInt, computeCount / 2);

        for (int32_t i = 0; i < computeCount / 2; i++) {
            gridPointIndex = curGridPointIndex + i;
            w = gridPointIndex % outW;
            h = (gridPointIndex / outW) % outH;
            n = gridPointIndex / (outH * outW);

            ncBaseOffset = n * dxStrideN;
            gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
            LocalTensor<T> gOutLocalTensor = dataInQueue[0].AllocTensor<T>();
            DataCopyParams copyParams = {1, 0, 0, 0};
            DataCopyPadParams padParams = {true, 0, 0, 0};
            int32_t alignCalCount = CeilAlign(channel, perBlockCount);
            copyParams.blockLen = channel * sizeof(T);
            padParams.rightPadding = alignCalCount - channel;
            padParams.paddingValue = GetScalarBitcodeValue((T)0);
            DataCopyPad(gOutLocalTensor, inputGm[0][gradGmOffset], copyParams, padParams);
            ComputeAfterTransposeGridGrad(nwIndex, iySe, yTensor, ixSe, xTensor, gOutLocalTensor, selTensor1, i, n);
            ComputeAfterTransposeXGrad(nwIndex2, nw, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(neIndex, yTensor, iySw, xTensor, ixSw, gOutLocalTensor, selTensor2, i, n);
            ComputeAfterTransposeXGrad(neIndex2, ne, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(swIndex, yTensor, iyNe, xTensor, ixNe, gOutLocalTensor, selTensor3, i, n);
            ComputeAfterTransposeXGrad(swIndex2, sw, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(seIndex, iyNw, yTensor, ixNw, xTensor, gOutLocalTensor, selTensor4, i, n);
            ComputeAfterTransposeXGrad(seIndex2, se, i, ncBaseOffset, gOutLocalTensor);
            ReduceSum<T>(sumY, sumY, sumY, channel);  // resue sumY as worklocal
            ReduceSum<T>(sumX, sumX, sumX, channel);
            gix -= sumX.GetValue(0);
            giy -= sumY.GetValue(0);
            dstLocal.SetValue(2 * i, gix * xGradIn.GetValue(i));
            dstLocal.SetValue(2 * i + 1, giy * yGradIn.GetValue(i));
            Duplicate<T>(sumX, 0, alignChannel);
            Duplicate<T>(sumY, 0, alignChannel);
            gix = static_cast<T>(0);
            giy = static_cast<T>(0);
            dataInQueue[0].FreeTensor(gOutLocalTensor);
        }
    }
    if (interpolation == 1) {
        LocalTensor<int32_t> ixNearInt = ixNearIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iyNearInt = iyNearIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<T> ixFloat = ixFloatBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iyFloat = iyFloatBuf.Get<T>(ubFactorElement);
        LocalTensor<int32_t> xIndex = computeIndexBuf.Get<int32_t>(ubFactorElement);
        Cast(ixNearInt, xTensor, RoundMode::CAST_RINT, computeCount / 2);
        Cast(iyNearInt, yTensor, RoundMode::CAST_RINT, computeCount / 2);
        Cast(ixFloat, ixNearInt, RoundMode::CAST_NONE, computeCount / 2);
        Cast(iyFloat, iyNearInt, RoundMode::CAST_NONE, computeCount / 2);
        WithinBounds2d(selTensor1, iyFloat, ixFloat, dupOneTensor, computeCount / 2);
        ComputeIndex(xIndex, xIndex, iyNearInt, ixNearInt, computeCount / 2);
        DataCopyParams copyParams = {1, 0, 0, 0};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        copyParams.blockLen = channel * sizeof(T);
        padParams.rightPadding = alignChannel - channel;
        padParams.paddingValue = GetScalarBitcodeValue((T)0);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        if (channel <= CHANNEL_1024) {
            int32_t times = (computeCount / 2) / group;
            int32_t tailNums = (computeCount / 2) % group;

            for (int32_t i = 0; i < times; i++) {
                LocalTensor<T> gOutLocalTensor = dataInQueue[0].AllocTensor<T>();
                for (int32_t j = 0; j < group; j++) {
                    gridPointIndex = curGridPointIndex + i * group + j;
                    w = gridPointIndex % outW;
                    h = (gridPointIndex / outW) % outH;
                    n = gridPointIndex / (outH * outW);
                    ncBaseOffset = n * dxStrideN;
                    gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                    DataCopyPad(gOutLocalTensor[j * alignChannel], inputGm[0][gradGmOffset], copyParams, padParams);
                }
                SetFlag<HardEvent::MTE2_S>(eventID);
                WaitFlag<HardEvent::MTE2_S>(eventID);
                for (int32_t k = 0; k < group; k++) {
                    gridPointIndex = curGridPointIndex + i * group + k;
                    n = gridPointIndex / (outH * outW);
                    ncBaseOffset = n * dxStrideN;
                    ComputeNearestXGrad(xIndex, dupOneTensor, i * group + k, k, ncBaseOffset, gOutLocalTensor);
                }
                dataInQueue[0].FreeTensor(gOutLocalTensor);
            }

            for (int32_t i = 0; i < tailNums; i++) {
                gridPointIndex = curGridPointIndex + times * group + i;
                w = gridPointIndex % outW;
                h = (gridPointIndex / outW) % outH;
                n = gridPointIndex / (outH * outW);
                ncBaseOffset = n * dxStrideN;
                gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                LocalTensor<T> gOutLocalTensor = dataInQueue[0].AllocTensor<T>();
                DataCopyPad(gOutLocalTensor, inputGm[0][gradGmOffset], copyParams, padParams);
                SetFlag<HardEvent::MTE2_S>(eventID);
                WaitFlag<HardEvent::MTE2_S>(eventID);
                ComputeNearestXGrad(xIndex, dupOneTensor, times * group + i, 0, ncBaseOffset, gOutLocalTensor);
                dataInQueue[0].FreeTensor(gOutLocalTensor);
            }
        } else {
            for (int32_t i = 0; i < computeCount / 2; i++) {
                gridPointIndex = curGridPointIndex + i;
                w = gridPointIndex % outW;
                h = (gridPointIndex / outW) % outH;
                n = gridPointIndex / (outH * outW);
                ncBaseOffset = n * dxStrideN;
                gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                LocalTensor<T> gOutLocalTensor = dataInQueue[0].AllocTensor<T>();
                DataCopyPad(gOutLocalTensor, inputGm[0][gradGmOffset], copyParams, padParams);
                SetFlag<HardEvent::MTE2_S>(eventID);
                WaitFlag<HardEvent::MTE2_S>(eventID);
                ComputeNearestXGrad(xIndex, dupOneTensor, i, 0, ncBaseOffset, gOutLocalTensor);
                dataInQueue[0].FreeTensor(gOutLocalTensor);
            }
        }
        Duplicate<T>(dstLocal, 0, 2 * ubFactorElement);
    }
    dataOutQueue[GRID_GRAD_OUTPUT_INDEX].EnQue(dstLocal);
    dataInQueue[GRID_INPUT_INDEX].FreeTensor(inputCoordinate);
}

template <typename T, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGrad<T, GridSamplerGradTilingData>::Process()
{
    uint32_t computePNum = 0;
    int64_t gridGmOffset = 0;
    int32_t gridOffset = 0;
    int32_t cycleOffset = 0;
    int64_t curGridPointIndex = 0;
    int32_t copyCountPerTime = 2 * ubFactorElement;
    int32_t actualComputNum = copyCountPerTime;
    if (blockIdx < tailPNum) {
        computePNum = pNumPerCore + 1;
        gridOffset = blockIdx * computePNum;
    } else {
        computePNum = pNumPerCore;
        gridOffset = blockIdx * pNumPerCore + tailPNum;
    }

    int32_t copyTimes = CeilDiv(computePNum * 2, copyCountPerTime);
    for (int j = 0; j < copyTimes; j++) {
        if (j == copyTimes - 1) {
            actualComputNum = computePNum * 2 - (copyTimes - 1) * copyCountPerTime;
        }
        cycleOffset = j * copyCountPerTime;
        gridGmOffset = cycleOffset + static_cast<int64_t>(gridOffset) * 2;
        curGridPointIndex = gridOffset + static_cast<int64_t>(j) * copyCountPerTime / 2;
        CopyIn(gridGmOffset, actualComputNum, GRID_INPUT_INDEX);
        Compute(actualComputNum, curGridPointIndex);
        CopyOut(gridGmOffset, actualComputNum);
    }
}
#endif  // GRID_SAMPLER_2D_GRAD_H_
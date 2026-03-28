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
 * \file upsample_trilinear3d_backward.h
 * \brief
 */
#ifndef UPSAMPLE_TRILINEAR3D_BACKWARD_H
#define UPSAMPLE_TRILINEAR3D_BACKWARD_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleTrilinear3dBackward {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t BUFFER_NUM = 1;
constexpr int8_t D_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;
constexpr int8_t W_DIRECTION = 2;

template <typename T>
class UpsampleTrilinear3dBackwardND {
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulH;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulD;

    __aicore__ inline UpsampleTrilinear3dBackwardND(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, UpsampleTrilinear3dBackwardTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(UpsampleTrilinear3dBackwardTilingData *tilingData);
    __aicore__ inline void GetSlideRange();
    __aicore__ inline void DirectionExpansion(int8_t direction, float scale);
    __aicore__ inline void CalculateRadioTensor(int8_t direction, int64_t index, int64_t length);
    __aicore__ inline float AreaPixelComputeSourceIndex(float scale, int32_t dstIndex);
    __aicore__ inline void CopyRadioTensorToGm();
    __aicore__ inline void CalculateWidthExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);
    __aicore__ inline void CalculateHeightExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);
    __aicore__ inline void CalculateDepthExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);

private:
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue;

    const TCubeTiling *__restrict matmulTilingW;
    const TCubeTiling *__restrict matmulTilingH;
    const TCubeTiling *__restrict matmulTilingD;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    int64_t blockIdx = 0;
    uint8_t dataType;
    int64_t batches = 0;
    int64_t inputShapes[3] = {0, 0, 0};
    int64_t outputShapes[3] = {0, 0, 0};

    float scaleW;
    float scaleH;
    float scaleD;
    bool alignCorners = true;
    bool needResizeW = true;
    bool needResizeH = true;
    bool needResizeD = true;

    int64_t slideSize = 0;
    int64_t radioMatrixSize;
    int64_t intermediateMatrixSizeW;
    int64_t intermediateMatrixSizeH;

    int64_t eachCoreSlideNums[3] = {0, 0, 0};
    int64_t remainders[3] = {0, 0, 0};
    int64_t tailStartSlideNums[3] = {0, 0, 0};
    int64_t groupCoreNums[3] = {0, 0, 0};
    int64_t inputRows[3] = {0, 0, 0};
    int64_t tailAvergingRows[3] = {0, 0, 0};
    int64_t needCoreNums[3] = {0, 0, 0};

    int64_t slideStarts[3] = {0, 0, 0};
    int64_t slideEnds[3] = {0, 0, 0};
    int64_t tailSlideStarts[3] = {0, 0, 0};
    int64_t tailSlideEnds[3] = {0, 0, 0};
    int64_t tailRowStarts[3] = {0, 0, 0};
    int64_t tailRowEnds[3] = {0, 0, 0};

    int64_t workSpaceRadioOffset = 0;
    int64_t xMin = 0;
    int64_t singleCoreK = 0;
};

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, UpsampleTrilinear3dBackwardTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;
    ParseTilingData(tilingData);
    GetSlideRange();

    pipe.InitBuffer(radioQueue, BUFFER_NUM, (radioMatrixSize * sizeof(float) + 31) / 32 * 32);

    inTensorsGM.SetGlobalBuffer((__gm__ T *)x);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)y);
    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        SyncAll();
        return;
    }

    if (needResizeW && blockIdx < needCoreNums[W_DIRECTION]) {
        DirectionExpansion(W_DIRECTION, scaleW);
    }
    SyncAll();

    if (needResizeH && blockIdx < needCoreNums[H_DIRECTION]) {
        DirectionExpansion(H_DIRECTION, scaleH);
    }
    SyncAll();

    if (needResizeD && blockIdx < needCoreNums[D_DIRECTION]) {
        DirectionExpansion(D_DIRECTION, scaleD);
    }
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::DirectionExpansion(int8_t direction, float scale)
{
    int64_t slideStart = slideStarts[direction];
    int64_t slideEnd = slideEnds[direction];
    // 计算批量分组的数据
    if (slideStart < slideEnd) {
        for (int64_t index = slideStart; index < slideEnd; index += slideSize) {
            int64_t length = MIN(slideSize, slideEnd - index);
            CalculateRadioTensor(direction, index, length);
            CopyRadioTensorToGm();
            if (direction == W_DIRECTION) {
                CalculateWidthExtension(index, 0, inputRows[direction], length);
            } else if (direction == H_DIRECTION) {
                CalculateHeightExtension(index, 0, inputRows[direction], length);
            } else {
                CalculateDepthExtension(index, 0, inputRows[direction], length);
            }
        }
    }

    int64_t tailSlideStart = tailSlideStarts[direction];
    int64_t tailSlideEnd = tailSlideEnds[direction];
    int64_t tailRowStart = tailRowStarts[direction];
    int64_t tailRowEnd = tailRowEnds[direction];
    // 处理尾块部分数据
    if (tailSlideStart < tailSlideEnd) {
        int64_t length = tailSlideEnd - tailSlideStart;
        CalculateRadioTensor(direction, tailSlideStart, length);
        CopyRadioTensorToGm();
        if (direction == W_DIRECTION) {
            CalculateWidthExtension(tailSlideStart, tailRowStart, tailRowEnd, length);
        } else if (direction == H_DIRECTION) {
            CalculateHeightExtension(tailSlideStart, tailRowStart, tailRowEnd, length);
        } else {
            CalculateDepthExtension(tailSlideStart, tailRowStart, tailRowEnd, length);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::CalculateRadioTensor(
    int8_t direction, int64_t xIndex, int64_t length)
{
    // 计算权重矩阵
    LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());

    float scale = scaleW;
    int64_t xlength = matmulTilingW->singleCoreK;
    if (direction == H_DIRECTION) {
        scale = scaleH;
        xlength = matmulTilingH->singleCoreK;
    } else if (direction == D_DIRECTION) {
        scale = scaleD;
        xlength = matmulTilingD->singleCoreK;
    }
    xMin = scale > (float)0.0 ? MAX(static_cast<int64_t>((float)(xIndex - 1) / scale) - 1, 0) : 0;
    xlength = MIN(outputShapes[direction] - xMin, xlength);
    singleCoreK = xlength;

    int64_t index0 = 0;
    int64_t index1 = 0;
    for (int64_t dstIndex = xMin; dstIndex < xMin + xlength; dstIndex++) {
        float realIndex = AreaPixelComputeSourceIndex(scale, dstIndex);
        int64_t srcIndex0 = MIN(static_cast<int64_t>(realIndex), inputShapes[direction] - 1);
        float lambda1 = MIN(MAX(realIndex - (float)srcIndex0, (float)0.0), (float)1.0);
        int64_t srcIndex1 = MIN(srcIndex0 + 1, inputShapes[direction] - 1);
        float lambda0 = srcIndex0 == srcIndex1 ? (float)1.0 : (float)1.0 - lambda1;

        if (direction == W_DIRECTION) {
            index0 = (dstIndex - xMin) * length + srcIndex0 - xIndex;
            index1 = (dstIndex - xMin) * length + srcIndex1 - xIndex;
        } else {
            index0 = (srcIndex0 - xIndex) * singleCoreK + dstIndex - xMin;
            index1 = (srcIndex1 - xIndex) * singleCoreK + dstIndex - xMin;
        }
        if (srcIndex0 - xIndex >= 0 && srcIndex0 - xIndex < length) {
            radioTensor.SetValue(index0, lambda0);
        }
        if (srcIndex0 != srcIndex1 && srcIndex1 - xIndex >= 0 && srcIndex1 - xIndex < length) {
            radioTensor.SetValue(index1, lambda1);
        }
    }

    if (dataType != 2) {
        Cast(radioTensor.ReinterpretCast<T>(), radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
    }
    radioQueue.EnQue(radioTensor);
}

template <typename T>
__aicore__ inline float UpsampleTrilinear3dBackwardND<T>::AreaPixelComputeSourceIndex(float scale, int32_t dstIndex)
{
    if (alignCorners) {
        return scale * (float)dstIndex;
    } else {
        return MAX(static_cast<float>(scale * ((float)dstIndex + (float)0.5) - (float)0.5), (float)0.0);
    }
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::CopyRadioTensorToGm()
{
    workSpaceRadioOffset = intermediateMatrixSizeW + intermediateMatrixSizeH + radioMatrixSize * blockIdx;

    int8_t size = 32 / sizeof(T);
    LocalTensor<T> radioTensor = radioQueue.DeQue<T>();
    DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, (radioTensor.GetSize() + size - 1) / size * size);
    radioQueue.FreeTensor(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::CalculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t xIndex = xMin + rowStart * outputShapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * inputShapes[2];

    int64_t singleCoreM = rowEnd - rowStart;
    int64_t singleCoreN = length;

    matmulW.SetOrgShape(singleCoreM, singleCoreN, outputShapes[2], singleCoreK, inputShapes[2]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > inputShapes[2]) {
        matmulW.SetTail(singleCoreM, inputShapes[2] - tensorCIndex, singleCoreK);
    }
    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    if (!needResizeH && !needResizeD) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }
    matmulW.End();
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::CalculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t singleCoreM = length;
    int64_t singleCoreN = matmulTilingH->singleCoreN;

    int64_t xIndex = xMin * inputShapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex * inputShapes[2];
    int64_t start = rowStart;
    int64_t end = rowEnd;

    matmulH.SetOrgShape(singleCoreM, inputShapes[2], singleCoreK, outputShapes[1], inputShapes[2]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > inputShapes[1]) {
        matmulH.SetTail(inputShapes[1] - tensorCIndex, singleCoreN, singleCoreK);
    }
    int64_t inStep = outputShapes[1] * inputShapes[2];
    int64_t outStep = inputShapes[1] * inputShapes[2];
    for (int64_t i = start, inOffset = start * inStep, outOffset = start * outStep; i < end;
         i++, inOffset += inStep, outOffset += outStep) {
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
        if (!needResizeW) {
            matmulH.SetTensorB(inTensorsGM[xIndex + inOffset], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + inOffset], false);
        }
        if (!needResizeD) {
            matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + outOffset], false);
        } else {
            matmulH.IterateAll(
                intermediateTensorGm[intermediateMatrixSizeW + tensorCIndexWithOffset + outOffset], false);
        }
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::CalculateDepthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t singleCoreM = length;
    int64_t singleCoreN = matmulTilingD->singleCoreN;

    int64_t xIndex = xMin * inputShapes[1] * inputShapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex * inputShapes[1] * inputShapes[2];
    int64_t start = rowStart;
    int64_t end = rowEnd;

    matmulD.SetOrgShape(
        singleCoreM, inputShapes[1] * inputShapes[2], singleCoreK, outputShapes[0], inputShapes[1] * inputShapes[2]);
    matmulD.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > inputShapes[0]) {
        matmulD.SetTail(inputShapes[0] - tensorCIndex, singleCoreN, singleCoreK);
    }
    int64_t inStep = outputShapes[0] * inputShapes[1] * inputShapes[2];
    int64_t outStep = inputShapes[0] * inputShapes[1] * inputShapes[2];
    for (int64_t i = start, inOffset = start * inStep, outOffset = start * outStep; i < end;
         i++, inOffset += inStep, outOffset += outStep) {
        matmulD.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
        if (!needResizeW && !needResizeH) {
            matmulD.SetTensorB(inTensorsGM[xIndex + inOffset], false);
        } else if (!needResizeH) {
            matmulD.SetTensorB(intermediateTensorGm[xIndex + inOffset], false);
        } else {
            matmulD.SetTensorB(intermediateTensorGm[intermediateMatrixSizeW + xIndex + inOffset], false);
        }
        matmulD.IterateAll(outTensorsGM[tensorCIndexWithOffset + outOffset], false);
        matmulD.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::ParseTilingData(
    UpsampleTrilinear3dBackwardTilingData *tilingData)
{
    dataType = tilingData->dataType;
    batches = tilingData->batches;
    for (int8_t i = 0; i < 3; i++) {
        inputShapes[i] = tilingData->inputShapes[i];
        outputShapes[i] = tilingData->outputShapes[i];

        eachCoreSlideNums[i] = tilingData->eachCoreSlideNums[i];
        remainders[i] = tilingData->remainders[i];
        tailStartSlideNums[i] = tilingData->tailStartSlideNums[i];
        groupCoreNums[i] = tilingData->groupCoreNums[i];
        inputRows[i] = tilingData->inputRows[i];
        tailAvergingRows[i] = tilingData->tailAvergingRows[i];
        needCoreNums[i] = tilingData->needCoreNums[i];
    }

    scaleW = tilingData->scaleW;
    scaleH = tilingData->scaleH;
    scaleD = tilingData->scaleD;
    alignCorners = tilingData->alignCorners;
    needResizeW = tilingData->needResizeW;
    needResizeH = tilingData->needResizeH;
    needResizeD = tilingData->needResizeD;

    slideSize = tilingData->slideSize;
    radioMatrixSize = tilingData->radioMatrixSize;
    intermediateMatrixSizeW = tilingData->intermediateMatrixSizeW;
    intermediateMatrixSizeH = tilingData->intermediateMatrixSizeH;

    matmulTilingW = &tilingData->matmulTilingW;
    matmulTilingH = &tilingData->matmulTilingH;
    matmulTilingD = &tilingData->matmulTilingD;
}

template <typename T>
__aicore__ inline void UpsampleTrilinear3dBackwardND<T>::GetSlideRange()
{
    for (int8_t i = 0; i < 3; i++) {
        slideStarts[i] = blockIdx * eachCoreSlideNums[i] * slideSize;
        slideEnds[i] = MIN(slideStarts[i] + eachCoreSlideNums[i] * slideSize, inputShapes[i]);

        int64_t groupIndex = blockIdx / groupCoreNums[i];
        if (groupIndex < remainders[i]) {
            tailSlideStarts[i] = (tailStartSlideNums[i] + groupIndex) * slideSize;
            tailSlideEnds[i] = MIN(tailSlideStarts[i] + slideSize, inputShapes[i]);
            int64_t blockIdxInGroup = blockIdx % groupCoreNums[i];
            tailRowStarts[i] = blockIdxInGroup * tailAvergingRows[i];
            tailRowEnds[i] = MIN(tailRowStarts[i] + tailAvergingRows[i], inputRows[i]);
        }
    }
}
}  // namespace UpsampleTrilinear3dBackward

#endif  // UPSAMPLE_TRILINEAR3D_BACKWARD_H

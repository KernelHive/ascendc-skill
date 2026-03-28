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
 * \file upsample_nearest_exact3d_grad.h
 * \brief
 */
#ifndef UPSAMPLE_NEARESTEXACT3D_GRAD_H
#define UPSAMPLE_NEARESTEXACT3D_GRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "upsample_nearest_exact3d_grad_common.h"

namespace UpsampleNearestExact3dGrad {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t BUFFER_NUM = 1;
constexpr int8_t D_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;
constexpr int8_t W_DIRECTION = 2;
constexpr int8_t RESERVED_LENGTH = 5;

template <typename T>
class UpsampleNearestExact3dGradND {
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

    __aicore__ inline UpsampleNearestExact3dGradND(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, bool isExact, GM_ADDR workspace, const UpsampleNearestExact3dGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };
    template <typename T1>
    __aicore__ inline int64_t CeilNum(T1 x)
    {
        int64_t floorX = int64_t(x);
        if (FloatEqual(x, floorX)) {
            return floorX;
        }
        return floorX + 1;
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 a, T1 b)
    {
        return a > b ? a : b;
    };
    __aicore__ inline void ParseTilingData(const UpsampleNearestExact3dGradTilingData *tilingData);
    __aicore__ inline void GetSlideRange();
    __aicore__ inline void ClearGM(const GlobalTensor<T> &dstGlobal, int64_t totalCount);
    __aicore__ inline void DirectionExpansion(int8_t direction, float scale);
    __aicore__ inline void CalculateIntermediateTensor(int8_t direction, int64_t index, int64_t length, float scale);
    __aicore__ inline void CalculateRadioTensor(int8_t direction, int64_t index, int64_t length, int64_t startIdx);
    __aicore__ inline void CopyRadioTensorToGm();
    __aicore__ inline void CalculateWidthExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);
    __aicore__ inline void CalculateHeightExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);
    __aicore__ inline void CalculateDepthExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);

private:
    TBuf<QuePosition::VECCALC> srcIndexQueue;
    TBuf<QuePosition::VECCALC> srcIndexOutQueue;
    TBuf<TPosition::VECCALC> UbBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue;

    const TCubeTiling *__restrict matmulTilingW;
    const TCubeTiling *__restrict matmulTilingH;
    const TCubeTiling *__restrict matmulTilingD;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> srcIndexTensor;
    LocalTensor<float> srcIndexOutTensor;
    int64_t blockIdx = 0;
    bool isExactFlag = false;
    uint8_t dataType;
    int64_t batches = 0;
    int64_t gradInputShapes[3] = {0, 0, 0};
    int64_t gradOutputShapes[3] = {0, 0, 0};
    float scaleArray[3] = {0.0, 0.0, 0.0};

    float scaleW;
    float scaleH;
    float scaleD;
    bool needResizeW = true;
    bool needResizeH = true;
    bool needResizeD = true;

    int64_t slideSize = 0;
    int64_t tensorSize = 0;
    int64_t tensorSizeMapping = 0;
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
    int64_t weightMin = 0;
    int64_t singleCoreK = 0;
};

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::Init(
    GM_ADDR x, GM_ADDR y, bool isExact, GM_ADDR workspace, const UpsampleNearestExact3dGradTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;
    isExactFlag = isExact;
    ParseTilingData(tilingData);
    GetSlideRange();

    pipe.InitBuffer(UbBuf, (64 * sizeof(T) + 31) / 32 * 32);
    pipe.InitBuffer(radioQueue, BUFFER_NUM, (radioMatrixSize * sizeof(T) + 31) / 32 * 32);
    pipe.InitBuffer(srcIndexQueue, (tensorSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(srcIndexOutQueue, (tensorSizeMapping * sizeof(float) + 31) / 32 * 32);

    inTensorsGM.SetGlobalBuffer((__gm__ T *)x);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)y);
    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);

    if (GetBlockIdx() == 0) {
        int64_t needCoreNum = Max(needCoreNums[0], needCoreNums[1]);
        needCoreNum = Max(needCoreNum, needCoreNums[2]);
        int64_t intermediateTensorSize =
            intermediateMatrixSizeW + intermediateMatrixSizeH + radioMatrixSize * needCoreNum;
        ClearGM(intermediateTensorGm, intermediateTensorSize);
        ClearGM(outTensorsGM, batches * gradInputShapes[0] * gradInputShapes[1] * gradInputShapes[2]);
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::ClearGM(const GlobalTensor<T> &dstGlobal, int64_t totalCount)
{
    int64_t baseN = 64;
    int64_t loop = totalCount / baseN;
    int64_t totalCountTail = totalCount % baseN;
    int64_t offset = 0;

    for (int i = 0; i < loop; i++) {
        InitGmZero<T>(dstGlobal, UbBuf, baseN, offset);
        offset += baseN;
    }
    if (totalCountTail > 0) {
        InitGmZero<T>(dstGlobal, UbBuf, totalCountTail, offset);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::Process()
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
__aicore__ inline void UpsampleNearestExact3dGradND<T>::DirectionExpansion(int8_t direction, float scale)
{
    srcIndexTensor = srcIndexQueue.AllocTensor<float>();
    srcIndexOutTensor = srcIndexOutQueue.AllocTensor<float>();

    int64_t slideStart = slideStarts[direction];
    int64_t slideEnd = slideEnds[direction];
    // 计算批量分组的数据
    if (slideStart < slideEnd) {
        CalculateIntermediateTensor(direction, slideStart, slideEnd - slideStart, scale);
        for (int64_t index = slideStart; index < slideEnd; index += slideSize) {
            int64_t length = Min(slideSize, slideEnd - index);
            CalculateRadioTensor(direction, index - slideStart, length, index);
            CopyRadioTensorToGm();
            if (singleCoreK > 0 && direction == W_DIRECTION) {
                CalculateWidthExtension(index, 0, inputRows[direction], length);
            } else if (singleCoreK > 0 && direction == H_DIRECTION) {
                CalculateHeightExtension(index, 0, inputRows[direction], length);
            } else if (singleCoreK > 0) {
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
        CalculateIntermediateTensor(direction, tailSlideStart, length, scale);
        CalculateRadioTensor(direction, 0, length, tailSlideStart);
        CopyRadioTensorToGm();
        if (singleCoreK > 0 && direction == W_DIRECTION) {
            CalculateWidthExtension(tailSlideStart, tailRowStart, tailRowEnd, length);
        } else if (singleCoreK > 0 && direction == H_DIRECTION) {
            CalculateHeightExtension(tailSlideStart, tailRowStart, tailRowEnd, length);
        } else if (singleCoreK > 0) {
            CalculateDepthExtension(tailSlideStart, tailRowStart, tailRowEnd, length);
        }
    }

    srcIndexQueue.FreeTensor(srcIndexTensor);
    srcIndexOutQueue.FreeTensor(srcIndexOutTensor);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::CalculateIntermediateTensor(
    int8_t direction, int64_t index, int64_t length, float scale)
{
    int64_t inputSize = gradInputShapes[direction];
    int64_t outputSize = gradOutputShapes[direction];
    int64_t actualLength = length + 1;

    ArithProgression(srcIndexTensor, static_cast<float>(index), static_cast<float>(1), actualLength);
    PipeBarrier<PIPE_V>();

    Muls(srcIndexTensor, srcIndexTensor, scale, actualLength);
    PipeBarrier<PIPE_V>();

    if (isExactFlag) {
        Adds(srcIndexTensor, srcIndexTensor, static_cast<float>(-0.5), actualLength);
        PipeBarrier<PIPE_V>();
    }

    Ceil(srcIndexTensor, srcIndexTensor, actualLength);
    PipeBarrier<PIPE_V>();
    Mins(srcIndexTensor, srcIndexTensor, static_cast<float>(outputSize), actualLength);
    PipeBarrier<PIPE_V>();

    Duplicate(srcIndexOutTensor, static_cast<float>(0.0), srcIndexOutTensor.GetSize());
    int64_t idxMin = srcIndexTensor.GetValue(0);
    weightMin = idxMin;
    for (int64_t i = 0; i < length; i++) {
        int64_t idx = srcIndexTensor.GetValue(i);
        int64_t idxUp = srcIndexTensor.GetValue(i + 1);
        for (int64_t weightIndex = idx; weightIndex < idxUp; weightIndex++) {
            srcIndexOutTensor.SetValue(weightIndex - idxMin, i + index);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::CalculateRadioTensor(
    int8_t direction, int64_t xIndex, int64_t length, int64_t startIdx)
{
    // 计算权重矩阵
    xMin = static_cast<int64_t>(srcIndexTensor.GetValue(xIndex));
    int64_t xOutIndex = xMin - weightMin;
    if (xOutIndex < 0) {
        xOutIndex = 0;
    }
    int64_t xOutlength = static_cast<int64_t>(srcIndexTensor.GetValue(xIndex + length)) - xMin;
    if (xOutlength < 0) {
        xOutlength = 0;
    }
    if (xOutlength > gradOutputShapes[direction] - xMin) {
        xOutlength = gradOutputShapes[direction] - xMin;
    }
    singleCoreK = xOutlength;

    LocalTensor<T> radioTensor = radioQueue.AllocTensor<T>();
    Duplicate(radioTensor, static_cast<T>(0.0), radioTensor.GetSize());

    for (int64_t i = xOutIndex; i < xOutIndex + xOutlength; i++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexOutTensor.GetValue(i));
        int64_t index = 0;

        if (srcIndex < startIdx || srcIndex >= (startIdx + length)) {
            continue;
        }

        if (direction == W_DIRECTION) {
            index = (i - xOutIndex) * length + srcIndex - startIdx;
        } else {
            index = (srcIndex - startIdx) * xOutlength + i - xOutIndex;
        }
        radioTensor.SetValue(index, static_cast<T>(1.0));
    }

    radioQueue.EnQue(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::CopyRadioTensorToGm()
{
    workSpaceRadioOffset = intermediateMatrixSizeW + intermediateMatrixSizeH + radioMatrixSize * blockIdx;

    int8_t size = 32 / sizeof(T);
    LocalTensor<T> radioTensor = radioQueue.DeQue<T>();
    DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, (radioTensor.GetSize() + size - 1) / size * size);
    radioQueue.FreeTensor(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::CalculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t xIndex = xMin + rowStart * gradOutputShapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * gradInputShapes[2];

    int64_t singleCoreM = rowEnd - rowStart;
    int64_t singleCoreN = length;

    matmulW.SetOrgShape(singleCoreM, singleCoreN, gradOutputShapes[2], singleCoreK, gradInputShapes[2]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > gradInputShapes[2]) {
        matmulW.SetTail(singleCoreM, gradInputShapes[2] - tensorCIndex, singleCoreK);
    }
    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    if (!needResizeH && !needResizeD) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], true);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], true);
    }
    matmulW.End();
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::CalculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t singleCoreM = length;
    int64_t singleCoreN = matmulTilingH->singleCoreN;

    int64_t xIndex = xMin * gradInputShapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex * gradInputShapes[2];
    int64_t start = rowStart;
    int64_t end = rowEnd;
    if (inputRows[H_DIRECTION] == gradInputShapes[2]) {
        singleCoreN = rowEnd - rowStart;
        start = 0;
        end = batches * gradOutputShapes[0];
        xIndex += rowStart;
        tensorCIndexWithOffset += rowStart;
    }

    matmulH.SetOrgShape(singleCoreM, gradInputShapes[2], singleCoreK, gradOutputShapes[1], gradInputShapes[2]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > gradInputShapes[1]) {
        matmulH.SetTail(gradInputShapes[1] - tensorCIndex, singleCoreN, singleCoreK);
    }
    int64_t inStep = gradOutputShapes[1] * gradInputShapes[2];
    int64_t outStep = gradInputShapes[1] * gradInputShapes[2];
    for (int64_t i = start, inOffset = start * inStep, outOffset = start * outStep; i < end;
         i++, inOffset += inStep, outOffset += outStep) {
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
        if (!needResizeW) {
            matmulH.SetTensorB(inTensorsGM[xIndex + inOffset], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + inOffset], false);
        }
        if (!needResizeD) {
            matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + outOffset], true);
        } else {
            matmulH.IterateAll(
                intermediateTensorGm[intermediateMatrixSizeW + tensorCIndexWithOffset + outOffset], true);
        }
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::CalculateDepthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t singleCoreM = length;
    int64_t singleCoreN = matmulTilingD->singleCoreN;

    int64_t xIndex = xMin * gradInputShapes[1] * gradInputShapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex * gradInputShapes[1] * gradInputShapes[2];
    int64_t start = rowStart;
    int64_t end = rowEnd;
    if (inputRows[D_DIRECTION] == gradInputShapes[1] * gradInputShapes[2]) {
        singleCoreN = rowEnd - rowStart;
        start = 0;
        end = batches;
        xIndex += rowStart;
        tensorCIndexWithOffset += rowStart;
    }

    matmulD.SetOrgShape(singleCoreM,
        gradInputShapes[1] * gradInputShapes[2],
        singleCoreK,
        gradOutputShapes[0],
        gradInputShapes[1] * gradInputShapes[2]);
    matmulD.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > gradInputShapes[0]) {
        matmulD.SetTail(gradInputShapes[0] - tensorCIndex, singleCoreN, singleCoreK);
    }
    int64_t inStep = gradOutputShapes[0] * gradInputShapes[1] * gradInputShapes[2];
    int64_t outStep = gradInputShapes[0] * gradInputShapes[1] * gradInputShapes[2];
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
        matmulD.IterateAll(outTensorsGM[tensorCIndexWithOffset + outOffset], true);
        matmulD.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::ParseTilingData(
    const UpsampleNearestExact3dGradTilingData *tilingData)
{
    dataType = tilingData->dataType;
    batches = tilingData->batches;
    for (int8_t i = 0; i < 3; i++) {
        gradOutputShapes[i] = tilingData->gradOutputShapes[i];
        gradInputShapes[i] = tilingData->gradInputShapes[i];

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

    scaleArray[0] = scaleD;
    scaleArray[1] = scaleH;
    scaleArray[2] = scaleW;
    needResizeW = tilingData->needResizeW;
    needResizeH = tilingData->needResizeH;
    needResizeD = tilingData->needResizeD;

    slideSize = tilingData->slideSize;
    tensorSize = tilingData->tensorSize;
    tensorSizeMapping = tilingData->tensorSizeMapping;
    radioMatrixSize = tilingData->radioMatrixSize;
    intermediateMatrixSizeW = tilingData->intermediateMatrixSizeW;
    intermediateMatrixSizeH = tilingData->intermediateMatrixSizeH;

    matmulTilingW = &tilingData->matmulTilingW;
    matmulTilingH = &tilingData->matmulTilingH;
    matmulTilingD = &tilingData->matmulTilingD;
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dGradND<T>::GetSlideRange()
{
    for (int8_t i = 0; i < 3; i++) {
        slideStarts[i] = blockIdx * eachCoreSlideNums[i] * slideSize;
        slideEnds[i] = Min(slideStarts[i] + eachCoreSlideNums[i] * slideSize, gradInputShapes[i]);

        int64_t groupIndex = groupCoreNums[i] == 0 ? 0 : (blockIdx / groupCoreNums[i]);
        if (groupIndex < remainders[i]) {
            tailSlideStarts[i] = (tailStartSlideNums[i] + groupIndex) * slideSize;
            tailSlideEnds[i] = Min(tailSlideStarts[i] + slideSize, gradInputShapes[i]);
            int64_t blockIdxInGroup = blockIdx % groupCoreNums[i];
            tailRowStarts[i] = blockIdxInGroup * tailAvergingRows[i];
            tailRowEnds[i] = Min(tailRowStarts[i] + tailAvergingRows[i], inputRows[i]);
        }
    }
}
}  // namespace UpsampleNearestExact3dGrad

#endif  // UPSAMPLE_NEARESTEXACT3D_GRAD_H

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
 * \file upsample_bilinear2d_aa_backward.h
 * \brief
 */
#ifndef UPSAMPLE_BILINEAR2D_AA_BACKWARD_H
#define UPSAMPLE_BILINEAR2D_AA_BACKWARD_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBilinear2dAABackward {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class UpsampleBilinear2dAABackwardND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulH;

    __aicore__ inline UpsampleBilinear2dAABackwardND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBilinear2dAABackwardTilingData *tilingData);
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
    template <typename T1>
    __aicore__ inline T1 weightCalculate(T1 x)
    {
        if (x < 0) {
            x = -1 * x;
        }
        if (x < (float)1.0) {
            return (float)1.0 - x;
        }
        return 0.0;
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
    __aicore__ inline void ParseTilingData(const UpsampleBilinear2dAABackwardTilingData *tilingData);
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void calculateIntermediateTensorW(int64_t index, int64_t length);
    __aicore__ inline void calculateIntermediateTensorH(int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorW(int64_t index, int64_t length, int64_t minIndex);
    __aicore__ inline void calculateRadioTensorH(int64_t index, int64_t length, int64_t minIndex);
    __aicore__ inline void calculateWidthExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);
    __aicore__ inline void calculateHeightExtension(
        int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length);

    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRadioTensor(int8_t direction);

    __aicore__ inline void releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xMinQueueW;
    TBuf<QuePosition::VECCALC> xSizeQueueW;
    TBuf<QuePosition::VECCALC> weightQueueW;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xMinQueueH;
    TBuf<QuePosition::VECCALC> xSizeQueueH;
    TBuf<QuePosition::VECCALC> weightQueueH;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueueH;

    const TCubeTiling *__restrict matmulTilingW;
    const TCubeTiling *__restrict matmulTilingH;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;
    LocalTensor<float> xSizeTensor;
    LocalTensor<float> weightTensor;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slideSize = 0;
    float scaleW;
    float scaleH;
    float invscaleW;
    float invscaleH;
    float supportW;
    float supportH;
    int64_t maxInterpSizeW;
    int64_t maxInterpSizeH;
    int64_t needCoreNumW;
    int64_t needCoreNumH;
    bool needResizeW = true;
    bool needResizeH = true;

    uint8_t dataType;
    uint64_t intermediateMatrixSize;
    uint32_t radioMatrixSizeW;
    uint32_t radioMatrixSizeH;

    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    int64_t slideStartH;
    int64_t slideEndH;
    int64_t tailSlideStartH;
    int64_t tailSlideEndH;
    int64_t tailRowStartH;
    int64_t tailRowEndH;

    int64_t inputShapes[4] = {0, 0, 0, 0};
    int64_t outputShapes[4] = {0, 0, 0, 0};
    int64_t workSpaceRadioOffset = 0;
    int64_t xMin = 0;
    int64_t singleCoreK = 0;
};

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBilinear2dAABackwardTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;

    inTensorPtr = input;
    outTensorPtr = output;
    ParseTilingData(tilingData);

    if (needResizeW) {
        int64_t tensorWidthSize = (matmulTilingW->singleCoreK * sizeof(float) + 31) / 32 * 32;
        pipe.InitBuffer(centerQueueW, tensorWidthSize);
        pipe.InitBuffer(xMinQueueW, tensorWidthSize);
        pipe.InitBuffer(xSizeQueueW, tensorWidthSize);
        pipe.InitBuffer(weightQueueW, (maxInterpSizeW * sizeof(float) + 31) / 32 * 32);
        pipe.InitBuffer(radioQueueW, BUFFER_NUM, (radioMatrixSizeW * sizeof(float) + 31) / 32 * 32);
    }

    if (needResizeH) {
        int64_t tensorHeightSize = (matmulTilingH->singleCoreK * sizeof(float) + 31) / 32 * 32;
        pipe.InitBuffer(centerQueueH, tensorHeightSize);
        pipe.InitBuffer(xMinQueueH, tensorHeightSize);
        pipe.InitBuffer(xSizeQueueH, tensorHeightSize);
        pipe.InitBuffer(weightQueueH, (maxInterpSizeH * sizeof(float) + 31) / 32 * 32);
        pipe.InitBuffer(radioQueueH, BUFFER_NUM, (radioMatrixSizeH * sizeof(float) + 31) / 32 * 32);
    }

    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorPtr);
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }

    // 先横向扩展
    if (needResizeW) {
        WDirectionExpansion();
    }

    SyncAll();

    // 再纵向扩展
    if (needResizeH) {
        HDirectionExpansion();
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::WDirectionExpansion()
{
    if (blockIdx < needCoreNumW) {
        centerTensor = centerQueueW.AllocTensor<float>();
        xMinTensor = xMinQueueW.AllocTensor<float>();
        xSizeTensor = xSizeQueueW.AllocTensor<float>();
        weightTensor = weightQueueW.AllocTensor<float>();
        // 计算批量分组的数据
        if (slideStartW < slideEndW) {
            for (int64_t index = slideStartW; index < slideEndW; index += slideSize) {
                int16_t length = Min(slideSize, slideEndW - index);
                calculateIntermediateTensorW(index, length);
                // 计算系数矩阵
                calculateRadioTensorW(0, length, index);
                copyRadioTensorToGm(0);
                calculateWidthExtension(index, 0, 0, length);
            }
        }

        // 处理尾块部分数据
        if (tailSlideStartW < tailSlideEndW) {
            for (int64_t index = tailSlideStartW; index < tailSlideEndW; index += slideSize) {
                int16_t length = Min(slideSize, tailSlideEndW - index);
                calculateIntermediateTensorW(index, length);
                calculateRadioTensorW(0, length, index);
                copyRadioTensorToGm(0);
                calculateWidthExtension(index, tailRowStartW, tailRowEndW, length);
            }
        }

        // 释放临时tensor
        centerQueueW.FreeTensor(centerTensor);
        xMinQueueW.FreeTensor(xMinTensor);
        xSizeQueueW.FreeTensor(xSizeTensor);
        weightQueueW.FreeTensor(weightTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::HDirectionExpansion()
{
    if (blockIdx < needCoreNumH) {
        centerTensor = centerQueueH.AllocTensor<float>();
        xMinTensor = xMinQueueH.AllocTensor<float>();
        xSizeTensor = xSizeQueueH.AllocTensor<float>();
        weightTensor = weightQueueH.AllocTensor<float>();
        // 计算批量分组的数据
        if (slideStartH < slideEndH) {
            for (int64_t index = slideStartH; index < slideEndH; index += slideSize) {
                int16_t length = Min(slideSize, slideEndH - index);
                calculateIntermediateTensorH(index, length);
                // 计算系数矩阵
                calculateRadioTensorH(0, length, index);
                copyRadioTensorToGm(1);
                calculateHeightExtension(index, 0, 0, length);
            }
        }

        // 处理尾块部分数据
        if (tailSlideStartH < tailSlideEndH) {
            for (int64_t index = tailSlideStartH; index < tailSlideEndH; index += slideSize) {
                int16_t length = Min(slideSize, tailSlideEndH - index);
                calculateIntermediateTensorH(index, length);
                calculateRadioTensorH(0, length, index);
                copyRadioTensorToGm(1);
                calculateHeightExtension(index, tailRowStartH, tailRowEndH, length);
            }
        }

        // 释放临时tensor
        centerQueueH.FreeTensor(centerTensor);
        xMinQueueH.FreeTensor(xMinTensor);
        xSizeQueueH.FreeTensor(xSizeTensor);
        weightQueueH.FreeTensor(weightTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::calculateIntermediateTensorW(int64_t index, int64_t length)
{
    if (scaleW > (float)0.0) {
        if (index > 0) {
            index = index - 1;
        }
        index = static_cast<int64_t>((float)index / scaleW - supportW) - 1;
        if (index < 0) {
            index = 0;
        }
    } else {
        index = 0;
    }
    length = centerTensor.GetSize();
    if (length > inputShapes[3] - index) {
        length = inputShapes[3] - index;
    }
    xMin = index;
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    Adds(centerTensor, centerTensor, (float)0.5, length);
    PipeBarrier<PIPE_V>();
    Muls(centerTensor, centerTensor, scaleW, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标最小映射值
    Adds(xMinTensor, centerTensor, (float)0.5 - supportW, length);
    PipeBarrier<PIPE_V>();
    Floor(xMinTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标映射的范围
    Adds(xSizeTensor, centerTensor, (float)0.5 + supportW, length);
    PipeBarrier<PIPE_V>();
    Floor(xSizeTensor, xSizeTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(outputShapes[3]), length);
    PipeBarrier<PIPE_V>();
    Sub(xSizeTensor, xSizeTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(maxInterpSizeW), length);
    PipeBarrier<PIPE_V>();
    Maxs(xSizeTensor, xSizeTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::calculateIntermediateTensorH(int64_t index, int64_t length)
{
    if (scaleH > (float)0.0) {
        if (index > 0) {
            index = index - 1;
        }
        index = static_cast<int64_t>((float)index / scaleH - supportH) - 1;
        if (index < 0) {
            index = 0;
        }
    } else {
        index = 0;
    }
    length = centerTensor.GetSize();
    if (length > inputShapes[2] - index) {
        length = inputShapes[2] - index;
    }
    xMin = index;
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    Adds(centerTensor, centerTensor, (float)0.5, length);
    PipeBarrier<PIPE_V>();
    Muls(centerTensor, centerTensor, scaleH, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标最小映射值
    Adds(xMinTensor, centerTensor, (float)0.5 - supportH, length);
    PipeBarrier<PIPE_V>();
    Floor(xMinTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标映射的范围
    Adds(xSizeTensor, centerTensor, (float)0.5 + supportH, length);
    PipeBarrier<PIPE_V>();
    Floor(xSizeTensor, xSizeTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(outputShapes[2]), length);
    PipeBarrier<PIPE_V>();
    Sub(xSizeTensor, xSizeTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(maxInterpSizeH), length);
    PipeBarrier<PIPE_V>();
    Maxs(xSizeTensor, xSizeTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::calculateRadioTensorW(
    int64_t xIndex, int64_t length, int64_t minIndex)
{
    LocalTensor<float> radioTensor = radioQueueW.AllocTensor<float>();
    // 计算横向系数矩阵
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    int64_t xlength = matmulTilingW->singleCoreK;
    if (xlength > inputShapes[3] - xMin) {
        xlength = inputShapes[3] - xMin;
    }
    singleCoreK = xlength;
    for (int64_t i = xIndex; i < xIndex + xlength; i++) {
        float totalW = 0.0;
        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
            float w = weightCalculate((j + xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5) * invscaleW);
            totalW += w;
            weightTensor.SetValue(j, w);
        }

        if (totalW > (float)0.0) {
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
                float weight = weightTensor.GetValue(j) / totalW;
                int64_t yIndexValue = j + xMinTensor.GetValue(i) - minIndex;
                if (yIndexValue >= 0 && yIndexValue < length) {
                    int64_t xIndexValue = i - xIndex;
                    int64_t index = xIndexValue * length + yIndexValue;
                    radioTensor.SetValue(index, weight);
                }
            }
        }
    }
    if (dataType != 2) {
        Cast(radioTensor.ReinterpretCast<T>(), radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
    }
    radioQueueW.EnQue(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::calculateRadioTensorH(
    int64_t xIndex, int64_t length, int64_t minIndex)
{
    LocalTensor<float> radioTensor = radioQueueH.AllocTensor<float>();
    // 计算纵向系数矩阵
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    int64_t xlength = matmulTilingH->singleCoreK;
    if (xlength > inputShapes[2] - xMin) {
        xlength = inputShapes[2] - xMin;
    }
    singleCoreK = xlength;
    for (int64_t i = xIndex; i < xIndex + xlength; i++) {
        float totalW = 0.0;
        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
            float w = weightCalculate((j + xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5) * invscaleH);
            totalW += w;
            weightTensor.SetValue(j, w);
        }

        if (totalW > (float)0.0) {
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
                float weight = weightTensor.GetValue(j) / totalW;
                int64_t yIndexValue = j + xMinTensor.GetValue(i) - minIndex;
                if (yIndexValue >= 0 && yIndexValue < length) {
                    int64_t xIndexValue = i - xIndex;
                    int64_t index = yIndexValue * singleCoreK + xIndexValue;
                    radioTensor.SetValue(index, weight);
                }
            }
        }
    }
    if (dataType != 2) {
        Cast(radioTensor.ReinterpretCast<T>(), radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
    }
    radioQueueH.EnQue(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::copyRadioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    if (direction == 0) {
        workSpaceRadioOffset = intermediateMatrixSize + radioMatrixSizeW * blockIdx;
    } else {
        workSpaceRadioOffset = intermediateMatrixSize + radioMatrixSizeH * blockIdx;
    }

    int8_t size = 32 / sizeof(T);
    LocalTensor<T> radioTensor = initRadioTensor(direction);
    DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, (radioTensor.GetSize() + size - 1) / size * size);
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    releaseRadioTensor(direction, radioTensor);
}

template <typename T>
__aicore__ inline LocalTensor<T> UpsampleBilinear2dAABackwardND<T>::initRadioTensor(int8_t direction)
{
    if (direction == 0) {
        return radioQueueW.DeQue<T>();
    } else {
        return radioQueueH.DeQue<T>();
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::releaseRadioTensor(
    int8_t direction, LocalTensor<T> radioTensor)
{
    if (direction == 0) {
        return radioQueueW.FreeTensor(radioTensor);
    } else {
        return radioQueueH.FreeTensor(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t singleCoreM = matmulTilingW->singleCoreM;
    int64_t singleCoreN = length;
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, inputShapes[3], singleCoreK, outputShapes[3]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slideSize > outputShapes[3]) {
        matmulW.SetTail(singleCoreM, outputShapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = xMin + rowStart * inputShapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * outputShapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    if (!needResizeH) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }
    matmulW.End();

    event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID3);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::calculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd, int64_t length)
{
    int64_t singleCoreM = length;
    int64_t singleCoreN = matmulTilingH->singleCoreN;
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreN = rowEnd - rowStart;
    }
    matmulH.SetOrgShape(singleCoreM, outputShapes[3], singleCoreK);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slideSize > outputShapes[2]) {
        matmulH.SetTail(outputShapes[2] - tensorCIndex, singleCoreN, singleCoreK);
    }
    int64_t xIndex = xMin * outputShapes[3] + rowStart;
    int64_t tensorCIndexWithOffset = tensorCIndex * outputShapes[3] + rowStart;

    for (int64_t i = 0; i < outputShapes[0] * outputShapes[1]; i++) {
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
        if (!needResizeW) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * inputShapes[2] * outputShapes[3]], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * inputShapes[2] * outputShapes[3]], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * outputShapes[2] * outputShapes[3]], false);
        matmulH.End();

        event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinear2dAABackwardND<T>::ParseTilingData(
    const UpsampleBilinear2dAABackwardTilingData *tilingData)
{
    slideSize = tilingData->slideSize;
    scaleW = tilingData->scaleW;
    scaleH = tilingData->scaleH;
    invscaleW = tilingData->invscaleW;
    invscaleH = tilingData->invscaleH;
    supportW = tilingData->supportW;
    supportH = tilingData->supportH;
    maxInterpSizeW = tilingData->maxInterpSizeW;
    maxInterpSizeH = tilingData->maxInterpSizeH;
    needCoreNumW = tilingData->needCoreNumW;
    needCoreNumH = tilingData->needCoreNumH;
    needResizeW = tilingData->needResizeW;
    needResizeH = tilingData->needResizeH;

    for (int8_t i = 0; i < 4; i++) {
        outputShapes[i] = tilingData->outputShapes[i];
    }
    for (int8_t i = 0; i < 4; i++) {
        inputShapes[i] = tilingData->inputShapes[i];
    }
    intermediateMatrixSize = tilingData->intermediateMatrixSize;
    radioMatrixSizeW = tilingData->radioMatrixSizeW;
    radioMatrixSizeH = tilingData->radioMatrixSizeH;

    slideStartW = tilingData->slideStartListW[blockIdx];
    slideEndW = tilingData->slideEndListW[blockIdx];
    tailSlideStartW = tilingData->tailSlideStartListW[blockIdx];
    tailSlideEndW = tilingData->tailSlideEndListW[blockIdx];
    tailRowStartW = tilingData->tailRowStartListW[blockIdx];
    tailRowEndW = tilingData->tailRowEndListW[blockIdx];

    slideStartH = tilingData->slideStartListH[blockIdx];
    slideEndH = tilingData->slideEndListH[blockIdx];
    tailSlideStartH = tilingData->tailSlideStartListH[blockIdx];
    tailSlideEndH = tilingData->tailSlideEndListH[blockIdx];
    tailRowStartH = tilingData->tailRowStartListH[blockIdx];
    tailRowEndH = tilingData->tailRowEndListH[blockIdx];

    dataType = tilingData->dataType;

    matmulTilingW = &tilingData->matmulTilingW;
    matmulTilingH = &tilingData->matmulTilingH;
}
}  // namespace UpsampleBilinear2dAABackward

#endif  // UPSAMPLE_BILINEAR2D_AA_BACKWARD

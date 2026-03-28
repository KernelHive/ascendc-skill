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
 * \file upsample_bicubic2d_grad_dc.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D_GRAD_DC
#define UPSAMPLE_BICUBIC2D_GRAD_DC

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2dGrad {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t NUMBER_TWO = 2;
constexpr int32_t NUMBER_THREE = 3;
constexpr int32_t NUMBER_FOUR = 4;
constexpr int32_t NUMBER_SIX = 6;

template <typename T>
class UpsampleBicubic2dGradDCND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulH;
    __aicore__ inline UpsampleBicubic2dGradDCND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcWeights(float (&weights)[4], float tValue)
    {
        float x1 = tValue; // tValue 为当前中心点偏移值，x1为左侧点偏移值
        weights[0] = CalcWeight1(x1 + 1);
        weights[1] = CalcWeight2(x1);
        float x2 = 1 - tValue; // tValue 为当前中心点偏移值，x2为右侧点偏移值
        weights[NUMBER_TWO] = CalcWeight2(x2);
        weights[NUMBER_THREE] = CalcWeight1(x2 + 1); // x2为右侧点偏移值，计算第二个点偏移值
    };
    // 计算weight,可将a替换为固定值
    __aicore__ inline float CalcWeight1(float x)
    {
        constexpr float COEFFICIENT_1 = -0.75f;
        constexpr float COEFFICIENT_2 = 3.75f;
        return ((x * COEFFICIENT_1 + COEFFICIENT_2) * x - static_cast<float>(NUMBER_SIX)) * x + static_cast<float>(NUMBER_THREE);
    };
    __aicore__ inline float CalcWeight2(float x)
    {
        constexpr float COEFFICIENT_1 = 1.25f;
        constexpr float COEFFICIENT_2 = 2.25f;
        return (x * COEFFICIENT_1 - COEFFICIENT_2) * x * x + 1.0f;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    template <typename T1>
    __aicore__ inline int64_t Ceil(T1 x)
    {
        if (x < 0) {
            x = x - 1;
        }
        int64_t floor_v = int64_t(x);
        return x == floor_v ? floor_v : (floor_v + 1);
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 x, T1 y)
    {
        return x >= y ? x : y;
    };
    __aicore__ inline void getQueueSize();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void ComputeIndexValue(int64_t index, int64_t length, int64_t instartIdx, int64_t maxIdx);
    __aicore__ inline void CalculateIntermediateTensor(
        int64_t slideStartW, int64_t slideEndW, int64_t instartIdx, int64_t maxIdx, float scale);
    __aicore__ inline int64_t CalculateInstartIdx(int64_t startIdx, float scale);
    __aicore__ inline void ParseTilingData(UpsampleBicubic2dGradTilingData *tilingData);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline __gm__ T *GetTensorAddr(int64_t index, GM_ADDR tensorPtr);
    __aicore__ inline void CalculateRadioTensor(int64_t index, int64_t length, int64_t direction);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm();
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;

    // 系数矩阵下标队列,横轴和纵轴范围
    TBuf<QuePosition::VECCALC> centerQueue;
    TBuf<QuePosition::VECCALC> xQueue;
    TBuf<QuePosition::VECCALC> tQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioCastQueue;

    const TCubeTiling *__restrict matmulTilingW;
    const TCubeTiling *__restrict matmulTilingH;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xTensor;
    LocalTensor<float> tTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slideSize = 0;

    int64_t alignCorners = 0;
    float scaleW;
    float scaleH;

    uint64_t intermediateMatrixSize = 16;
    uint32_t radioMatrixSize;
    // 切分块在原系数矩阵中的位置
    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    // 系数矩阵切块的宽度
    int64_t slidelen;
    int64_t slidelenH;
    int64_t queueSize = 0;

    int64_t slideStartH;
    int64_t slideEndH;
    int64_t tailSlideStartH;
    int64_t tailSlideEndH;
    int64_t tailRowStartH;
    int64_t tailRowEndH;
    int64_t dataType;

    float zeroScaleW = 0;
    float zeroScaleH = 0;
    int64_t inputShapes[4] = {0, 0, 0, 0};
    int64_t outputShapes[4] = {0, 0, 0, 0};

    uint32_t needCoreNumW;
    uint32_t needCoreNumH;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t instartW = 0;
    int64_t instartH = 0;

    int64_t instartIndex = 0;
    int64_t inendIndex = 0;

    int32_t singleCoreKH = 0;

    bool needExpandW = false;
    bool needExpandH = false;
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::getQueueSize()
{
    // 输入切块的长度
    int64_t wSlideSize = slideEndW - slideStartW;
    if (tailSlideEndW - tailSlideStartW > wSlideSize) {
        wSlideSize = tailSlideEndW - tailSlideStartW;
    }
    int64_t hSlideSize = slideEndH - slideStartH;
    if (tailSlideEndH - tailSlideStartH > hSlideSize) {
        hSlideSize = tailSlideEndH - tailSlideStartH;
    }

    zeroScaleW = inputShapes[3] > 0 ? scaleW : 1;
    zeroScaleH = inputShapes[2] > 0 ? scaleH : 1;

    int64_t inSlideW, inSlideH;

    inSlideW = scaleW > 0 ? static_cast<int64_t>((wSlideSize + NUMBER_FOUR) / scaleW)
                          : static_cast<int64_t>((wSlideSize + NUMBER_FOUR) / zeroScaleW);
    inSlideH = scaleH > 0 ? static_cast<int64_t>((hSlideSize + NUMBER_FOUR) / scaleH)
                          : static_cast<int64_t>((hSlideSize + NUMBER_FOUR) / zeroScaleH);

    if (inSlideW > inSlideH) {
        queueSize = inSlideW + 1;
    } else {
        queueSize = inSlideH + 1;
    }
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::WDirectionExpansion()
{
    if (blockIdx < needCoreNumW) {
        centerTensor = centerQueue.Get<float>();
        xTensor = xQueue.Get<float>();
        tTensor = tQueue.Get<float>();

        // 计算滑块映射范围
        if (slideStartW < slideEndW) {
            instartW = CalculateInstartIdx(slideStartW, scaleW > 0 ? scaleW : zeroScaleW);
            CalculateIntermediateTensor(slideStartW, slideEndW, instartW, outputShapes[3] - 1, scaleW);
            for (int64_t index = slideStartW; index < slideEndW; index += slideSize) {
                int64_t length = Min(slideSize, slideEndW - index);
                slidelen = length;
                CalculateRadioTensor(index, length, 0);
                copyRadioTensorToGm();
                calculateWidthExtension(index, 0, 0);
            }
        }
        if (tailRowStartW < tailRowEndW) {
            instartW = CalculateInstartIdx(tailSlideStartW, scaleW > 0 ? scaleW : zeroScaleW);
            CalculateIntermediateTensor(tailSlideStartW, tailSlideEndW, instartW, outputShapes[3] - 1, scaleW);
            for (int64_t index = tailSlideStartW; index < tailSlideEndW; index += slideSize) {
                int64_t length = Min(slideSize, tailSlideEndW - index);
                slidelen = length;
                CalculateRadioTensor(index, length, 0);
                copyRadioTensorToGm();
                calculateWidthExtension(index, tailRowStartW, tailRowEndW);
            }
        }
        // 处理尾块部分数据
        centerQueue.FreeTensor(centerTensor);
        xQueue.FreeTensor(xTensor);
        tQueue.FreeTensor(tTensor);
    }
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::HDirectionExpansion()
{
    if (blockIdx < needCoreNumH) {
        instartIndex = 0;
        inendIndex = 0;
        centerTensor = centerQueue.Get<float>();
        xTensor = xQueue.Get<float>();
        tTensor = tQueue.Get<float>();
        if (slideStartH < slideEndH) {
            instartH = CalculateInstartIdx(slideStartH, scaleH > 0 ? scaleH : zeroScaleH);
            CalculateIntermediateTensor(slideStartH, slideEndH, instartH, outputShapes[2] - 1, scaleH);
            for (int64_t index = slideStartH; index < slideEndH; index += slideSize) {
                int64_t length = Min(slideSize, slideEndH - index);
                slidelenH = length;
                CalculateRadioTensor(index, length, 1);
                copyRadioTensorToGm();
                calculateHeightExtension(index, 0, 0);
            }
        }
        if (tailRowStartH < tailRowEndH) {
            instartH = CalculateInstartIdx(tailSlideStartH, scaleH > 0 ? scaleH : zeroScaleH);
            CalculateIntermediateTensor(tailSlideStartH, tailSlideEndH, instartH, outputShapes[2] - 1, scaleH);
            for (int64_t index = tailSlideStartH; index < tailSlideEndH; index += slideSize) {
                int64_t length = Min(slideSize, tailSlideEndH - index);
                slidelenH = length;
                CalculateRadioTensor(index, length, 1);
                copyRadioTensorToGm();
                calculateHeightExtension(index, tailRowStartH, tailRowEndH);
            }
        }

        // 释放临时tensor
        centerQueue.FreeTensor(centerTensor);
        xQueue.FreeTensor(xTensor);
        tQueue.FreeTensor(tTensor);
    }
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dGradTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;
    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);

    getQueueSize();
    int64_t radioSize = radioMatrixSize;

    pipe.InitBuffer(centerQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(tQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioQueue, NO_BUFFER_NUM, (radioSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioCastQueue, NO_BUFFER_NUM, (radioSize * sizeof(T) + 31) / 32 * 32);

    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorsPtr);
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }

    // 先横向扩展
    if (needExpandW) {
        WDirectionExpansion();
    }

    SyncAll();

    // 再纵向扩展
    if (needExpandH || !needExpandW) {
        HDirectionExpansion();
    }
}

template <typename T>
__aicore__ inline int64_t UpsampleBicubic2dGradDCND<T>::CalculateInstartIdx(int64_t startIdx, float scale)
{
    if (alignCorners) {
        return Max(Ceil((startIdx - NUMBER_TWO) / scale), (int64_t)0);
    } else {
        return Max(Ceil((startIdx - float(1.5)) / scale - (float)0.5), (int64_t)0);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::CalculateIntermediateTensor(
    int64_t slideStart, int64_t slideEnd, int64_t instartIdx, int64_t maxIdx, float scale)
{
    int64_t length = static_cast<int64_t>(centerTensor.GetSize());
    // 使用标量计算中心点坐标，和cpu保持一致
    for (int64_t i = instartIdx; i < centerTensor.GetSize() + instartIdx; i++) {
        float value;
        if (alignCorners) {
            value = float(i) * scale;
        } else {
            value = (float(i) + float(0.5)) * scale - float(0.5);
        }
        centerTensor.SetValue(i - instartIdx, value);
    }
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);

    Floor(xTensor, centerTensor, length);
    PipeBarrier<PIPE_V>();

    Mins(xTensor, xTensor, (float)(maxIdx), length);
    PipeBarrier<PIPE_V>();

    Sub(tTensor, centerTensor, xTensor, length);
    PipeBarrier<PIPE_V>();

    Mins(tTensor, tTensor, (float)1, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::ComputeIndexValue(
    int64_t index, int64_t length, int64_t instartIdx, int64_t maxIdx)
{
    instartIndex = 0;
    inendIndex = 0;
    for (; instartIndex < xTensor.GetSize(); instartIndex++) {
        int64_t xmax = xTensor.GetValue(instartIndex) + NUMBER_TWO;
        if (xmax >= index) {
            break;
        }
    }

    for (inendIndex = instartIndex; inendIndex < xTensor.GetSize(); inendIndex++) {
        if (xTensor.GetValue(inendIndex) > index + length) {
            break;
        } else if (inendIndex + instartIdx > maxIdx) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::CalculateRadioTensor(int64_t index, int64_t length, int64_t direction)
{
    LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
    // 初始化为0
    Duplicate(radioTensor, float(0.0), radioTensor.GetSize());
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    // 计算影响该块的原始矩阵点的下标
    if (direction == 0){
        ComputeIndexValue(index, length, instartW, inputShapes[3] - 1);
        singleCoreK = inendIndex - instartIndex;
    } else {
        ComputeIndexValue(index, length, instartH, inputShapes[2] - 1);
        singleCoreKH = inendIndex - instartIndex;
    }

    float weights[4] = {0};
    
    for (int64_t i = instartIndex; i < inendIndex; i++) {
        int64_t xIdx = xTensor.GetValue(i) - index;

        CalcWeights(weights, tTensor.GetValue(i));

        // 计算当前参数关联的系数矩阵行数
        int64_t idxFirst = i - instartIndex;
        for (int64_t j = 0; j < NUMBER_FOUR; j++) {
            // 当前须赋予权重的点坐标
            int64_t idxSecond = xIdx - 1 + j;

            // 当前权重点不在输出块范围内时，进行处理
            if (idxSecond < 0) {
                if (index != 0) {
                    continue;
                }
                idxSecond = 0;
            } else if (idxSecond >= length) {
                if (index + length != (direction == 0 ? outputShapes[3] : outputShapes[2])) {
                    continue;
                }
                idxSecond = length - 1;
            }

            int64_t realIdx = direction == 0 ? (idxFirst * length + idxSecond) : (idxSecond * singleCoreKH + idxFirst);
            radioTensor.SetValue(realIdx, radioTensor.GetValue(realIdx) + weights[j]);
        }
    }

    if (dataType != 0) {
        LocalTensor<T> radioCastTensor = radioCastQueue.AllocTensor<T>();
        Cast(radioCastTensor, radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
        radioCastQueue.EnQue(radioCastTensor);
        radioQueue.FreeTensor(radioTensor);
    } else {
        radioQueue.EnQue(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::copyRadioTensorToGm()
{
    int64_t radioSize = radioMatrixSize;
    workSpaceRadioOffset = intermediateMatrixSize + radioSize * blockIdx;
    int8_t size = 32 / sizeof(T);
    if (dataType == 0) {
        LocalTensor<T> radioTensor = radioQueue.DeQue<T>();

        DataCopy(
            intermediateTensorGm[workSpaceRadioOffset], radioTensor, (radioTensor.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        radioQueue.FreeTensor(radioTensor);
    } else {
        LocalTensor<T> radioCastTensor = radioCastQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceRadioOffset],
            radioCastTensor,
            (radioCastTensor.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        radioCastQueue.FreeTensor(radioCastTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTilingW->singleCoreM;
    int64_t singleCoreN = matmulTilingW->singleCoreN;

    if (singleCoreK == 0) {
        singleCoreK++;
    }

    if (tensorCIndex + slideSize > outputShapes[3]) {
        singleCoreN = slidelen;
    }

    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }

    matmulW.SetOrgShape(singleCoreM, singleCoreN, inputShapes[3], singleCoreK, outputShapes[3]);

    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slideSize > outputShapes[3] - 1) {
        matmulW.SetTail(singleCoreM, outputShapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = instartIndex + instartW + rowStart * inputShapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * outputShapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);

    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);

    if (!needExpandH) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }
    matmulW.End();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::calculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTilingH->singleCoreM;
    int64_t singleCoreN = matmulTilingH->singleCoreN;

    if (singleCoreKH == 0) {
        singleCoreKH++;
    }
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreN = rowEnd - rowStart;
    }

    if (tensorCIndex + slideSize > outputShapes[2]) {
        singleCoreM = outputShapes[2] - tensorCIndex;
    }
    matmulH.SetOrgShape(singleCoreM, outputShapes[3], singleCoreKH, outputShapes[2], outputShapes[3]);

    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreKH);

    if (tensorCIndex + slideSize > outputShapes[2] - 1) {
        matmulH.SetTail(outputShapes[2] - tensorCIndex, singleCoreN, singleCoreKH);
    }

    int64_t xIndex = (instartIndex + instartH) * outputShapes[3] + rowStart;

    int64_t tensorCIndexWithOffset = tensorCIndex * outputShapes[3] + rowStart;

    for (int i = 0; i < outputShapes[0] * outputShapes[1]; i++) {
        // 系数矩阵起始位置
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);

        if (!needExpandW) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * inputShapes[2] * outputShapes[3]], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * inputShapes[2] * outputShapes[3]], false);
        }

        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * outputShapes[2] * outputShapes[3]], false);
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::ParseTilingData(UpsampleBicubic2dGradTilingData *tilingData)
{
    slideSize = tilingData->slideSize;
    scaleW = tilingData->scalesW;
    scaleH = tilingData->scalesH;
    alignCorners = tilingData->alignCorners;

    needCoreNumW = tilingData->CoreNumW;
    needCoreNumH = tilingData->CoreNumH;

    needExpandW = tilingData->needExpandW == 1 ? true : false;
    needExpandH = tilingData->needExpandH == 1 ? true : false;

    outputShapes[0] = tilingData->inputN;
    outputShapes[1] = tilingData->inputC;
    outputShapes[2] = tilingData->outputH;
    outputShapes[3] = tilingData->outputW;
    inputShapes[0] = tilingData->inputN;
    inputShapes[1] = tilingData->inputC;
    inputShapes[2] = tilingData->inputH;
    inputShapes[3] = tilingData->inputW;

    intermediateMatrixSize = tilingData->intermediateMatrixSize;
    radioMatrixSize = tilingData->radioMatrixSize;
    slideStartW = tilingData->slideStartListW[blockIdx];
    slideEndW = tilingData->slideEndListW[blockIdx];

    tailSlideStartW = tilingData->tailStartW;
    tailSlideEndW = tilingData->tailEndW;

    tailRowStartW = tilingData->tailSlideStartListW[blockIdx];
    tailRowEndW = tilingData->tailSlideEndListW[blockIdx];

    slideStartH = tilingData->slideStartListH[blockIdx];
    slideEndH = tilingData->slideEndListH[blockIdx];

    tailSlideStartH = tilingData->tailStartH;
    tailSlideEndH = tilingData->tailEndH;

    tailRowStartH = tilingData->tailSlideStartListH[blockIdx];
    tailRowEndH = tilingData->tailSlideEndListH[blockIdx];

    dataType = tilingData->dataType;

    matmulTilingW = &tilingData->MMParamW;
    matmulTilingH = &tilingData->MMParamH;
}

}  // namespace UpsampleBicubic2dGrad
#endif
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
 * \file upsample_bilinear2d_grad.h
 * \brief
 */
#ifndef UPSAMPLE_BILINEAR2D_GRAD
#define UPSAMPLE_BILINEAR2D_GRAD
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpSampleBilinear2dGrad {
using namespace AscendC;
constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);
constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t RESERVED_VALUE = 4;
constexpr float RESERVED_SCALE = 1.5;

template <typename T>
class UpSampleBilinear2dGradND {
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
    __aicore__ inline UpSampleBilinear2dGradND(){};
    __aicore__ inline void calculateIntermediateTensorX(LocalTensor<float> centerTensor,
        LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor, int64_t slideStart_w, int64_t slideEnd_w);
    __aicore__ inline void calculateIntermediateTensorY(LocalTensor<float> centerTensor,
        LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor, int64_t slideStart_h, int64_t slideEnd_h);
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBilinear2dGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
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
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };

    template <typename T1>
    __aicore__ inline T1 getMax(T1 x, T1 y)
    {
        if (x >= y) {
            return x;
        } else {
            return y;
        }
    }
    template <typename T1>
    __aicore__ inline T1 getMin(T1 x, T1 y)
    {
        if (x >= y) {
            return y;
        } else {
            return x;
        }
    }
    __aicore__ inline void setRadioValueW(LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor,
        LocalTensor<float> centerTensor, int64_t index, int64_t length);

    __aicore__ inline void setRadioValueH(LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor,
        LocalTensor<float> centerTensor, int64_t index, int64_t length);
    __aicore__ inline void setZeroRadioValue(LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor,
        LocalTensor<float> centerTensor, int64_t index, int64_t length);
    __aicore__ inline void getQueueSize();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void ParseTilingData(const UpsampleBilinear2dGradTilingData *tilingData);
    __aicore__ inline void calculateRadioTensor(LocalTensor<float> centerTensor, LocalTensor<float> xIndexTensor,
        LocalTensor<float> xLambdaTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorH(LocalTensor<float> centerTensor, LocalTensor<float> xIndexTensor,
        LocalTensor<float> xLambdaTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm(int64_t length);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

private:
    // 系数矩阵下标队列,横轴和纵轴范围
    TBuf<QuePosition::VECCALC> centerQueue;
    TBuf<QuePosition::VECCALC> xIndexQueue;
    TBuf<QuePosition::VECCALC> xLambdaQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> radioCastQueue;

    const TCubeTiling *__restrict matmulTiling_w;
    const TCubeTiling *__restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slide_size = 0;
    int64_t radioSize = 0;
    float scaleW;
    float scaleH;

    uint64_t intermediate_matrix_size = 16;
    uint32_t radio_matrix_size;
    uint32_t radio_matrix_size_h;
    // 切分块在原系数矩阵中的位置
    int64_t slideStart_w;
    int64_t slideEnd_w;
    int64_t tailSlideStart_w;
    int64_t tailSlideEnd_w;
    int64_t tailRowStart_w;
    int64_t tailRowEnd_w;

    // 系数矩阵切块的宽度
    int64_t queueSize = 0;
    int64_t cubeSize = 0;
    int64_t middleSize = 0;

    int64_t slideStart_h;
    int64_t slideEnd_h;
    int64_t tailSlideStart_h;
    int64_t tailSlideEnd_h;
    int64_t tailRowStart_h;
    int64_t tailRowEnd_h;
    int64_t dataType;

    float zeroScaleW = 0;
    float zeroScaleH = 0;
    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t workSpaceRadioOffset = 0;
    int64_t workSpaceLineOffset = 0;
    int64_t singleCoreK = 0;
    int64_t instart_w = 0;
    int64_t instart_h = 0;

    int64_t xMin = 0;
    int64_t instartIndex = 0;
    int64_t inendIndex = 0;
    int64_t align_corners = 0;
    int32_t singleCoreK_h = 0;
    int32_t tailBatchStart_h = 0;
    int32_t tailBatchEnd_h = 0;
    bool needExpendX = false;
    bool needExpendY = false;
};

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBilinear2dGradTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;

    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);
    middleSize = input_shapes[2] * output_shapes[3];
    cubeSize = output_shapes[2] * output_shapes[3];
    needExpendX = !FloatEqual(scaleW, 1.0);
    needExpendY = !FloatEqual(scaleH, 1.0);
    getQueueSize();
    radioSize = getMax(slide_size, queueSize);

    radio_matrix_size = getMax(radio_matrix_size, radio_matrix_size_h);

    int64_t a = (queueSize * sizeof(float) + 31) / 32 * 32;
    int64_t b = sizeof(float);
    pipe.InitBuffer(centerQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xIndexQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xLambdaQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioQueue, NO_BUFFER_NUM, (radioSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioCastQueue, NO_BUFFER_NUM, (radioSize * sizeof(T) + 31) / 32 * 32);

    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorsPtr);
};

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }

    // 先横向扩展
    if (needExpendX) {
        WDirectionExpansion();
    }

    SyncAll();

    // 再纵向扩展
    if (needExpendY || !needExpendX) {
        HDirectionExpansion();
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::getQueueSize()
{
    // 输入切块的长度
    int64_t xSlideSize = slideEnd_w - slideStart_w;
    if (tailSlideEnd_w - tailSlideStart_w > xSlideSize) {
        xSlideSize = tailSlideEnd_w - tailSlideStart_w;
    }
    int64_t ySlideSize = slideEnd_h - slideStart_h;

    if (tailSlideEnd_h - tailSlideStart_h > ySlideSize) {
        ySlideSize = tailSlideEnd_h - tailSlideStart_h;
    }

    zeroScaleW = input_shapes[3] > 0 ? static_cast<float>(output_shapes[3]) / input_shapes[3] : 1;
    zeroScaleH = input_shapes[2] > 0 ? static_cast<float>(output_shapes[2]) / input_shapes[2] : 1;

    int64_t inSlide_w = scaleW > 0 ? static_cast<int64_t>((slide_size + 4) / scaleW) + 2 * RESERVED_VALUE
                                    : static_cast<int64_t>((xSlideSize + 4) / zeroScaleW) + 2 * RESERVED_VALUE;
    int64_t inSlide_h = scaleH > 0 ? static_cast<int64_t>((slide_size + 4) / scaleH) + 2 * RESERVED_VALUE
                                    : static_cast<int64_t>((ySlideSize + 4) / zeroScaleH) + 2 * RESERVED_VALUE;

    if (inSlide_w > inSlide_h) {
        queueSize = inSlide_w;
    } else {
        queueSize = inSlide_h;
    }
};

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::WDirectionExpansion()
{
    if (blockIdx < need_core_num_w) {
        LocalTensor<float> centerTensor = centerQueue.Get<float>();
        LocalTensor<float> xIndexTensor = xIndexQueue.Get<float>();
        LocalTensor<float> xLambdaTensor = xLambdaQueue.Get<float>();

        // 计算滑块映射范围
        if (slideStart_w < slideEnd_w) {
            workSpaceRadioOffset = intermediate_matrix_size + radio_matrix_size * blockIdx;
            for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, output_shapes[3] - index);
                calculateIntermediateTensorX(centerTensor, xIndexTensor, xLambdaTensor, index, index + length);
                calculateRadioTensor(centerTensor, xIndexTensor, xLambdaTensor, index, length);
                calculateWidthExtension(index, 0, 0);
            }
        }
        if (tailSlideStart_w < tailSlideEnd_w) {
            workSpaceRadioOffset = intermediate_matrix_size + radio_matrix_size * blockIdx;
            for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, output_shapes[3] - index);
                calculateIntermediateTensorX(centerTensor, xIndexTensor, xLambdaTensor, index, index + length);
                calculateRadioTensor(centerTensor, xIndexTensor, xLambdaTensor, index, length);
                calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
            }
        }
        // 处理尾块部分数据
        centerQueue.FreeTensor(centerTensor);
        xIndexQueue.FreeTensor(xIndexTensor);
        xLambdaQueue.FreeTensor(xLambdaTensor);
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::HDirectionExpansion()
{
    if (blockIdx < need_core_num_h) {
        instartIndex = 0;
        inendIndex = 0;
        LocalTensor<float> centerTensor = centerQueue.Get<float>();
        LocalTensor<float> xIndexTensor = xIndexQueue.Get<float>();
        LocalTensor<float> xLambdaTensor = xLambdaQueue.Get<float>();

        if (slideStart_h < slideEnd_h) {
            workSpaceRadioOffset = intermediate_matrix_size + radio_matrix_size * blockIdx;
            for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, output_shapes[2] - index);
                calculateIntermediateTensorY(centerTensor, xIndexTensor, xLambdaTensor, index, index + length);
                calculateRadioTensorH(centerTensor, xIndexTensor, xLambdaTensor, index, length);
                calculateHeightExtension(index, 0, 0);
            }
        }

        if (tailSlideStart_h < tailSlideEnd_h) {
            workSpaceRadioOffset = intermediate_matrix_size + radio_matrix_size * blockIdx;
            for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, output_shapes[2] - index);
                calculateIntermediateTensorY(centerTensor, xIndexTensor, xLambdaTensor, index, index + length);
                calculateRadioTensorH(centerTensor, xIndexTensor, xLambdaTensor, index, length);
                calculateHeightExtension(index, tailBatchStart_h, tailBatchEnd_h);
            }
        }

        // 释放临时tensor
        centerQueue.FreeTensor(centerTensor);
        xIndexQueue.FreeTensor(xIndexTensor);
        xLambdaQueue.FreeTensor(xLambdaTensor);
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::calculateIntermediateTensorX(LocalTensor<float> centerTensor,
    LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor, int64_t slideStart_w, int64_t slideEnd_w)
{
    instart_w = scaleW > 0 ? static_cast<int64_t>((float)(slideStart_w - 2) / scaleW) - 1
                            : static_cast<int64_t>((float)(slideStart_w - 2) / zeroScaleW) - 1;

    if (instart_w < 0) {
        instart_w = 0;
    }

    int64_t length = static_cast<int64_t>(centerTensor.GetSize());
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor, static_cast<float>(instart_w), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    if (align_corners == 1) {
        Muls(centerTensor, centerTensor, scaleW, length);
        PipeBarrier<PIPE_V>();
    } else {
        for (int64_t i = 0; i < length; i++) {
            float centerValue = static_cast<float>(
                (centerTensor.GetValue(i) + static_cast<float>(0.5)) * scaleW - static_cast<float>(0.5));
            centerTensor.SetValue(i, centerValue);
        }
        PipeBarrier<PIPE_V>();
        // 中心点不能小于0
        Maxs(centerTensor, centerTensor, (float)0.0, length);
        PipeBarrier<PIPE_V>();
    }
    // 计算最近点下标
    Floor(xIndexTensor, centerTensor, length);
    PipeBarrier<PIPE_V>();

    // 计算权重
    Sub(xLambdaTensor, centerTensor, xIndexTensor, length);
    PipeBarrier<PIPE_V>();
    Duplicate(centerTensor, float(1.0), length);
    PipeBarrier<PIPE_V>();
    Sub(centerTensor, centerTensor, xLambdaTensor, length);
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::calculateIntermediateTensorY(LocalTensor<float> centerTensor_h,
    LocalTensor<float> xIndexTensor_h, LocalTensor<float> xLambdaTensor_h, int64_t slideStart_h, int64_t slideEnd_h)
{
    instart_h = scaleH > 0 ? static_cast<int64_t>((float)(slideStart_h - 2) / scaleH) - 1
                            : static_cast<int64_t>((float)(slideStart_h - 2) / zeroScaleH) - 1;

    if (instart_h < 0) {
        instart_h = 0;
    }

    int64_t length = static_cast<int64_t>(centerTensor_h.GetSize());
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor_h, static_cast<float>(instart_h), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    if (align_corners == 1) {
        Muls(centerTensor_h, centerTensor_h, scaleH, length);
        PipeBarrier<PIPE_V>();
    } else {
        for (int64_t i = 0; i < length; i++) {
            float centerValue = static_cast<float>(
                (centerTensor_h.GetValue(i) + static_cast<float>(0.5)) * scaleH - static_cast<float>(0.5));
            centerTensor_h.SetValue(i, centerValue);
        }
        PipeBarrier<PIPE_V>();
        // 中心点不能小于0
        Maxs(centerTensor_h, centerTensor_h, (float)0.0, length);
        PipeBarrier<PIPE_V>();
    }

    // 计算最近点下标
    Floor(xIndexTensor_h, centerTensor_h, length);
    PipeBarrier<PIPE_V>();

    // 计算权重
    Sub(xLambdaTensor_h, centerTensor_h, xIndexTensor_h, length);
    PipeBarrier<PIPE_V>();
    Duplicate(centerTensor_h, float(1.0), length);
    PipeBarrier<PIPE_V>();
    Sub(centerTensor_h, centerTensor_h, xLambdaTensor_h, length);
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::setRadioValueW(LocalTensor<float> xIndexTensor,
    LocalTensor<float> xLambdaTensor, LocalTensor<float> centerTensor, int64_t index, int64_t length)
{
    workSpaceLineOffset = workSpaceRadioOffset;
    for (int64_t i = instartIndex; i < inendIndex; i++) {
        LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
        Duplicate(radioTensor, float(0.0), radioTensor.GetSize());
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        float floorIndex = xIndexTensor.GetValue(i);
        float upIndex = xIndexTensor.GetValue(i) + 1;
        int64_t index_w = 0;
        if (floorIndex < index && upIndex >= index) {
            index_w = static_cast<int64_t>(upIndex) - index;
            float value = xLambdaTensor.GetValue(i);
            radioTensor.SetValue(index_w, value);
        } else if (upIndex >= index + length && floorIndex < index + length && floorIndex >= index) {
            if (floorIndex == output_shapes[3] - 1) {
                index_w = static_cast<int64_t>(floorIndex) - index;
                float value = static_cast<float>(1.0);
                radioTensor.SetValue(index_w, value);
            } else {
                index_w = static_cast<int64_t>(floorIndex) - index;
                float value = centerTensor.GetValue(i);
                radioTensor.SetValue(index_w, value);
            }
        } else {
            index_w = static_cast<int64_t>(floorIndex) - index;
            if (index_w >= 0) {
                float value1 = xLambdaTensor.GetValue(i);
                float value2 = centerTensor.GetValue(i);
                radioTensor.SetValue(index_w, value2);
                radioTensor.SetValue(index_w + 1, value1);
            }
        }

        if (dataType != 2) {
            LocalTensor<T> radioCastTensor_w = radioCastQueue.AllocTensor<T>();
            Cast(radioCastTensor_w, radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
            radioCastQueue.EnQue(radioCastTensor_w);
            radioQueue.FreeTensor(radioTensor);
        } else {
            radioQueue.EnQue(radioTensor);
        }
        copyRadioTensorToGm(length);
    }
}
template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::calculateRadioTensor(LocalTensor<float> centerTensor,
    LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor, int64_t index, int64_t length)
{
    for (int64_t i = 0; i < xIndexTensor.GetSize(); i++) {
        float downValue = xIndexTensor.GetValue(i);
        float upValue = downValue + 1;
        if ((upValue >= index && upValue < index + length) || (downValue >= index && downValue < index + length)) {
            instartIndex = i;
            break;
        }
    }
    inendIndex = instartIndex;
    for (int64_t i = instartIndex; i < xIndexTensor.GetSize(); i++) {
        float floorValue = xIndexTensor.GetValue(i);

        if (floorValue >= index + length) {
            inendIndex = i;
            break;
        }
        if (i + instart_w > input_shapes[3] - 1) {
            inendIndex = i;
            break;
        }
    }
    singleCoreK = inendIndex - instartIndex;
    if (singleCoreK == 0) {
        setZeroRadioValue(xIndexTensor, xLambdaTensor, centerTensor, index, length);
    }
    if (instartIndex + instart_w < input_shapes[3]) {
        setRadioValueW(xIndexTensor, xLambdaTensor, centerTensor, index, length);
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::setRadioValueH(LocalTensor<float> xIndexTensor,
    LocalTensor<float> xLambdaTensor, LocalTensor<float> centerTensor, int64_t index, int64_t length)
{
    workSpaceLineOffset = workSpaceRadioOffset;
    for (int64_t j = index; j < index + length; j++) {
        LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
        // 初始化为0
        Duplicate(radioTensor, float(0.0), radioTensor.GetSize());
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        for (int64_t i = instartIndex; i < inendIndex; i++) {
            float floorIndex = xIndexTensor.GetValue(i);
            float upIndex = xIndexTensor.GetValue(i) + 1;
            if (floorIndex == j) {
                float value = centerTensor.GetValue(i);
                if (floorIndex >= output_shapes[2] - 1) {
                    value = static_cast<float>(1.0);
                }
                radioTensor.SetValue(i - instartIndex, value);
            } else if (upIndex == j) {
                float value = xLambdaTensor.GetValue(i);
                radioTensor.SetValue(i - instartIndex, value);
            }
        }
        if (dataType != 2) {
            LocalTensor<T> radioCastTensor_h = radioCastQueue.AllocTensor<T>();
            Cast(radioCastTensor_h, radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
            radioCastQueue.EnQue(radioCastTensor_h);
            radioQueue.FreeTensor(radioTensor);
        } else {
            radioQueue.EnQue(radioTensor);
        }
        copyRadioTensorToGm(singleCoreK_h);
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::setZeroRadioValue(LocalTensor<float> xIndexTensor,
    LocalTensor<float> xLambdaTensor, LocalTensor<float> centerTensor, int64_t index, int64_t length)
{
    workSpaceLineOffset = workSpaceRadioOffset;

    for (int64_t j = index; j < index + length; j++) {
        LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
        // 初始化为0
        Duplicate(radioTensor, float(0.0), radioTensor.GetSize());
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        if (dataType != 2) {
            LocalTensor<T> radioCastTensor_h = radioCastQueue.AllocTensor<T>();
            Cast(radioCastTensor_h, radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
            radioCastQueue.EnQue(radioCastTensor_h);
            radioQueue.FreeTensor(radioTensor);
        } else {
            radioQueue.EnQue(radioTensor);
        }

        copyRadioTensorToGm(radioSize);
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::calculateRadioTensorH(LocalTensor<float> centerTensor,
    LocalTensor<float> xIndexTensor, LocalTensor<float> xLambdaTensor, int64_t index, int64_t length)
{
    // 计算影响该块的原始矩阵点的下标
    for (int64_t i = 0; i < xIndexTensor.GetSize(); i++) {
        float downValue = xIndexTensor.GetValue(i);
        float upValue = xIndexTensor.GetValue(i) + 1;
        if ((upValue >= index && upValue < index + length) || (downValue >= index && downValue < index + length)) {
            instartIndex = i;
            break;
        }
    }
    inendIndex = instartIndex;
    for (int64_t i = instartIndex; i < xIndexTensor.GetSize(); i++) {
        float floorValue = xIndexTensor.GetValue(i);
        if (i + instart_h > input_shapes[2] - 1) {
            inendIndex = i;
            break;
        }
        if (floorValue >= index + length) {
            inendIndex = i;
            break;
        }
    }
    singleCoreK_h = inendIndex - instartIndex;
    if (singleCoreK_h == 0) {
        setZeroRadioValue(xIndexTensor, xLambdaTensor, centerTensor, index, length);
    } else if (instartIndex + instart_h < input_shapes[2]) {
        setRadioValueH(xIndexTensor, xLambdaTensor, centerTensor, index, length);
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::copyRadioTensorToGm(int64_t length)
{
    int8_t size = 32 / sizeof(T);

    if (dataType == 2) {
        LocalTensor<T> radioTensor = radioQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceLineOffset], radioTensor, (length + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        radioQueue.FreeTensor(radioTensor);
    } else {
        LocalTensor<T> radioCastTensor = radioCastQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceLineOffset], radioCastTensor, (length + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        radioCastQueue.FreeTensor(radioCastTensor);
    }
    workSpaceLineOffset += length;
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;

    int64_t offset = instartIndex + instart_w;

    if (offset >= input_shapes[3]) {
        offset = 0;
    }
    if (singleCoreK == 0) {
        singleCoreK++;
    }
    if (tensorCIndex + slide_size < output_shapes[3]) {
        singleCoreN = slide_size;
    } else {
        singleCoreN = output_shapes[3] - tensorCIndex;
    }

    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);

    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[3] - 1) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = offset + rowStart * input_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * output_shapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);

    if (!needExpendY) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }

    matmulW.End();
}
template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::calculateHeightExtension(
    int64_t tensorCIndex, int64_t batchStart, int64_t batchEnd)
{
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    int64_t singleCoreN = matmulTiling_h->singleCoreN;
    int64_t offset = instartIndex + instart_h;
    if (offset >= input_shapes[2]) {
        offset = 0;
    }
    if (singleCoreK_h == 0) {
        singleCoreK_h++;
    }

    // 尾块batch分批处理
    singleCoreN = output_shapes[3];
    if (batchEnd == 0) {
        batchEnd = input_shapes[0] * input_shapes[1];
    }

    if (tensorCIndex + slide_size >= output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    } else {
        singleCoreM = slide_size;
    }
    matmulH.SetOrgShape(singleCoreM, output_shapes[3], singleCoreK_h, output_shapes[2], output_shapes[3]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK_h);
    if (tensorCIndex + slide_size > output_shapes[2] - 1) {
        matmulH.SetTail(output_shapes[2] - tensorCIndex, singleCoreN, singleCoreK_h);
    }

    int64_t xIndex = offset * output_shapes[3] + batchStart * middleSize;
    int64_t tensorCIndexWithOffset = tensorCIndex * output_shapes[3] + batchStart * cubeSize;
    matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
    for (int64_t i = batchStart; i < batchEnd; i++) {
        // 系数矩阵起始位置
        if (!needExpendX) {
            matmulH.SetTensorB(inTensorsGM[xIndex], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
        xIndex += middleSize;
        tensorCIndexWithOffset += cubeSize;
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpSampleBilinear2dGradND<T>::ParseTilingData(const UpsampleBilinear2dGradTilingData *tilingData)
{
    slide_size = tilingData->slide_size;
    scaleW = tilingData->scale_w;
    scaleH = tilingData->scale_h;
    align_corners = tilingData->align_corners;

    need_core_num_w = tilingData->need_core_num_w;
    need_core_num_h = tilingData->need_core_num_h;

    for (int8_t i = 0; i < 4; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
    }
    for (int8_t i = 0; i < 4; i++) {
        input_shapes[i] = tilingData->input_shapes[i];
    }

    intermediate_matrix_size = tilingData->intermediate_matrix_size;
    radio_matrix_size = tilingData->radio_matrix_size;
    radio_matrix_size_h = tilingData->radio_matrix_size_h;

    slideStart_w = tilingData->slideStartList_w[blockIdx];
    slideEnd_w = tilingData->slideEndList_w[blockIdx];
    tailSlideStart_w = tilingData->tailSlideStartList_w[blockIdx];
    tailSlideEnd_w = tilingData->tailSlideEndList_w[blockIdx];
    tailRowStart_w = tilingData->tailRowStartList_w[blockIdx];
    tailRowEnd_w = tilingData->tailRowEndList_w[blockIdx];

    slideStart_h = tilingData->slideStartList_h[blockIdx];
    slideEnd_h = tilingData->slideEndList_h[blockIdx];
    tailSlideStart_h = tilingData->tailSlideStartList_h[blockIdx];
    tailSlideEnd_h = tilingData->tailSlideEndList_h[blockIdx];
    tailRowStart_h = tilingData->tailRowStartList_h[blockIdx];
    tailRowEnd_h = tilingData->tailRowEndList_h[blockIdx];
    tailBatchStart_h = tilingData->tailBatchStartList_h[blockIdx];
    tailBatchEnd_h = tilingData->tailBatchEndList_h[blockIdx];
    dataType = tilingData->dataType;

    matmulTiling_w = &tilingData->matmulTiling_w;
    matmulTiling_h = &tilingData->matmulTiling_h;
}

}  // namespace UpSampleBilinear2dGrad

#endif
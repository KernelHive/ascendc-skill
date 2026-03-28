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
 * \file upsample_bicubic2d.h
 * \brief
 */
#ifndef UPSAMPLE_BICUBIC2D
#define UPSAMPLE_BICUBIC2D

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_NUM = 16;
constexpr uint32_t ADDR_ALIGN_SIZE = 128;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr int8_t MIN_SIZE = 1;
constexpr int8_t TWO_SIZE = 2;

template <typename T>
class UpsampleBicubic2dND {
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

    __aicore__ inline UpsampleBicubic2dND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 weightCalculate(T1 x, int64_t i, int64_t j, int64_t width)
    {
        float weight1 = 0;
        float weight2 = 0;
        float weight3 = 0;
        float weight4 = 0;
        float t = (float)1.0 - x;
        switch (j) {
            case 0:
                weight1 = calWeights2(1 + x);
                weight2 = calWeights1(x);
                weight3 = calWeights1(t);
                return getWeightIndex0(i, width, weight1, weight2, weight3);
            case 1:
                weight2 = calWeights1(x);
                weight3 = calWeights1(t);
                weight4 = calWeights2(1 + t);
                return getWeightIndex1(i, width, weight2, weight3, weight4);
            case 2:
                weight3 = calWeights1(t);
                weight4 = calWeights2(1 + t);
                return getWeightIndex2(i, width, weight3, weight4);
            case 3:
                weight4 = calWeights2(1 + t);
                return getWeightIndex3(i, width, weight4);
            default:
                return 0.0;
        }
    };

    template <typename T1>
    __aicore__ inline T1 getWeightIndex0(int64_t i, int64_t width, T1 weight1, T1 weight2, T1 weight3)
    {
        if (width == MIN_SIZE) {
            return 1.0;
        } else if (i < 0) {
            return (weight1 + weight2 + weight3);
        } else if (i == 0) {
            return (weight1 + weight2);
        } else if (out_of_range(i, width)) {
            return weight1;
        } else if (on_board(i, width)) {
            return weight1;
        } else {
            return weight1;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex1(int64_t i, int64_t width, T1 weight2, T1 weight3, T1 weight4)
    {
        if (width == MIN_SIZE) {
            return 0.0;
        } else if (i < 0) {
            return weight4;
        } else if (i == 0) {
            return (width == TWO_SIZE) ? (weight3 + weight4) : weight3;
        } else if (out_of_range(i, width)) {
            return (weight2 + weight3 + weight4);
        } else if (on_board(i, width)) {
            return weight2;
        } else {
            return weight2;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex2(int64_t i, int64_t width, T1 weight3, T1 weight4)
    {
        if (width == MIN_SIZE || i < 0) {
            return 0.0;
        } else if (i == 0) {
            return (width == TWO_SIZE) ? static_cast<float>(0.0) : weight4;
        } else if (out_of_range(i, width)) {
            return 0.0;
        } else if (on_board(i, width)) {
            return (weight3 + weight4);
        } else {
            return weight3;
        }
    }

    template <typename T1>
    __aicore__ inline T1 getWeightIndex3(int64_t i, int64_t width, T1 weight4)
    {
        if (width == MIN_SIZE || i <= 0) {
            return 0.0;
        } else if (out_of_range(i, width) || on_board(i, width)) {
            return 0.0;
        } else {
            return weight4;
        }
    }

    template <typename T1>
    __aicore__ inline T1 calWeights1(T1 x)
    {
        float res = ((T1)1.25 * x - (T1)2.25) * x * x + (T1)1.0;
        return res;
    }

    template <typename T1>
    __aicore__ inline T1 calWeights2(T1 x)
    {
        float res = (((T1)-0.75 * x + (T1)3.75) * x - (T1)6.0) * x + (T1)3.0;
        return res;
    }

    __aicore__ inline bool out_of_range(int64_t x, int64_t width)
    {
        return x >= (width - MIN_SIZE);
    };

    __aicore__ inline bool on_board(int64_t x, int64_t width)
    {
        if (x >= (width - TWO_SIZE) && x < (width - MIN_SIZE)) {
            return true;
        } else {
            return false;
        }
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

    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };

    __aicore__ inline void ParseTilingData(UpsampleBicubic2dTilingData *tilingData);
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();

    __aicore__ inline void calculateIntermediateTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void calculateRatioTensorW(int64_t index, int64_t length);
    __aicore__ inline void calculateRatioTensorH(int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

    __aicore__ inline void copyRatioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRatioTensor(int8_t direction);
    __aicore__ inline void releaseRatioTensor(int8_t direction, LocalTensor<T> ratioTensor);
    __aicore__ inline int64_t getWidthTensorSize();
    __aicore__ inline int64_t getHeightTensorSize();

private:
    // 系数矩阵下标队列
    TBuf<QuePosition::VECCALC> centerQueue_w;
    TBuf<QuePosition::VECCALC> xIntQueue_w;
    TBuf<QuePosition::VECCALC> xMinQueue_w;
    TBuf<QuePosition::VECCALC> xVQueue_w;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ratioQueue_w;

    TBuf<QuePosition::VECCALC> centerQueue_h;
    TBuf<QuePosition::VECCALC> xIntQueue_h;
    TBuf<QuePosition::VECCALC> xMinQueue_h;
    TBuf<QuePosition::VECCALC> xVQueue_h;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ratioQueue_h;

    const TCubeTiling *__restrict matmulTiling_w;
    const TCubeTiling *__restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;
    LocalTensor<float> xIntTensor;
    LocalTensor<float> xVTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slide_size = 0;
    float scale_w;
    float scale_h;
    bool align_corners;
    int64_t max_interp_size_w = 16;
    int64_t max_interp_size_h;
    int64_t need_core_num_w;
    int64_t need_core_num_h;
    int64_t dataType;

    bool floatEqual_h;
    bool floatEqual_w;

    uint64_t intermediate_matrix_size;
    uint32_t ratio_matrix_size_w;
    uint32_t ratio_matrix_size_h;

    int64_t slideStart_w;
    int64_t slideEnd_w;
    int64_t tailSlideStart_w;
    int64_t tailSlideEnd_w;
    int64_t tailRowStart_w;
    int64_t tailRowEnd_w;

    int64_t slideStart_h;
    int64_t slideEnd_h;
    int64_t tailSlideStart_h;
    int64_t tailSlideEnd_h;
    int64_t tailRowStart_h;
    int64_t tailRowEnd_h;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    int64_t workSpaceRatioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;

    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);
    int64_t tensorWidthSize = getWidthTensorSize();
    int64_t tensorHeightSize = getHeightTensorSize();

    floatEqual_h = FloatEqual(scale_h, 1.0);
    floatEqual_w = FloatEqual(scale_w, 1.0);

    if (!floatEqual_w) {
        pipe.InitBuffer(centerQueue_w, tensorWidthSize);
        pipe.InitBuffer(xIntQueue_w, tensorWidthSize);
        pipe.InitBuffer(xMinQueue_w, tensorWidthSize);
        pipe.InitBuffer(xVQueue_w, tensorWidthSize);
        pipe.InitBuffer(ratioQueue_w, BUFFER_NUM, ratio_matrix_size_w * sizeof(float));
    }

    if (!floatEqual_h || floatEqual_w) {
        pipe.InitBuffer(centerQueue_h, tensorHeightSize);
        pipe.InitBuffer(xIntQueue_h, tensorHeightSize);
        pipe.InitBuffer(xMinQueue_h, tensorHeightSize);
        pipe.InitBuffer(xVQueue_h, tensorHeightSize);
        pipe.InitBuffer(ratioQueue_h, BUFFER_NUM, ratio_matrix_size_h * sizeof(float));
    }

    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorsPtr);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }

    // 先横向扩展
    WDirectionExpansion();

    SyncAll();

    // 再纵向扩展
    HDirectionExpansion();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::WDirectionExpansion()
{
    if (!floatEqual_w) {
        if (blockIdx < need_core_num_w) {
            centerTensor = centerQueue_w.Get<float>();
            xIntTensor = xIntQueue_w.Get<float>();
            xMinTensor = xMinQueue_w.Get<float>();
            xVTensor = xVQueue_w.Get<float>();

            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_w < slideEnd_w) {
                for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                    int16_t length = Min(slide_size, slideEnd_w - index);
                    calculateIntermediateTensor(index, length, W_DIRECTION);
                    // 计算系数矩阵
                    calculateRatioTensorW(0, length);
                    copyRatioTensorToGm(0);
                    calculateWidthExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_w < tailSlideEnd_w) {
                calculateIntermediateTensor(tailSlideStart_w, tailSlideEnd_w - tailSlideStart_w, W_DIRECTION);
                for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                    int16_t length = Min(slide_size, tailSlideEnd_w - index);
                    calculateRatioTensorW(0, length);
                    copyRatioTensorToGm(0);
                    calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::HDirectionExpansion()
{
    if (!floatEqual_h || floatEqual_w) {
        if (blockIdx < need_core_num_h) {
            centerTensor = centerQueue_h.Get<float>();
            xIntTensor = xIntQueue_h.Get<float>();
            xMinTensor = xMinQueue_h.Get<float>();
            xVTensor = xVQueue_h.Get<float>();

            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_h < slideEnd_h) {
                for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                    int16_t length = Min(slide_size, slideEnd_h - index);
                    calculateIntermediateTensor(index, length, H_DIRECTION);
                    // 计算系数矩阵
                    calculateRatioTensorH(0, length);
                    copyRatioTensorToGm(1);
                    calculateHeightExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_h < tailSlideEnd_h) {
                calculateIntermediateTensor(tailSlideStart_h, tailSlideEnd_h - tailSlideStart_h, H_DIRECTION);
                for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                    int16_t length = Min(slide_size, tailSlideEnd_h - index);
                    calculateRatioTensorH(0, length);
                    copyRatioTensorToGm(1);
                    calculateHeightExtension(index, tailRowStart_h, tailRowEnd_h);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline int64_t UpsampleBicubic2dND<T>::getWidthTensorSize()
{
    int64_t size = slide_size;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline int64_t UpsampleBicubic2dND<T>::getHeightTensorSize()
{
    int64_t size = slide_size;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateIntermediateTensor(
    int64_t index, int64_t length, int8_t direction)
{
    length = Max(length, EACH_SLICE_HANDLE_NUM);
    float scale = scale_w;
    int64_t max_interp_size = max_interp_size_w;
    if (direction == H_DIRECTION) {
        scale = scale_h;
        max_interp_size = max_interp_size_h;
    }
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    if (align_corners) {
        // 角对齐
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        // 边对齐
        for (int64_t i = 0; i < length; i++) {
            float center = (static_cast<float>(0.5) + static_cast<float>(index + i)) * scale - static_cast<float>(0.5);
            centerTensor.SetValue(i, center);
        }
        PipeBarrier<PIPE_V>();
    }

    // 计算每个下标的int
    Floor(xIntTensor, centerTensor, length);

    // 计算每个下标的最小映射值
    Adds(xMinTensor, xIntTensor, (float)(-1.0), length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标的v
    Sub(xVTensor, centerTensor, xIntTensor, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateRatioTensorW(int64_t xIndex, int64_t length)
{
    LocalTensor<float> ratioTensor = ratioQueue_w.AllocTensor<float>();
    singleCoreK = 0;
    // 计算横向系数矩阵
    Duplicate(ratioTensor, (float)0.0, ratioTensor.GetSize());

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    xMin = static_cast<int64_t>(xMinTensor.GetValue(xIndex));
    for (int64_t i = xIndex; i < xIndex + length; i++) {
        int64_t xSize = 4;
        if (static_cast<int64_t>(xMinTensor.GetValue(i)) + 4 > input_shapes[3]) {
            xSize = input_shapes[3] - static_cast<int64_t>(xMinTensor.GetValue(i));
        }
        int64_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
        for (int64_t j = 0; j < xSize; j++) {
            float w = weightCalculate(xVTensor.GetValue(i), xIntTensor.GetValue(i), j, input_shapes[3]);
            int64_t yIndexValue = j + yIndexOffset;
            singleCoreK = singleCoreK < yIndexValue + 1 ? yIndexValue + 1 : singleCoreK;
            int64_t index = yIndexValue * slide_size + i - xIndex;
            ratioTensor.SetValue(index, w);
        }
    }

    if (dataType != 2) {
        Cast(ratioTensor.ReinterpretCast<T>(), ratioTensor, RoundMode::CAST_RINT, ratioTensor.GetSize());
        ratioQueue_w.EnQue(ratioTensor);
    } else {
        ratioQueue_w.EnQue(ratioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateRatioTensorH(int64_t yIndex, int64_t length)
{
    LocalTensor<float> ratioTensor = ratioQueue_h.AllocTensor<float>();
    xMin = static_cast<int64_t>(xMinTensor.GetValue(yIndex));
    // 计算纵向系数矩阵
    Duplicate(ratioTensor, (float)0.0, ratioTensor.GetSize());
    for (int64_t i = yIndex; i < yIndex + length; i++) {
        int64_t xSize = 4;
        if (static_cast<int64_t>(xMinTensor.GetValue(i)) + 4 > input_shapes[2]) {
            xSize = input_shapes[2] - static_cast<int64_t>(xMinTensor.GetValue(i));
        }
        singleCoreK = xMinTensor.GetValue(yIndex + length - 1) - xMin + xSize;
        int64_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
        for (int64_t j = 0; j < xSize; j++) {
            float w = weightCalculate(xVTensor.GetValue(i), xIntTensor.GetValue(i), j, input_shapes[2]);
            int64_t yIndexValue = j + yIndexOffset;
            int64_t index = yIndexValue + (i - yIndex) * matmulTiling_h->singleCoreK;
            ratioTensor.SetValue(index, w);
        }
    }

    if (dataType != 2) {
        Cast(ratioTensor.ReinterpretCast<T>(), ratioTensor, RoundMode::CAST_RINT, ratioTensor.GetSize());
        ratioQueue_h.EnQue(ratioTensor);
    } else {
        ratioQueue_h.EnQue(ratioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::copyRatioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    if (direction == 0) {
        workSpaceRatioOffset = intermediate_matrix_size + ratio_matrix_size_w * blockIdx;
    } else {
        workSpaceRatioOffset = intermediate_matrix_size + ratio_matrix_size_h * blockIdx;
    }

    if (dataType == 2) {
        LocalTensor<T> ratioTensor = initRatioTensor(direction);
        DataCopy(intermediateTensorGm[workSpaceRatioOffset], ratioTensor, ratioTensor.GetSize());
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        releaseRatioTensor(direction, ratioTensor);
    } else {
        int8_t size = 32 / sizeof(T);
        LocalTensor<T> ratioTensor = initRatioTensor(direction);
        DataCopy(
            intermediateTensorGm[workSpaceRatioOffset], ratioTensor, (ratioTensor.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        releaseRatioTensor(direction, ratioTensor);
    }
}

template <typename T>
__aicore__ inline LocalTensor<T> UpsampleBicubic2dND<T>::initRatioTensor(int8_t direction)
{
    if (direction == 0) {
        return ratioQueue_w.DeQue<T>();
    } else {
        return ratioQueue_h.DeQue<T>();
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::releaseRatioTensor(int8_t direction, LocalTensor<T> ratioTensor)
{
    if (direction == 0) {
        return ratioQueue_w.FreeTensor(ratioTensor);
    } else {
        return ratioQueue_h.FreeTensor(ratioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[3]) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = xMin + rowStart * input_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * output_shapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRatioOffset], false);
    if (floatEqual_h) {
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
__aicore__ inline void UpsampleBicubic2dND<T>::calculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    int64_t singleCoreN = matmulTiling_h->singleCoreN;
    if (tensorCIndex + slide_size > output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    }

    matmulH.SetOrgShape(singleCoreM, output_shapes[3], matmulTiling_h->singleCoreK, output_shapes[2], output_shapes[3]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[2]) {
        matmulH.SetTail(output_shapes[2] - tensorCIndex, singleCoreN, singleCoreK);
    }
    if (rowEnd == 0) {
        rowEnd = input_shapes[0] * input_shapes[1];
    }

    int64_t xIndex = xMin * output_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex * output_shapes[3];

    int64_t middleHWSize = input_shapes[2] * output_shapes[3];
    int64_t outputHWSize = output_shapes[2] * output_shapes[3];

    matmulH.SetTensorA(intermediateTensorGm[workSpaceRatioOffset], false);
    for (int i = rowStart; i < rowEnd; i++) {
        if (floatEqual_w) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * middleHWSize], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * middleHWSize], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * outputHWSize], false);
        matmulH.End();

        event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND<T>::ParseTilingData(UpsampleBicubic2dTilingData *tilingData)
{
    slide_size = tilingData->slide_size;
    scale_w = tilingData->scale_w;
    scale_h = tilingData->scale_h;
    align_corners = tilingData->align_corners;
    max_interp_size_w = tilingData->max_interp_size_w;
    max_interp_size_h = tilingData->max_interp_size_h;

    need_core_num_w = tilingData->need_core_num_w;
    need_core_num_h = tilingData->need_core_num_h;

    for (int8_t i = 0; i < 4; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
    }
    for (int8_t i = 0; i < 4; i++) {
        input_shapes[i] = tilingData->input_shapes[i];
    }

    intermediate_matrix_size = tilingData->intermediate_matrix_size / sizeof(T);
    ratio_matrix_size_w = (tilingData->ratio_matrix_size_w + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    ratio_matrix_size_h = (tilingData->ratio_matrix_size_h + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

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

    dataType = tilingData->dataType;

    matmulTiling_w = &tilingData->matmulTiling_w;
    matmulTiling_h = &tilingData->matmulTiling_h;
}
}  // namespace UpsampleBicubic2d

#endif  // UPSAMPLE_BICUBIC2D

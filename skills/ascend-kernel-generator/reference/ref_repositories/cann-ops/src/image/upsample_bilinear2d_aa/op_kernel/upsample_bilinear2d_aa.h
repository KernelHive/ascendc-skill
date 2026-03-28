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
 * \file upsample_bilinear2d_aa.h
 * \brief
 */
#ifndef UPSAMPLE_BILINEAR2D_AA
#define UPSAMPLE_BILINEAR2D_AA

#include <type_traits>
#include "lib/matmul_intf.h"

namespace UpsampleBilinear2dAA {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr uint32_t ADDR_ALIGN_SIZE = 128;

template <typename T>
class UpsampleBilinearAAND {
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

    __aicore__ inline UpsampleBilinearAAND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBilinearAATilingData *tilingData);
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
    __aicore__ inline bool FloatEqual(float a, float b)
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };
    __aicore__ inline void ParseTilingData(const UpsampleBilinearAATilingData *tilingData);
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void calculateIntermediateTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void calculateRadioTensorW(int64_t index, int64_t length, float invscale);
    __aicore__ inline void calculateRadioTensorH(int64_t index, int64_t length, float invscale);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRadioTensor(int8_t direction);
    __aicore__ inline void getSlideRange();

    __aicore__ inline void releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor);
    __aicore__ inline int64_t getWidthTensorSize();
    __aicore__ inline int64_t getHeightTensorSize();

private:
    // 系数矩阵下标队列

    TBuf<QuePosition::VECCALC> centerQueue_w;
    TBuf<QuePosition::VECCALC> xMinQueue_w;
    TBuf<QuePosition::VECCALC> xSizeQueue_w;
    TBuf<QuePosition::VECCALC> weightQueue_w;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_w;

    TBuf<QuePosition::VECCALC> centerQueue_h;
    TBuf<QuePosition::VECCALC> xMinQueue_h;
    TBuf<QuePosition::VECCALC> xSizeQueue_h;
    TBuf<QuePosition::VECCALC> weightQueue_h;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_h;

    const TCubeTiling *__restrict matmulTiling_w;
    const TCubeTiling *__restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;
    LocalTensor<float> xSizeTensor;
    LocalTensor<float> weightTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    int64_t slide_size = 0;
    float scaleW;
    float scaleH;
    float invscaleW;
    float invscaleH;
    float supportW;
    float supportH;
    int64_t max_interp_size_w = 16;
    int64_t max_interp_size_h;
    int64_t need_core_num_w;
    int64_t need_core_num_h;
    int64_t dataType;

    uint64_t intermediate_matrix_size;
    uint32_t radio_matrix_size_h;
    uint32_t radio_matrix_size_w;

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t eachCoreSlideNumH;
    int64_t tailStartSlideNumH;
    int64_t slideNumH;
    int64_t groupCoreNumH;
    int64_t tailAvergingRowsH;
    int64_t remainderH;

    int64_t slideStart_w = 0;
    int64_t slideEnd_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailSlideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t tailRowEnd_w = 0;

    int64_t slideStart_h = 0;
    int64_t slideEnd_h = 0;
    int64_t tailSlideStart_h = 0;
    int64_t tailSlideEnd_h = 0;
    int64_t tailRowStart_h = 0;
    int64_t tailRowEnd_h = 0;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;
};

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBilinearAATilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;
    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);
    getSlideRange();
    int64_t tensorWidthSize = getWidthTensorSize();
    int64_t tensorHeightSize = getHeightTensorSize();

    if (!FloatEqual(scaleW, 1.0)) {
        pipe.InitBuffer(centerQueue_w, tensorWidthSize);
        pipe.InitBuffer(xMinQueue_w, tensorWidthSize);
        pipe.InitBuffer(xSizeQueue_w, tensorWidthSize);
        pipe.InitBuffer(weightQueue_w, (max_interp_size_w * sizeof(float) + 31) / 32 * 32);
        pipe.InitBuffer(radioQueue_w, BUFFER_NUM, radio_matrix_size_w * sizeof(float));
    }

    if (!FloatEqual(scaleH, 1.0) || FloatEqual(scaleW, 1.0)) {
        pipe.InitBuffer(centerQueue_h, tensorHeightSize);
        pipe.InitBuffer(xMinQueue_h, tensorHeightSize);
        pipe.InitBuffer(xSizeQueue_h, tensorHeightSize);
        pipe.InitBuffer(weightQueue_h, (max_interp_size_h * sizeof(float) + 31) / 32 * 32);
        pipe.InitBuffer(radioQueue_h, BUFFER_NUM, radio_matrix_size_h * sizeof(float));
    }

    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorsPtr);
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::Process()
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
__aicore__ inline int64_t UpsampleBilinearAAND<T>::getWidthTensorSize()
{
    int64_t size = slide_size;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline int64_t UpsampleBilinearAAND<T>::getHeightTensorSize()
{
    int64_t size = slide_size;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::WDirectionExpansion()
{
    if (!FloatEqual(scaleW, 1.0)) {
        if (blockIdx < need_core_num_w) {
            centerTensor = centerQueue_w.Get<float>();
            xMinTensor = xMinQueue_w.Get<float>();
            xSizeTensor = xSizeQueue_w.Get<float>();
            weightTensor = weightQueue_w.Get<float>();
            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_w < slideEnd_w) {
                for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                    int16_t length = Min(slide_size, slideEnd_w - index);
                    calculateIntermediateTensor(index, length, W_DIRECTION);

                    // 计算系数矩阵
                    calculateRadioTensorW(index - slideStart_w, length, invscaleW);
                    copyRadioTensorToGm(0);
                    calculateWidthExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_w < tailSlideEnd_w) {
                calculateIntermediateTensor(tailSlideStart_w, tailSlideEnd_w - tailSlideStart_w, W_DIRECTION);
                for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                    int16_t length = Min(slide_size, tailSlideEnd_w - index);
                    calculateRadioTensorW(index - tailSlideStart_w, length, invscaleW);
                    copyRadioTensorToGm(0);
                    calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::HDirectionExpansion()
{
    if (!FloatEqual(scaleH, 1.0) || FloatEqual(scaleW, 1.0)) {
        if (blockIdx < need_core_num_h) {
            centerTensor = centerQueue_h.Get<float>();
            xMinTensor = xMinQueue_h.Get<float>();
            xSizeTensor = xSizeQueue_h.Get<float>();
            weightTensor = weightQueue_h.Get<float>();

            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_h < slideEnd_h) {
                for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                    int16_t length = Min(slide_size, slideEnd_h - index);
                    calculateIntermediateTensor(index, length, H_DIRECTION);
                    // 计算系数矩阵
                    calculateRadioTensorH(index - slideStart_h, length, invscaleH);
                    copyRadioTensorToGm(1);
                    calculateHeightExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_h < tailSlideEnd_h) {
                calculateIntermediateTensor(tailSlideStart_h, tailSlideEnd_h - tailSlideStart_h, H_DIRECTION);
                for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                    int16_t length = Min(slide_size, tailSlideEnd_h - index);
                    calculateRadioTensorH(index - tailSlideStart_h, length, invscaleH);
                    copyRadioTensorToGm(1);
                    calculateHeightExtension(index, tailRowStart_h, tailRowEnd_h);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::calculateIntermediateTensor(
    int64_t index, int64_t length, int8_t direction)
{
    length = Max(length, EACH_SLICE_HANDLE_NUM);
    float scale = scaleW;
    float support = supportW;
    int64_t max_interp_size = max_interp_size_w;
    int64_t maxSize = input_shapes[3];
    if (direction == H_DIRECTION) {
        scale = scaleH;
        support = supportH;
        max_interp_size = max_interp_size_h;
        maxSize = input_shapes[2];
    }
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    Adds(centerTensor, centerTensor, (float)0.5, length);
    PipeBarrier<PIPE_V>();
    Muls(centerTensor, centerTensor, scale, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标最小映射值
    Adds(xMinTensor, centerTensor, (float)0.5 - support, length);
    PipeBarrier<PIPE_V>();
    Floor(xMinTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标映射的范围
    Adds(xSizeTensor, centerTensor, (float)0.5 + support, length);
    PipeBarrier<PIPE_V>();
    Floor(xSizeTensor, xSizeTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(maxSize), length);
    PipeBarrier<PIPE_V>();
    Sub(xSizeTensor, xSizeTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(max_interp_size), length);
    PipeBarrier<PIPE_V>();
    Maxs(xSizeTensor, xSizeTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::calculateRadioTensorW(int64_t xIndex, int64_t length, float invscale)
{
    LocalTensor<float> radioTensor = radioQueue_w.AllocTensor<float>();

    xIndex = 0;
    singleCoreK = 0;
    // 计算横向系数矩阵
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());

    event_t eventIDVToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    xMin = static_cast<int64_t>(xMinTensor.GetValue(xIndex));
    for (int64_t i = xIndex; i < xIndex + length; i++) {
        float total_w = 0.0;
        float tmpValue = xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5;
        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
            float singleW = weightCalculate((j + tmpValue) * invscale);
            weightTensor.SetValue(j, singleW);
            total_w += singleW;
        }
        int64_t offset = i - xIndex;

        if (!FloatEqual(total_w, (float)0.0)) {
            int64_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
                float weight = weightTensor.GetValue(j) / total_w;
                int64_t yIndexValue = j + yIndexOffset;
                singleCoreK = singleCoreK < yIndexValue + 1 ? yIndexValue + 1 : singleCoreK;
                int64_t index = yIndexValue * slide_size + offset;
                radioTensor.SetValue(index, weight);
            }
        }
    }

    if (dataType != 2) {
        Cast(radioTensor.ReinterpretCast<T>(), radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
        radioQueue_w.EnQue(radioTensor);
    } else {
        radioQueue_w.EnQue(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::calculateRadioTensorH(int64_t xIndex, int64_t length, float invscale)
{
    LocalTensor<float> radioTensor = radioQueue_h.AllocTensor<float>();

    xIndex = 0;
    // 计算横向系数矩阵
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());
    xMin = static_cast<int64_t>(xMinTensor.GetValue(xIndex));
    singleCoreK = xMinTensor.GetValue(xIndex + length - 1) - xMinTensor.GetValue(xIndex) +
                  xSizeTensor.GetValue(xIndex + length - 1);

    for (int64_t i = xIndex; i < xIndex + length; i++) {
        float total_w = 0.0;
        float tmpValue = xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5;
        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
            float w = weightCalculate((j + tmpValue) * invscale);
            weightTensor.SetValue(j, w);
            total_w += w;
        }

        int64_t offset = (i - xIndex) * matmulTiling_h->singleCoreK;
        if (!FloatEqual(total_w, (float)0.0)) {
            int64_t yIndexOffset = static_cast<int64_t>(xMinTensor.GetValue(i)) - xMin;
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
                float weight = weightTensor.GetValue(j) / total_w;
                int64_t yIndexValue = j + yIndexOffset;
                int64_t index = yIndexValue + offset;
                radioTensor.SetValue(index, weight);
            }
        }
    }

    if (dataType != 2) {
        Cast(radioTensor.ReinterpretCast<T>(), radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
        radioQueue_h.EnQue(radioTensor);
    } else {
        radioQueue_h.EnQue(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::copyRadioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    if (direction == 0) {
        workSpaceRadioOffset = intermediate_matrix_size + radio_matrix_size_w * blockIdx;
    } else {
        workSpaceRadioOffset = intermediate_matrix_size + radio_matrix_size_h * blockIdx;
    }

    if (dataType == 2) {
        LocalTensor<T> radioTensor = initRadioTensor(direction);
        DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, radioTensor.GetSize());
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        releaseRadioTensor(direction, radioTensor);
    } else {
        int8_t size = 32 / sizeof(T);
        LocalTensor<T> radioTensor = initRadioTensor(direction);
        DataCopy(
            intermediateTensorGm[workSpaceRadioOffset], radioTensor, (radioTensor.GetSize() + size - 1) / size * size);
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        releaseRadioTensor(direction, radioTensor);
    }
}

template <typename T>
__aicore__ inline LocalTensor<T> UpsampleBilinearAAND<T>::initRadioTensor(int8_t direction)
{
    if (direction == 0) {
        return radioQueue_w.DeQue<T>();
    } else {
        return radioQueue_h.DeQue<T>();
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor)
{
    if (direction == 0) {
        return radioQueue_w.FreeTensor(radioTensor);
    } else {
        return radioQueue_h.FreeTensor(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::calculateWidthExtension(
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
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    if (FloatEqual(scaleH, 1.0)) {
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
__aicore__ inline void UpsampleBilinearAAND<T>::calculateHeightExtension(
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

    matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
    for (int i = rowStart; i < rowEnd; i++) {
        if (FloatEqual(scaleW, 1.0)) {
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
__aicore__ inline void UpsampleBilinearAAND<T>::ParseTilingData(const UpsampleBilinearAATilingData *tilingData)
{
    slide_size = tilingData->slide_size;
    scaleW = tilingData->scale_w;
    scaleH = tilingData->scale_h;
    invscaleW = tilingData->invscale_w;
    invscaleH = tilingData->invscale_h;
    supportW = tilingData->support_w;
    supportH = tilingData->support_h;
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
    radio_matrix_size_w = (tilingData->radio_matrix_size_w + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    radio_matrix_size_h = (tilingData->radio_matrix_size_h + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

    eachCoreSlideNumW = tilingData->eachCoreSlideNumW;
    tailStartSlideNumW = tilingData->tailStartSlideNumW;
    slideNumW = tilingData->slideNumW;
    groupCoreNumW = tilingData->groupCoreNumW;
    tailAvergingRowsW = tilingData->tailAvergingRowsW;
    remainderW = tilingData->remainderW;

    eachCoreSlideNumH = tilingData->eachCoreSlideNumH;
    tailStartSlideNumH = tilingData->tailStartSlideNumH;
    slideNumH = tilingData->slideNumH;
    groupCoreNumH = tilingData->groupCoreNumH;
    tailAvergingRowsH = tilingData->tailAvergingRowsH;
    remainderH = tilingData->remainderH;

    dataType = tilingData->dataType;

    matmulTiling_w = &tilingData->matmulTiling_w;
    matmulTiling_h = &tilingData->matmulTiling_h;
}

template <typename T>
__aicore__ inline void UpsampleBilinearAAND<T>::getSlideRange()
{
    slideStart_w = blockIdx * eachCoreSlideNumW * slide_size;
    slideEnd_w = (Min((blockIdx + 1) * eachCoreSlideNumW, slideNumW)) * slide_size;
    int64_t groupIndex = blockIdx / groupCoreNumW;
    if (groupIndex < remainderW) {
        tailSlideStart_w = (tailStartSlideNumW + groupIndex) * slide_size;
        tailSlideEnd_w = Min(tailSlideStart_w + slide_size, output_shapes[3]);
        int64_t blockIdxInGroup = blockIdx % groupCoreNumW;
        tailRowStart_w = blockIdxInGroup * tailAvergingRowsW;
        tailRowEnd_w = Min(tailRowStart_w + tailAvergingRowsW, input_shapes[0] * input_shapes[1] * input_shapes[2]);
    }

    slideStart_h = blockIdx * eachCoreSlideNumH * slide_size;
    slideEnd_h = (Min((blockIdx + 1) * eachCoreSlideNumH, slideNumH)) * slide_size;
    groupIndex = blockIdx / groupCoreNumH;
    if (groupIndex < remainderH) {
        tailSlideStart_h = (tailStartSlideNumH + groupIndex) * slide_size;
        tailSlideEnd_h = Min(tailSlideStart_h + slide_size, output_shapes[2]);
        int64_t blockIdxInGroup = blockIdx % groupCoreNumH;
        tailRowStart_h = blockIdxInGroup * tailAvergingRowsH;
        tailRowEnd_h = Min(tailRowStart_h + tailAvergingRowsH, input_shapes[0] * input_shapes[1]);
    }
}

}  // namespace UpsampleBilinear2dAA

#endif  // UPSAMPLE_BILINEAR2D_AA

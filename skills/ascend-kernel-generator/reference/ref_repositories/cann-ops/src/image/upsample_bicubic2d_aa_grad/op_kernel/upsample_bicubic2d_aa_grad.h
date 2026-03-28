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
 * \file upsample_bicubic2d_aa_grad.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC_AA_GRAD
#define UPSAMPLE_BICUBIC_AA_GRAD

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpSampleBicubic2dAAGrad {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class UpSampleBicubic2dAAGradND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>>
        matmulH;
    __aicore__ inline UpSampleBicubic2dAAGradND(){};
    __aicore__ inline void calculateIntermediateTensorX(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
        LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor, int64_t slideStart_w, int64_t slideEnd_w);
    __aicore__ inline void calculateIntermediateTensorY(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
        LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor, int64_t slideStart_h, int64_t slideEnd_h);
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBicubicAAGradTilingData *tilingData);
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
    template <typename T1>
    __aicore__ inline T1 getWeight(T1 x)
    {
        if (x < 0) {
            x = -1 * x;
        }
        if (x < (float)1.0) {
            return ((float)1.5 * x - (float)2.5) * x * x + (float)1.0;
        }
        return (((float)2.5 - (float)0.5 * x) * x - (float)4.0) * x + (float)2.0;
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
    __aicore__ inline void getQueueSize();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void computeIndexValueH(
        LocalTensor<float> xMinTensor_h, LocalTensor<float> xSizeTensor_h, int64_t index, int64_t length);
    __aicore__ inline void computeIndexValueW(
        LocalTensor<float> xMinTensor, LocalTensor<float> xSizeTensor, int64_t index, int64_t length);
    __aicore__ inline void ParseTilingData(const UpsampleBicubicAAGradTilingData *tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline void ComputeAndCopyOut(int64_t index, int64_t dataCount, LocalTensor<float> &float32Tensor);
    __aicore__ inline __gm__ T *GetTensorAddr(int64_t index, GM_ADDR tensorPtr);
    __aicore__ inline void calculateRadioTensor(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
        LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateRadioTensorH(LocalTensor<float> centerTensor, LocalTensor<float> xMinTensor,
        LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor, int64_t index, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm();
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;

    // 系数矩阵下标队列,横轴和纵轴范围
    TBuf<QuePosition::VECCALC> centerQueue;
    TBuf<QuePosition::VECCALC> xMinQueue;
    TBuf<QuePosition::VECCALC> xSizeQueue;
    TBuf<QuePosition::VECCALC> weightQueue;
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

    float scaleW;
    float scaleH;
    float invscaleW;
    float invscaleH;
    float supportW;
    float supportH;
    int64_t max_interp_size_w;
    int64_t max_interp_size_h;

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
    int64_t slidelen;
    int64_t slidelen_h;
    int64_t queueSize = 0;

    int64_t slideStart_h;
    int64_t slideEnd_h;
    int64_t tailSlideStart_h;
    int64_t tailSlideEnd_h;
    int64_t tailRowStart_h;
    int64_t tailRowEnd_h;
    int64_t dataType;

    float zeroScaleW = 0;
    float zeroScaleH = 0;
    int64_t tensorRemainerStartOffset_w;
    int64_t tensorRemainerEndOffset_w;
    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    uint32_t maxDataCount = {0};

    uint64_t inputsTensorUbSize = 0;
    int64_t *tensorDataCountList = nullptr;
    uint32_t tensorStart_w = {0};
    uint32_t tensorEnd_w = {0};
    int64_t tensorStartOffset_w = {0};
    int64_t tensorEndOffset_w = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;

    uint32_t maxCastDataCount = {0};

    uint32_t need_core_num_w;
    uint32_t need_core_num_h;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t instart_w = 0;
    int64_t instart_h = 0;

    int64_t xMin = 0;
    int64_t instartIndex = 0;
    int64_t inendIndex = 0;

    int32_t singleCoreK_h = 0;

    bool needExpendX = false;
    bool needExpendY = false;
};

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::getQueueSize()
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

    int64_t inSlide_w = scaleW > 0 ? static_cast<int64_t>(2 * (xSlideSize + supportW) / scaleW) + 1
                                   : static_cast<int64_t>(2 * (xSlideSize + supportW) / zeroScaleW) + 1;
    int64_t inSlide_h = scaleH > 0 ? static_cast<int64_t>(2 * (ySlideSize + supportH) / scaleH) + 1
                                   : static_cast<int64_t>(2 * (ySlideSize + supportH) / zeroScaleH) + 1;

    if (inSlide_w > inSlide_h) {
        queueSize = inSlide_w;
    } else {
        queueSize = inSlide_h;
    }
};

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::WDirectionExpansion()
{
    if (blockIdx < need_core_num_w) {
        LocalTensor<float> centerTensor = centerQueue.Get<float>();
        LocalTensor<float> xMinTensor = xMinQueue.Get<float>();
        LocalTensor<float> xSizeTensor = xSizeQueue.Get<float>();
        LocalTensor<float> weightTensor = weightQueue.Get<float>();

        // 计算滑块映射范围
        if (slideStart_w < slideEnd_w) {
            calculateIntermediateTensorX(centerTensor, xMinTensor, xSizeTensor, weightTensor, slideStart_w, slideEnd_w);
            for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, slideEnd_w - index);
                slidelen = length;
                calculateRadioTensor(centerTensor, xMinTensor, xSizeTensor, weightTensor, index, length);
                copyRadioTensorToGm();
                calculateWidthExtension(index, 0, 0);
            }
        }
        if (tailSlideStart_w < tailSlideEnd_w) {
            calculateIntermediateTensorX(
                centerTensor, xMinTensor, xSizeTensor, weightTensor, tailSlideStart_w, tailSlideEnd_w);
            for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, tailSlideEnd_w - index);
                slidelen = length;
                calculateRadioTensor(centerTensor, xMinTensor, xSizeTensor, weightTensor, index, length);
                copyRadioTensorToGm();
                calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
            }
        }
        // 处理尾块部分数据
        centerQueue.FreeTensor(centerTensor);
        xMinQueue.FreeTensor(xMinTensor);
        xSizeQueue.FreeTensor(xSizeTensor);
        weightQueue.FreeTensor(weightTensor);
    }
    // 获取要计算系数矩阵的下标
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::HDirectionExpansion()
{
    if (blockIdx < need_core_num_h) {
        instartIndex = 0;
        inendIndex = 0;
        LocalTensor<float> centerTensor_h = centerQueue.Get<float>();
        LocalTensor<float> xMinTensor_h = xMinQueue.Get<float>();
        LocalTensor<float> xSizeTensor_h = xSizeQueue.Get<float>();
        LocalTensor<float> weightTensor_h = weightQueue.Get<float>();
        if (slideStart_h < slideEnd_h) {
            calculateIntermediateTensorY(
                centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, slideStart_h, slideEnd_h);
            for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, slideEnd_h + 1 - index);
                slidelen_h = length;
                calculateRadioTensorH(centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, index, length);
                copyRadioTensorToGm();
                calculateHeightExtension(index, 0, 0);
            }
        }

        if (tailSlideStart_h < tailSlideEnd_h) {
            calculateIntermediateTensorY(
                centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, tailSlideStart_h, tailSlideEnd_h);
            for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, tailSlideEnd_h + 1 - index);
                slidelen_h = length;
                calculateRadioTensorH(centerTensor_h, xMinTensor_h, xSizeTensor_h, weightTensor_h, index, length);
                copyRadioTensorToGm();
                calculateHeightExtension(index, tailRowStart_h, tailRowEnd_h);
            }
        }

        // 释放临时tensor
        centerQueue.FreeTensor(centerTensor_h);
        xMinQueue.FreeTensor(xMinTensor_h);
        xSizeQueue.FreeTensor(xSizeTensor_h);
        weightQueue.FreeTensor(weightTensor_h);
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleBicubicAAGradTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;

    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);

    needExpendX = !FloatEqual(scaleW, 1.0);
    needExpendY = !FloatEqual(scaleH, 1.0);

    getQueueSize();
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    int64_t interpsize = getMax(max_interp_size_h, max_interp_size_w);

    pipe.InitBuffer(centerQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xMinQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(xSizeQueue, (queueSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioQueue, NO_BUFFER_NUM, (radioSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(weightQueue, (interpsize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioCastQueue, NO_BUFFER_NUM, (radioSize * sizeof(T) + 31) / 32 * 32);

    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorsPtr);
};

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::Process()
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
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateIntermediateTensorX(LocalTensor<float> centerTensor,
    LocalTensor<float> xMinTensor, LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor,
    int64_t slideStart_w, int64_t slideEnd_w)
{
    instart_w = scaleW > 0 ? static_cast<int64_t>((float)(slideStart_w - supportW) / scaleW) - 1
                           : static_cast<int64_t>((float)(slideStart_w - supportW) / zeroScaleW) - 1;

    if (instart_w < 0) {
        instart_w = 0;
    }

    int64_t length = static_cast<int64_t>(centerTensor.GetSize());
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor, static_cast<float>(instart_w), static_cast<float>(1), length);
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
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(output_shapes[3]), length);
    PipeBarrier<PIPE_V>();
    Sub(xSizeTensor, xSizeTensor, xMinTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xSizeTensor, xSizeTensor, static_cast<float>(max_interp_size_w), length);
    PipeBarrier<PIPE_V>();
    Maxs(xSizeTensor, xSizeTensor, (float)0.0, length);
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateIntermediateTensorY(LocalTensor<float> centerTensor_h,
    LocalTensor<float> xMinTensor_h, LocalTensor<float> xSizeTensor_h, LocalTensor<float> weightTensor_h,
    int64_t slideStart_h, int64_t slideEnd_h)
{
    instart_h = scaleH > 0 ? static_cast<int64_t>((float)(slideStart_h - supportH) / scaleH) - 1
                           : static_cast<int64_t>((float)(slideStart_h - supportH) / zeroScaleH) - 1;
    int64_t length = static_cast<int64_t>(centerTensor_h.GetSize());
    if (instart_h < 0) {
        instart_h = 0;
    }
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor_h, static_cast<float>(instart_h), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    // 计算center下标
    Adds(centerTensor_h, centerTensor_h, (float)0.5, length);
    PipeBarrier<PIPE_V>();
    Muls(centerTensor_h, centerTensor_h, scaleH, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标最小映射值
    Adds(xMinTensor_h, centerTensor_h, (float)0.5 - supportH, length);
    PipeBarrier<PIPE_V>();
    Floor(xMinTensor_h, xMinTensor_h, length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor_h, xMinTensor_h, (float)0.0, length);
    PipeBarrier<PIPE_V>();

    // 计算每个下标映射的范围
    Adds(xSizeTensor_h, centerTensor_h, (float)0.5 + supportH, length);
    PipeBarrier<PIPE_V>();

    Floor(xSizeTensor_h, xSizeTensor_h, length);
    PipeBarrier<PIPE_V>();

    Mins(xSizeTensor_h, xSizeTensor_h, static_cast<float>(output_shapes[2]), length);
    PipeBarrier<PIPE_V>();

    Sub(xSizeTensor_h, xSizeTensor_h, xMinTensor_h, length);
    PipeBarrier<PIPE_V>();

    Mins(xSizeTensor_h, xSizeTensor_h, static_cast<float>(max_interp_size_h), length);
    PipeBarrier<PIPE_V>();

    Maxs(xSizeTensor_h, xSizeTensor_h, (float)0.0, length);
    // 计算批量分组的数据
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::computeIndexValueH(
    LocalTensor<float> xMinTensor_h, LocalTensor<float> xSizeTensor_h, int64_t index, int64_t length)
{
    instartIndex = 0;
    inendIndex = 0;
    for (; instartIndex < xMinTensor_h.GetSize(); instartIndex++) {
        int64_t ymax = xMinTensor_h.GetValue(instartIndex) + xSizeTensor_h.GetValue(instartIndex);
        if (ymax >= index) {
            break;
        }
    }
    for (inendIndex = instartIndex; inendIndex < xMinTensor_h.GetSize(); inendIndex++) {
        if (xMinTensor_h.GetValue(inendIndex) > index + length - 1) {
            break;
        } else if (inendIndex + instart_h > input_shapes[2] - 1) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateRadioTensorH(LocalTensor<float> centerTensor_h,
    LocalTensor<float> xMinTensor_h, LocalTensor<float> xSizeTensor_h, LocalTensor<float> weightTensor_h, int64_t index,
    int64_t length)
{
    LocalTensor<float> radioTensor_h = radioQueue.AllocTensor<float>();
    // 初始化为0
    Duplicate(radioTensor_h, float(0.0), radioTensor_h.GetSize());

    // 计算影响该块的原始矩阵点的下标

    computeIndexValueH(xMinTensor_h, xSizeTensor_h, index, length);
    singleCoreK_h = inendIndex - instartIndex;
    for (int64_t i = instartIndex; i < inendIndex; i++) {
        float total_w = 0.0;
        int64_t xmin = xMinTensor_h.GetValue(i);
        int64_t xmax = xmin + xSizeTensor_h.GetValue(i);
        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor_h.GetValue(i)); j++) {
            float w = getWeight((j + xMinTensor_h.GetValue(i) - centerTensor_h.GetValue(i) + (float)0.5) * invscaleH);
            weightTensor_h.SetValue(j, w);
            total_w += w;
        }
        int64_t insertx = i - instartIndex;
        singleCoreK_h = singleCoreK_h < insertx + 1 ? insertx + 1 : singleCoreK_h;
        int64_t xstart = getMax(index, xmin) - index;
        int64_t xend = getMin(index + slidelen_h, xmax) - index;
        if (!FloatEqual(total_w, 0.0)) {
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor_h.GetValue(i)); j++) {
                float weight = weightTensor_h.GetValue(j) / total_w;
                // 求更新系数矩阵中行的位置

                int64_t yIndexValue = xmin + j - index;

                if (yIndexValue < xend && yIndexValue >= 0) {
                    int64_t index = yIndexValue * matmulTiling_h->singleCoreK + insertx;
                    radioTensor_h.SetValue(index, weight);
                }
            }
        }
    }

    if (dataType != 2) {
        LocalTensor<T> radioCastTensor_h = radioCastQueue.AllocTensor<T>();
        Cast(radioCastTensor_h, radioTensor_h, RoundMode::CAST_RINT, radioTensor_h.GetSize());
        radioCastQueue.EnQue(radioCastTensor_h);
        radioQueue.FreeTensor(radioTensor_h);
    } else {
        radioQueue.EnQue(radioTensor_h);
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::computeIndexValueW(
    LocalTensor<float> xMinTensor, LocalTensor<float> xSizeTensor, int64_t index, int64_t length)
{
    instartIndex = 0;
    inendIndex = 0;
    for (; instartIndex < xMinTensor.GetSize(); instartIndex++) {
        int64_t xmax = xMinTensor.GetValue(instartIndex) + xSizeTensor.GetValue(instartIndex);
        if (xmax >= index) {
            break;
        }
    }

    for (inendIndex = instartIndex; inendIndex < xMinTensor.GetSize(); inendIndex++) {
        if (xMinTensor.GetValue(inendIndex) > index + length - 1) {
            break;
        } else if (inendIndex + instart_w > input_shapes[3] - 1) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateRadioTensor(LocalTensor<float> centerTensor,
    LocalTensor<float> xMinTensor, LocalTensor<float> xSizeTensor, LocalTensor<float> weightTensor, int64_t index,
    int64_t length)
{
    LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
    // 初始化为0
    Duplicate(radioTensor, float(0.0), radioTensor.GetSize());
    // 计算影响该块的原始矩阵点的下标
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    computeIndexValueW(xMinTensor, xSizeTensor, index, length);

    for (int64_t i = instartIndex; i < inendIndex; i++) {
        float total_w = 0.0;
        int64_t xmin = xMinTensor.GetValue(i);
        int64_t xmax = xmin + xSizeTensor.GetValue(i);

        for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {

            float w = getWeight((j + xMinTensor.GetValue(i) - centerTensor.GetValue(i) + (float)0.5) * invscaleW);

            weightTensor.SetValue(j, w);
            total_w += w;
        }

        if (!FloatEqual(total_w, 0.0)) {
            int64_t xstart = getMax(index, xmin) - index;
            int64_t xend = getMin(index + length, xmax) - index;
            for (int64_t j = 0; j < static_cast<int64_t>(xSizeTensor.GetValue(i)); j++) {
                float weight = weightTensor.GetValue(j) / total_w;
                // 求更新系数矩阵中行的位置
                int64_t insertx = xmin + j - index;

                if (insertx < xend && insertx >= 0) {
                    int64_t yIndexValue = 0;

                    yIndexValue = i - instartIndex;

                    singleCoreK = singleCoreK < yIndexValue + 1 ? yIndexValue + 1 : singleCoreK;
                    if (instartIndex + singleCoreK > input_shapes[3]) {
                        singleCoreK = input_shapes[3] - instartIndex;
                    }
                    int64_t index = yIndexValue * length + insertx;

                    radioTensor.SetValue(index, weight);
                }
            }
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
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::copyRadioTensorToGm()
{
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    workSpaceRadioOffset = intermediate_matrix_size + radioSize * blockIdx;
    int8_t size = 32 / sizeof(T);

    if (dataType == 2) {
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
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    if (singleCoreK == 0) {
        singleCoreK++;
    }

    if (tensorCIndex + slide_size > output_shapes[3]) {
        singleCoreN = slidelen;
    }

    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);

    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size > output_shapes[3] - 1) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = instartIndex + instart_w + rowStart * input_shapes[3];
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
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::calculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    int64_t singleCoreN = matmulTiling_h->singleCoreN;
    if (singleCoreK_h == 0) {
        singleCoreK_h++;
    }
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreN = rowEnd - rowStart;
    }

    if (tensorCIndex + slide_size > output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    }
    matmulH.SetOrgShape(singleCoreM, output_shapes[3], matmulTiling_h->singleCoreK, output_shapes[2], output_shapes[3]);

    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK_h);

    if (tensorCIndex + slide_size > output_shapes[2] - 1) {
        matmulH.SetTail(output_shapes[2] - tensorCIndex, singleCoreN, singleCoreK_h);
    }

    int64_t xIndex = (instartIndex + instart_h) * output_shapes[3] + rowStart;

    int64_t tensorCIndexWithOffset = tensorCIndex * output_shapes[3] + rowStart;

    for (int i = 0; i < output_shapes[0] * output_shapes[1]; i++) {
        // 系数矩阵起始位置
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
        if (!needExpendX) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * input_shapes[2] * output_shapes[3]], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * input_shapes[2] * output_shapes[3]], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * output_shapes[2] * output_shapes[3]], false);
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpSampleBicubic2dAAGradND<T>::ParseTilingData(const UpsampleBicubicAAGradTilingData *tilingData)
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
    dataType = tilingData->dataType;

    matmulTiling_w = &tilingData->matmulTiling_w;
    matmulTiling_h = &tilingData->matmulTiling_h;
}

}  // namespace UpSampleBicubic2dAAGrad
#endif
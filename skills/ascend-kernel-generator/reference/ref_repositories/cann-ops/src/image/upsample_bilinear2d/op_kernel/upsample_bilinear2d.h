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
 * \file upsample_bilinear2d.h
 * \brief
 */
#ifndef UPSAMPLE_BILINEAR2D
#define UPSAMPLE_BILINEAR2D

#include <type_traits>
#include "lib/matmul_intf.h"
#include "upsample_linear_common.h"

namespace UpsampleLinear2d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr uint32_t ADDR_ALIGN_SIZE = 128;

template <typename T>
class UpsampleLinear2dND {
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

    __aicore__ inline UpsampleLinear2dND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear2dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearGM(const GlobalTensor<T> &dstGlobal, int64_t totalCount);
    __aicore__ inline void ParseTilingData(const UpsampleLinear2dTilingData *tilingData);
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void calculateRadioTensorW(int64_t loopIndex, int64_t length);
    __aicore__ inline void calculateRadioTensorH(int64_t loopIndex, int64_t length);
    __aicore__ inline void calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void calculateHeightExtension(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);

    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline LocalTensor<T> initRadioTensor(int8_t direction);
    __aicore__ inline void getSlideRange();

    __aicore__ inline void releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor);
    __aicore__ inline int64_t getWidthTensorSize();
    __aicore__ inline int64_t getHeightTensorSize();

private:
    TBuf<TPosition::VECCALC> UbBuf;

    // 系数矩阵下标队列

    TBuf<QuePosition::VECCALC> centerQueue_w;
    TBuf<QuePosition::VECCALC> xMinQueue_w;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_w;

    TBuf<QuePosition::VECCALC> centerQueue_h;
    TBuf<QuePosition::VECCALC> xMinQueue_h;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue_h;

    const TCubeTiling *__restrict matmulTiling_w;
    const TCubeTiling *__restrict matmulTiling_h;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xMinTensor;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    bool alignCorners = false;
    int64_t mode = 1;
    int64_t blockIdx = 0;
    int64_t slide_size_w = 0;
    int64_t slide_size_h = 0;
    float scaleW;
    float scaleH;

    float support = 1.0;
    int64_t max_interp_size = 2;

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

    int64_t wInMaxIdx = 0;
    int64_t hInMaxIdx = 0;
};

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear2dTilingData *tilingData)
{
    blockIdx = GetBlockIdx() / 2;
    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);
    getSlideRange();
    int64_t tensorWidthSize = getWidthTensorSize();
    int64_t tensorHeightSize = getHeightTensorSize();
    pipe.InitBuffer(UbBuf, (64 * sizeof(T) + 31) / 32 * 32);
    if (!FloatEqual(scaleW, 1.0) || mode == 1) {
        pipe.InitBuffer(centerQueue_w, tensorWidthSize);

        pipe.InitBuffer(xMinQueue_w, tensorWidthSize);
        pipe.InitBuffer(radioQueue_w, BUFFER_NUM, radio_matrix_size_w * sizeof(float));
    }
    if (mode == 2 && (!FloatEqual(scaleH, 1.0) || FloatEqual(scaleW, 1.0))) {
        pipe.InitBuffer(centerQueue_h, tensorHeightSize);
        pipe.InitBuffer(xMinQueue_h, tensorHeightSize);
        pipe.InitBuffer(radioQueue_h, BUFFER_NUM, radio_matrix_size_h * sizeof(float));
    }
    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)outTensorsPtr);
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::ClearGM(const GlobalTensor<T> &dstGlobal, int64_t totalCount)
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
__aicore__ inline void UpsampleLinear2dND<T>::Process()
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
__aicore__ inline int64_t UpsampleLinear2dND<T>::getWidthTensorSize()
{
    int64_t size = slide_size_w;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline int64_t UpsampleLinear2dND<T>::getHeightTensorSize()
{
    int64_t size = slide_size_h;
    size = (size * sizeof(float) + 31) / 32 * 32;
    return size;
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::WDirectionExpansion()
{
    if (!FloatEqual(scaleW, 1.0) || mode == 1) {
        if (blockIdx < need_core_num_w) {
            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_w < slideEnd_w) {
                for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size_w) {
                    int16_t length = Min(slide_size_w, slideEnd_w - index);
                    // 计算系数矩阵
                    calculateRadioTensorW(index, length);
                    copyRadioTensorToGm(0);
                    calculateWidthExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_w < tailSlideEnd_w) {
                for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size_w) {
                    int16_t length = Min(slide_size_w, tailSlideEnd_w - index);
                    calculateRadioTensorW(index, length);
                    copyRadioTensorToGm(0);
                    calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::HDirectionExpansion()
{
    if (mode == 2 && (!FloatEqual(scaleH, 1.0) || FloatEqual(scaleW, 1.0))) {
        if (blockIdx < need_core_num_h) {
            centerTensor = centerQueue_h.Get<float>();
            xMinTensor = xMinQueue_h.Get<float>();
            // 获取要计算系数矩阵的下标
            // 计算批量分组的数据
            if (slideStart_h < slideEnd_h) {
                for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size_h) {
                    int16_t length = Min(slide_size_h, slideEnd_h - index);
                    // 计算系数矩阵
                    calculateRadioTensorH(index, length);
                    copyRadioTensorToGm(1);
                    calculateHeightExtension(index, 0, 0);
                }
            }

            // 处理尾块部分数据
            if (tailSlideStart_h < tailSlideEnd_h) {
                for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size_h) {
                    int16_t length = Min(slide_size_h, tailSlideEnd_h - index);
                    calculateRadioTensorH(index, length);
                    copyRadioTensorToGm(1);
                    calculateHeightExtension(index, tailRowStart_h, tailRowEnd_h);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::calculateRadioTensorW(int64_t loopIndex, int64_t length)
{
    LocalTensor<float> radioTensor = radioQueue_w.AllocTensor<float>();
    singleCoreK = 0;
    // 计算横向系数矩阵
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());
    event_t eventIDVToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    xMin = getCenterValue(loopIndex, scaleW, alignCorners);
    int64_t xMax = getCenterValue(loopIndex + length - 1, scaleW, alignCorners);
    int64_t xMaxNext = Min(xMax + (int64_t)2, input_shapes[3]);
    int64_t xMaxSize = Min(Max(xMaxNext - xMax, static_cast<int64_t>(0)), static_cast<int64_t>(2));
    singleCoreK = Max(xMax - xMin + xMaxSize, (int64_t)1);
    for (int64_t i = 0; i < length; i++) {
        float i_rel_idx = getCenterValue(i + loopIndex, scaleW, alignCorners);
        int64_t i_min = Min(static_cast<int64_t>(i_rel_idx), wInMaxIdx);
        int64_t i_max = Min(i_min + (int64_t)1, wInMaxIdx);
        int64_t yIndexOffset = i_min - xMin;
        int64_t indexMin = yIndexOffset * slide_size_w + i;
        float i_lambda_1 = 0;
        float i_lambda_0 = 0;
        int64_t indexMax = 0;
        if (i_min == i_max) {
            radioTensor.SetValue(indexMin, 1);
        } else {
            i_lambda_1 = getLambda(i_rel_idx, i_min);
            i_lambda_0 = 1 - i_lambda_1;
            radioTensor.SetValue(indexMin, i_lambda_0);
            indexMax = (1 + yIndexOffset) * slide_size_w + i;
            radioTensor.SetValue(indexMax, i_lambda_1);
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
__aicore__ inline void UpsampleLinear2dND<T>::calculateRadioTensorH(int64_t loopIndex, int64_t length)
{
    LocalTensor<float> radioTensor = radioQueue_h.AllocTensor<float>();
    // 计算横向系数矩阵
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());
    event_t eventIDVToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    xMin = static_cast<int64_t>(getCenterValue(loopIndex, scaleH, alignCorners));
    int64_t xMinMaxIdx = Min(xMin + (int64_t)2, input_shapes[2]);
    int64_t xMinSize = Min(Max(xMinMaxIdx - xMin, static_cast<int64_t>(0)), static_cast<int64_t>(2));

    int64_t xMax = static_cast<int64_t>(getCenterValue(loopIndex + length - 1, scaleH, alignCorners));
    singleCoreK = Min(xMax - xMin + xMinSize, input_shapes[2]);
    if ((singleCoreK + xMin) > input_shapes[2]) {
        singleCoreK = input_shapes[2] - xMin;
    }

    for (int64_t i = 0; i < length; i++) {
        float i_rel_idx = getCenterValue(i + loopIndex, scaleH, alignCorners);
        int64_t i_min = Min(static_cast<int64_t>(i_rel_idx), hInMaxIdx);
        int64_t i_max = Min(i_min + (int64_t)1, hInMaxIdx);
        int64_t yIndexOffset = i_min - xMin;
        int64_t offset = i * matmulTiling_h->singleCoreK;
        int64_t indexMin = yIndexOffset + offset;
        if (i_min == i_max) {
            radioTensor.SetValue(indexMin, 1);
        } else {
            float i_lambda_1 = getLambda(i_rel_idx, i_min);
            float i_lambda_0 = 1 - i_lambda_1;
            radioTensor.SetValue(indexMin, i_lambda_0);
            int64_t indexMax = 1 + yIndexOffset + offset;
            radioTensor.SetValue(indexMax, i_lambda_1);
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
__aicore__ inline void UpsampleLinear2dND<T>::copyRadioTensorToGm(int8_t direction)
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
__aicore__ inline LocalTensor<T> UpsampleLinear2dND<T>::initRadioTensor(int8_t direction)
{
    if (direction == 0) {
        return radioQueue_w.DeQue<T>();
    } else {
        return radioQueue_h.DeQue<T>();
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::releaseRadioTensor(int8_t direction, LocalTensor<T> radioTensor)
{
    if (direction == 0) {
        return radioQueue_w.FreeTensor(radioTensor);
    } else {
        return radioQueue_h.FreeTensor(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::calculateWidthExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    if (singleCoreK > 0) {
        int64_t singleCoreM = matmulTiling_w->singleCoreM;
        int64_t singleCoreN = matmulTiling_w->singleCoreN;
        // 尾块batch分批处理
        if (rowEnd != 0) {
            singleCoreM = rowEnd - rowStart;
        }
        matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);
        matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

        if (tensorCIndex + slide_size_w > output_shapes[3]) {
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
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::calculateHeightExtension(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    int64_t singleCoreN = matmulTiling_h->singleCoreN;

    if (tensorCIndex + slide_size_h > output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    }
    matmulH.SetOrgShape(singleCoreM, output_shapes[3], matmulTiling_h->singleCoreK, output_shapes[2], output_shapes[3]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size_h > output_shapes[2]) {
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
__aicore__ inline void UpsampleLinear2dND<T>::ParseTilingData(const UpsampleLinear2dTilingData *tilingData)
{
    mode = tilingData->mode;
    alignCorners = tilingData->align_corners;
    slide_size_w = tilingData->slide_size_w;
    slide_size_h = tilingData->slide_size_h;
    scaleW = tilingData->scale_w;
    scaleH = tilingData->scale_h;

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

    wInMaxIdx = input_shapes[3] - 1;
    hInMaxIdx = input_shapes[2] - 1;
}

template <typename T>
__aicore__ inline void UpsampleLinear2dND<T>::getSlideRange()
{
    slideStart_w = blockIdx * eachCoreSlideNumW * slide_size_w;
    slideEnd_w = (Min((blockIdx + 1) * eachCoreSlideNumW, slideNumW)) * slide_size_w;
    int64_t groupIndex = groupCoreNumW > 0 ? blockIdx / groupCoreNumW : 0;
    if (groupIndex < remainderW) {
        tailSlideStart_w = (tailStartSlideNumW + groupIndex) * slide_size_w;
        tailSlideEnd_w = Min(tailSlideStart_w + slide_size_w, output_shapes[3]);
        int64_t blockIdxInGroup = groupCoreNumW > 0 ? blockIdx % groupCoreNumW : 0;
        tailRowStart_w = blockIdxInGroup * tailAvergingRowsW;
        tailRowEnd_w = Min(tailRowStart_w + tailAvergingRowsW, input_shapes[0] * input_shapes[1] * input_shapes[2]);
    }

    slideStart_h = blockIdx * eachCoreSlideNumH * slide_size_h;
    slideEnd_h = (Min((blockIdx + 1) * eachCoreSlideNumH, slideNumH)) * slide_size_h;
    groupIndex = groupCoreNumH > 0 ? blockIdx / groupCoreNumH : 0;
    if (groupIndex < remainderH) {
        tailSlideStart_h = (tailStartSlideNumH + groupIndex) * slide_size_h;
        tailSlideEnd_h = Min(tailSlideStart_h + slide_size_h, output_shapes[2]);
        int64_t blockIdxInGroup = groupCoreNumH > 0 ? blockIdx % groupCoreNumH : 0;
        tailRowStart_h = blockIdxInGroup * tailAvergingRowsH;
        tailRowEnd_h = Min(tailRowStart_h + tailAvergingRowsH, input_shapes[0] * input_shapes[1]);
    }
}

}  // namespace UpsampleLinear2d

#endif  // UPSAMPLE_BILINEAR2D_AA

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
 * \file upsample_bicubic2d_310p.h
 * \brief
 */
#ifndef UPSAMPLE_BICUBIC2D_310P
#define UPSAMPLE_BICUBIC2D_310P

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2d {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t EACH_SLICEHANDLE_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

constexpr int8_t MIN_SIZE = 1;
constexpr int8_t TWO_SIZE = 2;

const int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
const int32_t DEFAULT_SLICE_SIZE = 16;
const int32_t DEFAULT_CLEAR_UB_SIZE = 10 * 1024;
const int64_t DEFAULT_UB_MAX_DATA_COUNT = 512;

template <typename T>
class UpsampleBicubic2dND310p {
public:
    TPipe pipe;

    __aicore__ inline UpsampleBicubic2dND310p(){};
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

    __aicore__ inline void CalculateIntermediateTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void CalculateRatioTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void CalculateConvolution(int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH);
    __aicore__ inline void ClearGM();
    __aicore__ inline void BicubicComputeBatch();
    __aicore__ inline void BicubicComputeTail();
    __aicore__ inline void CopyIn(int64_t indexInput, int64_t calCount);
    __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CubicInterp2d(int64_t indexW, int64_t indexH, int64_t offsetW, int64_t offsetH);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xIntQueueW;
    TBuf<QuePosition::VECCALC> xMinQueueW;
    TBuf<QuePosition::VECCALC> xVQueueW;
    TBuf<QuePosition::VECCALC> ratioQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xIntQueueH;
    TBuf<QuePosition::VECCALC> xMinQueueH;
    TBuf<QuePosition::VECCALC> xVQueueH;
    TBuf<QuePosition::VECCALC> ratioQueueH;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    TBuf<TPosition::VECCALC> cacheTensorBuff;
    TBuf<TPosition::VECCALC> castInputBuff;
    TBuf<TPosition::VECCALC> castOutputBuff;
    TBuf<TPosition::VECCALC> clearTensorBuff;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    LocalTensor<float> xMinTensorW;
    LocalTensor<float> xMinTensorH;
    LocalTensor<float> ratioTensorW;
    LocalTensor<float> ratioTensorH;
    LocalTensor<float> cacheTensor;
    LocalTensor<float> castInputTensor;
    LocalTensor<float> castOutputTensor;
    LocalTensor<T> clearTensor;

    int64_t blockIdx = 0;
    int64_t slideSize = 0;
    float scaleW;
    float scaleH;
    bool alignCorners;
    int64_t dataType;

    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int32_t blockSize = 8;
    int64_t startIdxW;
    int64_t startIdxH;
    int64_t batchLength;
    int64_t needCoreNum;

    uint32_t maxDataCount = DEFAULT_UB_MAX_DATA_COUNT;
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dTilingData *tilingData)
{
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    pipe.InitBuffer(centerQueueW, maxDataCount * sizeof(float));           // 2k
    pipe.InitBuffer(xIntQueueW, maxDataCount * sizeof(float));             // 2k
    pipe.InitBuffer(xMinQueueW, maxDataCount * sizeof(float));             // 2k
    pipe.InitBuffer(xVQueueW, maxDataCount * sizeof(float));               // 2k
    pipe.InitBuffer(ratioQueueW, DEFAULT_SLICE_SIZE * 4 * sizeof(float));  // 256

    pipe.InitBuffer(centerQueueH, maxDataCount * sizeof(float));           // 2k
    pipe.InitBuffer(xIntQueueH, maxDataCount * sizeof(float));             // 2k
    pipe.InitBuffer(xMinQueueH, maxDataCount * sizeof(float));             // 2k
    pipe.InitBuffer(xVQueueH, maxDataCount * sizeof(float));               // 2k
    pipe.InitBuffer(ratioQueueH, DEFAULT_SLICE_SIZE * 4 * sizeof(float));  // 256

    pipe.InitBuffer(inputQueue, BUFFER_NUM, maxDataCount * sizeof(float));   // 4k
    pipe.InitBuffer(outputQueue, BUFFER_NUM, maxDataCount * sizeof(float));  // 4k
    pipe.InitBuffer(cacheTensorBuff, maxDataCount * sizeof(float));          // 2k
    pipe.InitBuffer(castInputBuff, maxDataCount * sizeof(float));            // 2k
    pipe.InitBuffer(castOutputBuff, maxDataCount * sizeof(float));           // 2k
    pipe.InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));     // 20k or 40k

    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::Process()
{
    ClearGM();
    SyncAll();

    BicubicComputeBatch();
    BicubicComputeTail();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::ClearGM()
{
    LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
    int64_t totalNum = outputH * outputW * inputN * inputC;
    int64_t totalBlockNum = (totalNum + blockSize - 1) / blockSize;
    int64_t preCoreBlockCnt = totalBlockNum / needCoreNum;
    int64_t tailBlockCnt = totalBlockNum % needCoreNum;
    int32_t realNeedCore = 1;
    if (preCoreBlockCnt > 0) {
        realNeedCore = needCoreNum;
    }
    if (blockIdx >= realNeedCore) {
        return;
    }
    int64_t preCoreDataCnt = preCoreBlockCnt * blockSize;
    int32_t loopCnt = preCoreDataCnt / DEFAULT_CLEAR_UB_SIZE;
    int64_t tailCnt = preCoreDataCnt % DEFAULT_CLEAR_UB_SIZE;
    int64_t offset = blockIdx * preCoreDataCnt;

    Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

    event_t eventIdVToMTE3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    for (int i = 0; i < loopCnt; i++) {
        DataCopy(outTensorsGM[offset], clearUb, DEFAULT_CLEAR_UB_SIZE);
        offset += DEFAULT_CLEAR_UB_SIZE;
    }
    if (tailCnt > 0) {
        tailCnt = (tailCnt + blockSize - 1) / blockSize * blockSize;
        DataCopy(outTensorsGM[offset], clearUb, tailCnt);
    }
    if ((tailBlockCnt > 0) && (blockIdx == 0)) {
        tailCnt = tailBlockCnt * blockSize;
        offset = preCoreDataCnt * realNeedCore;
        DataCopy(outTensorsGM[offset], clearUb, tailCnt);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::BicubicComputeBatch()
{
    // 计算批量分组的数据
    if (slideStartW >= slideEndW) {
        return;
    }
    slideEndW = Min(slideEndW, outputW);
    int64_t slideOffset = slideEndW - slideStartW;
    int64_t loopCntW = (slideOffset + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    int64_t loopCntH = (outputH + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    for (int64_t loopIdxW = 0; loopIdxW < loopCntW; loopIdxW++) {
        startIdxW = slideStartW + loopIdxW * DEFAULT_UB_MAX_DATA_COUNT;
        int64_t ratioLengthW = Min(slideEndW - startIdxW, DEFAULT_UB_MAX_DATA_COUNT);
        int64_t endIdxW = Min(slideEndW, startIdxW + DEFAULT_UB_MAX_DATA_COUNT);
        CalculateIntermediateTensor(startIdxW, ratioLengthW, W_DIRECTION);
        for (int64_t loopIdxH = 0; loopIdxH < loopCntH; loopIdxH++) {
            startIdxH = loopIdxH * DEFAULT_UB_MAX_DATA_COUNT;
            int64_t ratioLengthH = Min(outputH - startIdxH, DEFAULT_UB_MAX_DATA_COUNT);
            int64_t endIdxH = Min(outputH, startIdxH + DEFAULT_UB_MAX_DATA_COUNT);
            CalculateIntermediateTensor(startIdxH, ratioLengthH, H_DIRECTION);
            for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
                int64_t lengthW = Min(slideSize, endIdxW - indexW);
                CalculateRatioTensor(indexW - startIdxW, lengthW, W_DIRECTION);
                for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
                    int64_t lengthH = Min(slideSize, endIdxH - indexH);
                    CalculateRatioTensor(indexH - startIdxH, lengthH, H_DIRECTION);
                    CalculateConvolution(indexW, indexH, lengthW, lengthH);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::BicubicComputeTail()
{
    // 处理尾块部分数据
    if (tailSlideStartW >= tailSlideEndW) {
        return;
    }
    int64_t slideOffset = tailSlideEndW - tailSlideStartW;
    int64_t loopCntW = (slideOffset + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    int64_t tailRowCnt = tailRowEndW - tailRowStartW;
    int64_t loopCntH = (tailRowCnt + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;
    for (int64_t loopIdxW = 0; loopIdxW < loopCntW; loopIdxW++) {
        startIdxW = tailSlideStartW + loopIdxW * DEFAULT_UB_MAX_DATA_COUNT;
        int64_t ratioLengthW = Min(DEFAULT_UB_MAX_DATA_COUNT, tailSlideEndW - startIdxW);
        int64_t endIdxW = Min(tailSlideEndW, startIdxW + DEFAULT_UB_MAX_DATA_COUNT);
        CalculateIntermediateTensor(startIdxW, ratioLengthW, W_DIRECTION);
        for (int64_t loopIdxH = 0; loopIdxH < loopCntH; loopIdxH++) {
            startIdxH = tailRowStartW + loopIdxH * DEFAULT_UB_MAX_DATA_COUNT;
            int64_t ratioLengthH = Min(DEFAULT_UB_MAX_DATA_COUNT, tailRowEndW - startIdxH);
            int64_t endIdxH = Min(tailRowEndW, startIdxH + DEFAULT_UB_MAX_DATA_COUNT);
            CalculateIntermediateTensor(startIdxH, ratioLengthH, H_DIRECTION);
            for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
                int64_t lengthW = Min(slideSize, endIdxW - indexW);
                CalculateRatioTensor(indexW - startIdxW, lengthW, W_DIRECTION);
                for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
                    int64_t lengthH = Min(slideSize, endIdxH - indexH);
                    CalculateRatioTensor(indexH - startIdxH, lengthH, H_DIRECTION);
                    CalculateConvolution(indexW, indexH, lengthW, lengthH);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CalculateIntermediateTensor(
    int64_t index, int64_t length, int8_t direction)
{
    length = Max(length, EACH_SLICEHANDLE_NUM);
    float scale = scaleW;
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> xIntTensor = xIntQueueW.Get<float>();
    LocalTensor<float> xMinTensor = xMinQueueW.Get<float>();
    LocalTensor<float> xVTensor = xVQueueW.Get<float>();
    if (direction == H_DIRECTION) {
        scale = scaleH;
        centerTensor = centerQueueH.Get<float>();
        xIntTensor = xIntQueueH.Get<float>();
        xMinTensor = xMinQueueH.Get<float>();
        xVTensor = xVQueueH.Get<float>();
    }
#if __CCE_AICORE__ == 200
    ArithProgression(centerTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
#else
    for (int32_t i = 0; i < length; i++) {
        centerTensor.SetValue(i, static_cast<float>(index + i));
    }
#endif

    // 计算center下标
    if (alignCorners) {
        // 角对齐
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        // 边对齐
        for (int64_t i = 0; i < length; i++) {
            float center = ((float)0.5 + static_cast<float>(index + i)) * scale - (float)0.5;
            centerTensor.SetValue(i, center);
        }
    }

    Floor(xIntTensor, centerTensor, length);
    Adds(xMinTensor, xIntTensor, (float)(-1.0), length);
    PipeBarrier<PIPE_V>();
    Maxs(xMinTensor, xMinTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
    Sub(xVTensor, centerTensor, xIntTensor, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CalculateRatioTensor(
    int64_t xIndex, int64_t length, int8_t direction)
{
    LocalTensor<float> ratioTensor = ratioQueueW.Get<float>();
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> xIntTensor = xIntQueueW.Get<float>();
    LocalTensor<float> xMinTensor = xMinQueueW.Get<float>();
    LocalTensor<float> xVTensor = xVQueueW.Get<float>();
    int64_t boundSize = inputW;
    if (direction == H_DIRECTION) {
        ratioTensor = ratioQueueH.Get<float>();
        centerTensor = centerQueueH.Get<float>();
        xIntTensor = xIntQueueH.Get<float>();
        xMinTensor = xMinQueueH.Get<float>();
        xVTensor = xVQueueH.Get<float>();
        boundSize = inputH;
    }

    // 计算系数矩阵
    Duplicate(ratioTensor, (float)0.0, ratioTensor.GetSize());

    int64_t xMin = static_cast<int64_t>(xMinTensor.GetValue(xIndex));
    for (int64_t i = 0; i < length; i++) {
        int64_t xSize = 4;
        int64_t idx = i + xIndex;
        if (static_cast<int64_t>(xMinTensor.GetValue(idx)) + 4 > boundSize) {
            xSize = boundSize - static_cast<int64_t>(xMinTensor.GetValue(idx));
        }
        for (int64_t j = 0; j < xSize; j++) {
            float w = weightCalculate(xVTensor.GetValue(idx), xIntTensor.GetValue(idx), j, boundSize);
            int64_t weightIndex = j + i * 4;
            ratioTensor.SetValue(weightIndex, w);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CalculateConvolution(
    int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH)
{
    xMinTensorW = xMinQueueW.Get<float>();
    xMinTensorH = xMinQueueH.Get<float>();
    ratioTensorW = ratioQueueW.Get<float>();
    ratioTensorH = ratioQueueH.Get<float>();
    cacheTensor = cacheTensorBuff.Get<float>();
    castInputTensor = castInputBuff.Get<float>();
    castOutputTensor = castOutputBuff.Get<float>();

    for (int64_t i = 0; i < lengthH; i++) {
        for (int64_t j = 0; j < lengthW; j++) {
            CubicInterp2d(indexW, indexH, j, i);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CopyIn(int64_t indexInput, int64_t calCount)
{
    LocalTensor<T> srcDataLocal = inputQueue.AllocTensor<T>();
    DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount);
    inputQueue.EnQue(srcDataLocal);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CopyOut(int64_t indexOutput, int64_t calCount)
{
    LocalTensor<T> dstDataLocal = outputQueue.DeQue<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
    } else {
        int64_t blockCalCount = (calCount + blockSize - 1) / blockSize * blockSize;
        SetAtomicAdd<T>();
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, blockCalCount);
        SetAtomicNone();
    }

    outputQueue.FreeTensor(dstDataLocal);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::CubicInterp2d(
    int64_t indexW, int64_t indexH, int64_t offsetW, int64_t offsetH)
{
    int64_t startX = static_cast<int64_t>(xMinTensorW.GetValue(indexW + offsetW - startIdxW));
    int64_t startY = static_cast<int64_t>(xMinTensorH.GetValue(indexH + offsetH - startIdxH));
    int32_t xSize = (inputW - startX) > 4 ? 4 : (inputW - startX);
    int32_t ySize = (inputH - startY) > 4 ? 4 : (inputH - startY);
    int32_t loopCnt = (batchLength + DEFAULT_UB_MAX_DATA_COUNT - 1) / DEFAULT_UB_MAX_DATA_COUNT;

    for (int32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        int64_t startIdx = loopIdx * DEFAULT_UB_MAX_DATA_COUNT;
        int64_t calCount = Min(DEFAULT_UB_MAX_DATA_COUNT, batchLength - startIdx);
        int64_t blockCalCount = (calCount + blockSize - 1) / blockSize * blockSize;

        if (dataType == 2) {
            LocalTensor<float> dstDataLocal = outputQueue.AllocTensor<float>();
            Duplicate(dstDataLocal, (float)0, DEFAULT_UB_MAX_DATA_COUNT);
            for (int32_t y = 0; y < ySize; y++) {
                Duplicate(cacheTensor, (float)0, blockCalCount);
                for (int32_t x = 0; x < xSize; x++) {
                    int64_t indexInput = ((startX + x) + (startY + y) * inputW) * batchLength + startIdx;
                    CopyIn(indexInput, blockCalCount);
                    LocalTensor<float> srcDataLocal = inputQueue.DeQue<float>();
                    float weightW = ratioTensorW.GetValue(offsetW * 4 + x);
                    Muls(srcDataLocal, srcDataLocal, weightW, calCount);
                    Add(cacheTensor, cacheTensor, srcDataLocal, calCount);
                    inputQueue.FreeTensor(srcDataLocal);
                }
                float weightH = ratioTensorH.GetValue(offsetH * 4 + y);
                Muls(cacheTensor, cacheTensor, weightH, calCount);
                Add(dstDataLocal, dstDataLocal, cacheTensor, calCount);
            }
            outputQueue.EnQue(dstDataLocal);
            int64_t indexOutput = ((indexW + offsetW) + (indexH + offsetH) * outputW) * batchLength + startIdx;
            CopyOut(indexOutput, calCount);
        } else {
            LocalTensor<T> dstDataLocal = outputQueue.AllocTensor<T>();
            Duplicate(dstDataLocal, (T)0, DEFAULT_UB_MAX_DATA_COUNT);
            Duplicate(castOutputTensor, (float)0, DEFAULT_UB_MAX_DATA_COUNT);
            for (int32_t y = 0; y < ySize; y++) {
                Duplicate(cacheTensor, (float)0, blockCalCount);
                for (int32_t x = 0; x < xSize; x++) {
                    int64_t indexInput = ((startX + x) + (startY + y) * inputW) * batchLength + startIdx;
                    CopyIn(indexInput, blockCalCount);
                    LocalTensor<T> srcDataLocal = inputQueue.DeQue<T>();
                    float weightW = ratioTensorW.GetValue(offsetW * 4 + x);
                    Cast(castInputTensor, srcDataLocal, RoundMode::CAST_NONE, blockCalCount);
                    Muls(castInputTensor, castInputTensor, weightW, calCount);
                    Add(cacheTensor, cacheTensor, castInputTensor, calCount);
                    inputQueue.FreeTensor(srcDataLocal);
                }
                float weightH = ratioTensorH.GetValue(offsetH * 4 + y);
                Muls(cacheTensor, cacheTensor, weightH, calCount);
                Add(castOutputTensor, castOutputTensor, cacheTensor, calCount);
            }
            Cast(dstDataLocal, castOutputTensor, RoundMode::CAST_NONE, blockCalCount);
            outputQueue.EnQue(dstDataLocal);
            int64_t indexOutput = ((indexW + offsetW) + (indexH + offsetH) * outputW) * batchLength + startIdx;
            CopyOut(indexOutput, calCount);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dND310p<T>::ParseTilingData(UpsampleBicubic2dTilingData *tilingData)
{
    slideSize = DEFAULT_SLICE_SIZE;
    scaleW = tilingData->scale_w;
    scaleH = tilingData->scale_h;
    alignCorners = tilingData->align_corners;
    needCoreNum = tilingData->need_core_num_w;

    inputH = tilingData->input_shapes[0];
    inputW = tilingData->input_shapes[1];
    inputN = tilingData->input_shapes[2];
    inputC = tilingData->input_shapes[3];
    outputH = tilingData->output_shapes[0];
    outputW = tilingData->output_shapes[1];

    batchLength = inputN * inputC;

    slideStartW = tilingData->slideStartList_w[blockIdx];
    slideEndW = tilingData->slideEndList_w[blockIdx];
    tailSlideStartW = tilingData->tailSlideStartList_w[blockIdx];
    tailSlideEndW = tilingData->tailSlideEndList_w[blockIdx];
    tailRowStartW = tilingData->tailRowStartList_w[blockIdx];
    tailRowEndW = tilingData->tailRowEndList_w[blockIdx];

    dataType = tilingData->dataType;

    blockSize = 32 / sizeof(T);
}
}  // namespace UpsampleBicubic2d

#endif  // UPSAMPLE_BICUBIC2D_310P
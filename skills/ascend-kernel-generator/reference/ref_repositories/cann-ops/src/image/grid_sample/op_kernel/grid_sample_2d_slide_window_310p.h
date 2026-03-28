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
 * \file grid_sampler_2d_slide_window_310p.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_SLIDE_WINDOW_310P
#define GRID_SAMPLER_2D_SLIDE_WINDOW_310P

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

struct Mte2Param {
    int64_t coordVal_0;
    int64_t xLocation_0;
    int64_t coordVal_1;
    int64_t xLocation_1;

    __aicore__ inline Mte2Param()
    {}

    __aicore__ inline Mte2Param(int64_t coordVal_0, int64_t xLocation_0)
        : coordVal_0(coordVal_0), xLocation_0(xLocation_0)
    {
        coordVal_1 = 0;
        xLocation_1 = 0;
    }

    __aicore__ inline Mte2Param(int64_t coordVal_0, int64_t xLocation_0, int64_t coordVal_1, int64_t xLocation_1)
        : coordVal_0(coordVal_0), xLocation_0(xLocation_0), coordVal_1(coordVal_1), xLocation_1(xLocation_1)
    {}
};

template <typename T>
class GridSampler2DSlideWindow310P {
public:
    __aicore__ inline GridSampler2DSlideWindow310P(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign);
    __aicore__ inline void ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub, LocalTensor<float> x1Ub,
        LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
        LocalTensor<uint8_t> weightMaskUb, uint16_t id, int32_t nIdx, int32_t hwIdx);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb, LocalTensor<uint8_t> maskTmpXUb,
        LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
        LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum);
    __aicore__ inline void CoordinatesSelectTensor(
        LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb, LocalTensor<uint8_t> maskUb,
        LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow, const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal);
    __aicore__ inline void MTE2ForNHWCType1(int32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal);
    __aicore__ inline Mte2Param GetMte2ParamForType1(int64_t base, int64_t doubleChannelAlign,
        int64_t forthChannelAlign, uint64_t wcLength, LocalTensor<int32_t> coorUb, int64_t offset, int64_t indexOffset,
        int16_t index);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);
    __aicore__ inline void calculateEachPointValue(int32_t nIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueUbSum);
    __aicore__ inline void PointBilinear2(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<float> weightUb2,
        LocalTensor<float> weightUb3, LocalTensor<float> weightUb4, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<uint8_t> weightMaskUb2, LocalTensor<uint8_t> weightMaskUb3, LocalTensor<uint8_t> weightMaskUb4,
        LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void MTE3ForNCHW(int16_t cIdx, int32_t calCElems, int32_t loopOffset, int16_t loopElems,
        int64_t outBaseOffset, LocalTensor<float> outValueUbSum);
    __aicore__ inline void ResetGMToZero();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gridQueue_;

    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> inputXYFPBuf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputXFpBuf_;
    TBuf<QuePosition::VECCALC> inputYFpBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> weightTmpBuf_;
    TBuf<QuePosition::VECCALC> weightTmp1Buf_;
    TBuf<QuePosition::VECCALC> weightTmp2Buf_;
    TBuf<QuePosition::VECCALC> weightTmp3Buf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> coorBuf2_;
    TBuf<QuePosition::VECCALC> coorBuf3_;
    TBuf<QuePosition::VECCALC> coorBuf4_;

    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf2_;

    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf2_;

    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> maskBuf3_;
    TBuf<QuePosition::VECCALC> maskBuf4_;
    TBuf<QuePosition::VECCALC> maskBuf6_;
    TBuf<QuePosition::VECCALC> maskBuf8_;
    TBuf<QuePosition::VECCALC> maskBuf9_;

    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf2_;
    TBuf<QuePosition::VECCALC> weightMaskBuf3_;
    TBuf<QuePosition::VECCALC> weightMaskBuf4_;
    TBuf<QuePosition::VECCALC> weightMaskBuf5_;
    TBuf<QuePosition::VECCALC> weightMaskBuf6_;
    TBuf<QuePosition::VECCALC> weightMaskBuf7_;
    TBuf<QuePosition::VECCALC> weightMaskBuf8_;
    TBuf<QuePosition::VECCALC> weightMaskBuf9_;

    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;
    TBuf<QuePosition::VECCALC> inputMaxXYFpBuf_;
    TBuf<QuePosition::VECCALC> inputMaxXYIntBuf_;
    TBuf<QuePosition::VECCALC> dupBuf_;
    TBuf<QuePosition::VECCALC> dupBuf3_;
    TBuf<QuePosition::VECCALC> dupBuf4_;
    TBuf<QuePosition::VECCALC> dupBuf6_;
    TBuf<QuePosition::VECCALC> dupBuf8_;
    TBuf<QuePosition::VECCALC> dupBuf9_;

    TBuf<QuePosition::VECCALC> bufferMaskBuf_;
    TBuf<QuePosition::VECCALC> bufferBuf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t CHANNEL_BLOCK = 32;
    const int64_t BLOCK_SIZE = 32;
    const int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);

    const int64_t B32_MASK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;
    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / BLOCK_NUM;
    const int64_t MASK_SIZE = 960;
    const int64_t WEIGHT_MASK_SIZE = 320;

    const int64_t OUT_UB_SIZE_4_GENERAL = 65536;
    const int64_t OUT_UB_SIZE_GENERAL = 16384;
    const int64_t X_UB_SIZE_4_GENERAL = 65536;

    int64_t blockIDX = 0;

    // tiling params
    int64_t coreNum_ = 0;
    int64_t inputN_ = 0;
    int64_t inputC_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t interpolationMode_ = 0;
    int64_t paddingMode_ = 0;
    int64_t alignCorners_ = 0;
    int64_t channelLast_ = 0;
    int64_t needCoreNum_ = 0;

    int64_t gridHW_ = 0;
    int64_t lastLoopHW_ = 0;
    int64_t lastLoopHWAlign_ = 0;
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;

    int64_t alignmentType_ = 0;
    int64_t dataCopyType = 0;

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;
    constexpr static int64_t LAYOUT_NHWC = 1;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;

    constexpr static int64_t SLIDING_WINDOW_C_LIMIT = 16;

    constexpr static int64_t ALIGNMENT_TYPE_1 = 1;
    constexpr static int64_t DATA_COPY_TYPE_1 = 1;

    constexpr static uint16_t U_INT16_T_MAX_VALUE = 65535;
};

/**
 * @description: 解析tiling数据，计算分核数据
 * @param {GridSampleTilingData*} tilingData
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::ParseTilingData(const GridSampleTilingData *tilingData)
{
    coreNum_ = tilingData->coreNumVar;
    inputN_ = tilingData->inN;
    inputC_ = tilingData->inC;
    inputH_ = tilingData->inH;
    inputW_ = tilingData->inW;
    outputH_ = tilingData->outH;
    outputW_ = tilingData->outW;
    interpolationMode_ = tilingData->interpolationMode;
    paddingMode_ = tilingData->paddingMode;
    alignCorners_ = tilingData->alignCorners;
    channelLast_ = tilingData->channelLast;
    needCoreNum_ = tilingData->needCoreNum;
    gridHW_ = outputH_ * outputW_;
    preNUbLoop_ = Ceil(gridHW_, CAL_H_W_BLOCK);                        // 1
    lastLoopHW_ = gridHW_ - CAL_H_W_BLOCK * (preNUbLoop_ - 1);         // 300
    lastLoopHWAlign_ = Ceil(lastLoopHW_, BLOCK_NUM) * BLOCK_NUM;       // 32位对齐 304
    totalUbLoop_ = preNUbLoop_ * inputN_;                              // 9
    preCoreLoop_ = Ceil(totalUbLoop_, needCoreNum_);                   // 2
    needCoreNum_ = Ceil(totalUbLoop_, preCoreLoop_);                   // 5
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);  //

    channelLoop_ = Ceil(inputC_, CHANNEL_BLOCK);
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);

    if (gridHW_ % TRANSE_REP_STRIDE < BLOCK_NUM && gridHW_ % TRANSE_REP_STRIDE != 0) {
        alignmentType_ = ALIGNMENT_TYPE_1;
    }

    if (inputW_ * inputC_ / BLOCK_NUM > U_INT16_T_MAX_VALUE) {
        dataCopyType = DATA_COPY_TYPE_1;
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ T *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer申请初始化
    pipe.InitBuffer(gridQueue_, 1, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(dupBuf_, 2048);                          // 2KB
    pipe.InitBuffer(dupBuf3_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf4_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf6_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf8_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf9_, 2048);                         // 2KB

    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);              // 64KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputXIntBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputYIntBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputXFpBuf_, GRID_UB_SIZE_4_GENERAL);    // 4KB
    pipe.InitBuffer(inputYFpBuf_, GRID_UB_SIZE_4_GENERAL);    // 4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);     // 8KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);  // 8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);         // 2KB
    pipe.InitBuffer(intTmpBuf2_, Y_UB_SIZE_4_GENERAL);        // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);           // 2KB

    pipe.InitBuffer(outValueBuf_, OUT_UB_SIZE_4_GENERAL);  // 64KB
    pipe.InitBuffer(outValueBuf2_, OUT_UB_SIZE_GENERAL);   // 64KB

    pipe.InitBuffer(maskBuf_, MASK_SIZE);   // 960B
    pipe.InitBuffer(maskBuf3_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf4_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf6_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf8_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf9_, MASK_SIZE);  // 960B

    pipe.InitBuffer(weightMaskBuf_, WEIGHT_MASK_SIZE);   // 320B
    pipe.InitBuffer(weightMaskBuf2_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf3_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf4_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf5_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf6_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf7_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf8_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf9_, WEIGHT_MASK_SIZE);  // 320B

    pipe.InitBuffer(modBuf_, Y_UB_SIZE_4_GENERAL);        // 2KB
    pipe.InitBuffer(extraBuf_, Y_UB_SIZE_4_GENERAL);      // 2KB
    pipe.InitBuffer(outTmpBuf_, GRID_UB_SIZE_4_GENERAL);  // 4KB

    pipe.InitBuffer(bufferMaskBuf_, BLOCK_SIZE);              // 32B
    pipe.InitBuffer(bufferBuf_, BLOCK_SIZE * CHANNEL_BLOCK);  // 4K

    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
    bufPattern.SetValue(0, 0b11111111);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::ComputeWeightSub(LocalTensor<float> w1Ub,
    LocalTensor<float> w2Ub, LocalTensor<float> x1Ub, LocalTensor<float> x2Ub, LocalTensor<float> y1Ub,
    LocalTensor<float> y2Ub)
{
    Sub(w1Ub, x1Ub, x2Ub, CAL_H_W_BLOCK);
    Sub(w2Ub, y1Ub, y2Ub, CAL_H_W_BLOCK);
}

/**
 * @description: 计算grid中的x、y的一维坐标和越界点的mask值
 * @param {LocalTensor<float>} iXFpUb
 * @param {LocalTensor<float>} iYFpUb
 * @param {LocalTensor<int32_t>} iXIntUb
 * @param {LocalTensor<int32_t>} iYIntUb
 * @param {LocalTensor<int32_t>} out coorUb
 * @param {LocalTensor<uint8_t>} out wMaskUb
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::ClipCoordinates(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> wMaskUb, uint16_t id, int32_t nIdx, int32_t hwIdx)
{
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);

    if (id == 1) {
        LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
        LocalTensor<int32_t> inputXIntTmpUb = coorUb;
        LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
        Adds<int32_t, false>(inputXIntTmpUb, iXIntUb, 0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
        Adds<int32_t, false>(inputYIntTmpUb, iYIntUb, 0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        // 重计算坐标，使坐标不超过边界
        CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ - 1));
        CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));

        PipeBarrier<PIPE_V>();

        // cood = y + x * IW
        Muls<int32_t, false>(
            inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<int32_t, false>(
            coorUb, coorUb, inputYIntTmpUb, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});
    }
}

/**
 * @description: 原坐标越界时计算新坐标
 * @param {LocalTensor<float>} X坐标
 * @param {LocalTensor<float>} Y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    // 这里只计算border和reflection的场景，zeros的场景对标cpu实现，这边先不处理，后面再处理
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(iXFpUb, iYFpUb);
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(iXFpUb, iYFpUb);
    }
}

/**
 * @description: 坐标超过上下界的处理
 * @param {LocalTensor<int32_t>} x or y
 * @param {int32_t} 上界
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::CoordinatesFrameRange(
    LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iIntUb, iIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

/**
 * @description: 取出合法坐标点：maskXUb，maskYUb
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    // maskTmpXUb存的是大于0的合法点
    CompareScalar<float, uint8_t, false>(
        maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});

    // maskXUb存的是小于inputW_的合法点
    CompareScalar<float, uint8_t, false>(maskXUb,
        iXFpUb,
        static_cast<float>(inputW_ - 1),
        CMPMODE::LE,
        MASK_PLACEHOLDER,
        CAL_H_W_BLOCK / B32_MASK,
        {1, 1, 8, 8});
    // maskTmpYUb存的是大于0的合法点
    CompareScalar<float, uint8_t, false>(
        maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    // maskYUb存的是小于inputH_的合法点
    CompareScalar<float, uint8_t, false>(maskYUb,
        iYFpUb,
        static_cast<float>(inputH_ - 1),
        CMPMODE::LE,
        MASK_PLACEHOLDER,
        CAL_H_W_BLOCK / B32_MASK,
        {1, 1, 8, 8});

    PipeBarrier<PIPE_V>();

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // 合并上面的两个结果，得到最终合法点
    And<uint16_t, false>(
        maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    And<uint16_t, false>(
        maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    And<uint16_t, false>(
        maskXUbTmp, maskYUbTmp, maskXUbTmp, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
    LocalTensor<float> oFpUb, LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = Ceil(calNum, B32_VECTOR_MASK);
    Select<float, uint8_t, false>(
        oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::CoordinatesSelectTensor(
    LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = Ceil(CAL_H_W_BLOCK, B32_VECTOR_MASK);
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

/**
 * @description: PaddingMode：Border
 * @param {LocalTensor<float>} x坐标
 * @param {LocalTensor<float>} y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

/**
 * @description: PaddingMode：Reflection
 * @param {LocalTensor<float>} x坐标
 * @param {LocalTensor<float>} y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::ReflectClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    LocalTensor<float> extraFpUb = extraBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> fmodFpUb = modBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    // coorUb = Y * inputW_ + X
    int64_t twiceLow = (alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowY = REFLECT_RATIO * (inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (inputW_ - 1);
    if (alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowY = REFLECT_RATIO * inputH_ - 1;
        twiceLowX = REFLECT_RATIO * inputW_ - 1;
    }
    ReflectCoordinatesGeneral(iYFpUb, iYFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowY);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(iXFpUb, iXFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowX);
    PipeBarrier<PIPE_V>();

    LocalTensor<T> tmpUb = inputXYFPBuf_.Get<T>();
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
    LocalTensor<float> coorSubUb, LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
    LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
    const int64_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        Duplicate(coorSubUb, (float)0.0, CAL_H_W_BLOCK);
        return;
    }

    float minS = static_cast<float>(twiceLow) / 2;
    float negMinS = static_cast<float>(-1.0) * minS;
    float spanS = static_cast<float>(twiceHigh - twiceLow) / 2;

    // new relative position
    Adds(coorSubUb, iFpUb, negMinS, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Abs(coorSubUb, coorSubUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(extraFpUb, extraFpUb, spanS, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(extraFpUb, coorSubUb, extraFpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // coordinate
    /*
     S1: get two results for both possibilities, out1: extra + min, out2: muls(extra, -1.0) + span + min
     S2: get mod val, mods: flips % 2
     S3: get mask tensor, masks: CompareScalar(mods, 0.0)
     S4: select val from out1 and out2 by mask tensor, out: Select(out1, out2, mask)
    */
    LocalTensor<float> out1 = tmpFpUb;
    LocalTensor<float> out2 = extraFpUb;
    LocalTensor<float> mods = fmodFpUb;

    Adds(out1, extraFpUb, minS, CAL_H_W_BLOCK);
    Muls(out2, extraFpUb, -1.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, spanS, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, minS, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(mods, coorSubUb, static_cast<float>(1 / 2.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(mods, mods, 2.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(mods, coorSubUb, mods, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CoordinatesSelectTensor(out1, out2, coorSubUb, maskUb);
}

/**
 * @description: x是NCHW的format，一个点一个点的搬，搬align(c)次，会多搬进来align(c)-c个数
 * @param {int32_t} nIdx
 * @param {int32_t} cIdx
 * @param {int32_t} calCElems
 * @param {int32_t} channelAlign
 * @param {int32_t} loopOffset
 * @param {int32_t} loopElems
 * @param {LocalTensor<int32_t>} coorUb
 * @param {LocalTensor<float>} xLocal
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::MTE2ForNCHW(int32_t nIdx, int16_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal)
{
    for (int16_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation =
            (int64_t)nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int16_t cIter = 0; cIter < channelAlign; cIter++) {
            int32_t xLocalOffset = i * channelAlign + cIter;
            if (cIter >= calCElems) {
                xLocal.SetValue(xLocalOffset, 0.0f);
                continue;
            }

            int64_t coordinate = baseLocation + cIter * inputH_ * inputW_;
            xLocal.SetValue(xLocalOffset, gmX_.GetValue(coordinate));
        }
    }
}

/**
 * @description: x是NHWC的format，连续搬运align(c)个，按hw循环
 * @param {int32_t} nIdx
 * @param {int32_t} cIdx
 * @param {int32_t} calCElems
 * @param {int32_t} channelAlign
 * @param {int32_t} loopOffset
 * @param {int32_t} loopElems
 * @param {LocalTensor<int32_t>} coorUb
 * @param {LocalTensor<float>} xLocal
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::MTE2ForNHWC(int32_t nIdx, int16_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal)
{
    int64_t base = (int64_t)nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;

    auto timeStep = loopElems / 8;

    int32_t calCElemsAlign = Ceil(calCElems, 8) * 8;
    int64_t doubleChannelAlign = static_cast<int64_t>(channelAlign) * 4;
    uint16_t blockCount = 2;
    uint16_t blockLen = calCElemsAlign * 2 / 8;
    uint16_t wcLength = inputW_ * inputC_ / 8;
    uint16_t srcStride = wcLength - blockLen;
    uint16_t dstStride = 0;
    if (wcLength < blockLen) {
        blockLen = calCElemsAlign / 8;
        srcStride = 0;
        dstStride = blockLen;
    }
    DataCopyParams params{blockCount, blockLen, srcStride, dstStride};

    // 这边为了不打断流水，提高性能
    for (int16_t i = 0; i < timeStep; i++) {
        int64_t offset = loopOffset + i * 8;
        int64_t coordVal_0 = base + coorUb.GetValue(offset);
        int64_t coordVal_1 = base + coorUb.GetValue(offset + 1);
        int64_t coordVal_2 = base + coorUb.GetValue(offset + 2);
        int64_t coordVal_3 = base + coorUb.GetValue(offset + 3);
        int64_t coordVal_4 = base + coorUb.GetValue(offset + 4);
        int64_t coordVal_5 = base + coorUb.GetValue(offset + 5);
        int64_t coordVal_6 = base + coorUb.GetValue(offset + 6);
        int64_t coordVal_7 = base + coorUb.GetValue(offset + 7);

        int64_t xLocation_0 = (i * 8) * doubleChannelAlign;
        int64_t xLocation_1 = xLocation_0 + doubleChannelAlign;
        int64_t xLocation_2 = xLocation_1 + doubleChannelAlign;
        int64_t xLocation_3 = xLocation_2 + doubleChannelAlign;
        int64_t xLocation_4 = xLocation_3 + doubleChannelAlign;
        int64_t xLocation_5 = xLocation_4 + doubleChannelAlign;
        int64_t xLocation_6 = xLocation_5 + doubleChannelAlign;
        int64_t xLocation_7 = xLocation_6 + doubleChannelAlign;

        DataCopy(xLocal[xLocation_0], gmX_[coordVal_0], params);
        DataCopy(xLocal[xLocation_1], gmX_[coordVal_1], params);
        DataCopy(xLocal[xLocation_2], gmX_[coordVal_2], params);
        DataCopy(xLocal[xLocation_3], gmX_[coordVal_3], params);
        DataCopy(xLocal[xLocation_4], gmX_[coordVal_4], params);
        DataCopy(xLocal[xLocation_5], gmX_[coordVal_5], params);
        DataCopy(xLocal[xLocation_6], gmX_[coordVal_6], params);
        DataCopy(xLocal[xLocation_7], gmX_[coordVal_7], params);
    }

    for (int16_t i = loopElems / 8 * 8; i < loopElems; i++) {
        int64_t coordVal_0 = base + coorUb.GetValue(loopOffset + i);
        DataCopy(xLocal[i * doubleChannelAlign], gmX_[coordVal_0], params);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::MTE2ForNHWCType1(int32_t nIdx, int16_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal)
{
    int64_t base = (int64_t)nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;

    auto timeStep = loopElems / 8;

    int32_t calCElemsAlign = Ceil(calCElems, 8) * 8;
    int64_t doubleChannelAlign = static_cast<int64_t>(channelAlign) * 2;
    int64_t forthChannelAlign = static_cast<int64_t>(channelAlign) * 4;
    uint16_t blockLen = calCElemsAlign * 2;
    uint64_t wcLength = inputW_ * inputC_;

    // 这边为了不打断流水，提高性能
    for (int16_t i = 0; i < timeStep; i++) {
        int64_t offset = loopOffset + i * 8;
        int64_t indexOffset = i * 8;
        Mte2Param mte2Param0 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 0);
        Mte2Param mte2Param1 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 1);
        Mte2Param mte2Param2 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 2);
        Mte2Param mte2Param3 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 3);
        Mte2Param mte2Param4 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 4);
        Mte2Param mte2Param5 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 5);
        Mte2Param mte2Param6 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 6);
        Mte2Param mte2Param7 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, coorUb, offset, indexOffset, 7);

        DataCopy(xLocal[mte2Param0.xLocation_0], gmX_[mte2Param0.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param1.xLocation_0], gmX_[mte2Param1.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param2.xLocation_0], gmX_[mte2Param2.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param3.xLocation_0], gmX_[mte2Param3.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param4.xLocation_0], gmX_[mte2Param4.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param5.xLocation_0], gmX_[mte2Param5.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param6.xLocation_0], gmX_[mte2Param6.coordVal_0], blockLen);
        DataCopy(xLocal[mte2Param7.xLocation_0], gmX_[mte2Param7.coordVal_0], blockLen);

        DataCopy(xLocal[mte2Param0.xLocation_1], gmX_[mte2Param0.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param1.xLocation_1], gmX_[mte2Param1.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param2.xLocation_1], gmX_[mte2Param2.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param3.xLocation_1], gmX_[mte2Param3.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param4.xLocation_1], gmX_[mte2Param4.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param5.xLocation_1], gmX_[mte2Param5.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param6.xLocation_1], gmX_[mte2Param6.coordVal_1], blockLen);
        DataCopy(xLocal[mte2Param7.xLocation_1], gmX_[mte2Param7.coordVal_1], blockLen);
    }

    for (int16_t i = loopElems / 8 * 8; i < loopElems; i++) {
        int64_t coordVal_0 = base + coorUb.GetValue(loopOffset + i);
        DataCopy(xLocal[i * forthChannelAlign], gmX_[coordVal_0], blockLen);
        DataCopy(xLocal[i * forthChannelAlign + doubleChannelAlign], gmX_[coordVal_0 + wcLength], blockLen);
    }
}

template <typename T>
__aicore__ inline Mte2Param GridSampler2DSlideWindow310P<T>::GetMte2ParamForType1(int64_t base,
    int64_t doubleChannelAlign, int64_t forthChannelAlign, uint64_t wcLength, LocalTensor<int32_t> coorUb,
    int64_t offset, int64_t indexOffset, int16_t index)
{
    int64_t coordVal_0 = base + coorUb.GetValue(offset + index);
    int64_t xLocation_0 = (indexOffset + index) * forthChannelAlign;
    int64_t coordVal_0_1 = coordVal_0 + wcLength;
    int64_t xLocation_0_1 = xLocation_0 + doubleChannelAlign;
    Mte2Param mte2Param = Mte2Param(coordVal_0, xLocation_0, coordVal_0_1, xLocation_0_1);
    return mte2Param;
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::OutTranspose(
    int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb)
{
    if (channelAlign == 8) {
        v4dtrans(((__ubuf__ uint32_t *)outValueUb.GetPhyAddr()),
            ((__ubuf__ uint32_t *)xLocal.GetPhyAddr()),
            TRANSE_REP_STRIDE,
            channelAlign * 4,
            1);
    } else if (channelAlign <= 64) {
        v4dtrans(((__ubuf__ uint32_t *)outValueUb.GetPhyAddr()),
            ((__ubuf__ uint32_t *)xLocal.GetPhyAddr()),
            TRANSE_REP_STRIDE,
            channelAlign * 4,
            1);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::calculateEachPointValue(int32_t nIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
    LocalTensor<float> outValueUbSum)
{
    for (int16_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
        int32_t outOffset = i * B32_MASK;
        int32_t weightOffset = loopOffset + i * B32_MASK;
        Mul<float, false>(outValueUb[outOffset],
            outValueUb[outOffset],
            weightUb[weightOffset],
            MASK_PLACEHOLDER,
            calCElems,
            {1, 1, 1, 16, 16, 0});
    }
    PipeBarrier<PIPE_V>();
    Add<float, false>(outValueUbSum,
        outValueUbSum,
        outValueUb,
        MASK_PLACEHOLDER,
        TRANSE_REP_STRIDE * channelAlign / B32_MASK,
        {1, 1, 1, 8, 8, 8});
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::PointBilinear2(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
    int32_t calHWElemsAlign, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<float> weightUb2, LocalTensor<float> weightUb3, LocalTensor<float> weightUb4,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<uint8_t> weightMaskUb2, LocalTensor<uint8_t> weightMaskUb3,
    LocalTensor<uint8_t> weightMaskUb4, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式

    Muls<int32_t, false>(
        coordinatesUb, coordinatesUb, (int32_t)inputC_, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});

    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(weightUb2, weightUb2, weightMaskUb2, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(weightUb3, weightUb3, weightMaskUb3, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(weightUb4, weightUb4, weightMaskUb4, 0.0f, CAL_H_W_BLOCK);
    }

    LocalTensor<float> outValueUbSum = outValueBuf2_.Get<float>();

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半

    auto weightMaskUbTmp1 = weightMaskUb.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp2 = weightMaskUb2.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp3 = weightMaskUb3.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp4 = weightMaskUb4.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb5 = weightMaskBuf5_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp5 = weightMaskUb5.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb6 = weightMaskBuf6_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp6 = weightMaskUb6.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb7 = weightMaskBuf7_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp7 = weightMaskUb7.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb8 = weightMaskBuf8_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp8 = weightMaskUb8.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb9 = weightMaskBuf9_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp9 = weightMaskUb9.ReinterpretCast<uint16_t>();

    if (calHWElemsAlign < CAL_H_W_BLOCK) {
        int maskOffset = Ceil(calHWElemsAlign, 16);
        Duplicate(weightMaskUbTmp1[maskOffset], (uint16_t)0, maskNum - maskOffset);
    }

    And<uint16_t, false>(weightMaskUbTmp5,
        weightMaskUbTmp1,
        weightMaskUbTmp3,
        MASK_PLACEHOLDER,
        MASK_UB_SIZE / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    weightMaskUb5 = weightMaskUbTmp5.ReinterpretCast<uint8_t>();
    And<uint16_t, false>(weightMaskUbTmp7,
        weightMaskUbTmp1,
        weightMaskUbTmp2,
        MASK_PLACEHOLDER,
        MASK_UB_SIZE / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    weightMaskUb7 = weightMaskUbTmp7.ReinterpretCast<uint8_t>();
    And<uint16_t, false>(weightMaskUbTmp6,
        weightMaskUbTmp3,
        weightMaskUbTmp4,
        MASK_PLACEHOLDER,
        MASK_UB_SIZE / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    weightMaskUb6 = weightMaskUbTmp6.ReinterpretCast<uint8_t>();
    And<uint16_t, false>(weightMaskUbTmp9,
        weightMaskUbTmp2,
        weightMaskUbTmp4,
        MASK_PLACEHOLDER,
        MASK_UB_SIZE / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    weightMaskUb9 = weightMaskUbTmp9.ReinterpretCast<uint8_t>();
    Or<uint16_t, false>(weightMaskUbTmp8,
        weightMaskUbTmp2,
        weightMaskUbTmp3,
        MASK_PLACEHOLDER,
        MASK_UB_SIZE / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    weightMaskUb8 = weightMaskUbTmp8.ReinterpretCast<uint8_t>();

    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb3 = maskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb4 = maskBuf4_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb6 = maskBuf6_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb8 = maskBuf8_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb9 = maskBuf9_.Get<uint8_t>(MASK_UB_SIZE);

    auto weightMaskUbTmp = weightMaskUb7.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_3 = weightMaskUb5.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_4 = weightMaskUb4.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_6 = weightMaskUb6.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_8 = weightMaskUb8.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_9 = weightMaskUb9.ReinterpretCast<uint64_t>();

    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();
    auto maskUbTmp3 = maskUb3.ReinterpretCast<uint64_t>();
    auto maskUbTmp4 = maskUb4.ReinterpretCast<uint64_t>();
    auto maskUbTmp6 = maskUb6.ReinterpretCast<uint64_t>();
    auto maskUbTmp8 = maskUb8.ReinterpretCast<uint64_t>();
    auto maskUbTmp9 = maskUb9.ReinterpretCast<uint64_t>();

    int32_t trans_loop = Ceil(calHWElemsAlign, TRANSE_REP_STRIDE);
    int16_t loopElems = TRANSE_REP_STRIDE;
    int32_t loopOffset = 0;
    int64_t outBaseOffset = (int64_t)nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK;
    int32_t ubOffset = 0;
    int32_t maskOffset = 0;
    BinaryRepeatParams repParams{1, 1, 1, 8, 8, 8};

    LocalTensor<float> xLocal = xBuf_.AllocTensor<float>();
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

    auto dupUb4 = dupBuf4_.Get<float>();
    auto dupUbU32_4 = dupUb4.ReinterpretCast<uint32_t>();
    auto dupUb6 = dupBuf6_.Get<float>();
    auto dupUbU32_6 = dupUb6.ReinterpretCast<uint32_t>();
    auto dupUb8 = dupBuf8_.Get<float>();
    auto dupUbU32_8 = dupUb8.ReinterpretCast<uint32_t>();
    auto dupUb = dupBuf_.Get<float>();
    auto dupUbU32 = dupUb.ReinterpretCast<uint32_t>();
    auto dupUb3 = dupBuf3_.Get<float>();
    auto dupUbU32_3 = dupUb3.ReinterpretCast<uint32_t>();
    auto dupUb9 = dupBuf9_.Get<float>();
    auto dupUbU32_9 = dupUb9.ReinterpretCast<uint32_t>();

    // 按vmask(128)分块，循环处理
    for (int16_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            loopElems = calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        loopOffset = loop_idx * TRANSE_REP_STRIDE;
        maskOffset = loop_idx * 2;

        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));
        maskUbTmp.SetValue(2, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(3, weightMaskUbTmp.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32 = maskUbTmp.ReinterpretCast<float>();

        maskUbTmp3.SetValue(0, weightMaskUbTmp_3.GetValue(maskOffset));
        maskUbTmp3.SetValue(1, weightMaskUbTmp_3.GetValue(maskOffset + 1));
        maskUbTmp3.SetValue(2, weightMaskUbTmp_3.GetValue(maskOffset));
        maskUbTmp3.SetValue(3, weightMaskUbTmp_3.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_3 = maskUbTmp3.ReinterpretCast<float>();

        maskUbTmp4.SetValue(0, weightMaskUbTmp_4.GetValue(maskOffset));
        maskUbTmp4.SetValue(1, weightMaskUbTmp_4.GetValue(maskOffset + 1));
        maskUbTmp4.SetValue(2, weightMaskUbTmp_4.GetValue(maskOffset));
        maskUbTmp4.SetValue(3, weightMaskUbTmp_4.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_4 = maskUbTmp4.ReinterpretCast<float>();

        maskUbTmp6.SetValue(0, weightMaskUbTmp_6.GetValue(maskOffset));
        maskUbTmp6.SetValue(1, weightMaskUbTmp_6.GetValue(maskOffset + 1));
        maskUbTmp6.SetValue(2, weightMaskUbTmp_6.GetValue(maskOffset));
        maskUbTmp6.SetValue(3, weightMaskUbTmp_6.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_6 = maskUbTmp6.ReinterpretCast<float>();

        maskUbTmp8.SetValue(0, weightMaskUbTmp_8.GetValue(maskOffset));
        maskUbTmp8.SetValue(1, weightMaskUbTmp_8.GetValue(maskOffset + 1));
        maskUbTmp8.SetValue(2, weightMaskUbTmp_8.GetValue(maskOffset));
        maskUbTmp8.SetValue(3, weightMaskUbTmp_8.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_8 = maskUbTmp8.ReinterpretCast<float>();

        maskUbTmp9.SetValue(0, weightMaskUbTmp_9.GetValue(maskOffset));
        maskUbTmp9.SetValue(1, weightMaskUbTmp_9.GetValue(maskOffset + 1));
        maskUbTmp9.SetValue(2, weightMaskUbTmp_9.GetValue(maskOffset));
        maskUbTmp9.SetValue(3, weightMaskUbTmp_9.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_9 = maskUbTmp9.ReinterpretCast<float>();

        // channel先按64大小循环
        for (int16_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }
            int32_t channelAlign = Ceil(calCElems, BLOCK_NUM) * BLOCK_NUM;
            if (channelLast_ == LAYOUT_NHWC && dataCopyType != DATA_COPY_TYPE_1) {
                MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal);
            } else if (channelLast_ == LAYOUT_NHWC && dataCopyType == DATA_COPY_TYPE_1) {
                MTE2ForNHWCType1(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal);
            } else {
                MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal);
            }

            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);
            OutTranspose(channelAlign, xLocal, outValueUb);

            LocalTensor<float> outValueUb2 = outValueUb[channelAlign * (TRANSE_REP_STRIDE)];
            LocalTensor<float> outValueUb3 = outValueUb2[channelAlign * (TRANSE_REP_STRIDE)];
            LocalTensor<float> outValueUb4 = outValueUb3[channelAlign * (TRANSE_REP_STRIDE)];

            if (loop_idx > 0) {
                WaitFlag<HardEvent::MTE3_V>(eventMte3V);
            }

            Duplicate(outValueUbSum, (float)0.0, outValueUbSum.GetSize());
            uint32_t dstShape[2] = {Ceil(calCElems, 32 * 8 / TRANSE_REP_STRIDE), (uint32_t)8};
            uint32_t srcShape[2] = {1, (uint32_t)8};

            BroadCast<float, 2, 0>(dupUb9, weightMaskUbTmpfp32_9, dstShape, srcShape);
            PipeBarrier<PIPE_V>();

            Select<float, uint32_t, false>(outValueUb4,
                dupUbU32_9,
                outValueUb4,
                outValueUb2,
                SELMODE::VSEL_TENSOR_TENSOR_MODE,
                64,
                calCElems * (TRANSE_REP_STRIDE / 64),
                repParams);

            BroadCast<float, 2, 0>(dupUb6, weightMaskUbTmpfp32_6, dstShape, srcShape);

            PipeBarrier<PIPE_V>();
            Select<float, uint32_t, false>(outValueUb4,
                dupUbU32_6,
                outValueUb4,
                outValueUb3,
                SELMODE::VSEL_TENSOR_TENSOR_MODE,
                64,
                calCElems * (TRANSE_REP_STRIDE / 64),
                repParams);

            BroadCast<float, 2, 0>(dupUb8, weightMaskUbTmpfp32_8, dstShape, srcShape);

            PipeBarrier<PIPE_V>();
            Select<float, uint32_t, false>(outValueUb4,
                dupUbU32_8,
                outValueUb4,
                outValueUb,
                SELMODE::VSEL_TENSOR_TENSOR_MODE,
                64,
                calCElems * (TRANSE_REP_STRIDE / 64),
                repParams);

            BroadCast<float, 2, 0>(dupUb4, weightMaskUbTmpfp32_4, dstShape, srcShape);

            BroadCast<float, 2, 0>(dupUb, weightMaskUbTmpfp32, dstShape, srcShape);

            BroadCast<float, 2, 0>(dupUb3, weightMaskUbTmpfp32_3, dstShape, srcShape);

            PipeBarrier<PIPE_V>();
            Select<float, uint32_t, false>(outValueUb2,
                dupUbU32,
                outValueUb2,
                outValueUb,
                SELMODE::VSEL_TENSOR_TENSOR_MODE,
                64,
                calCElems * (TRANSE_REP_STRIDE / 64),
                repParams);
            Select<float, uint32_t, false>(outValueUb3,
                dupUbU32_3,
                outValueUb3,
                outValueUb,
                SELMODE::VSEL_TENSOR_TENSOR_MODE,
                64,
                calCElems * (TRANSE_REP_STRIDE / 64),
                repParams);
            Select<float, uint32_t, false>(outValueUb4,
                dupUbU32_4,
                outValueUb4,
                (float)0.0,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                64,
                calCElems * (TRANSE_REP_STRIDE / 64),
                repParams);
            PipeBarrier<PIPE_V>();

            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb, outValueUb, outValueUbSum);
            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb2, outValueUb2, outValueUbSum);
            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb3, outValueUb3, outValueUbSum);
            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb4, outValueUb4, outValueUbSum);

            MTE3ForNCHW(cIdx, calCElems, loopOffset, loopElems, outBaseOffset, outValueUbSum);
            if (loop_idx < trans_loop - 1 && trans_loop != 1) {
                SetFlag<HardEvent::MTE3_V>(eventMte3V);
            }
        }
    }
    ResetMask();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::MTE3ForNCHW(int16_t cIdx, int32_t calCElems, int32_t loopOffset,
    int16_t loopElems, int64_t outBaseOffset, LocalTensor<float> outValueUbSum)
{
    int64_t gmYBaseOffset = outBaseOffset + loopOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    int16_t loopElemsAlign = Ceil(loopElems, 8) * 8;
    uint32_t mask = 8;
    uint64_t rsvdCnt = 0;
    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
    LocalTensor<float> bufTensor = bufferBuf_.Get<float>();

    // 这个提出来时间优化略等于无
    if (alignmentType_ != ALIGNMENT_TYPE_1) {
        if (loopElemsAlign != loopElems) {
            PipeBarrier<PIPE_V>();
            for (int16_t i = 0; i < calCElems; i++) {
                int64_t outputOffset = i * TRANSE_REP_STRIDE;
                GatherMask(bufTensor[i * BLOCK_NUM],
                    outValueUbSum[outputOffset + loopElems - BLOCK_NUM],
                    bufPattern,
                    true,
                    mask,
                    {1, 1, 8, 8},
                    rsvdCnt);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (int16_t i = 0; i < calCElems; i++) {
            int64_t gmYOffset = gmYBaseOffset + i * gridHW_;
            int64_t outputOffset = i * TRANSE_REP_STRIDE;
            DataCopy(gmY_[gmYOffset], outValueUbSum[outputOffset], loopElems);
            if (loopElemsAlign != loopElems) {
                DataCopy(gmY_[gmYOffset + loopElems - BLOCK_NUM], bufTensor[i * BLOCK_NUM], BLOCK_NUM);
            }
        }
    } else {
        if (loopElemsAlign != loopElems) {
            PipeBarrier<PIPE_V>();
            for (int16_t i = 0; i < calCElems; i++) {
                int64_t outputOffset = i * TRANSE_REP_STRIDE;
                Duplicate(outValueUbSum[outputOffset + loopElems], 0.0f, loopElemsAlign - loopElems);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (int16_t i = 0; i < calCElems; i++) {
            int64_t gmYOffset = gmYBaseOffset + i * gridHW_;
            int64_t outputOffset = i * TRANSE_REP_STRIDE;
            if (loopElemsAlign == loopElems) {
                DataCopy(gmY_[gmYOffset], outValueUbSum[outputOffset], loopElems);
            } else {
                SetAtomicAdd<float>();
                DataCopy(gmY_[gmYOffset], outValueUbSum[outputOffset], loopElemsAlign);
                SetAtomicNone();
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::PerLoopCompute(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;

    LocalTensor<T> gridLocal = gridQueue_.AllocTensor<T>();
    DataCopy(gridLocal, gmGrid_[gridGmOffset], calHWElemsAlign * 2);
    gridQueue_.EnQue(gridLocal);
    gridQueue_.DeQue();

    LocalTensor<T> inputXYUb = inputXYFPBuf_.Get<T>();
    // 加1后，grid的datarange从-1~1到0~2
    Adds(inputXYUb, gridLocal, (float)1.0, CAL_H_W_BLOCK * 2);

    uint32_t mask = CAL_H_W_BLOCK * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;

    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;
    PipeBarrier<PIPE_V>();
    LocalTensor<float> inputXFpLocal = gridLocal;
    LocalTensor<float> inputYFpLocal = gridLocal[CAL_H_W_BLOCK];
    // 分别取x和y
    GatherMask(inputXFpLocal, inputXYUb, xPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYUb, yPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式

    // 不同alignCorners_的unnormlize处理
    if (alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (inputW_ - (float)1.0)), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * (inputH_ - (float)1.0)), CAL_H_W_BLOCK);
    } else {
        Muls<float, false>(inputXFpLocal,
            inputXFpLocal,
            (float)((float)0.5 * inputW_),
            MASK_PLACEHOLDER,
            CAL_H_W_BLOCK / B32_MASK,
            {1, 1, 8, 8});
        Muls<float, false>(inputYFpLocal,
            inputYFpLocal,
            (float)((float)0.5 * inputH_),
            MASK_PLACEHOLDER,
            CAL_H_W_BLOCK / B32_MASK,
            {1, 1, 8, 8});

        PipeBarrier<PIPE_V>();
        Adds<float, false>(
            inputXFpLocal, inputXFpLocal, (float)(-0.5), MASK_PLACEHOLDER, CAL_H_W_BLOCK * 2 / B32_MASK, {1, 1, 8, 8});
    }
    PipeBarrier<PIPE_V>();

    // 处理越界坐标
    Clip(inputXFpLocal, inputYFpLocal);

    LocalTensor<int32_t> inputXWIntLocal = inputXIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXEIntLocal = inputXIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<int32_t> inputYWIntLocal = inputYIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputYEIntLocal = inputYIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);

    LocalTensor<float> inputXWFpLocal = inputXFpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> inputXEFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> inputYWFpLocal = inputYFpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> inputYEFpLocal = inputYFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);

    Cast(inputXWIntLocal, inputXFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    Cast(inputYWIntLocal, inputYFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(inputXWFpLocal, inputXWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputYWFpLocal, inputYWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    // 分别计算左上，右上，左下，右下的坐标
    Adds<int32_t, false>(inputXEIntLocal, inputXWIntLocal, 1, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    Adds<int32_t, false>(inputYEIntLocal, inputYWIntLocal, 1, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    Adds<float, false>(
        inputXEFpLocal, inputXWFpLocal, (float)1.0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    Adds<float, false>(
        inputYEFpLocal, inputYWFpLocal, (float)1.0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();

    LocalTensor<float> nwWeightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> neWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> swWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> seWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    LocalTensor<float> weightTmpLocal = weightTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> weightTmp1Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> weightTmp2Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> weightTmp3Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    Sub<float, false>(
        weightTmpLocal, inputXEFpLocal, inputXFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(
        weightTmp1Local, inputXFpLocal, inputXWFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(
        weightTmp2Local, inputYEFpLocal, inputYFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(
        weightTmp3Local, inputYFpLocal, inputYWFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});

    PipeBarrier<PIPE_V>();
    Mul<float, false>(
        nwWeightLocal, weightTmpLocal, weightTmp2Local, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(neWeightLocal,
        weightTmp1Local,
        weightTmp2Local,
        MASK_PLACEHOLDER,
        CAL_H_W_BLOCK / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    Mul<float, false>(
        swWeightLocal, weightTmpLocal, weightTmp3Local, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(seWeightLocal,
        weightTmp1Local,
        weightTmp3Local,
        MASK_PLACEHOLDER,
        CAL_H_W_BLOCK / B32_MASK,
        {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> weightMaskUb2 = weightMaskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> weightMaskUb3 = weightMaskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> weightMaskUb4 = weightMaskBuf4_.Get<uint8_t>(MASK_UB_SIZE);

    // 划窗条件不满足，走兜底分支
    ClipCoordinates(inputXWFpLocal,
        inputYWFpLocal,
        inputXWIntLocal,
        inputYWIntLocal,
        coordinatesLocal,
        weightMaskUb,
        1,
        nIdx,
        hwIdx);
    ClipCoordinates(inputXEFpLocal,
        inputYWFpLocal,
        inputXEIntLocal,
        inputYWIntLocal,
        coordinatesLocal,
        weightMaskUb2,
        2,
        nIdx,
        hwIdx);
    ClipCoordinates(inputXWFpLocal,
        inputYEFpLocal,
        inputXWIntLocal,
        inputYEIntLocal,
        coordinatesLocal,
        weightMaskUb3,
        3,
        nIdx,
        hwIdx);
    ClipCoordinates(inputXEFpLocal,
        inputYEFpLocal,
        inputXEIntLocal,
        inputYEIntLocal,
        coordinatesLocal,
        weightMaskUb4,
        4,
        nIdx,
        hwIdx);
    ResetMask();

    PointBilinear2(nIdx,
        hwIdx,
        calHWElems,
        calHWElemsAlign,
        coordinatesLocal,
        nwWeightLocal,
        neWeightLocal,
        swWeightLocal,
        seWeightLocal,
        weightMaskUb,
        weightMaskUb2,
        weightMaskUb3,
        weightMaskUb4,
        outValueLocal,
        true);

    gridQueue_.FreeTensor(gridLocal);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::ResetGMToZero()
{
    LocalTensor<float> outValueLocal = outValueBuf2_.AllocTensor<float>();
    Duplicate(outValueLocal, (float)0.0, outValueLocal.GetSize());
    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;  // 每个核开始的block数

    int64_t loopSize = preCoreLoop_;  // 要处理的block数量
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;   // N维的index
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;  // h、w在block中位置
        if (hwIdx == preNUbLoop_ - 1) {
            for (int64_t cIdx = 0; cIdx < inputC_; cIdx++) {
                int64_t gmYBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK + cIdx * gridHW_;
                DataCopy(gmY_[gmYBaseOffset], outValueLocal, lastLoopHWAlign_);
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow310P<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calHWElems = 0;
    int32_t calHWElemsAlign = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    if (gridHW_ < BLOCK_NUM || alignmentType_ == ALIGNMENT_TYPE_1) {
        ResetGMToZero();
    }

    for (int16_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = CAL_H_W_BLOCK;
        calHWElemsAlign = CAL_H_W_BLOCK;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
            calHWElemsAlign = lastLoopHWAlign_;
        }
        PerLoopCompute(nIdx, hwIdx, calHWElems, calHWElemsAlign);
    }
}

}  // namespace GridSample
#endif  // GRID_SAMPLER_2D_SLIDE_WINDOW_310P
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
 * \file grid_sampler_2d_slide_window.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_SLIDE_WINDOW
#define GRID_SAMPLER_2D_SLIDE_WINDOW

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grid_sample_2d_common.h"

namespace GridSample {

using namespace AscendC;

struct SlideCoorParam {
    int32_t xMin = 0;
    int32_t xMax = 0;
    int32_t yMin = 0;
    int32_t yMax = 0;

    __aicore__ inline SlideCoorParam()
    {}

    __aicore__ inline SlideCoorParam(int32_t xMin, int32_t xMax, int32_t yMin, int32_t yMax)
        : xMin(xMin), xMax(xMax), yMin(yMin), yMax(yMax)
    {}
};

template <typename T>
class GridSampler2DSlideWindow {
public:
    __aicore__ inline GridSampler2DSlideWindow(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(ProcessParam2D processParam);
    __aicore__ inline void ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub, LocalTensor<float> x1Ub,
        LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
        LocalTensor<uint8_t> weightMaskUb);

    __aicore__ inline void ClipXYCoordinates(
        InputTensorStruct2D InputTensorStruct2D, LocalTensor<int32_t> inputXIntTmpUb, LocalTensor<uint8_t> wMaskUb);

    __aicore__ inline void ClipCoordinatesXInLocal(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
        LocalTensor<uint8_t> weightMaskUb, int32_t xMin, int32_t xMax, int32_t yMin, int32_t yMax);
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

    __aicore__ inline void ReflectCoordinatesGeneralSelect(LocalTensor<float> coorSubUb, float minS, float spanS);

    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb, LocalTensor<uint8_t> maskUb,
        LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow, const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal, int32_t idx);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);
    __aicore__ inline void MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset, LocalTensor<float> outValueUb);

    __aicore__ inline void calculatePointBilinear(int32_t nIdx, LocalTensor<int32_t> coordinatesUb,
        LocalTensor<float> outValueUb, LocalTensor<float> outValueTotalLocal, LocalTensor<float> weightUb,
        LocalTensor<uint64_t> maskUbTmp, int32_t loopElems, int32_t loopOffset, LocalTensor<float> weightMaskUbTmpfp32,
        LocalTensor<float> xLocal, int32_t cIdx, int32_t calCElems, int32_t channelAlign, bool isAutomicAdd,
        int32_t idx);
    __aicore__ inline void initTensor();
    __aicore__ inline void initMaskTensor();

    __aicore__ inline void PointBilinearSetMask(int32_t maskOffset);

    __aicore__ inline void PointBilinearEachChannel(ProcessParam2D processParam, LocalTensor<float> outValueUb,
        PointParam2D pointBilinearParam, LocalTensor<float> xLocal, LocalTensor<float> outValueTotalLocal);

    __aicore__ inline void PointBilinear(ProcessParam2D processParam, LocalTensor<float> outValueUb);

    __aicore__ inline void calculatePointBilinearXInLocal(int32_t calHWElems, LocalTensor<int32_t> coordinatesUb,
        LocalTensor<float> weightUb, LocalTensor<float> outValueUb, LocalTensor<float> outValueTotalLocal,
        bool isAutomicAdd, LocalTensor<float> xLocal, LocalTensor<uint64_t> weightMaskUbTmp, int32_t cIdx,
        int32_t calCElems);

    __aicore__ inline void CalculateGrid(ProcessParam2D processParam, int64_t gridGmOffset, LocalTensor<T> gridLocal);

    __aicore__ inline void GetInputTensor();

    __aicore__ inline void calculateGridWeight();

    __aicore__ inline void GetNoSlideWindow(ProcessParam2D processParam, LocalTensor<T> inputMaxXYFpUb,
        LocalTensor<int32_t> inputMaxXYIntUb, SlideCoorParam &slideCoorParam, bool &noSlideWindow);

    __aicore__ inline void PointBilinearInSlideWindow(
        ProcessParam2D processParam, LocalTensor<float> outValueLocal, SlideCoorParam slideCoorParam);

    __aicore__ inline void PointBilinearXInLocal(ProcessParam2D processParam, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueTotalLocal, bool isAutomicAdd, LocalTensor<float> xLocal);

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
    TBuf<QuePosition::VECCALC> coorTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> maskBuf2_;
    TBuf<QuePosition::VECCALC> maskBuf3_;
    TBuf<QuePosition::VECCALC> maskBuf4_;

    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf2_;
    TBuf<QuePosition::VECCALC> weightMaskBuf3_;
    TBuf<QuePosition::VECCALC> weightMaskBuf4_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;
    TBuf<QuePosition::VECCALC> inputMaxXYFpBuf_;
    TBuf<QuePosition::VECCALC> inputMaxXYIntBuf_;
    TBuf<QuePosition::VECCALC> dupBuf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmY_;

    LocalTensor<float> inputXFpLocal;
    LocalTensor<float> inputYFpLocal;
    LocalTensor<float> nwWeightLocal;
    LocalTensor<float> neWeightLocal;
    LocalTensor<float> swWeightLocal;
    LocalTensor<float> seWeightLocal;
    LocalTensor<int32_t> coordinatesLocal;
    LocalTensor<int32_t> coordinatesLocal2;
    LocalTensor<int32_t> coordinatesLocal3;
    LocalTensor<int32_t> coordinatesLocal4;
    LocalTensor<uint8_t> weightMaskUb;
    LocalTensor<uint8_t> weightMaskUb2;
    LocalTensor<uint8_t> weightMaskUb3;
    LocalTensor<uint8_t> weightMaskUb4;
    LocalTensor<uint64_t> weightMaskUbTmp;
    LocalTensor<uint64_t> weightMaskUbTmp2;
    LocalTensor<uint64_t> weightMaskUbTmp3;
    LocalTensor<uint64_t> weightMaskUbTmp4;

    LocalTensor<uint8_t> maskUb;
    LocalTensor<uint64_t> maskUbTmp;
    LocalTensor<float> weightMaskUbTmpfp32;
    LocalTensor<uint8_t> maskUb2;
    LocalTensor<uint64_t> maskUbTmp2;
    LocalTensor<float> weightMaskUbTmpfp32_2;
    LocalTensor<uint8_t> maskUb3;
    LocalTensor<uint64_t> maskUbTmp3;
    LocalTensor<float> weightMaskUbTmpfp32_3;
    LocalTensor<uint8_t> maskUb4;
    LocalTensor<uint64_t> maskUbTmp4;
    LocalTensor<float> weightMaskUbTmpfp32_4;

    LocalTensor<int32_t> inputXWIntLocal;
    LocalTensor<int32_t> inputXEIntLocal;
    LocalTensor<int32_t> inputYWIntLocal;
    LocalTensor<int32_t> inputYEIntLocal;
    LocalTensor<float> inputXWFpLocal;
    LocalTensor<float> inputXEFpLocal;
    LocalTensor<float> inputYWFpLocal;
    LocalTensor<float> inputYEFpLocal;

    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;
    const int64_t CHANNEL_BLOCK_X_IN_LOCAL = 8;

    const int64_t X_UB_SIZE_4_GENERAL = 81920;
    const int64_t OUT_UB_SIZE_4_GENERAL = 32768;
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t OUT_VAL_NUM = 4096;
    const int64_t X_UB_OFFSET = 512;
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t BLOCK_SIZE = 32;
    const int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / BLOCK_NUM;

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
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;
    int64_t channelXInLocalLoop_ = 0;
    int64_t perLoopChannelXInLocal_ = 0;
    int64_t lastLoopChannelXInLocal_ = 0;

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
};

/**
 * @description: 解析tiling数据，计算分核数据
 * @param {GridSampleTilingData*} tilingData
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ParseTilingData(const GridSampleTilingData *tilingData)
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
    preNUbLoop_ = Ceil(gridHW_, CAL_H_W_BLOCK);
    lastLoopHW_ = gridHW_ - CAL_H_W_BLOCK * (preNUbLoop_ - 1);
    totalUbLoop_ = preNUbLoop_ * inputN_;
    preCoreLoop_ = Ceil(totalUbLoop_, needCoreNum_);
    needCoreNum_ = Ceil(totalUbLoop_, preCoreLoop_);
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = Ceil(inputC_, CHANNEL_BLOCK);
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);

    channelXInLocalLoop_ = Ceil(inputC_, CHANNEL_BLOCK_X_IN_LOCAL);
    perLoopChannelXInLocal_ = CHANNEL_BLOCK_X_IN_LOCAL;
    lastLoopChannelXInLocal_ = inputC_ - perLoopChannelXInLocal_ * (channelXInLocalLoop_ - 1);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::Init(
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

    pipe.InitBuffer(inputMaxXYFpBuf_, 32);   // 32B
    pipe.InitBuffer(inputMaxXYIntBuf_, 32);  // 32B

    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);              // 80KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputXIntBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputYIntBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputXFpBuf_, GRID_UB_SIZE_4_GENERAL);    // 4KB
    pipe.InitBuffer(inputYFpBuf_, GRID_UB_SIZE_4_GENERAL);    // 4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);     // 8KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);  // 8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);         // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);           // 2KB
    pipe.InitBuffer(coorTmpBuf_, Y_UB_SIZE_4_GENERAL);        // 2KB
    pipe.InitBuffer(outValueBuf_, OUT_UB_SIZE_4_GENERAL);     // 32KB
    pipe.InitBuffer(maskBuf_, 960);                           // 960B
    pipe.InitBuffer(maskBuf2_, 960);                          // 960B
    pipe.InitBuffer(maskBuf3_, 960);                          // 960B
    pipe.InitBuffer(maskBuf4_, 960);                          // 960B

    pipe.InitBuffer(weightMaskBuf_, 320);   // 320B
    pipe.InitBuffer(weightMaskBuf2_, 320);  // 320B
    pipe.InitBuffer(weightMaskBuf3_, 320);  // 320B
    pipe.InitBuffer(weightMaskBuf4_, 320);  // 320B

    pipe.InitBuffer(modBuf_, Y_UB_SIZE_4_GENERAL);        // 2KB
    pipe.InitBuffer(extraBuf_, Y_UB_SIZE_4_GENERAL);      // 2KB
    pipe.InitBuffer(outTmpBuf_, GRID_UB_SIZE_4_GENERAL);  // 4KB
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub,
    LocalTensor<float> x1Ub, LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub)
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
__aicore__ inline void GridSampler2DSlideWindow<T>::ClipCoordinates(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
    LocalTensor<int32_t> inputXIntTmpUb, LocalTensor<uint8_t> wMaskUb)
{
    LocalTensor<int32_t> inputYIntTmpUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    InputTensorStruct2D inputTensorStruct2D{iXFpUb, iYFpUb, iXIntUb, iYIntUb};
    ClipXYCoordinates(inputTensorStruct2D, inputXIntTmpUb, wMaskUb);

    // cood = y + x * IW
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(inputXIntTmpUb, inputXIntTmpUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ClipXYCoordinates(
    InputTensorStruct2D inputTensorStruct2D, LocalTensor<int32_t> inputXIntTmpUb, LocalTensor<uint8_t> wMaskUb)
{
    LocalTensor<int32_t> inputYIntTmpUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    Adds(inputXIntTmpUb, inputTensorStruct2D.iXIntUb, 0, CAL_H_W_BLOCK);
    Adds(inputYIntTmpUb, inputTensorStruct2D.iYIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Cast(inputTensorStruct2D.iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputTensorStruct2D.iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(
        inputTensorStruct2D.iXFpUb, inputTensorStruct2D.iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    // 合法的x的mask
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    // 合法的y的mask
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // maskXUbTmp：合法的点的mask
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    wMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();

    // 重计算坐标，使坐标不超过边界
    CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ - 1));
    CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));

    PipeBarrier<PIPE_V>();
}

/**
 * @description: 滑框场景下计算坐标，相对滑框坐标系
 * @param {LocalTensor<float>} iXFpUb
 * @param {LocalTensor<float>} iYFpUb
 * @param {LocalTensor<int32_t>} iXIntUb
 * @param {LocalTensor<int32_t>} iYIntUb
 * @param {LocalTensor<int32_t>} out coorUb
 * @param {LocalTensor<uint8_t>} out wMaskUb
 * @param {int32_t} x_min
 * @param {int32_t} x_max
 * @param {int32_t} y_min
 * @param {int32_t} y_max
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ClipCoordinatesXInLocal(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
    LocalTensor<int32_t> inputXIntTmpUb, LocalTensor<uint8_t> wMaskUb, int32_t xMin, int32_t xMax, int32_t yMin,
    int32_t yMax)
{
    LocalTensor<int32_t> inputYIntTmpUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    InputTensorStruct2D inputTensorStruct2D{iXFpUb, iYFpUb, iXIntUb, iYIntUb};
    ClipXYCoordinates(inputTensorStruct2D, inputXIntTmpUb, wMaskUb);

    // 将原坐标系的坐标（相对顶点）转换成划窗坐标系坐标
    Adds(inputXIntTmpUb, inputXIntTmpUb, (int32_t)(-1 * xMin), CAL_H_W_BLOCK);
    Adds(inputYIntTmpUb, inputYIntTmpUb, (int32_t)(-1 * yMin), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // 坐标一维化
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)(Ceil(xMax - xMin + 1, BLOCK_NUM) * BLOCK_NUM), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(inputXIntTmpUb, inputXIntTmpUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

/**
 * @description: 原坐标越界时计算新坐标
 * @param {LocalTensor<float>} X坐标
 * @param {LocalTensor<float>} Y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
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
__aicore__ inline void GridSampler2DSlideWindow<T>::CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound)
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
__aicore__ inline void GridSampler2DSlideWindow<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    // maskTmpXUb存的是大于0的合法点
    CompareScalar(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    // maskXUb存的是小于inputW_的合法点
    CompareScalar(maskXUb, iXFpUb, static_cast<float>(inputW_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);
    // maskTmpYUb存的是大于0的合法点
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    // maskYUb存的是小于inputH_的合法点
    CompareScalar(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);

    PipeBarrier<PIPE_V>();

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // 合并上面的两个结果，得到最终合法点
    And(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, maskNum);
    And(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, maskNum);
    PipeBarrier<PIPE_V>();
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
    LocalTensor<float> oFpUb, LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = Ceil(calNum, B32_VECTOR_MASK);
    Select(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::CoordinatesSelectTensor(
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
__aicore__ inline void GridSampler2DSlideWindow<T>::BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // weightMaskBuf_作tmpBuf用，和weight无关
    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<T> tmpUb = inputXYFPBuf_.Get<T>();
    // +INF/-INF/NAN 场景下，+INF/-INF * 0 = NAN，消INF
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    // NAN eq NAN = FALSE，maskUb是NAN的mask
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    // 对上一步mask的位置置0，即+INF/-INF/NAN 全置0
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    // Y同理
    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

/**
 * @description: PaddingMode：Reflection
 * @param {LocalTensor<float>} x坐标
 * @param {LocalTensor<float>} y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    LocalTensor<float> coorSubUb = coorTmpBuf_.Get<float>(CAL_H_W_BLOCK);
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

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
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
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ReflectCoordinatesGeneralSelect(
    LocalTensor<float> coorSubUb, float minS, float spanS)
{
    // coordinate
    /*
     S1: get two results for both possibilities, out1: extra + min, out2: muls(extra, -1.0) + span + min
     S2: get mod val, mods: flips % 2
     S3: get mask tensor, masks: CompareScalar(mods, 0.0)
     S4: select val from out1 and out2 by mask tensor, out: Select(out1, out2, mask)
    */
    LocalTensor<float> fmodFpUb = modBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * NUM_3);
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<float> extraFpUb = extraBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);

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

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
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

    ReflectCoordinatesGeneralSelect(coorSubUb, minS, spanS);
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
__aicore__ inline void GridSampler2DSlideWindow<T>::MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal)
{
    for (int32_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation =
            (int64_t)nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int cIter = 0; cIter < channelAlign; cIter++) {
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
__aicore__ inline void GridSampler2DSlideWindow<T>::MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal,
    int32_t idx)
{
    int64_t base = (int64_t)nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;
    auto timeStep = loopElems / 8;
    auto timeStepRes = loopElems - loopElems / 8 * 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    // 这边为了不打断流水，提高性能
    for (int32_t i = 0; i < timeStep; i++) {
        int64_t coordVal_0 = coorUb.GetValue(loopOffset + i * 8) * inputC_;
        int64_t coordVal_1 = coorUb.GetValue(loopOffset + i * 8 + 1) * inputC_;
        int64_t coordVal_2 = coorUb.GetValue(loopOffset + i * 8 + 2) * inputC_;
        int64_t coordVal_3 = coorUb.GetValue(loopOffset + i * 8 + 3) * inputC_;
        int64_t coordVal_4 = coorUb.GetValue(loopOffset + i * 8 + 4) * inputC_;
        int64_t coordVal_5 = coorUb.GetValue(loopOffset + i * 8 + 5) * inputC_;
        int64_t coordVal_6 = coorUb.GetValue(loopOffset + i * 8 + 6) * inputC_;
        int64_t coordVal_7 = coorUb.GetValue(loopOffset + i * 8 + 7) * inputC_;
        int64_t location_0 = base + coordVal_0;
        int64_t location_1 = base + coordVal_1;
        int64_t location_2 = base + coordVal_2;
        int64_t location_3 = base + coordVal_3;
        int64_t location_4 = base + coordVal_4;
        int64_t location_5 = base + coordVal_5;
        int64_t location_6 = base + coordVal_6;
        int64_t location_7 = base + coordVal_7;

        DataCopyPad(xLocal[(i * 8) * channelAlign], gmX_[location_0], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 1) * channelAlign], gmX_[location_1], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 2) * channelAlign], gmX_[location_2], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 3) * channelAlign], gmX_[location_3], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 4) * channelAlign], gmX_[location_4], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 5) * channelAlign], gmX_[location_5], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 6) * channelAlign], gmX_[location_6], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 7) * channelAlign], gmX_[location_7], params, padParams);
    }
    for (auto i = loopElems / 8 * 8; i < loopElems; i++) {
        int64_t coordVal_0 = coorUb.GetValue(loopOffset + i) * inputC_;
        int64_t location_0 = base + coordVal_0;

        DataCopyPad(xLocal[i * channelAlign], gmX_[location_0], params, padParams);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::OutTranspose(
    int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb)
{
    uint64_t dstList[16];
    uint64_t srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.srcHighHalf = false;
    transDataParams.dstHighHalf = false;
    if (channelAlign == NUM_8) {
        transDataParams.repeatTimes = MIN_CHANNEL_ALIGN;
        transDataParams.dstRepStride = NUM_2;
        transDataParams.srcRepStride = NUM_16;

        for (int32_t i = 0; i < NUM_8; i++) {
            dstList[i * NUM_2] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE].GetPhyAddr());
            dstList[i * NUM_2 + 1] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + NUM_8].GetPhyAddr());
        }

        for (int32_t i = 0; i < NUM_16; i++) {
            srcList[i] = (uint64_t)(xLocal[i * NUM_8].GetPhyAddr());
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= NUM_64) {
        transDataParams.repeatTimes = channelAlign / NUM_8;
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        for (int32_t j = 0; j < NUM_8; j++) {
            for (int32_t i = 0; i < NUM_8; i++) {
                dstList[i * NUM_2] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + j * NUM_16].GetPhyAddr());
                dstList[i * NUM_2 + NUM_1] =
                    (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + NUM_8 + j * NUM_16].GetPhyAddr());
            }

            for (int32_t i = 0; i < NUM_16; i++) {
                srcList[i] = (uint64_t)(xLocal[i * channelAlign + j * NUM_16 * channelAlign].GetPhyAddr());
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<float>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset,
    LocalTensor<float> outValueUb)
{
    int64_t gmYBaseOffset = outBaseOffset + loopOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    if (calCElems == 1) {
        DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
    } else {
        for (int32_t i = 0; i < calCElems; i++) {
            int64_t gmYOffset = gmYBaseOffset + i * gridHW_;
            DataCopyPad(
                gmY_[gmYOffset], outValueUb[i * TRANSE_REP_STRIDE], {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::initTensor()
{
    nwWeightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    neWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    swWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    seWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);
    coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    coordinatesLocal2 = coorTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    coordinatesLocal3 = modBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    coordinatesLocal4 = extraBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb2 = weightMaskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb3 = weightMaskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb4 = weightMaskBuf4_.Get<uint8_t>(MASK_UB_SIZE);

    weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();
    weightMaskUbTmp2 = weightMaskUb2.ReinterpretCast<uint64_t>();
    weightMaskUbTmp3 = weightMaskUb3.ReinterpretCast<uint64_t>();
    weightMaskUbTmp4 = weightMaskUb4.ReinterpretCast<uint64_t>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::initMaskTensor()
{
    maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp = maskUb.ReinterpretCast<uint64_t>();
    weightMaskUbTmpfp32 = maskUbTmp.ReinterpretCast<float>();
    maskUb2 = maskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp2 = maskUb2.ReinterpretCast<uint64_t>();
    weightMaskUbTmpfp32_2 = maskUbTmp2.ReinterpretCast<float>();

    maskUb3 = maskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp3 = maskUb3.ReinterpretCast<uint64_t>();
    weightMaskUbTmpfp32_3 = maskUbTmp3.ReinterpretCast<float>();
    maskUb4 = maskBuf4_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp4 = maskUb4.ReinterpretCast<uint64_t>();
    weightMaskUbTmpfp32_4 = maskUbTmp4.ReinterpretCast<float>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::PointBilinearSetMask(int32_t maskOffset)
{
    maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
    maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));
    maskUbTmp.SetValue(2, weightMaskUbTmp.GetValue(maskOffset));
    maskUbTmp.SetValue(3, weightMaskUbTmp.GetValue(maskOffset + 1));

    maskUbTmp2.SetValue(0, weightMaskUbTmp2.GetValue(maskOffset));
    maskUbTmp2.SetValue(1, weightMaskUbTmp2.GetValue(maskOffset + 1));
    maskUbTmp2.SetValue(2, weightMaskUbTmp2.GetValue(maskOffset));
    maskUbTmp2.SetValue(3, weightMaskUbTmp2.GetValue(maskOffset + 1));

    maskUbTmp3.SetValue(0, weightMaskUbTmp3.GetValue(maskOffset));
    maskUbTmp3.SetValue(1, weightMaskUbTmp3.GetValue(maskOffset + 1));
    maskUbTmp3.SetValue(2, weightMaskUbTmp3.GetValue(maskOffset));
    maskUbTmp3.SetValue(3, weightMaskUbTmp3.GetValue(maskOffset + 1));

    maskUbTmp4.SetValue(0, weightMaskUbTmp4.GetValue(maskOffset));
    maskUbTmp4.SetValue(1, weightMaskUbTmp4.GetValue(maskOffset + 1));
    maskUbTmp4.SetValue(2, weightMaskUbTmp4.GetValue(maskOffset));
    maskUbTmp4.SetValue(3, weightMaskUbTmp4.GetValue(maskOffset + 1));
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::PointBilinearEachChannel(ProcessParam2D processParam,
    LocalTensor<float> outValueUb, PointParam2D pointBilinearParam, LocalTensor<float> xLocal,
    LocalTensor<float> outValueTotalLocal)
{
    pointBilinearParam.calCElems = perLoopChannel_;
    if (pointBilinearParam.cIdx == channelLoop_ - 1) {
        pointBilinearParam.calCElems = lastLoopChannel_;
    }

    pointBilinearParam.channelAlign = Ceil(pointBilinearParam.calCElems, BLOCK_NUM) * BLOCK_NUM;

    calculatePointBilinear(processParam.nIdx,
        coordinatesLocal,
        outValueUb,
        outValueTotalLocal,
        nwWeightLocal,
        maskUbTmp,
        pointBilinearParam.loopElems,
        pointBilinearParam.loopOffset,
        weightMaskUbTmpfp32,
        xLocal,
        pointBilinearParam.cIdx,
        pointBilinearParam.calCElems,
        pointBilinearParam.channelAlign,
        false,
        1);

    calculatePointBilinear(processParam.nIdx,
        coordinatesLocal2,
        outValueUb,
        outValueTotalLocal,
        neWeightLocal,
        maskUbTmp2,
        pointBilinearParam.loopElems,
        pointBilinearParam.loopOffset,
        weightMaskUbTmpfp32_2,
        xLocal,
        pointBilinearParam.cIdx,
        pointBilinearParam.calCElems,
        pointBilinearParam.channelAlign,
        true,
        2);
    calculatePointBilinear(processParam.nIdx,
        coordinatesLocal3,
        outValueUb,
        outValueTotalLocal,
        swWeightLocal,
        maskUbTmp3,
        pointBilinearParam.loopElems,
        pointBilinearParam.loopOffset,
        weightMaskUbTmpfp32_3,
        xLocal,
        pointBilinearParam.cIdx,
        pointBilinearParam.calCElems,
        pointBilinearParam.channelAlign,
        true,
        3);
    calculatePointBilinear(processParam.nIdx,
        coordinatesLocal4,
        outValueUb,
        outValueTotalLocal,
        seWeightLocal,
        maskUbTmp4,
        pointBilinearParam.loopElems,
        pointBilinearParam.loopOffset,
        weightMaskUbTmpfp32_4,
        xLocal,
        pointBilinearParam.cIdx,
        pointBilinearParam.calCElems,
        pointBilinearParam.channelAlign,
        true,
        4);
    MTE3ForNCHW(processParam.nIdx,
        pointBilinearParam.cIdx,
        pointBilinearParam.calCElems,
        pointBilinearParam.channelAlign,
        processParam.hwIdx,
        pointBilinearParam.loopOffset,
        pointBilinearParam.loopElems,
        pointBilinearParam.outBaseOffset,
        outValueTotalLocal);
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventMte3V);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::PointBilinear(
    ProcessParam2D processParam, LocalTensor<float> outValueUb)
{
    initTensor();
    initMaskTensor();

    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(nwWeightLocal, nwWeightLocal, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(neWeightLocal, neWeightLocal, weightMaskUb2, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(swWeightLocal, swWeightLocal, weightMaskUb3, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(seWeightLocal, seWeightLocal, weightMaskUb4, 0.0f, CAL_H_W_BLOCK);
    }

    PointParam2D pointBilinearParam{};
    int32_t trans_loop = Ceil(processParam.calHWElems, TRANSE_REP_STRIDE);
    pointBilinearParam.loopElems = TRANSE_REP_STRIDE;
    pointBilinearParam.loopOffset = 0;
    pointBilinearParam.outBaseOffset =
        (int64_t)processParam.nIdx * gridHW_ * inputC_ + processParam.hwIdx * CAL_H_W_BLOCK;
    pointBilinearParam.maskOffset = 0;
    PipeBarrier<PIPE_ALL>();

    // 按vmask(128)分块，循环处理
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            pointBilinearParam.loopElems = processParam.calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        pointBilinearParam.loopOffset = loop_idx * TRANSE_REP_STRIDE;
        pointBilinearParam.maskOffset = loop_idx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        PointBilinearSetMask(pointBilinearParam.maskOffset);

        LocalTensor<float> xLocal = xBuf_.AllocTensor<float>();
        LocalTensor<float> outValueTotalLocal = xLocal[CHANNEL_BLOCK * TRANSE_REP_STRIDE];
        // channel先按64大小循环
        for (pointBilinearParam.cIdx = 0; pointBilinearParam.cIdx < channelLoop_; pointBilinearParam.cIdx++) {
            PointBilinearEachChannel(processParam, outValueUb, pointBilinearParam, xLocal, outValueTotalLocal);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::calculatePointBilinear(int32_t nIdx,
    LocalTensor<int32_t> coordinatesUb, LocalTensor<float> outValueUb, LocalTensor<float> outValueTotalLocal,
    LocalTensor<float> weightUb, LocalTensor<uint64_t> maskUbTmp, int32_t loopElems, int32_t loopOffset,
    LocalTensor<float> weightMaskUbTmpfp32, LocalTensor<float> xLocal, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, bool isAtomicAdd, int32_t idx)
{
    auto outValueLocal = outValueTotalLocal;
    if (isAtomicAdd) {
        outValueLocal = outValueUb;
    }

    int32_t ubOffset = 0;
    if (channelLast_ == LAYOUT_NHWC) {
        MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal, idx);
    } else {
        MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal);
    }
    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    OutTranspose(channelAlign, xLocal, outValueLocal);
    PipeBarrier<PIPE_V>();
    if (calCElems >= 16) {
        BinaryRepeatParams repParams{1, 1, 0, 8, 8, 0};
        auto dupUb = dupBuf_.Get<float>();
        auto dupUbU32 = dupUb.ReinterpretCast<uint32_t>();
        uint32_t dstShape[2] = {Ceil(calCElems, 32 * 8 / TRANSE_REP_STRIDE), (uint32_t)8};
        uint32_t srcShape[2] = {1, (uint32_t)8};
        BroadCast<float, 2, 0>(dupUb, weightMaskUbTmpfp32, dstShape, srcShape);

        PipeBarrier<PIPE_V>();
        Select(outValueLocal,
            dupUbU32,
            outValueLocal,
            0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            64,
            calCElems * (TRANSE_REP_STRIDE / 64),
            repParams);
    } else {
        for (size_t i = 0; i < calCElems; i++) {
            ubOffset = i * TRANSE_REP_STRIDE;
            // 把非法值再select出来
            Select(outValueLocal[ubOffset],
                maskUbTmp,
                outValueLocal[ubOffset],
                0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                TRANSE_REP_STRIDE);
        }
    }
    PipeBarrier<PIPE_V>();

    if (calCElems == 1) {
        // 乘以权重

        Mul(outValueLocal, outValueLocal, weightUb[loopOffset], TRANSE_REP_STRIDE);
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = loopOffset + i * B32_MASK;

            Mul(outValueLocal[outOffset],
                outValueLocal[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, 16, 16, 0});
        }
    }

    if (isAtomicAdd) {
        Add(outValueTotalLocal, outValueTotalLocal, outValueLocal, calCElems * TRANSE_REP_STRIDE);
    }
}

/**
 * @description: 滑窗的PointBilinear方法，按坐标从xLocal中gather出对应的值，乘以权重后搬出
 * @param {int32_t} nIdx
 * @param {int32_t} hwIdx
 * @param {int32_t} calHWElems
 * @param {LocalTensor<int32_t>} coordinatesUb
 * @param {LocalTensor<float>} weightUb
 * @param {LocalTensor<uint8_t>} weightMaskUb
 * @param {LocalTensor<float>} outValueUb
 * @param {bool} isAutomicAdd
 * @param {LocalTensor<float>} xLocal
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::PointBilinearXInLocal(ProcessParam2D processParam,
    LocalTensor<float> outValueUb, LocalTensor<float> outValueTotalLocal, bool isAutomicAdd, LocalTensor<float> xLocal)
{
    initTensor();

    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(nwWeightLocal, nwWeightLocal, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(neWeightLocal, neWeightLocal, weightMaskUb2, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(swWeightLocal, swWeightLocal, weightMaskUb3, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(seWeightLocal, seWeightLocal, weightMaskUb4, 0.0f, CAL_H_W_BLOCK);
    }

    Muls(coordinatesLocal, coordinatesLocal, (int32_t)(4 * inputC_), processParam.calHWElems);
    Muls(coordinatesLocal2, coordinatesLocal2, (int32_t)(4 * inputC_), processParam.calHWElems);
    Muls(coordinatesLocal3, coordinatesLocal3, (int32_t)(4 * inputC_), processParam.calHWElems);
    Muls(coordinatesLocal4, coordinatesLocal4, (int32_t)(4 * inputC_), processParam.calHWElems);

    PipeBarrier<PIPE_V>();

    for (int32_t cIdx = 0; cIdx < channelXInLocalLoop_; cIdx++) {
        int32_t calCElems = perLoopChannelXInLocal_;
        if (cIdx == channelXInLocalLoop_ - 1) {
            calCElems = lastLoopChannelXInLocal_;
        }

        calculatePointBilinearXInLocal(processParam.calHWElems,
            coordinatesLocal,
            nwWeightLocal,
            outValueUb,
            outValueTotalLocal,
            false,
            xLocal,
            weightMaskUbTmp,
            cIdx,
            calCElems);
        calculatePointBilinearXInLocal(processParam.calHWElems,
            coordinatesLocal2,
            neWeightLocal,
            outValueUb,
            outValueTotalLocal,
            true,
            xLocal,
            weightMaskUbTmp2,
            cIdx,
            calCElems);
        calculatePointBilinearXInLocal(processParam.calHWElems,
            coordinatesLocal3,
            swWeightLocal,
            outValueUb,
            outValueTotalLocal,
            true,
            xLocal,
            weightMaskUbTmp3,
            cIdx,
            calCElems);
        calculatePointBilinearXInLocal(processParam.calHWElems,
            coordinatesLocal4,
            seWeightLocal,
            outValueUb,
            outValueTotalLocal,
            true,
            xLocal,
            weightMaskUbTmp4,
            cIdx,
            calCElems);

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 搬出，outValueUb里面是CHW，搬出也是CHW
        DataCopyExtParams params;
        params.blockCount = calCElems;
        params.blockLen = processParam.calHWElems * sizeof(float);
        params.srcStride = CAL_H_W_BLOCK / BLOCK_NUM - Ceil(processParam.calHWElems, BLOCK_NUM);
        params.dstStride = (outputH_ * outputW_ - processParam.calHWElems) * sizeof(float);
        int64_t gmYOffset = (int64_t)processParam.nIdx * outputH_ * outputW_ * inputC_ +
                            (int64_t)processParam.hwIdx * CAL_H_W_BLOCK +
                            cIdx * outputH_ * outputW_ * perLoopChannelXInLocal_;
        DataCopyPad(gmY_[gmYOffset], outValueTotalLocal, params);

        event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::calculatePointBilinearXInLocal(int32_t calHWElems,
    LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
    LocalTensor<float> outValueTotalLocal, bool isAutomicAdd, LocalTensor<float> xLocal,
    LocalTensor<uint64_t> weightMaskUbTmp, int32_t cIdx, int32_t calCElems)
{
    auto outValueLocal = outValueTotalLocal;
    if (isAutomicAdd) {
        outValueLocal = outValueUb;
    }

    for (int32_t loop_c = 0; loop_c < calCElems; loop_c++) {
        int32_t loop_c_idx = cIdx * perLoopChannelXInLocal_ + loop_c;
        LocalTensor<uint32_t> coordUb = coordinatesUb.ReinterpretCast<uint32_t>();
        Gather(outValueLocal[CAL_H_W_BLOCK * loop_c], xLocal, coordUb, (uint32_t)0, (uint32_t)calHWElems);
        PipeBarrier<PIPE_V>();
        if (loop_c_idx != inputC_ - 1) {
            // Gather的indices是按类型的，所以float32的indices，两个元素之间需要加4
            Adds(coordinatesUb, coordinatesUb, 4, (uint32_t)calHWElems);
            PipeBarrier<PIPE_V>();
        }
    }

    for (size_t i = 0; i < calCElems; i++) {
        auto ubOffset = i * CAL_H_W_BLOCK;
        Select(outValueLocal[ubOffset],
            weightMaskUbTmp,
            outValueLocal[ubOffset],
            0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            CAL_H_W_BLOCK);
    }
    PipeBarrier<PIPE_V>();

    int32_t trans_loop = Ceil(calHWElems, B32_MASK);
    // 权重处理
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        int64_t loopOffset = loop_idx * B32_MASK;
        uint8_t repeatStride = CAL_H_W_BLOCK / BLOCK_NUM;
        Mul(outValueLocal[loopOffset],
            outValueLocal[loopOffset],
            weightUb[loopOffset],
            B32_MASK,
            calCElems,
            {1, 1, 1, repeatStride, repeatStride, 0});
    }
    PipeBarrier<PIPE_V>();
    if (isAutomicAdd) {
        Add(outValueTotalLocal, outValueTotalLocal, outValueLocal, calCElems * CAL_H_W_BLOCK);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::CalculateGrid(
    ProcessParam2D processParam, int64_t gridGmOffset, LocalTensor<T> gridLocal)
{
    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = processParam.calHWElems * 2 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    DataCopyPadExtParams<float> padParamsGrid{false, 0, 0, 0};
    DataCopyPad(gridLocal, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
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

    // 分别取x和y
    GatherMask(inputXFpLocal, inputXYUb, xPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYUb, yPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // 不同alignCorners_的unnormlize处理
    if (alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (inputW_ - (float)1.0)), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * (inputH_ - (float)1.0)), CAL_H_W_BLOCK);
    } else {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * inputW_), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * inputH_), CAL_H_W_BLOCK);
        PipeBarrier<PIPE_V>();
        Adds(inputXFpLocal, inputXFpLocal, (float)(-0.5), CAL_H_W_BLOCK);
        Adds(inputYFpLocal, inputYFpLocal, (float)(-0.5), CAL_H_W_BLOCK);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::GetInputTensor()
{
    inputXWIntLocal = inputXIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    inputXEIntLocal = inputXIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    inputYWIntLocal = inputYIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    inputYEIntLocal = inputYIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);

    inputXWFpLocal = inputXFpBuf_.Get<float>(CAL_H_W_BLOCK);
    inputXEFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    inputYWFpLocal = inputYFpBuf_.Get<float>(CAL_H_W_BLOCK);
    inputYEFpLocal = inputYFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::calculateGridWeight()
{
    LocalTensor<float> weightTmpLocal = weightTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> weightTmp1Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> weightTmp2Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> weightTmp3Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);
    // 计算4个点的权重
    ComputeWeightSub(nwWeightLocal, weightTmpLocal, inputXEFpLocal, inputXFpLocal, inputYEFpLocal, inputYFpLocal);
    ComputeWeightSub(neWeightLocal, weightTmp1Local, inputXFpLocal, inputXWFpLocal, inputYEFpLocal, inputYFpLocal);
    ComputeWeightSub(swWeightLocal, weightTmp2Local, inputXEFpLocal, inputXFpLocal, inputYFpLocal, inputYWFpLocal);
    ComputeWeightSub(seWeightLocal, weightTmp3Local, inputXFpLocal, inputXWFpLocal, inputYFpLocal, inputYWFpLocal);
    PipeBarrier<PIPE_V>();
    Mul(nwWeightLocal, nwWeightLocal, weightTmpLocal, CAL_H_W_BLOCK);
    Mul(neWeightLocal, neWeightLocal, weightTmp1Local, CAL_H_W_BLOCK);
    Mul(swWeightLocal, swWeightLocal, weightTmp2Local, CAL_H_W_BLOCK);
    Mul(seWeightLocal, seWeightLocal, weightTmp3Local, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::GetNoSlideWindow(ProcessParam2D processParam,
    LocalTensor<T> inputMaxXYFpUb, LocalTensor<int32_t> inputMaxXYIntUb, SlideCoorParam &slideCoorParam,
    bool &noSlideWindow)
{
    slideCoorParam = {0, (int32_t)(inputW_ - 1), 0, (int32_t)(inputH_ - 1)};
    noSlideWindow = (inputC_ > SLIDING_WINDOW_C_LIMIT);
    if (!noSlideWindow) {
        LocalTensor<T> tmpFpUb = outTmpBuf_.Get<T>(CAL_H_W_BLOCK);
        // 计算x的最小值
        ReduceMin(inputMaxXYFpUb, inputXWFpLocal, tmpFpUb, processParam.calHWElems, false);
        PipeBarrier<PIPE_V>();
        // 计算x的最大值
        ReduceMax(inputMaxXYFpUb[1], inputXEFpLocal, tmpFpUb, processParam.calHWElems, false);
        PipeBarrier<PIPE_V>();
        // 计算y的最小值
        ReduceMin(inputMaxXYFpUb[2], inputYWFpLocal, tmpFpUb, processParam.calHWElems, false);
        PipeBarrier<PIPE_V>();
        // 计算y的最大值
        ReduceMax(inputMaxXYFpUb[3], inputYEFpLocal, tmpFpUb, processParam.calHWElems, false);
        PipeBarrier<PIPE_V>();
        Cast(inputMaxXYIntUb, inputMaxXYFpUb, RoundMode::CAST_FLOOR, BLOCK_NUM);
        PipeBarrier<PIPE_V>();
        event_t eventIdV_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdV_S);
        WaitFlag<HardEvent::V_S>(eventIdV_S);
        slideCoorParam.xMin = inputMaxXYIntUb.GetValue(0) < 0 ? 0 : inputMaxXYIntUb.GetValue(0);
        slideCoorParam.xMax = inputMaxXYIntUb.GetValue(1) > (inputW_ - 1) ? (inputW_ - 1) : inputMaxXYIntUb.GetValue(1);
        slideCoorParam.yMin = inputMaxXYIntUb.GetValue(2) < 0 ? 0 : inputMaxXYIntUb.GetValue(2);
        slideCoorParam.yMax = inputMaxXYIntUb.GetValue(3) > (inputH_ - 1) ? (inputH_ - 1) : inputMaxXYIntUb.GetValue(3);

        noSlideWindow =
            noSlideWindow || (slideCoorParam.xMin > slideCoorParam.xMax || slideCoorParam.yMin > slideCoorParam.yMax);
        // 滑框总面积超过UB分配大小，这里要注意UB分配大小要和xBuf一致或比xBuf小，否则搬入可能越界
        noSlideWindow = noSlideWindow || Ceil(slideCoorParam.xMax - slideCoorParam.xMin + 1, BLOCK_NUM) * BLOCK_NUM *
                                                 (slideCoorParam.yMax - slideCoorParam.yMin + 1) * inputC_ >
                                             X_UB_SIZE_4_GENERAL / sizeof(float);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::PointBilinearInSlideWindow(
    ProcessParam2D processParam, LocalTensor<float> outValueLocal, SlideCoorParam slideCoorParam)
{
    event_t eventIdS_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdS_MTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIdS_MTE2);
    LocalTensor<float> xLocal = xBuf_.AllocTensor<float>();
    DataCopyExtParams params;
    // 按行搬，重复y次
    params.blockCount = slideCoorParam.yMax - slideCoorParam.yMin + 1;
    params.blockLen = (slideCoorParam.xMax - slideCoorParam.xMin + 1) * inputC_ * sizeof(float);
    params.srcStride =
        (inputW_ * inputC_) * sizeof(float) - (slideCoorParam.xMax - slideCoorParam.xMin + 1) * inputC_ * sizeof(float);
    // UB空间按aligh(X) * C对齐
    params.dstStride = Ceil(slideCoorParam.xMax - slideCoorParam.xMin + 1, BLOCK_NUM) * inputC_ -
                       Ceil((slideCoorParam.xMax - slideCoorParam.xMin + 1) * inputC_, BLOCK_NUM);
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    int64_t gmOffset = (int64_t)processParam.nIdx * inputH_ * inputW_ * inputC_ +
                       (slideCoorParam.xMin + slideCoorParam.yMin * inputW_) * inputC_;
    DataCopyPad(xLocal, gmX_[gmOffset], params, padParams);

    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    LocalTensor<float> outValueTotalLocal = outValueLocal[CHANNEL_BLOCK_X_IN_LOCAL * CAL_H_W_BLOCK];

    ClipCoordinatesXInLocal(inputXWFpLocal,
        inputYWFpLocal,
        inputXWIntLocal,
        inputYWIntLocal,
        coordinatesLocal,
        weightMaskUb,
        slideCoorParam.xMin,
        slideCoorParam.xMax,
        slideCoorParam.yMin,
        slideCoorParam.yMax);
    ClipCoordinatesXInLocal(inputXEFpLocal,
        inputYWFpLocal,
        inputXEIntLocal,
        inputYWIntLocal,
        coordinatesLocal2,
        weightMaskUb2,
        slideCoorParam.xMin,
        slideCoorParam.xMax,
        slideCoorParam.yMin,
        slideCoorParam.yMax);
    ClipCoordinatesXInLocal(inputXWFpLocal,
        inputYEFpLocal,
        inputXWIntLocal,
        inputYEIntLocal,
        coordinatesLocal3,
        weightMaskUb3,
        slideCoorParam.xMin,
        slideCoorParam.xMax,
        slideCoorParam.yMin,
        slideCoorParam.yMax);
    ClipCoordinatesXInLocal(inputXEFpLocal,
        inputYEFpLocal,
        inputXEIntLocal,
        inputYEIntLocal,
        coordinatesLocal4,
        weightMaskUb4,
        slideCoorParam.xMin,
        slideCoorParam.xMax,
        slideCoorParam.yMin,
        slideCoorParam.yMax);
    PointBilinearXInLocal(processParam, outValueLocal, outValueTotalLocal, true, xLocal);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::PerLoopCompute(ProcessParam2D processParam)
{
    int64_t gridGmOffset = processParam.nIdx * gridHW_ * 2 + processParam.hwIdx * CAL_H_W_BLOCK * 2;

    LocalTensor<T> gridLocal = gridQueue_.AllocTensor<T>();
    inputXFpLocal = gridLocal;
    inputYFpLocal = gridLocal[CAL_H_W_BLOCK];
    CalculateGrid(processParam, gridGmOffset, gridLocal);

    // 处理越界坐标
    Clip(inputXFpLocal, inputYFpLocal);

    GetInputTensor();

    Cast(inputXWIntLocal, inputXFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    Cast(inputYWIntLocal, inputYFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(inputXWFpLocal, inputXWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputYWFpLocal, inputYWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    // 分别计算左上，右上，左下，右下的坐标
    Adds(inputXEIntLocal, inputXWIntLocal, 1, CAL_H_W_BLOCK);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Adds(inputXEFpLocal, inputXWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    Adds(inputYEFpLocal, inputYWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    initTensor();
    calculateGridWeight();

    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();

    LocalTensor<T> inputMaxXYFpUb = inputMaxXYFpBuf_.Get<T>();
    LocalTensor<int32_t> inputMaxXYIntUb = inputMaxXYIntBuf_.Get<int32_t>();

    SlideCoorParam slideCoorParam;
    bool noSlideWindow = false;
    GetNoSlideWindow(processParam, inputMaxXYFpUb, inputMaxXYIntUb, slideCoorParam, noSlideWindow);
    if (noSlideWindow) {
        // 划窗条件不满足，走兜底分支
        ClipCoordinates(
            inputXWFpLocal, inputYWFpLocal, inputXWIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb);
        ClipCoordinates(
            inputXEFpLocal, inputYWFpLocal, inputXEIntLocal, inputYWIntLocal, coordinatesLocal2, weightMaskUb2);
        ClipCoordinates(
            inputXWFpLocal, inputYEFpLocal, inputXWIntLocal, inputYEIntLocal, coordinatesLocal3, weightMaskUb3);
        ClipCoordinates(
            inputXEFpLocal, inputYEFpLocal, inputXEIntLocal, inputYEIntLocal, coordinatesLocal4, weightMaskUb4);

        PointBilinear(processParam, outValueLocal);

        gridQueue_.FreeTensor(gridLocal);
        return;
    }

    PointBilinearInSlideWindow(processParam, outValueLocal, slideCoorParam);

    gridQueue_.FreeTensor(gridLocal);
}

template <typename T>
__aicore__ inline void GridSampler2DSlideWindow<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    ProcessParam2D processParam;
    int32_t preLoopNum = blockIDX * preCoreLoop_;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        processParam.nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        processParam.hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        processParam.calHWElems = CAL_H_W_BLOCK;
        if (processParam.hwIdx == preNUbLoop_ - 1) {
            processParam.calHWElems = lastLoopHW_;
        }
        PerLoopCompute(processParam);
    }
}

}  // namespace GridSample
#endif  // GRID_SAMPLER_2D_SLIDE_WINDOW
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
 * \file grid_sampler_2d_fp16_slide_window.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_FP16_SLIDE_WINDOW
#define GRID_SAMPLER_2D_FP16_SLIDE_WINDOW

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler2DFP16SlideWindow {
public:
    __aicore__ inline GridSampler2DFP16SlideWindow(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);
    __aicore__ inline void ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub, LocalTensor<float> x1Ub,
        LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
        LocalTensor<uint8_t> weightMaskUb);
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
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb, LocalTensor<uint8_t> maskUb,
        LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow, const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);
    __aicore__ inline void MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset, LocalTensor<float> weightUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void PointBilinearXInLocal(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd, LocalTensor<T> xLocal);

    __aicore__ inline void CopyOut(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);

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
    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;
    TBuf<QuePosition::VECCALC> inputMaxXYFpBuf_;
    TBuf<QuePosition::VECCALC> inputMaxXYIntBuf_;
    TBuf<QuePosition::VECCALC> dupBuf_;

    TBuf<QuePosition::VECCALC> inputGridFp32Buf_;
    TBuf<QuePosition::VECCALC> outValueFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;

    const int64_t X_UB_SIZE = 65536;            // 64k
    const int64_t X_UB_SIZE_4_GENERAL = 32768;  // 32k
    const int64_t X_UB_SIZE_4_FP16 = 16384;     // 16k
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t GRID_UB_SIZE_4_FP16 = 2048;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t OUT_VAL_NUM = 4096;
    const int64_t X_UB_OFFSET = 512;
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / 8;

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

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;
    constexpr static int64_t LAYOUT_NHWC = 1;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;
    constexpr static int64_t B32_ALIGN_FACTOR = 8;
    constexpr static int64_t B16_ALIGN_FACTOR = 16;

    constexpr static int64_t SLIDING_WINDOW_C_LIMIT = 16;
};

/**
 * @description: 解析tiling数据，计算分核数据
 * @param {GridSampleTilingData*} tilingData
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::ParseTilingData(const GridSampleTilingData *tilingData)
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
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer申请初始化
    pipe.InitBuffer(gridQueue_, 1, GRID_UB_SIZE_4_FP16);  // 2KB
    pipe.InitBuffer(dupBuf_, 2048);                       // 2KB
    pipe.InitBuffer(inputMaxXYFpBuf_, 32);                // 32B
    pipe.InitBuffer(inputMaxXYIntBuf_, 32);               // 32B

    pipe.InitBuffer(xBuf_, X_UB_SIZE);                           // 64KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);      // 4KB
    pipe.InitBuffer(inputXIntBuf_, GRID_UB_SIZE_4_GENERAL);      // 4KB
    pipe.InitBuffer(inputYIntBuf_, GRID_UB_SIZE_4_GENERAL);      // 4KB
    pipe.InitBuffer(inputXFpBuf_, GRID_UB_SIZE_4_GENERAL);       // 4KB
    pipe.InitBuffer(inputGridFp32Buf_, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(inputYFpBuf_, GRID_UB_SIZE_4_GENERAL);       // 4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);        // 8KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);     // 8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);            // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);              // 2KB
    pipe.InitBuffer(coorTmpBuf_, Y_UB_SIZE_4_GENERAL);           // 2KB
    pipe.InitBuffer(outValueBuf_, X_UB_SIZE_4_GENERAL);          // 32KB
    pipe.InitBuffer(outValueFp16Buf_, X_UB_SIZE_4_FP16);         // 16KB
    pipe.InitBuffer(maskBuf_, 960);                              // 960B
    pipe.InitBuffer(weightMaskBuf_, 320);                        // 320B
    pipe.InitBuffer(modBuf_, Y_UB_SIZE_4_GENERAL);               // 2KB
    pipe.InitBuffer(extraBuf_, Y_UB_SIZE_4_GENERAL);             // 2KB
    pipe.InitBuffer(outTmpBuf_, GRID_UB_SIZE_4_GENERAL);         // 4KB
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::ComputeWeightSub(LocalTensor<float> w1Ub,
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::ClipCoordinates(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> wMaskUb)
{
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
    PipeBarrier<PIPE_V>();
    Adds(inputXIntTmpUb, iXIntUb, 0, CAL_H_W_BLOCK);
    Adds(inputYIntTmpUb, iYIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Cast(iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);
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

    // cood = y + x * IW
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, CAL_H_W_BLOCK);
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::ClipCoordinatesXInLocal(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> wMaskUb, int32_t xMin, int32_t xMax, int32_t yMin, int32_t yMax)
{
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
    PipeBarrier<PIPE_V>();
    Adds(inputXIntTmpUb, iXIntUb, 0, CAL_H_W_BLOCK);
    Adds(inputYIntTmpUb, iYIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Cast(iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);
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

    // 将原坐标系的坐标（相对顶点）转换成划窗坐标系坐标
    Adds(inputXIntTmpUb, inputXIntTmpUb, (int32_t)(-1 * xMin), CAL_H_W_BLOCK);
    Adds(inputYIntTmpUb, inputYIntTmpUb, (int32_t)(-1 * yMin), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // 坐标一维化
    Muls(inputYIntTmpUb,
        inputYIntTmpUb,
        (int32_t)(Ceil(xMax - xMin + 1, B16_ALIGN_FACTOR) * B16_ALIGN_FACTOR),
        CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

/**
 * @description: 原坐标越界时计算新坐标
 * @param {LocalTensor<float>} X坐标
 * @param {LocalTensor<float>} Y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::CoordinatesFrameRange(
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::CoordinatesSelectTensor(
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
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
    LocalTensor<float> tmpUb = inputXYFPBuf_.Get<float>();
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::ReflectClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
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

    LocalTensor<float> tmpUb = inputXYFPBuf_.Get<float>();
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    for (int32_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation =
            (int64_t)nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int cIter = 0; cIter < channelAlign; cIter++) {
            int32_t xLocalOffset = i * channelAlign + cIter;
            if (cIter >= calCElems) {
                xLocal.SetValue(xLocalOffset, (half)0.0);
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    int64_t base = (int64_t)nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;
    auto timeStep = loopElems / 8;
    auto timeStepRes = loopElems - loopElems / 8 * 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::OutTranspose(
    int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb)
{
    LocalTensor<half> dstList[16];
    LocalTensor<half> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == 16) {
        transDataParams.repeatTimes = 8;
        transDataParams.dstRepStride = 1;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 16];
        }

        for (int32_t i = 0; i < 16; i++) {
            dstList[i] = outValueUb[i * TRANSE_REP_STRIDE];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<half>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 16;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * channelAlign + j * 16 * channelAlign];
            }

            for (int32_t i = 0; i < 16; i++) {
                dstList[i] = outValueUb[i * TRANSE_REP_STRIDE + j * 16];
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<half>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset,
    LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    // 512 * inputC_ * blockIDX 每个核的地址
    // loopOffset 偏移的是几个128
    int64_t gmYBaseOffset = CAL_H_W_BLOCK * inputC_ * blockIDX + loopOffset + cIdx * CHANNEL_BLOCK * CAL_H_W_BLOCK;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    if (calCElems == 1) {
        // 乘以权重
        Mul(outValueUb, outValueUb, weightUb[loopOffset], TRANSE_REP_STRIDE);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopyPad(gmWorkspace_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            SetAtomicNone();
        } else {
            DataCopyPad(gmWorkspace_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        }
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = loopOffset + i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, 16, 16, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * CAL_H_W_BLOCK;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(gmWorkspace_[gmYOffset],
                    outValueUb[i * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
            SetAtomicNone();
        } else {
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * CAL_H_W_BLOCK;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(gmWorkspace_[gmYOffset],
                    outValueUb[i * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
    LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
    LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
    }

    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();
    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();

    int32_t trans_loop = Ceil(calHWElems, TRANSE_REP_STRIDE);
    int32_t loop_elems = TRANSE_REP_STRIDE;
    int32_t loop_offset = 0;
    int64_t outBaseOffset = (int64_t)nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK;
    int32_t ubOffset = 0;
    int32_t maskOffset = 0;
    PipeBarrier<PIPE_ALL>();
    // 按vmask(128)分块，循环处理
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            loop_elems = calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        loop_offset = loop_idx * TRANSE_REP_STRIDE;
        maskOffset = loop_idx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));
        maskUbTmp.SetValue(2, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(3, weightMaskUbTmp.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32 = maskUbTmp.ReinterpretCast<float>();
        LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
        // channel先按64大小循环
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }
            int32_t channelAlign = Ceil(calCElems, 16) * 16;
            if (channelLast_ == LAYOUT_NHWC) {
                MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loop_offset, loop_elems, coordinatesUb, xLocal);
            } else {
                MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loop_offset, loop_elems, coordinatesUb, xLocal);
            }
            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);

            LocalTensor<half> outValueFp16Ub = outValueFp16Buf_.Get<half>();
            OutTranspose(channelAlign, xLocal, outValueFp16Ub);
            PipeBarrier<PIPE_V>();
            Cast(outValueUb, outValueFp16Ub, RoundMode::CAST_NONE, calCElems * TRANSE_REP_STRIDE);
            PipeBarrier<PIPE_V>();
            if (calCElems >= 16) {
                BinaryRepeatParams repParams{1, 1, 0, 8, 8, 0};
                auto dupUb = dupBuf_.Get<float>();
                auto dupUbU32 = dupUb.ReinterpretCast<uint32_t>();
                uint32_t dstShape[2] = {Ceil(calCElems, 32 * 8 / TRANSE_REP_STRIDE), (uint32_t)8};
                uint32_t srcShape[2] = {1, (uint32_t)8};
                BroadCast<float, 2, 0>(dupUb, weightMaskUbTmpfp32, dstShape, srcShape);

                PipeBarrier<PIPE_V>();
                Select(outValueUb,
                    dupUbU32,
                    outValueUb,
                    0.0f,
                    SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    64,
                    calCElems * (TRANSE_REP_STRIDE / 64),
                    repParams);
            } else {
                for (size_t i = 0; i < calCElems; i++) {
                    ubOffset = i * TRANSE_REP_STRIDE;
                    // 把非法值再select出来
                    Select(outValueUb[ubOffset],
                        maskUbTmp,
                        outValueUb[ubOffset],
                        0.0f,
                        SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        TRANSE_REP_STRIDE);
                }
            }
            PipeBarrier<PIPE_V>();
            MTE3ForNCHW(nIdx,
                cIdx,
                calCElems,
                channelAlign,
                hwIdx,
                loop_offset,
                loop_elems,
                outBaseOffset,
                weightUb,
                outValueUb,
                isAutomicAdd);
            event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventMte3V);
        }
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
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::PointBilinearXInLocal(int32_t nIdx, int32_t hwIdx,
    int32_t calHWElems, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb, bool isAutomicAdd, LocalTensor<T> xLocal)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
    }

    auto weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();

    Muls(coordinatesUb, coordinatesUb, (int32_t)(sizeof(T) * inputC_), calHWElems);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> outValueFp16Ub = outValueFp16Buf_.Get<half>();

    for (int32_t loop_c = 0; loop_c < inputC_; loop_c++) {
        LocalTensor<uint32_t> coordUb = coordinatesUb.ReinterpretCast<uint32_t>();
        Gather(outValueFp16Ub[CAL_H_W_BLOCK * loop_c], xLocal, coordUb, (uint32_t)0, (uint32_t)calHWElems);
        PipeBarrier<PIPE_V>();
        if (loop_c != inputC_ - 1) {
            // Gather的indices是按类型的，所以float16的indices，两个元素之间需要加2
            Adds(coordinatesUb, coordinatesUb, 2, (uint32_t)calHWElems);
            PipeBarrier<PIPE_V>();
        }
    }

    for (size_t i = 0; i < inputC_; i++) {
        auto ubOffset = i * CAL_H_W_BLOCK;
        Select(outValueFp16Ub[ubOffset],
            weightMaskUbTmp,
            outValueFp16Ub[ubOffset],
            (half)0.0,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            CAL_H_W_BLOCK);
    }
    PipeBarrier<PIPE_V>();

    Cast(outValueUb, outValueFp16Ub, RoundMode::CAST_NONE, inputC_ * CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    int32_t trans_loop = Ceil(calHWElems, B32_MASK);
    // 权重处理
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        int64_t loop_offset = loop_idx * B32_MASK;
        uint8_t repeatStride = CAL_H_W_BLOCK / 8;
        Mul(outValueUb[loop_offset],
            outValueUb[loop_offset],
            weightUb[loop_offset],
            B32_MASK,
            inputC_,
            {1, 1, 1, repeatStride, repeatStride, 0});
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    // 搬出，outValueUb里面是CHW，搬出也是CHW
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = inputC_ * CAL_H_W_BLOCK * sizeof(float);
    params.srcStride = 0;
    params.dstStride = 0;
    int64_t gmYOffset = (int64_t)CAL_H_W_BLOCK * inputC_ * blockIDX;
    event_t eventIdS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdS_MTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdS_MTE3);

    if (isAutomicAdd) {
        SetAtomicAdd<float>();
        DataCopyPad(gmWorkspace_[gmYOffset], outValueUb, params);
        SetAtomicNone();
    } else {
        DataCopyPad(gmWorkspace_[gmYOffset], outValueUb, params);
    }
    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);

    event_t eventIdV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::CopyOut(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    LocalTensor<float> outLocal = xBuf_.AllocTensor<float>();
    LocalTensor<T> outLocalFP16 = outValueFp16Buf_.AllocTensor<half>();
    // 每次处理16*512个数据
    int64_t loopTime = Ceil(inputC_, 16);
    int64_t lastC = inputC_ - 16 * (loopTime - 1);
    int64_t dataCount = CAL_H_W_BLOCK * 16;
    int64_t basegmWorkSpaceAddr = blockIDX * CAL_H_W_BLOCK * inputC_;

    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    for (size_t i = 0; i < loopTime - 1; i++) {
        DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + dataCount * i], dataCount);

        SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
        Cast(outLocalFP16, outLocal, RoundMode::CAST_NONE, dataCount);

        SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
        DataCopyExtParams params;
        params.blockCount = 16;
        params.blockLen = calHWElems * sizeof(T);
        params.srcStride = CAL_H_W_BLOCK / 16 - Ceil(calHWElems, 16);
        params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
        int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * CAL_H_W_BLOCK +
                            i * 16 * outputH_ * outputW_;
        DataCopyPad(gmY_[gmYOffset], outLocalFP16, params);

        SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);

        SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    }

    dataCount = CAL_H_W_BLOCK * lastC;
    DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + CAL_H_W_BLOCK * 16 * (loopTime - 1)], dataCount);

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    Cast(outLocalFP16, outLocal, RoundMode::CAST_NONE, dataCount);

    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    DataCopyExtParams params;
    params.blockCount = lastC;
    params.blockLen = calHWElems * sizeof(T);
    params.srcStride = CAL_H_W_BLOCK / 16 - Ceil(calHWElems, 16);
    params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
    int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * CAL_H_W_BLOCK +
                        (loopTime - 1) * 16 * outputH_ * outputW_;
    DataCopyPad(gmY_[gmYOffset], outLocalFP16, params);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;

    LocalTensor<T> gridLocal = gridQueue_.AllocTensor<T>();
    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = calHWElems * 2 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    DataCopyPadExtParams<T> padParamsGrid{false, 0, 0, 0};
    DataCopyPad(gridLocal, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
    gridQueue_.EnQue(gridLocal);
    gridQueue_.DeQue<T>();

    LocalTensor<float> gridFp32Local = inputGridFp32Buf_.Get<float>();
    Cast(gridFp32Local, gridLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK * 2);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> inputXYUb = inputXYFPBuf_.Get<float>();
    // 加1后，grid的datarange从-1~1到0~2
    Adds(inputXYUb, gridFp32Local, (float)1.0, CAL_H_W_BLOCK * 2);

    uint32_t mask = CAL_H_W_BLOCK * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;

    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;
    PipeBarrier<PIPE_V>();
    LocalTensor<float> inputXFpLocal = gridFp32Local;
    LocalTensor<float> inputYFpLocal = gridFp32Local[CAL_H_W_BLOCK];
    // 分别取x和y(inputXFpLocal, inputXYUb, xPattern, true, mask,
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
    Adds(inputXEIntLocal, inputXWIntLocal, 1, CAL_H_W_BLOCK);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Adds(inputXEFpLocal, inputXWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    Adds(inputYEFpLocal, inputYWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> nwWeightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> neWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> swWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> seWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

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

    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);

    LocalTensor<float> inputMaxXYFpUb = inputMaxXYFpBuf_.Get<float>();
    LocalTensor<int32_t> inputMaxXYIntUb = inputMaxXYIntBuf_.Get<int32_t>();

    // 滑窗逻辑开始
    int32_t xMin = 0;
    int32_t xMax = inputW_ - 1;
    int32_t yMin = 0;
    int32_t yMax = inputH_ - 1;
    // inputC较大，不走滑窗，性能收益小
    bool noSlideWindow = (inputC_ > SLIDING_WINDOW_C_LIMIT);
    if (!noSlideWindow) {
        LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK);
        // 计算x的最小值
        ReduceMin(inputMaxXYFpUb, inputXWFpLocal, tmpFpUb, calHWElems, false);
        PipeBarrier<PIPE_V>();
        // 计算x的最大值
        ReduceMax(inputMaxXYFpUb[1], inputXEFpLocal, tmpFpUb, calHWElems, false);
        PipeBarrier<PIPE_V>();
        // 计算y的最小值
        ReduceMin(inputMaxXYFpUb[2], inputYWFpLocal, tmpFpUb, calHWElems, false);
        PipeBarrier<PIPE_V>();
        // 计算y的最大值
        ReduceMax(inputMaxXYFpUb[3], inputYEFpLocal, tmpFpUb, calHWElems, false);
        PipeBarrier<PIPE_V>();
        Cast(inputMaxXYIntUb, inputMaxXYFpUb, RoundMode::CAST_FLOOR, 8);
        PipeBarrier<PIPE_V>();
        event_t eventIdV_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdV_S);
        WaitFlag<HardEvent::V_S>(eventIdV_S);
        xMin = inputMaxXYIntUb.GetValue(0);
        xMax = inputMaxXYIntUb.GetValue(1);
        yMin = inputMaxXYIntUb.GetValue(2);
        yMax = inputMaxXYIntUb.GetValue(3);

        // 滑框边界超过x的边界时，将最大值最小值归到边界内
        // 越界的点在Clip中有处理，只剩padding:zereos时还可能会越界，后面对这些点会处理，划窗时不需要考虑
        xMin = xMin < 0 ? 0 : xMin;
        xMax = xMax > (inputW_ - 1) ? (inputW_ - 1) : xMax;
        yMin = yMin < 0 ? 0 : yMin;
        yMax = yMax > (inputH_ - 1) ? (inputH_ - 1) : yMax;

        // 滑框边界超过x的边界
        noSlideWindow = noSlideWindow || (xMin > xMax || yMin > yMax);
        // 滑框总面积超过UB大小
        noSlideWindow = noSlideWindow || Ceil(xMax - xMin + 1, B16_ALIGN_FACTOR) * B16_ALIGN_FACTOR *
                                                 (yMax - yMin + 1) * inputC_ * sizeof(T) >
                                             X_UB_SIZE;
        // 滑窗只支持channel last场景
        noSlideWindow = noSlideWindow || (channelLast_ != LAYOUT_NHWC);
    }
    if (noSlideWindow) {
        // 划窗条件不满足，走兜底分支
        ClipCoordinates(
            inputXWFpLocal, inputYWFpLocal, inputXWIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, nwWeightLocal, weightMaskUb, outValueLocal, false);
        ClipCoordinates(
            inputXEFpLocal, inputYWFpLocal, inputXEIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, neWeightLocal, weightMaskUb, outValueLocal, true);
        ClipCoordinates(
            inputXWFpLocal, inputYEFpLocal, inputXWIntLocal, inputYEIntLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, swWeightLocal, weightMaskUb, outValueLocal, true);
        ClipCoordinates(
            inputXEFpLocal, inputYEFpLocal, inputXEIntLocal, inputYEIntLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, seWeightLocal, weightMaskUb, outValueLocal, true);
        PipeBarrier<PIPE_ALL>();
        CopyOut(nIdx, hwIdx, calHWElems);
        PipeBarrier<PIPE_ALL>();
        gridQueue_.FreeTensor(gridLocal);
        return;
    }

    event_t eventIdS_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdS_MTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIdS_MTE2);
    LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
    DataCopyExtParams params;
    // 按行搬，重复y次
    params.blockCount = yMax - yMin + 1;
    params.blockLen = (xMax - xMin + 1) * inputC_ * sizeof(T);
    params.srcStride = (inputW_ * inputC_) * sizeof(T) - (xMax - xMin + 1) * inputC_ * sizeof(T);
    // UB空间按aligh(X) * C对齐
    params.dstStride = Ceil(xMax - xMin + 1, 16) * inputC_ - Ceil((xMax - xMin + 1) * inputC_, 16);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    int64_t gmOffset = (int64_t)nIdx * inputH_ * inputW_ * inputC_ + (xMin + yMin * inputW_) * inputC_;
    DataCopyPad(xLocal, gmX_[gmOffset], params, padParams);

    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    ClipCoordinatesXInLocal(inputXWFpLocal,
        inputYWFpLocal,
        inputXWIntLocal,
        inputYWIntLocal,
        coordinatesLocal,
        weightMaskUb,
        xMin,
        xMax,
        yMin,
        yMax);
    PointBilinearXInLocal(
        nIdx, hwIdx, calHWElems, coordinatesLocal, nwWeightLocal, weightMaskUb, outValueLocal, false, xLocal);
    ClipCoordinatesXInLocal(inputXEFpLocal,
        inputYWFpLocal,
        inputXEIntLocal,
        inputYWIntLocal,
        coordinatesLocal,
        weightMaskUb,
        xMin,
        xMax,
        yMin,
        yMax);
    PointBilinearXInLocal(
        nIdx, hwIdx, calHWElems, coordinatesLocal, neWeightLocal, weightMaskUb, outValueLocal, true, xLocal);
    ClipCoordinatesXInLocal(inputXWFpLocal,
        inputYEFpLocal,
        inputXWIntLocal,
        inputYEIntLocal,
        coordinatesLocal,
        weightMaskUb,
        xMin,
        xMax,
        yMin,
        yMax);
    PointBilinearXInLocal(
        nIdx, hwIdx, calHWElems, coordinatesLocal, swWeightLocal, weightMaskUb, outValueLocal, true, xLocal);
    ClipCoordinatesXInLocal(inputXEFpLocal,
        inputYEFpLocal,
        inputXEIntLocal,
        inputYEIntLocal,
        coordinatesLocal,
        weightMaskUb,
        xMin,
        xMax,
        yMin,
        yMax);
    PointBilinearXInLocal(
        nIdx, hwIdx, calHWElems, coordinatesLocal, seWeightLocal, weightMaskUb, outValueLocal, true, xLocal);
    PipeBarrier<PIPE_ALL>();
    CopyOut(nIdx, hwIdx, calHWElems);
    PipeBarrier<PIPE_ALL>();
    gridQueue_.FreeTensor(gridLocal);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calHWElems = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = CAL_H_W_BLOCK;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
        }
        PerLoopCompute(nIdx, hwIdx, calHWElems);
    }
}

}  // namespace GridSample
#endif  // GRID_SAMPLER_2D_FP16_SLIDE_WINDOWs
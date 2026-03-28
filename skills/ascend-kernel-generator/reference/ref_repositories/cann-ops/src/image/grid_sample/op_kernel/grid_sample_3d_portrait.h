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
 * \file grid_sampler_3d_portrait.h
 * \brief
 */
#ifndef GIRD_SAMPLER_3D_PORTRAIT
#define GIRD_SAMPLER_3D_PORTRAIT

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {
using namespace AscendC;
template <typename T>
class GridSampler3DPortrait {
public:
    __aicore__ inline GridSampler3DPortrait(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR gird, GM_ADDR output, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitLocalVars();
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems);

    __aicore__ inline void ComputeWeightMul(LocalTensor<float> weightXBasic, LocalTensor<float> weightTempUb);
    __aicore__ inline void ClipCoordinates(
        int32_t offsetX, int32_t offsetY, int32_t offsetZ, LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> wMaskUb);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<float> iZFpUb, LocalTensor<uint8_t> wMaskUb, LocalTensor<uint8_t> maskTmpUb);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
        LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum);
    __aicore__ inline void CoordinatesSelectTensor(
        LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);
    __aicore__ inline void ComputeMask(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb,
        LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpUb);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void ReflectCoordinatesBefore(
        LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb, float negMinS, float spanS);
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<uint8_t> maskUb, const int64_t twiceLow, const int64_t twiceHigh);

    __aicore__ inline void GatherPtsFp32(
        int32_t loopIdx, int64_t gmOutOffset, int32_t calDHWElems, LocalTensor<float> weightUb, bool isAutomicAdd);
    __aicore__ inline void GatherPtsFp16(
        int32_t loopIdx, int32_t cIdx, int32_t calDHWElems, LocalTensor<float> weightUb, bool isAutomicAdd);
    __aicore__ inline void MvGatherRange(int32_t loopIdx, LocalTensor<int32_t> coorUb32I,
        LocalTensor<int32_t> coorTempUb32I, int32_t calDHWElems, int32_t inputElems);
    __aicore__ inline void PointBilinear(
        int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems, LocalTensor<float> weightUb, bool isAutomicAdd);

    __aicore__ inline void GatherGridAndClip(
        int64_t gridGmOffset, int32_t calDHWElems, LocalTensor<float> gridFp32Local);
    __aicore__ inline void ComputeWeightAndBilinear(int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems,
        LocalTensor<float> gridFp32Local, LocalTensor<float> inputX1FpLocal);
    __aicore__ inline void CopyOutFp16(int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> gridFp32Buf_;
    TBuf<QuePosition::VECCALC> inputXYZFPBuf_;
    TBuf<QuePosition::VECCALC> inputIntBuf_;
    TBuf<QuePosition::VECCALC> inputFpBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> weightTmpBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> coorTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> gridMaskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;
    TBuf<QuePosition::VECCALC> bufOutValueFp32_;
    TBuf<QuePosition::VECCALC> bufOutValueTmpFp32_;

    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmGrid_;

    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmOutput_;

    LocalTensor<T> xLocal_;
    LocalTensor<uint32_t> gridMaskLocal_;

    int64_t TRANSE_REP_STRIDE = 256;
    int64_t CAL_D_H_W_BLOCK = 512;
    int64_t MASK_UB_SIZE = 0;
    int64_t CHANNEL_BLOCK = 16384;
    int64_t channelBlockSize_ = 0;
    int64_t channelBlockSizeMins_ = 0;
    int64_t channelBlockSizeMaxs_ = 0;

    const int64_t X_UB_SIZE_4_GENERAL = 65536;
    const int64_t X_UB_SIZE_4_FP16 = 32768;
    const int64_t GRID_UB_SIZE_6_GENERAL = 6144;
    const int64_t GRID_UB_SIZE_6_FP16 = 3072;
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t GRID_UB_SIZE_4_FP16 = 2048;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t OUT_VAL_NUM = 4096;
    const int64_t X_UB_OFFSET = 512;

    const int32_t UB_ALIGN = 32;
    const int32_t DIMS = 3;

    int64_t blockIDX = 0;

    // tiling params
    int64_t coreNum_ = 0;
    int64_t inputN_ = 0;
    int64_t inputC_ = 0;
    int64_t inputD_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputD_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t interpolationMode_ = 0;
    int64_t paddingMode_ = 0;
    int64_t alignCorners_ = 0;
    int64_t channelLast_ = 0;
    int64_t needCoreNum_ = 0;

    int64_t gridDHW_ = 0;
    int64_t lastLoopHW_ = 0;
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;
    int32_t alignT = 0;
    bool setAtomicAdd_ = false;

    int64_t inputDHW_ = 0;
    int64_t inputDHWSize_ = 0;
    int32_t ubAlignElems_ = 0;
    int32_t calDHWGridElems_ = 0;
    float gridMulBeginX_ = 0.0f;
    float gridMulBeginY_ = 0.0f;
    float gridMulBeginZ_ = 0.0f;
    int32_t inputXYZFPBufOffset_ = 0;
    int32_t weightTmpBufOffset_ = 0;
    int32_t modBufOffset_ = 0;
    int32_t extraBufOffset_ = 0;

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
};

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::InitLocalVars()
{
    // 场景特化优化
    if (inputC_ == 32) {
        CHANNEL_BLOCK = 32768;
        TRANSE_REP_STRIDE = 512;
    } else {
        CAL_D_H_W_BLOCK = 1024;
        TRANSE_REP_STRIDE = 1024;
        if constexpr (IsSameType<T, half>::value) {
            CHANNEL_BLOCK = 32768;
        }
    }
    MASK_UB_SIZE = CAL_D_H_W_BLOCK / 8;
    // 标量计算提前算好赋值
    inputDHW_ = inputD_ * inputH_ * inputW_;
    inputDHWSize_ = inputDHW_ * sizeof(T);
    ubAlignElems_ = UB_ALIGN / sizeof(T);
    calDHWGridElems_ = CAL_D_H_W_BLOCK * DIMS;
    if (alignCorners_ == 1) {
        gridMulBeginX_ = (float)0.5 * (inputW_ - (float)1.0);
        gridMulBeginY_ = (float)0.5 * (inputH_ - (float)1.0);
        gridMulBeginZ_ = (float)0.5 * (inputD_ - (float)1.0);
    } else {
        gridMulBeginX_ = (float)0.5 * inputW_;
        gridMulBeginY_ = (float)0.5 * inputH_;
        gridMulBeginZ_ = (float)0.5 * inputD_;
    }
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ParseTilingData(const GridSampleTilingData *tilingData)
{
    coreNum_ = tilingData->coreNumVar;
    inputN_ = tilingData->inN;
    inputC_ = tilingData->inC;
    inputD_ = tilingData->inD;
    inputH_ = tilingData->inH;
    inputW_ = tilingData->inW;
    outputD_ = tilingData->outD;
    outputH_ = tilingData->outH;
    outputW_ = tilingData->outW;
    interpolationMode_ = tilingData->interpolationMode;
    paddingMode_ = tilingData->paddingMode;
    alignCorners_ = tilingData->alignCorners;
    channelLast_ = tilingData->channelLast;
    needCoreNum_ = tilingData->needCoreNum;

    InitLocalVars();

    gridDHW_ = outputD_ * outputH_ * outputW_;
    preNUbLoop_ = (gridDHW_ + CAL_D_H_W_BLOCK - 1) / CAL_D_H_W_BLOCK;  // 每个N的grid需要循环
    lastLoopHW_ = gridDHW_ - CAL_D_H_W_BLOCK * (preNUbLoop_ - 1);      // 最后一个尾块的grid
    totalUbLoop_ = preNUbLoop_ * inputN_;                              // 总共的grid循环次数
    // 每个核需要循环处理grid的次数 (总共的grid循环次数/核数并向上取整)
    preCoreLoop_ = (totalUbLoop_ + needCoreNum_ - 1) / needCoreNum_;
    // 实际需要的核数
    needCoreNum_ = (totalUbLoop_ + preCoreLoop_ - 1) / preCoreLoop_;
    // 最后一个核需要遍历grid的次数
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = (gridDHW_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    if (channelLoop_ != 1) {
        alignT = 32 / sizeof(T);
        perLoopChannel_ = CHANNEL_BLOCK;
        lastLoopChannel_ = gridDHW_ - perLoopChannel_ * (channelLoop_ - 1);

        channelBlockSize_ = CHANNEL_BLOCK * sizeof(T);
        channelBlockSizeMins_ = (CHANNEL_BLOCK + alignT) * sizeof(T);
        channelBlockSizeMaxs_ = (CHANNEL_BLOCK + alignT - 1) * sizeof(T);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::Init(
    GM_ADDR input, GM_ADDR gird, GM_ADDR output, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();

    // 初始化tiling
    ParseTilingData(tilingData);

    gmInput_.SetGlobalBuffer((__gm__ T *)input);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmOutput_.SetGlobalBuffer((__gm__ T *)output);

    // buffer申请初始化，offset用于空间复用，另外算子中的Select方法还要预留8K缓存(必要)
    // 0 ~ 32byte 用于临界值取值
    int64_t gridCalSize = CAL_D_H_W_BLOCK * sizeof(float) * DIMS;
    inputXYZFPBufOffset_ = 32 + gridCalSize;
    weightTmpBufOffset_ = inputXYZFPBufOffset_ + gridCalSize;
    modBufOffset_ = weightTmpBufOffset_ + CAL_D_H_W_BLOCK * sizeof(float) * 4;
    extraBufOffset_ = modBufOffset_ + CAL_D_H_W_BLOCK * sizeof(float);

    pipe.InitBuffer(weightBuf_, CAL_D_H_W_BLOCK * sizeof(float) * 8);
    pipe.InitBuffer(intTmpBuf_, CAL_D_H_W_BLOCK * sizeof(int32_t) * 2);
    pipe.InitBuffer(coorTmpBuf_, CAL_D_H_W_BLOCK * sizeof(float));
    pipe.InitBuffer(coorBuf_, CAL_D_H_W_BLOCK * sizeof(float));
    pipe.InitBuffer(inputFpBuf_, CAL_D_H_W_BLOCK * sizeof(float) * 6);
    pipe.InitBuffer(inputIntBuf_, CAL_D_H_W_BLOCK * sizeof(int32_t) * 6);

    if constexpr (IsSameType<T, half>::value) {
        // 64kb+64B(用于临界值取值)
        pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL + 64);
        // outValueBuf_要承担保留历史数据并做累加, C=32时此处占64K，C=4时是16K
        pipe.InitBuffer(outValueBuf_, CAL_D_H_W_BLOCK * inputC_ * sizeof(float));
        // half要先做float转换，需要一块额外空间, C=32时此处占3K，C=4时是6K
        pipe.InitBuffer(outTmpBuf_, CAL_D_H_W_BLOCK * (sizeof(half) + sizeof(float)));
    } else {
        // C=32时 此处各占 2KB；C=4时,此处各占4KB
        pipe.InitBuffer(outValueBuf_, CAL_D_H_W_BLOCK * sizeof(T));
        pipe.InitBuffer(outTmpBuf_, CAL_D_H_W_BLOCK * sizeof(T));
        if (inputC_ == 32) {
            pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL * 2 + 64);
        } else {
            pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL + 64);
        }
    }

    pipe.InitBuffer(maskBuf_, 640);        // 640B
    pipe.InitBuffer(gridMaskBuf_, 96);     // 96B
    pipe.InitBuffer(weightMaskBuf_, 128);  // 128B

    xLocal_ = xBuf_.AllocTensor<T>();
    gridMaskLocal_ = gridMaskBuf_.AllocTensor<uint32_t>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ComputeWeightMul(
    LocalTensor<float> weightXBasic, LocalTensor<float> weightTempUb)
{
    // weightXBasic是000或100
    auto weightx01 = weightXBasic[CAL_D_H_W_BLOCK];
    auto weightx10 = weightx01[CAL_D_H_W_BLOCK];
    auto weightx11 = weightx10[CAL_D_H_W_BLOCK];

    auto weightTmp2Local = weightTempUb[CAL_D_H_W_BLOCK];
    auto weightTmp1Local = weightTmp2Local[CAL_D_H_W_BLOCK];
    auto weightTmp3Local = weightTmp1Local[CAL_D_H_W_BLOCK];

    Mul(weightx10, weightXBasic, weightTmp1Local, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(weightx11, weightx10, weightTmp3Local, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(weightx10, weightx10, weightTmp2Local, CAL_D_H_W_BLOCK);
    Mul(weightXBasic, weightXBasic, weightTempUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(weightx01, weightXBasic, weightTmp3Local, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(weightXBasic, weightXBasic, weightTmp2Local, CAL_D_H_W_BLOCK);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ClipCoordinates(
    int32_t offsetX, int32_t offsetY, int32_t offsetZ, LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> wMaskUb)
{
    LocalTensor<float> inputFpUb = inputFpBuf_.Get<float>();
    LocalTensor<int32_t> inputIntUb = inputIntBuf_.Get<int32_t>();
    LocalTensor<float> iXFpUb = inputFpUb[calDHWGridElems_ * offsetX];
    LocalTensor<float> iYFpUb = inputFpUb[calDHWGridElems_ * offsetY + CAL_D_H_W_BLOCK];
    LocalTensor<float> iZFpUb = inputFpUb[calDHWGridElems_ * offsetZ + CAL_D_H_W_BLOCK * 2];
    LocalTensor<int32_t> iXIntUb = inputIntUb[calDHWGridElems_ * offsetX];
    LocalTensor<int32_t> iYIntUb = inputIntUb[calDHWGridElems_ * offsetY + CAL_D_H_W_BLOCK];
    LocalTensor<int32_t> iZIntUb = inputIntUb[calDHWGridElems_ * offsetZ + CAL_D_H_W_BLOCK * 2];

    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK * 2);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
    LocalTensor<int32_t> inputZIntTmpUb = tmpIntUb[CAL_D_H_W_BLOCK];
    PipeBarrier<PIPE_V>();
    // 此处加alignT，是为了方便后面做偏移取值
    Adds(inputXIntTmpUb, iXIntUb, alignT, CAL_D_H_W_BLOCK);
    Adds(inputYIntTmpUb, iYIntUb, 0, CAL_D_H_W_BLOCK);
    Adds(inputZIntTmpUb, iZIntUb, 0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(iZFpUb, inputZIntTmpUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 5);
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, iZFpUb, wMaskUb, maskUb);

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskXUbTmp = wMaskUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskUb.ReinterpretCast<uint16_t>();
    auto maskZUbTmp = maskUb[MASK_UB_SIZE].ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    And(maskXUbTmp, maskZUbTmp, maskXUbTmp, maskNum);
    wMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();

    CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ + alignT - 1));
    CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));
    CoordinatesFrameRange(inputZIntTmpUb, (int32_t)(inputD_ - 1));

    PipeBarrier<PIPE_V>();
    Muls(inputXIntTmpUb, inputXIntTmpUb, (int32_t)(sizeof(T)), CAL_D_H_W_BLOCK);
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)(inputW_ * sizeof(T)), CAL_D_H_W_BLOCK);
    Muls(inputZIntTmpUb, inputZIntTmpUb, (int32_t)(inputW_ * inputH_ * sizeof(T)), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputZIntTmpUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::Clip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb)
{
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(iXFpUb, iYFpUb, iZFpUb);
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(iXFpUb, iYFpUb, iZFpUb);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iIntUb, iIntUb, 0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb, LocalTensor<uint8_t> wMaskUb, LocalTensor<uint8_t> maskTmpUb)
{
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskTmpUb;
    LocalTensor<uint8_t> maskZUb = maskTmpUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpXUb = maskTmpUb[MASK_UB_SIZE * 2];
    LocalTensor<uint8_t> maskTmpYUb = maskTmpUb[MASK_UB_SIZE * 3];
    LocalTensor<uint8_t> maskTmpZUb = maskTmpUb[MASK_UB_SIZE * 4];
    CompareScalar(maskTmpXUb, iXFpUb, static_cast<float>(alignT), CMPMODE::GE, CAL_D_H_W_BLOCK);
    CompareScalar(maskXUb, iXFpUb, static_cast<float>(inputW_ + alignT - 1), CMPMODE::LE, CAL_D_H_W_BLOCK);
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, CAL_D_H_W_BLOCK);
    CompareScalar(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE, CAL_D_H_W_BLOCK);
    CompareScalar(maskTmpZUb, iZFpUb, 0.0f, CMPMODE::GE, CAL_D_H_W_BLOCK);
    CompareScalar(maskZUb, iZFpUb, static_cast<float>(inputD_ - 1), CMPMODE::LE, CAL_D_H_W_BLOCK);

    PipeBarrier<PIPE_V>();

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    auto maskTmpZUbTmp = maskTmpZUb.ReinterpretCast<uint16_t>();
    auto maskZUbTmp = maskZUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, maskNum);
    And(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, maskNum);
    And(maskZUbTmp, maskTmpZUbTmp, maskZUbTmp, maskNum);
    PipeBarrier<PIPE_V>();
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
    maskZUb = maskZUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
    LocalTensor<float> oFpUb, LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (calNum + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::CoordinatesSelectTensor(
    LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (CAL_D_H_W_BLOCK + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ComputeMask(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
    LocalTensor<float> iZFpUb, LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpUb)
{
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(tmpUb, iZFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iZFpUb, iZFpUb, maskUb, 0.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::BorderClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb)
{
    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_D_H_W_BLOCK);
    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_D_H_W_BLOCK);
    Mins(iZFpUb, iZFpUb, (float)(inputD_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_D_H_W_BLOCK);
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_D_H_W_BLOCK);
    Maxs(iZFpUb, iZFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<float> tmpUb = xBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK * 3, inputXYZFPBufOffset_);
    ComputeMask(iXFpUb, iYFpUb, iZFpUb, maskUb, tmpUb);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ReflectClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb)
{
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);

    int64_t twiceLow = (alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowY = REFLECT_RATIO * (inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (inputW_ - 1);
    int64_t twiceLowZ = REFLECT_RATIO * (inputD_ - 1);
    if (alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowY = REFLECT_RATIO * inputH_ - 1;
        twiceLowX = REFLECT_RATIO * inputW_ - 1;
        twiceLowZ = REFLECT_RATIO * inputD_ - 1;
    }
    ReflectCoordinatesGeneral(iYFpUb, iYFpUb, maskUb, twiceLow, twiceLowY);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(iXFpUb, iXFpUb, maskUb, twiceLow, twiceLowX);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(iZFpUb, iZFpUb, maskUb, twiceLow, twiceLowZ);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> tmpUb = xBuf_.GetWithOffset<float>(calDHWGridElems_, inputXYZFPBufOffset_);
    ComputeMask(iXFpUb, iYFpUb, iZFpUb, maskUb, tmpUb);
    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_D_H_W_BLOCK);
    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_D_H_W_BLOCK);
    Mins(iZFpUb, iZFpUb, (float)(inputD_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_D_H_W_BLOCK);
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_D_H_W_BLOCK);
    Maxs(iZFpUb, iZFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ReflectCoordinatesBefore(
    LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb, float negMinS, float spanS)
{
    LocalTensor<float> extraFpUb = xBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, extraBufOffset_);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    // new relative position
    Adds(coorSubUb, iFpUb, negMinS, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Abs(coorSubUb, coorSubUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(extraFpUb, extraFpUb, spanS, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(extraFpUb, coorSubUb, extraFpUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
    LocalTensor<float> coorSubUb, LocalTensor<uint8_t> maskUb, const int64_t twiceLow, const int64_t twiceHigh)
{
    LocalTensor<float> fmodFpUb = xBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, modBufOffset_);
    LocalTensor<float> extraFpUb = xBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, extraBufOffset_);
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    if (twiceLow == twiceHigh) {
        Duplicate(coorSubUb, (float)0.0, CAL_D_H_W_BLOCK);
        return;
    }

    float minS = static_cast<float>(twiceLow) / 2;
    float negMinS = static_cast<float>(-1.0) * minS;
    float spanS = static_cast<float>(twiceHigh - twiceLow) / 2;
    ReflectCoordinatesBefore(iFpUb, coorSubUb, negMinS, spanS);

    LocalTensor<float> out1 = tmpFpUb;
    LocalTensor<float> out2 = extraFpUb;
    LocalTensor<float> mods = fmodFpUb;

    Adds(out1, extraFpUb, minS, CAL_D_H_W_BLOCK);
    Muls(out2, extraFpUb, -1.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, spanS, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, minS, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(mods, coorSubUb, static_cast<float>(1 / 2.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(mods, mods, 2.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(mods, coorSubUb, mods, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CoordinatesSelectTensor(out1, out2, coorSubUb, maskUb);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::GatherPtsFp32(
    int32_t loopIdx, int64_t gmOutOffset, int32_t calDHWElems, LocalTensor<float> weightUb, bool isAutomicAdd)
{
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    LocalTensor<float> outValueUb = outValueBuf_.Get<float>();
    LocalTensor<uint32_t> coorTempUb = coorTmpBuf_.Get<uint32_t>();

    Gather(outValueUb, xLocal_, coorTempUb, 0, calDHWElems);
    PipeBarrier<PIPE_V>();
    Mul(outValueUb, outValueUb, weightUb, calDHWElems);
    if (loopIdx != 0 && !isAutomicAdd && !setAtomicAdd_) {
        SetAtomicAdd<float>();
        setAtomicAdd_ = true;
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(gmOutput_[gmOutOffset], outValueUb, {1, (uint16_t)(calDHWElems * sizeof(float)), 0, 0});
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::GatherPtsFp16(
    int32_t loopIdx, int32_t cIdx, int32_t calDHWElems, LocalTensor<float> weightUb, bool isAutomicAdd)
{
    LocalTensor<float> outValueUb = outValueBuf_.Get<float>();
    LocalTensor<float> outValueUbFp32 =
        outTmpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * sizeof(half));
    LocalTensor<half> outValueTempUb = outTmpBuf_.Get<half>(CAL_D_H_W_BLOCK);
    LocalTensor<uint32_t> coorTempUb = coorTmpBuf_.Get<uint32_t>();

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    Gather(outValueTempUb, xLocal_, coorTempUb, 0, calDHWElems);
    PipeBarrier<PIPE_V>();
    Cast(outValueUbFp32, outValueTempUb, RoundMode::CAST_NONE, calDHWElems);
    PipeBarrier<PIPE_V>();
    Mul(outValueUbFp32, outValueUbFp32, weightUb, calDHWElems);
    PipeBarrier<PIPE_V>();
    if (loopIdx == 0 && !isAutomicAdd) {
        DataCopy(outValueUb[cIdx * calDHWElems], outValueUbFp32, calDHWElems);
    } else {
        Add(outValueUb[cIdx * calDHWElems], outValueUb[cIdx * calDHWElems], outValueUbFp32, calDHWElems);
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::MvGatherRange(int32_t loopIdx, LocalTensor<int32_t> coorUb32I,
    LocalTensor<int32_t> coorTempUb32I, int32_t calDHWElems, int32_t inputElems)
{
    // 挪到一块取值区间内，超出的临界值都取0
    if (loopIdx != 0) {
        Maxs(coorTempUb32I, coorUb32I, (int32_t)((CHANNEL_BLOCK * loopIdx + alignT - 1) * sizeof(T)), calDHWElems);
        PipeBarrier<PIPE_V>();
        Adds(coorTempUb32I, coorTempUb32I, (int32_t)(-CHANNEL_BLOCK * sizeof(T) * loopIdx), calDHWElems);
        PipeBarrier<PIPE_V>();
        Mins(coorTempUb32I, coorTempUb32I, (int32_t)((alignT + inputElems) * sizeof(T)), calDHWElems);
    } else {
        Mins(coorTempUb32I, coorUb32I, (int32_t)(channelBlockSizeMins_), calDHWElems);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::PointBilinear(
    int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems, LocalTensor<float> weightUb, bool isAutomicAdd)
{
    LocalTensor<int32_t> coorUb32I = coorBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    auto coorUbF32 = coorUb32I.ReinterpretCast<float>();
    LocalTensor<uint32_t> coorTempUb = coorTmpBuf_.Get<uint32_t>();
    auto coorTempUb32I = coorTempUb.ReinterpretCast<int32_t>();
    LocalTensor<uint64_t> weightMaskUb = weightMaskBuf_.Get<uint64_t>();

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int64_t base = nIdx * gridDHW_ * inputC_;
    int64_t outBaseOffset = base + dhwIdx * CAL_D_H_W_BLOCK;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    int32_t exceptIdx = (alignT - 1) * sizeof(T);
    Select(coorUbF32, weightMaskUb, coorUbF32, (float)exceptIdx, SELMODE::VSEL_TENSOR_SCALAR_MODE, calDHWElems);
    PipeBarrier<PIPE_V>();
    for (int32_t loopIdx = 0; loopIdx < channelLoop_; loopIdx++) {
        int32_t inputElems = loopIdx == channelLoop_ - 1 ? lastLoopChannel_ : perLoopChannel_;
        params.blockLen = inputElems * sizeof(T);
        // 22,4,16,64,64场景会遇到尾块偏移没法达到临界值
        if (inputElems != CHANNEL_BLOCK) {
            xLocal_.SetValue(inputElems + alignT, static_cast<T>(0.0));
        }
        MvGatherRange(loopIdx, coorUb32I, coorTempUb32I, calDHWElems, inputElems);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        for (int32_t cIdx = 0; cIdx < inputC_; cIdx++) {
            int32_t cOffset = cIdx * gridDHW_;
            int32_t gmOffset = base + cOffset + CHANNEL_BLOCK * loopIdx;
            // 此处DataCopyPad性能比datacopy好
            DataCopyPad(xLocal_[alignT], gmInput_[gmOffset], params, padParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if constexpr (IsSameType<T, float>::value) {
                GatherPtsFp32(loopIdx, outBaseOffset + cOffset, calDHWElems, weightUb, isAutomicAdd);
            } else {
                GatherPtsFp16(loopIdx, cIdx, calDHWElems, weightUb, isAutomicAdd);
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::CopyOutFp16(int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems)
{
    LocalTensor<float> outLocalFP32 = outValueBuf_.AllocTensor<float>();
    LocalTensor<half> outLocalHf = xBuf_.AllocTensor<half>();

    Cast(outLocalHf[alignT], outLocalFP32, RoundMode::CAST_NONE, calDHWElems * inputC_);
    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    DataCopyExtParams params;
    params.blockCount = inputC_;
    params.blockLen = calDHWElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = (gridDHW_ - calDHWElems) * sizeof(T);
    int64_t gmYOffset = (int64_t)nIdx * gridDHW_ * inputC_ + (int64_t)dhwIdx * CAL_D_H_W_BLOCK;
    DataCopyPad(gmOutput_[gmYOffset], outLocalHf[alignT], params);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::ComputeWeightAndBilinear(int32_t nIdx, int32_t dhwIdx,
    int32_t calDHWElems, LocalTensor<float> gridFp32Local, LocalTensor<float> inputX1FpLocal)
{
    auto inputY1FpLocal = inputX1FpLocal[CAL_D_H_W_BLOCK];
    auto inputX2FpLocal = inputX1FpLocal[CAL_D_H_W_BLOCK * DIMS];
    auto inputY2FpLocal = inputX2FpLocal[CAL_D_H_W_BLOCK];
    auto inputXFpLocal = gridFp32Local;
    auto inputYFpLocal = gridFp32Local[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal000 = weightBuf_.Get<float>();
    LocalTensor<float> weightLocal001 = weightLocal000[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal010 = weightLocal001[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal011 = weightLocal010[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal100 = weightLocal011[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal101 = weightLocal100[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal110 = weightLocal101[CAL_D_H_W_BLOCK];
    LocalTensor<float> weightLocal111 = weightLocal110[CAL_D_H_W_BLOCK];
    // 此处顺序这样排列是为了后面合并sub操作 0 2 1 3
    LocalTensor<float> weightTmpLocal = xBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK * 4, weightTmpBufOffset_);
    LocalTensor<float> weightTmp1Local = weightTmpLocal[CAL_D_H_W_BLOCK * 2];
    // 先计算x的部分，复用权重申请空间
    Sub(weightLocal000, inputX2FpLocal, inputXFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightLocal100, inputXFpLocal, inputX1FpLocal, CAL_D_H_W_BLOCK);
    // 再计算y/z的部分，此处tensor合并，一并计算
    Sub(weightTmpLocal, inputY2FpLocal, inputYFpLocal, CAL_D_H_W_BLOCK * 2);
    Sub(weightTmp1Local, inputYFpLocal, inputY1FpLocal, CAL_D_H_W_BLOCK * 2);
    PipeBarrier<PIPE_V>();
    ComputeWeightMul(weightLocal000, weightTmpLocal);
    ComputeWeightMul(weightLocal100, weightTmpLocal);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    ClipCoordinates(0, 0, 0, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal000, false);
    ClipCoordinates(0, 0, 1, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal001, true);
    ClipCoordinates(0, 1, 0, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal010, true);
    ClipCoordinates(0, 1, 1, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal011, true);
    ClipCoordinates(1, 0, 0, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal100, true);
    ClipCoordinates(1, 0, 1, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal101, true);
    ClipCoordinates(1, 1, 0, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal110, true);
    ClipCoordinates(1, 1, 1, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, dhwIdx, calDHWElems, weightLocal111, true);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::GatherGridAndClip(
    int64_t gridGmOffset, int32_t calDHWElems, LocalTensor<float> gridFp32Local)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t calGridElems = calDHWElems * DIMS;
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<T> gridFp16Local = outTmpBuf_.Get<T>();
        DataCopy(gridFp16Local, gmGrid_[gridGmOffset], calGridElems);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(gridFp32Local, gridFp16Local, RoundMode::CAST_NONE, calDHWGridElems_);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopy(gridFp32Local, gmGrid_[gridGmOffset], calGridElems);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    LocalTensor<float> inputXYZUb = xBuf_.GetWithOffset<float>(calDHWGridElems_, inputXYZFPBufOffset_);
    Adds(inputXYZUb, gridFp32Local, (float)1.0, calDHWGridElems_);

    uint64_t rsvdCnt;
    uint32_t mask = 24;
    uint8_t repeatTimes = (calDHWElems + 8 - 1) / 8;
    LocalTensor<float> inputXFpLocal = gridFp32Local;
    LocalTensor<float> inputYFpLocal = gridFp32Local[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputZFpLocal = gridFp32Local[CAL_D_H_W_BLOCK * 2];
    PipeBarrier<PIPE_V>();
    GatherMask(inputZFpLocal,
        inputXYZUb,
        gridMaskLocal_,
        true,
        mask,
        {1, repeatTimes, static_cast<uint16_t>(DIMS), 0},
        rsvdCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(inputYFpLocal,
        inputXYZUb,
        gridMaskLocal_[8],
        true,
        mask,
        {1, repeatTimes, static_cast<uint16_t>(DIMS), 0},
        rsvdCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(inputXFpLocal,
        inputXYZUb,
        gridMaskLocal_[16],
        true,
        mask,
        {1, repeatTimes, static_cast<uint16_t>(DIMS), 0},
        rsvdCnt);
    PipeBarrier<PIPE_V>();
    if (alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, gridMulBeginX_, CAL_D_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, gridMulBeginY_, CAL_D_H_W_BLOCK);
        Muls(inputZFpLocal, inputZFpLocal, gridMulBeginZ_, CAL_D_H_W_BLOCK);
    } else {
        Muls(inputXFpLocal, inputXFpLocal, gridMulBeginX_, CAL_D_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, gridMulBeginY_, CAL_D_H_W_BLOCK);
        Muls(inputZFpLocal, inputZFpLocal, gridMulBeginZ_, CAL_D_H_W_BLOCK);
        PipeBarrier<PIPE_V>();
        Adds(gridFp32Local, gridFp32Local, (float)(-0.5), calDHWGridElems_);
    }
    PipeBarrier<PIPE_V>();
    Clip(inputXFpLocal, inputYFpLocal, inputZFpLocal);
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::PerLoopCompute(int32_t nIdx, int32_t dhwIdx, int32_t calDHWElems)
{
    int64_t gridGmOffset = nIdx * gridDHW_ * DIMS + dhwIdx * calDHWGridElems_;
    LocalTensor<float> gridFp32Local = xBuf_.GetWithOffset<float>(calDHWGridElems_, 32);
    GatherGridAndClip(gridGmOffset, calDHWElems, gridFp32Local);
    LocalTensor<float> inputXFpLocal = gridFp32Local;
    LocalTensor<float> inputYFpLocal = gridFp32Local[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputZFpLocal = gridFp32Local[CAL_D_H_W_BLOCK * 2];

    // 三维用1代表相邻的较小坐标，2代表相邻的较大坐标
    // 将此处tensor改为同纬度连续，便于后面合并cast和add操作
    LocalTensor<int32_t> inputX1IntLocal = inputIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> inputY1IntLocal = inputX1IntLocal[CAL_D_H_W_BLOCK];
    LocalTensor<int32_t> inputZ1IntLocal = inputY1IntLocal[CAL_D_H_W_BLOCK];
    LocalTensor<int32_t> inputX2IntLocal = inputZ1IntLocal[CAL_D_H_W_BLOCK];
    LocalTensor<int32_t> inputY2IntLocal = inputX2IntLocal[CAL_D_H_W_BLOCK];
    LocalTensor<int32_t> inputZ2IntLocal = inputY2IntLocal[CAL_D_H_W_BLOCK];

    LocalTensor<float> inputX1FpLocal = inputFpBuf_.Get<float>();
    LocalTensor<float> inputY1FpLocal = inputX1FpLocal[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputZ1FpLocal = inputY1FpLocal[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputX2FpLocal = inputZ1FpLocal[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputY2FpLocal = inputX2FpLocal[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputZ2FpLocal = inputY2FpLocal[CAL_D_H_W_BLOCK];

    // 此处xyz一次处理
    Cast(inputX1IntLocal, gridFp32Local, RoundMode::CAST_FLOOR, calDHWGridElems_);
    PipeBarrier<PIPE_V>();
    Cast(inputX1FpLocal, inputX1IntLocal, RoundMode::CAST_NONE, calDHWGridElems_);
    Adds(inputX2IntLocal, inputX1IntLocal, 1, calDHWGridElems_);
    PipeBarrier<PIPE_V>();
    Adds(inputX2FpLocal, inputX1FpLocal, (float)1.0, calDHWGridElems_);
    PipeBarrier<PIPE_V>();

    ComputeWeightAndBilinear(nIdx, dhwIdx, calDHWElems, gridFp32Local, inputX1FpLocal);
    if constexpr (IsSameType<T, half>::value) {
        CopyOutFp16(nIdx, dhwIdx, calDHWElems);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void GridSampler3DPortrait<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    int32_t nIdx = 0;
    int32_t dhwIdx = 0;

    int32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calDHWElems = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    // 为取值做准备，两个value是临界值
    if (channelLoop_ != 1) {
        xLocal_.SetValue(alignT - 1, static_cast<T>(0.0));
        xLocal_.SetValue(CHANNEL_BLOCK + alignT, static_cast<T>(0.0));
    }
    uint32_t patternX = 0b100100100100100100100100;
    uint32_t patternY = 0b010010010010010010010010;
    uint32_t patternZ = 0b001001001001001001001001;
    gridMaskLocal_.SetValue(0, patternX);
    gridMaskLocal_.SetValue(8, patternY);
    gridMaskLocal_.SetValue(16, patternZ);

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        int32_t nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        dhwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calDHWElems = CAL_D_H_W_BLOCK;
        if (dhwIdx == preNUbLoop_ - 1) {
            calDHWElems = lastLoopHW_;
        }
        PerLoopCompute(nIdx, dhwIdx, calDHWElems);
        SetAtomicNone();
        setAtomicAdd_ = false;
    }
}

}  // namespace GridSample
#endif
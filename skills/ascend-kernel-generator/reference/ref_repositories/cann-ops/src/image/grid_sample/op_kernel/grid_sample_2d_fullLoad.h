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
 * \file grid_sampler_2d_fullLoad.h
 * \brief
 */
#ifndef GIRD_SAMPLER_2D_FULLLOAD
#define GIRD_SAMPLER_2D_FULLLOAD

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T, int templateCNum>
class GridSampler2DFullLoad {
public:
    __aicore__ inline GridSampler2DFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);

    __aicore__ inline void ProcessingCoordinates(
        int32_t nIdx, int32_t hwIdx, int32_t calHWElems, LocalTensor<float> tmpLocal);

    __aicore__ inline void ClipCoordinates(LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
        LocalTensor<float> tmpLocal, LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb);

    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb, LocalTensor<uint8_t> maskTmpXUb,
        LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
        LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum);
    __aicore__ inline void CoordinatesSelectTensor(
        LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);

    __aicore__ inline void Clip(LocalTensor<float> tmpLocal);
    __aicore__ inline void BorderClip(LocalTensor<float> tmpLocal);
    __aicore__ inline void ReflectClip(LocalTensor<float> tmpLocal);

    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb, LocalTensor<uint8_t> maskUb,
        LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow, const int64_t twiceHigh);

    __aicore__ inline void MTE3ForNCHW(int32_t cIdx, int32_t calCElems, int32_t loopElems, int64_t outBaseOffset,
        LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void MTE3ForNCHWToWorkSpace(int32_t cIdx, int32_t calCElems, int32_t loopElems,
        LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void MTE3ForC32(GlobalTensor<float> gm_, int32_t calCElems, int32_t loopElems,
        LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);
    __aicore__ inline void PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);

    __aicore__ inline void PointBilinearC1(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> addUb, LocalTensor<float> tmpLocal);
    __aicore__ inline void PointBilinearC32(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void CopyOut(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);

private:
    TPipe pipe;
    // 输入
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    // 存储坐标
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;

    // 存储权重
    TBuf<QuePosition::VECCALC> weightBuf_;

    // 存放mask值
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;

    // 存放搬出数据
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> outAddBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;

    // FP16 场景使用的空间
    TBuf<QuePosition::VECCALC> gridFP16Buf_;
    TBuf<QuePosition::VECCALC> outValueFP16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 8;
    const int64_t ORI_X_SIZE = 80 * 1024;
    const int64_t FP16_X_SIZE = 40 * 1024;
    const int64_t C1_X_SIZE = 16 * 1024;
    const int64_t C1_X_COUNT = 4096;
    const int64_t ORI_H_W_BLOCK = 1024;
    const int64_t C1_H_W_BLOCK = 2048;
    const int64_t MASK_COUNT = 8;
    const int64_t BLOCK_SIZE = 32;
    const int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    const int64_t C32_H_W_BLOCK = 512;

    int64_t xUbSize = ORI_X_SIZE;
    int64_t calHWBlock = ORI_H_W_BLOCK;
    int32_t mulWeightLoop = calHWBlock / B32_MASK;

    int64_t calHWSize = calHWBlock * sizeof(float);
    int64_t CalOutputSize = calHWSize * CHANNEL_BLOCK;
    int64_t maskUbSize = calHWBlock / MASK_COUNT;

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

    int64_t lastXNIdx_ = -1;

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;
};

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::ParseTilingData(const GridSampleTilingData *tilingData)
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

    // 当c=1且input_h*input_w小于4k的时候，我们选择全载定制模板1；
    // 此时x全载的大小调整为16k，基本块大小调整为2k，在UB进行累加，其余操作不变
    if constexpr (templateCNum == 1) {
        xUbSize = C1_X_SIZE;
        calHWBlock = C1_H_W_BLOCK;
        mulWeightLoop = calHWBlock / B32_MASK;
        calHWSize = calHWBlock * sizeof(float);
        CalOutputSize = calHWSize * CHANNEL_BLOCK;
        maskUbSize = calHWBlock / MASK_COUNT;
    }

    gridHW_ = outputH_ * outputW_;
    preNUbLoop_ = (gridHW_ + calHWBlock - 1) / calHWBlock;
    lastLoopHW_ = gridHW_ - calHWBlock * (preNUbLoop_ - 1);
    totalUbLoop_ = preNUbLoop_ * inputN_;
    preCoreLoop_ = (totalUbLoop_ + needCoreNum_ - 1) / needCoreNum_;
    needCoreNum_ = (totalUbLoop_ + preCoreLoop_ - 1) / preCoreLoop_;
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = (inputC_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    if constexpr (IsSameType<T, half>::value) {
        xUbSize = FP16_X_SIZE;
        pipe.InitBuffer(gridFP16Buf_, calHWSize);
        if constexpr (templateCNum == 1) {
            pipe.InitBuffer(outValueFP16Buf_, calHWSize / 2);
        } else {
            pipe.InitBuffer(outValueFP16Buf_, CalOutputSize / 2);
        }
    }

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer申请初始化
    pipe.InitBuffer(xBuf_, xUbSize);
    pipe.InitBuffer(tmpBuf_, calHWSize * 6);        // 6倍的calHWSize
    pipe.InitBuffer(inputXIntBuf_, calHWSize * 2);  // 2倍的calHWSize
    pipe.InitBuffer(inputYIntBuf_, calHWSize * 2);  // 2倍的calHWSize
    pipe.InitBuffer(weightBuf_, calHWSize * 4);     // 4倍的calHWSize

    // weight buf多申请了一些
    pipe.InitBuffer(maskBuf_, maskUbSize * 8);        // 1k
    pipe.InitBuffer(weightMaskBuf_, maskUbSize * 4);  // 512B

    // C不等于1场景，输出是带上了C通道，需要使用outValueBuf_进行计算 C通道当前使用的是8
    if constexpr (templateCNum != 1) {
        pipe.InitBuffer(outValueBuf_, CalOutputSize);  //  8 * calHWSize
    }

    pipe.InitBuffer(outAddBuf_, calHWSize);  // calHWSize
    pipe.InitBuffer(coorBuf_, calHWSize);    // calHWSize
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::ClipCoordinates(LocalTensor<int32_t> iXIntUb,
    LocalTensor<int32_t> iYIntUb, LocalTensor<float> tmpLocal, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> weightMaskUb)
{
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<float> iXFpUb = tmpLocal;
    LocalTensor<float> iYFpUb = tmpLocal[calHWBlock];

    auto inputYIntTmpUb = tmpLocal[calHWBlock * 2].ReinterpretCast<int32_t>();
    auto tmpIntLocal1 = tmpLocal[calHWBlock * 3].ReinterpretCast<int32_t>();
    auto tmpIntLocal2 = tmpLocal[calHWBlock * 4].ReinterpretCast<int32_t>();

    PipeBarrier<PIPE_V>();
    Adds(inputXIntTmpUb, iXIntUb, 0, calHWBlock);
    Adds(inputYIntTmpUb, iYIntUb, 0, calHWBlock);
    PipeBarrier<PIPE_V>();

    Cast(iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, calHWBlock);
    Cast(iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, calHWBlock);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(maskUbSize * 3);
    LocalTensor<uint8_t> maskXUb = weightMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[maskUbSize];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[maskUbSize * 2];  // 2: iY temp mask

    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);
    int32_t maskNum = (maskUbSize + 1) / 2;  // 除2数据量按照uint16类型折半
    // 合法的X的mask
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    // 合法的Y的mask
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // maskXUbTmp：合法的点的mask
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    weightMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();

    // 重计算坐标，使坐标不超过边界
    Mins(tmpIntLocal1, inputXIntTmpUb, (int32_t)(inputW_ - 1), calHWBlock);
    Mins(tmpIntLocal2, inputYIntTmpUb, (int32_t)(inputH_ - 1), calHWBlock);
    PipeBarrier<PIPE_V>();
    Maxs(inputXIntTmpUb, tmpIntLocal1, 0, calHWBlock);
    Maxs(inputYIntTmpUb, tmpIntLocal2, 0, calHWBlock);
    PipeBarrier<PIPE_V>();

    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, calHWBlock);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, calHWBlock);
    PipeBarrier<PIPE_V>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::Clip(LocalTensor<float> tmpLocal)
{
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(tmpLocal);
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(tmpLocal);
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    CompareScalar(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, calHWBlock);
    CompareScalar(maskXUb, iXFpUb, static_cast<float>(inputW_ - 1), CMPMODE::LE, calHWBlock);
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, calHWBlock);
    CompareScalar(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE, calHWBlock);

    PipeBarrier<PIPE_V>();

    int32_t maskNum = (maskUbSize + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, maskNum);
    And(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, maskNum);
    PipeBarrier<PIPE_V>();
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
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

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::CoordinatesSelectTensor(
    LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (calHWBlock + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::BorderClip(LocalTensor<float> tmpLocal)
{
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlock];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlock * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlock * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlock * 4];

    // tmpBuf[x,x,3,4,x,x]->tmpBuf[1,x,x,4,x,x]
    Mins(tmpLocal1, tmpLocal3, (float)(inputW_ - 1), calHWBlock);
    // tmpBuf[1,x,x,4,x,x]->tmpBuf[1,2,x,x,x,x]
    Mins(tmpLocal2, tmpLocal4, (float)(inputH_ - 1), calHWBlock);
    PipeBarrier<PIPE_V>();

    // tmpBuf[1,2,X,x,x,x]->tmpBuf[x,2,3,x,x,x]
    Maxs(tmpLocal3, tmpLocal1, (float)0, calHWBlock);
    // tmpBuf[x,2,3,x,x,x]->tmpBuf[x,x,3,4,x,x]
    Maxs(tmpLocal4, tmpLocal2, (float)0, calHWBlock);
    PipeBarrier<PIPE_V>();

    // weightMaskBuf_作tmpBuf用，和weight无关
    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(maskUbSize);
    // +INF/-INF/NAN 场景下，+INF/-INF * 0 = NAN，消INF
    Muls(tmpLocal5, tmpLocal3, (float)(0.0), calHWBlock);
    PipeBarrier<PIPE_V>();
    // NAN eq NAN = FALSE，maskUb是NAN的mask
    Compare(maskUb, tmpLocal5, tmpLocal5, CMPMODE::EQ, calHWBlock);
    PipeBarrier<PIPE_V>();
    // 对上一步mask的位置置0，即+INF/-INF/NAN 全置0
    CoordinatesSelectScalar(tmpLocal3, tmpLocal3, maskUb, 0.0f, calHWBlock);
    PipeBarrier<PIPE_V>();

    // Y同理
    Muls(tmpLocal5, tmpLocal4, (float)(0.0), calHWBlock);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpLocal5, tmpLocal5, CMPMODE::EQ, calHWBlock);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(tmpLocal4, tmpLocal4, maskUb, 0.0f, calHWBlock);
    PipeBarrier<PIPE_V>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::ReflectClip(LocalTensor<float> tmpLocal)
{
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlock];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlock * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlock * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlock * 4];
    LocalTensor<float> tmpLocal6 = tmpLocal[calHWBlock * 5];
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(maskUbSize * 3);

    // coorUb = Y * inputW_ + X
    int64_t twiceLow = (alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowY = REFLECT_RATIO * (inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (inputW_ - 1);
    if (alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowY = REFLECT_RATIO * inputH_ - 1;
        twiceLowX = REFLECT_RATIO * inputW_ - 1;
    }
    LocalTensor<int32_t> tmpLocal6tmp = tmpLocal6.ReinterpretCast<int32_t>();

    ReflectCoordinatesGeneral(
        tmpLocal4, tmpLocal4, tmpLocal1, tmpLocal2, maskUb, tmpLocal5, tmpLocal6tmp, twiceLow, twiceLowY);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(
        tmpLocal3, tmpLocal3, tmpLocal1, tmpLocal2, maskUb, tmpLocal5, tmpLocal6tmp, twiceLow, twiceLowX);
    PipeBarrier<PIPE_V>();

    tmpLocal6 = tmpLocal6tmp.ReinterpretCast<float>();

    // LocalTensor<T> tmpUb = inputXYFPBuf_.Get<float>();
    Muls(tmpLocal1, tmpLocal3, (float)(0.0), calHWBlock);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpLocal1, tmpLocal1, CMPMODE::EQ, calHWBlock);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(tmpLocal3, tmpLocal3, maskUb, 0.0f, calHWBlock);
    PipeBarrier<PIPE_V>();
    Muls(tmpLocal1, tmpLocal4, (float)(0.0), calHWBlock);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpLocal1, tmpLocal1, CMPMODE::EQ, calHWBlock);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(tmpLocal4, tmpLocal4, maskUb, 0.0f, calHWBlock);
    PipeBarrier<PIPE_V>();

    Mins(tmpLocal3, tmpLocal3, (float)(inputW_ - 1), calHWBlock);
    PipeBarrier<PIPE_V>();
    Maxs(tmpLocal3, tmpLocal3, (float)0, calHWBlock);
    PipeBarrier<PIPE_V>();

    Mins(tmpLocal4, tmpLocal4, (float)(inputH_ - 1), calHWBlock);
    PipeBarrier<PIPE_V>();
    Maxs(tmpLocal4, tmpLocal4, (float)0, calHWBlock);
    PipeBarrier<PIPE_V>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
    LocalTensor<float> coorSubUb, LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
    LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
    const int64_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        Duplicate(coorSubUb, (float)0.0, calHWBlock);
        return;
    }

    float minS = static_cast<float>(twiceLow) / 2;
    float negMinS = static_cast<float>(-1.0) * minS;
    float spanS = static_cast<float>(twiceHigh - twiceLow) / 2;

    // new relative position
    Adds(coorSubUb, iFpUb, negMinS, calHWBlock);
    PipeBarrier<PIPE_V>();
    Abs(coorSubUb, coorSubUb, calHWBlock);
    PipeBarrier<PIPE_V>();

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanS), calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, calHWBlock);
    PipeBarrier<PIPE_V>();
    Muls(extraFpUb, extraFpUb, spanS, calHWBlock);
    PipeBarrier<PIPE_V>();
    Sub(extraFpUb, coorSubUb, extraFpUb, calHWBlock);
    PipeBarrier<PIPE_V>();

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanS), calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, calHWBlock);
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

    Adds(out1, extraFpUb, minS, calHWBlock);
    Muls(out2, extraFpUb, -1.0f, calHWBlock);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, spanS, calHWBlock);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, minS, calHWBlock);
    PipeBarrier<PIPE_V>();

    Muls(mods, coorSubUb, static_cast<float>(1 / 2.0), calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, calHWBlock);
    PipeBarrier<PIPE_V>();
    Muls(mods, mods, 2.0f, calHWBlock);
    PipeBarrier<PIPE_V>();
    Sub(mods, coorSubUb, mods, calHWBlock);
    PipeBarrier<PIPE_V>();

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, calHWBlock);
    PipeBarrier<PIPE_V>();

    CoordinatesSelectTensor(out1, out2, coorSubUb, maskUb);
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::MTE3ForNCHW(int32_t cIdx, int32_t calCElems,
    int32_t loopElems, int64_t outBaseOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
    bool isAutomicAdd)
{
    int64_t gmYBaseOffset = outBaseOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    if (calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb, calHWBlock);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            SetAtomicNone();
        } else {
            DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        }
    } else {
        for (int32_t i = 0; i < mulWeightLoop; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, 128, 128, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 搬出，outValueUb里面是CHW，搬出也是CHW
        DataCopyExtParams params;
        params.blockCount = calCElems;
        params.blockLen = loopElems * sizeof(float);
        params.srcStride = calHWBlock / BLOCK_NUM - Ceil(loopElems, BLOCK_NUM);
        ;
        params.dstStride = (outputH_ * outputW_ - loopElems) * sizeof(float);

        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopyPad(gmY_[gmYBaseOffset], outValueUb, params);
            SetAtomicNone();
        } else {
            DataCopyPad(gmY_[gmYBaseOffset], outValueUb, params);
        }
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::MTE3ForNCHWToWorkSpace(int32_t cIdx, int32_t calCElems,
    int32_t loopElems, LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    // 1024 * inputC_ * blockIDX 每个核的地址
    int64_t gmYBaseOffset = calHWBlock * inputC_ * blockIDX + cIdx * CHANNEL_BLOCK * calHWBlock;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    if (calCElems == 1) {
        // 乘以权重
        Mul(outValueUb, outValueUb, weightUb, calHWBlock);
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
        for (int32_t i = 0; i < mulWeightLoop; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, 128, 128, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 搬出，outValueUb里面是CHW，搬出也是CHW
        // 一把将C * 1024(hw)都搬到workspace
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = calCElems * calHWBlock * sizeof(float);
        params.srcStride = 0;
        params.dstStride = 0;

        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopyPad(gmWorkspace_[gmYBaseOffset], outValueUb, params);
            SetAtomicNone();
        } else {
            DataCopyPad(gmWorkspace_[gmYBaseOffset], outValueUb, params);
        }
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::CopyOut(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    LocalTensor<float> outLocal = outValueBuf_.Get<float>();
    LocalTensor<T> outLocalFP16 = outValueFP16Buf_.Get<T>();
    // 每次处理8(C)*1024(HW)个数据
    int64_t loopTime = Ceil(inputC_, 8);
    int64_t lastC = inputC_ - 8 * (loopTime - 1);
    int64_t dataCount = calHWBlock * 8;
    int64_t basegmWorkSpaceAddr = blockIDX * calHWBlock * inputC_;

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
        params.blockCount = 8;
        params.blockLen = calHWElems * sizeof(T);
        params.srcStride = calHWBlock / 16 - Ceil(calHWElems, 16);
        params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
        int64_t gmYOffset =
            (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * calHWBlock + i * 8 * outputH_ * outputW_;
        DataCopyPad(gmY_[gmYOffset], outLocalFP16, params);

        SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);

        SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    }

    dataCount = calHWBlock * lastC;
    DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + calHWBlock * 8 * (loopTime - 1)], dataCount);

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    Cast(outLocalFP16, outLocal, RoundMode::CAST_NONE, dataCount);

    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    DataCopyExtParams params;
    params.blockCount = lastC;
    params.blockLen = calHWElems * sizeof(T);
    params.srcStride = calHWBlock / 16 - Ceil(calHWElems, 16);
    params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
    int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * calHWBlock +
                        (loopTime - 1) * 8 * outputH_ * outputW_;
    DataCopyPad(gmY_[gmYOffset], outLocalFP16, params);
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::MTE3ForC32(GlobalTensor<float> gm_, int32_t calCElems,
    int32_t loopElems, LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    auto mulWeightLoop = C32_H_W_BLOCK / 64;
    for (int32_t i = 0; i < mulWeightLoop; i++) {
        int32_t outOffset = i * B32_MASK;
        int32_t weightOffset = i * B32_MASK;
        Mul(outValueUb[outOffset],
            outValueUb[outOffset],
            weightUb[weightOffset],
            B32_MASK,
            calCElems,
            {1, 1, 1, (uint8_t)(mulWeightLoop * 8), (uint8_t)(mulWeightLoop * 8), 0});
    }

    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    // 搬出，outValueUb里面是CHW，搬出也是CHW
    DataCopyExtParams params;
    params.blockCount = calCElems;
    if constexpr (IsSameType<T, half>::value) {
        params.blockLen = C32_H_W_BLOCK * sizeof(float);
        params.srcStride = 0;
        params.dstStride = (1024 - C32_H_W_BLOCK) * sizeof(float);
    } else {
        params.blockLen = loopElems * sizeof(float);
        params.srcStride = C32_H_W_BLOCK / BLOCK_NUM - Ceil(loopElems, BLOCK_NUM);
        params.dstStride = (outputH_ * outputW_ - loopElems) * sizeof(float);
    }

    if (isAutomicAdd) {
        SetAtomicAdd<float>();
        DataCopyPad(gm_, outValueUb, params);
        SetAtomicNone();
    } else {
        DataCopyPad(gm_, outValueUb, params);
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::OutTranspose(
    int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb)
{
    const int64_t TRANSE_REP_STRIDE = 512;
    LocalTensor<T> dstList[16];
    LocalTensor<T> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;

    if (channelAlign == 32 / sizeof(T)) {
        transDataParams.repeatTimes = 8 * 4;
        transDataParams.dstRepStride = sizeof(T) / 2;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 32 / sizeof(T)];
        }
        if constexpr (IsSameType<T, half>::value) {
            for (int32_t i = 0; i < 16; i++) {
                dstList[i] = outValueUb[i * TRANSE_REP_STRIDE];
            }
        } else {
            for (int32_t i = 0; i < 8; i++) {
                dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE];
                dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8];
            }
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::PointBilinear(int32_t nIdx, int32_t hwIdx,
    int32_t calHWElems, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, calHWBlock);
    }

    PipeBarrier<PIPE_V>();
    Muls(coordinatesUb, coordinatesUb, (int32_t)(sizeof(T) * inputC_), calHWBlock);
    int64_t outBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * calHWBlock;
    auto coorUb = coordinatesUb.ReinterpretCast<uint32_t>();
    int32_t loop_elems = calHWElems;
    int32_t ubOffset = 0;
    LocalTensor<T> xLocal = xBuf_.Get<T>();

    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<half> outValueFP16Local = outValueFP16Buf_.Get<half>();
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }
            PipeBarrier<PIPE_V>();
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            for (int32_t c_idx = 0; c_idx < calCElems; c_idx++) {
                uint32_t srcBaseAddr = cIdx * perLoopChannel_ * sizeof(T) + (uint32_t)c_idx * sizeof(T);
                Gather(outValueFP16Local[c_idx * calHWBlock], xLocal, coorUb, srcBaseAddr, calHWBlock);
            }

            PipeBarrier<PIPE_V>();
            for (size_t i = 0; i < calCElems; i++) {
                ubOffset = i * calHWBlock;
                Select(outValueFP16Local[ubOffset],
                    weightMaskUb,
                    outValueFP16Local[ubOffset],
                    half(0.0),
                    SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    calHWBlock);
            }

            PipeBarrier<PIPE_V>();
            Cast(outValueUb, outValueFP16Local, RoundMode::CAST_NONE, calCElems * calHWBlock);

            PipeBarrier<PIPE_V>();
            MTE3ForNCHWToWorkSpace(cIdx, calCElems, loop_elems, weightUb, outValueUb, isAutomicAdd);
            event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventMte3V);
        }
    } else {
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }
            PipeBarrier<PIPE_V>();
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            for (int32_t c_idx = 0; c_idx < calCElems; c_idx++) {
                uint32_t srcBaseAddr = cIdx * perLoopChannel_ * sizeof(T) + (uint32_t)c_idx * sizeof(T);
                Gather(outValueUb[c_idx * calHWBlock], xLocal, coorUb, srcBaseAddr, calHWBlock);
            }

            PipeBarrier<PIPE_V>();
            for (size_t i = 0; i < calCElems; i++) {
                ubOffset = i * calHWBlock;
                Select(outValueUb[ubOffset],
                    weightMaskUb,
                    outValueUb[ubOffset],
                    0.0f,
                    SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    calHWBlock);
            }

            PipeBarrier<PIPE_V>();
            MTE3ForNCHW(cIdx, calCElems, loop_elems, outBaseOffset, weightUb, outValueUb, isAutomicAdd);
            event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventMte3V);
        }
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::PointBilinearC32(int32_t nIdx, int32_t hwIdx,
    int32_t calHWElems, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, calHWBlock);
    }

    PipeBarrier<PIPE_V>();
    Muls(coordinatesUb, coordinatesUb, (int32_t)(sizeof(T) * inputC_), calHWBlock);
    int64_t outBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * calHWBlock;
    auto coorUb = coordinatesUb.ReinterpretCast<uint32_t>();
    int32_t loop_elems = calHWElems;
    int32_t ubOffset = 0;
    LocalTensor<T> xLocal = xBuf_.Get<T>();

    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<uint16_t> xLocalUint = xBuf_.Get<uint16_t>();
        LocalTensor<half> tmpBufTotal = tmpBuf_.Get<half>();
        LocalTensor<uint16_t> tmpBufUint = tmpBufTotal.ReinterpretCast<uint16_t>();
        LocalTensor<half> outValueFP16Local = outValueFP16Buf_.Get<half>();
        perLoopChannel_ = 16;
        channelLoop_ = Ceil(inputC_, perLoopChannel_);
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            int32_t loop_num_tmp = Ceil(loop_elems, C32_H_W_BLOCK);
            int32_t loop_elems_tmp = C32_H_W_BLOCK;

            for (auto HWLoop = 0; HWLoop < loop_num_tmp; HWLoop++) {
                if (HWLoop == loop_num_tmp - 1) {
                    loop_elems_tmp = loop_elems - HWLoop * C32_H_W_BLOCK;
                }
                PipeBarrier<PIPE_V>();
                event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);

                uint8_t repeatTime = (C32_H_W_BLOCK * perLoopChannel_ * sizeof(uint16_t)) / 256;
                GatherRepeatParams params{1, 8};

                Gatherb<uint16_t>(tmpBufUint[2 * 1024 / sizeof(uint16_t)],
                    xLocalUint[cIdx * perLoopChannel_],
                    coorUb[HWLoop * C32_H_W_BLOCK],
                    repeatTime,
                    params);

                PipeBarrier<PIPE_V>();
                OutTranspose(32 / sizeof(T), tmpBufTotal[2 * 1024 / sizeof(uint16_t)], outValueFP16Local);
                PipeBarrier<PIPE_V>();
                for (size_t i = 0; i < calCElems; i++) {
                    ubOffset = i * C32_H_W_BLOCK;
                    Select(outValueFP16Local[ubOffset],
                        weightMaskUb[HWLoop * C32_H_W_BLOCK / 8],
                        outValueFP16Local[ubOffset],
                        half(0.0),
                        SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        C32_H_W_BLOCK);
                }

                PipeBarrier<PIPE_V>();
                Cast(outValueUb, outValueFP16Local, RoundMode::CAST_NONE, calCElems * C32_H_W_BLOCK);

                PipeBarrier<PIPE_V>();
                int64_t gmYBaseOffset = calHWBlock * inputC_ * blockIDX + (int64_t)cIdx * perLoopChannel_ * calHWBlock +
                                        HWLoop * C32_H_W_BLOCK;
                MTE3ForC32(gmWorkspace_[gmYBaseOffset],
                    calCElems,
                    loop_elems_tmp,
                    weightUb[HWLoop * C32_H_W_BLOCK],
                    outValueUb,
                    isAutomicAdd);
                event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventMte3V);
                WaitFlag<HardEvent::MTE3_V>(eventMte3V);
            }
        }
    } else {
        auto outValueUbUint = outValueUb.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> xLocalUint = xBuf_.Get<uint32_t>();
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            int32_t loop_num_tmp = Ceil(loop_elems, C32_H_W_BLOCK);
            int32_t loop_elems_tmp = C32_H_W_BLOCK;
            for (auto HWLoop = 0; HWLoop < loop_num_tmp; HWLoop++) {
                if (HWLoop == loop_num_tmp - 1) {
                    loop_elems_tmp = loop_elems - HWLoop * C32_H_W_BLOCK;
                }
                PipeBarrier<PIPE_V>();
                event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);

                uint8_t repeatTime = (C32_H_W_BLOCK * perLoopChannel_ * sizeof(uint32_t)) / 256;
                GatherRepeatParams params{1, 8};

                Gatherb<uint32_t>(outValueUbUint[perLoopChannel_ * C32_H_W_BLOCK],
                    xLocalUint[cIdx * perLoopChannel_],
                    coorUb[HWLoop * C32_H_W_BLOCK],
                    repeatTime,
                    params);

                PipeBarrier<PIPE_V>();
                OutTranspose(32 / sizeof(T), outValueUb[perLoopChannel_ * C32_H_W_BLOCK], outValueUb);
                PipeBarrier<PIPE_V>();
                for (size_t i = 0; i < calCElems; i++) {
                    ubOffset = i * C32_H_W_BLOCK;
                    Select(outValueUb[ubOffset],
                        weightMaskUb[HWLoop * C32_H_W_BLOCK / 8],
                        outValueUb[ubOffset],
                        0.0f,
                        SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        C32_H_W_BLOCK);
                }

                PipeBarrier<PIPE_V>();
                int64_t gmYBaseOffset = outBaseOffset + HWLoop * C32_H_W_BLOCK + cIdx * CHANNEL_BLOCK * gridHW_;
                MTE3ForC32(gmY_[gmYBaseOffset],
                    calCElems,
                    loop_elems_tmp,
                    weightUb[HWLoop * C32_H_W_BLOCK],
                    outValueUb,
                    isAutomicAdd);
                event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventMte3V);
                WaitFlag<HardEvent::MTE3_V>(eventMte3V);
            }
        }
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::PointBilinearC1(int32_t nIdx, int32_t hwIdx,
    int32_t calHWElems, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> addUb, LocalTensor<float> tmpLocal)
{
    LocalTensor<float> tmpFPLocal = tmpLocal.ReinterpretCast<float>();
    LocalTensor<float> outValueUb = tmpFPLocal[calHWBlock * 5];

    PipeBarrier<PIPE_V>();
    Muls(coordinatesUb, coordinatesUb, (int32_t)(sizeof(T) * inputC_), calHWBlock);
    PipeBarrier<PIPE_V>();

    auto coorUb = coordinatesUb.ReinterpretCast<uint32_t>();
    LocalTensor<T> xLocal = xBuf_.Get<T>();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<half> outValueFP16Local = outValueFP16Buf_.Get<half>();
        Gather(outValueFP16Local, xLocal, coorUb, 0, calHWBlock);
        PipeBarrier<PIPE_V>();

        Cast(outValueUb, outValueFP16Local, RoundMode::CAST_NONE, inputC_ * calHWBlock);
    } else {
        Gather(outValueUb, xLocal, coorUb, 0, calHWBlock);
    }

    PipeBarrier<PIPE_V>();
    Mul(outValueUb, outValueUb, weightUb, calHWBlock);

    PipeBarrier<PIPE_V>();
    Select(outValueUb, weightMaskUb, outValueUb, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, calHWBlock);

    PipeBarrier<PIPE_V>();
    Add(addUb, addUb, outValueUb, calHWBlock);
    PipeBarrier<PIPE_V>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::ProcessingCoordinates(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems, LocalTensor<float> tmpLocal)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * calHWBlock * 2;
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlock];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlock * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlock * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlock * 4];
    LocalTensor<float> tmpLocal6 = tmpLocal[calHWBlock * 5];

    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = calHWElems * 2 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<T> gridFp16Local = gridFP16Buf_.Get<T>();
        DataCopyPad(gridFp16Local, gmGrid_[gridGmOffset], paramsGrid, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        Cast(tmpLocal1, gridFp16Local, RoundMode::CAST_NONE, calHWBlock * 2);
        PipeBarrier<PIPE_V>();
    } else {
        // grid put in tmpBuf[1,2,x,x,x,x]
        //  搬入2份calHWBlock
        DataCopyPad(tmpLocal1, gmGrid_[gridGmOffset], paramsGrid, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    }

    // put in tmpBuf[x,x,3,4,x,x]
    Adds(tmpLocal3, tmpLocal1, (float)1.0, calHWBlock * 2);
    PipeBarrier<PIPE_V>();

    uint32_t mask = calHWBlock * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;
    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;

    // weight put in tmpBuf[1,2,x,x,x,x]   1->x  2->y
    GatherMask(tmpLocal1, tmpLocal3, xPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(tmpLocal2, tmpLocal3, yPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // alignCorner流程
    if (alignCorners_ == 1) {
        // put in tmpBuf[x,x,3,4,x,x]
        Muls(tmpLocal3, tmpLocal1, (float)((float)0.5 * (inputW_ - (float)1.0)), calHWBlock);
        Muls(tmpLocal4, tmpLocal2, (float)((float)0.5 * (inputH_ - (float)1.0)), calHWBlock);
    } else {
        // put in tmpBuf[x,x,x,x,5,6]
        Muls(tmpLocal5, tmpLocal1, (float)((float)0.5 * inputW_), calHWBlock);
        Muls(tmpLocal6, tmpLocal2, (float)((float)0.5 * inputH_), calHWBlock);
        PipeBarrier<PIPE_V>();
        // put in tmpBuf[x,x,3,4,x,x]
        Adds(tmpLocal3, tmpLocal5, (float)(-0.5), calHWBlock);
        Adds(tmpLocal4, tmpLocal6, (float)(-0.5), calHWBlock);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::PerLoopCompute(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
    // 处理坐标以及alignCorner流程，将坐标存储在tmpBuf[x,x,3,4,x,x]
    ProcessingCoordinates(nIdx, hwIdx, calHWElems, tmpLocal);

    // clip后的结果坐标存储在tmpBuf[x,x,3,4,x,x]
    // tmpBuf_3是X坐标,tmpBuf_4是Y坐标
    Clip(tmpLocal);

    // 存储X和Y的int型坐标
    LocalTensor<int32_t> inputXIntLocal = inputXIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> inputYIntLocal = inputYIntBuf_.Get<int32_t>();

    LocalTensor<int32_t> inputXWIntLocal = inputXIntLocal;
    LocalTensor<int32_t> inputXEIntLocal = inputXIntLocal[calHWBlock];
    LocalTensor<int32_t> inputYWIntLocal = inputYIntLocal;
    LocalTensor<int32_t> inputYEIntLocal = inputYIntLocal[calHWBlock];

    // 临时变量空间
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlock];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlock * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlock * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlock * 4];
    LocalTensor<float> tmpLocal6 = tmpLocal[calHWBlock * 5];

    // 存储weight值
    LocalTensor<float> weightLocal = weightBuf_.Get<float>();
    LocalTensor<float> WeightTmpLocal1 = weightLocal;
    LocalTensor<float> WeightTmpLocal2 = weightLocal[calHWBlock];
    LocalTensor<float> WeightTmpLocal3 = weightLocal[calHWBlock * 2];
    LocalTensor<float> WeightTmpLocal4 = weightLocal[calHWBlock * 3];

    Cast(inputXWIntLocal, tmpLocal3, RoundMode::CAST_FLOOR, calHWBlock);
    Cast(inputYWIntLocal, tmpLocal4, RoundMode::CAST_FLOOR, calHWBlock);
    PipeBarrier<PIPE_V>();
    Cast(WeightTmpLocal1, inputXWIntLocal, RoundMode::CAST_NONE, calHWBlock);
    Cast(WeightTmpLocal2, inputYWIntLocal, RoundMode::CAST_NONE, calHWBlock);

    // int型X坐标从左到右 inputXWIntLocal inputXEIntLocal
    // int型Y坐标从左到右 inputYWIntLocal inputYEIntLocal
    Adds(inputXEIntLocal, inputXWIntLocal, 1, calHWBlock);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, calHWBlock);
    PipeBarrier<PIPE_V>();

    // float型X坐标从左到右 WeightTmpLocal1 tmpLocal3 WeightTmpLocal3
    // float型Y坐标从左到右 WeightTmpLocal2 tmpLocal4 WeightTmpLocal4
    Adds(WeightTmpLocal3, WeightTmpLocal1, (float)1.0, calHWBlock);
    Adds(WeightTmpLocal4, WeightTmpLocal2, (float)1.0, calHWBlock);
    PipeBarrier<PIPE_V>();

    // tmpLocal1:ceilX   tmpLocal2:ceilY
    // tmpLocal5:floorX  tmpLocal6:floorY
    Sub(tmpLocal1, WeightTmpLocal3, tmpLocal3, calHWBlock);
    Sub(tmpLocal2, WeightTmpLocal4, tmpLocal4, calHWBlock);
    Sub(tmpLocal5, tmpLocal3, WeightTmpLocal1, calHWBlock);
    Sub(tmpLocal6, tmpLocal4, WeightTmpLocal2, calHWBlock);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> nwWeightLocal = WeightTmpLocal1;
    LocalTensor<float> neWeightLocal = WeightTmpLocal2;
    LocalTensor<float> swWeightLocal = WeightTmpLocal3;
    LocalTensor<float> seWeightLocal = WeightTmpLocal4;

    // nwWeightLocal:ceilX * ceilY,  neWeightLocal:floorX * ceilY
    // swWeightLocal:ceilX * floorY, seWeightLocal:floorX * floorY
    Mul(nwWeightLocal, tmpLocal1, tmpLocal2, calHWBlock);
    Mul(neWeightLocal, tmpLocal5, tmpLocal2, calHWBlock);
    Mul(swWeightLocal, tmpLocal1, tmpLocal6, calHWBlock);
    Mul(seWeightLocal, tmpLocal5, tmpLocal6, calHWBlock);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>();
    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>();
    if constexpr (templateCNum == 1) {
        LocalTensor<float> outAddLocal = outAddBuf_.Get<float>();

        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

        Duplicate(outAddLocal, (float)0.0, calHWBlock);

        ClipCoordinates(inputXWIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx, hwIdx, calHWElems, coordinatesLocal, nwWeightLocal, weightMaskUb, outAddLocal, tmpLocal);
        ClipCoordinates(inputXEIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx, hwIdx, calHWElems, coordinatesLocal, neWeightLocal, weightMaskUb, outAddLocal, tmpLocal);
        ClipCoordinates(inputXWIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx, hwIdx, calHWElems, coordinatesLocal, swWeightLocal, weightMaskUb, outAddLocal, tmpLocal);
        ClipCoordinates(inputXEIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx, hwIdx, calHWElems, coordinatesLocal, seWeightLocal, weightMaskUb, outAddLocal, tmpLocal);

        int64_t gmYBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * calHWBlock;
        event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<T> outValueFP16Local = outValueFP16Buf_.Get<T>();
            Cast(outValueFP16Local, outAddLocal, RoundMode::CAST_NONE, calHWBlock);
            SetFlag<HardEvent::V_MTE3>(eventVMte3);
            WaitFlag<HardEvent::V_MTE3>(eventVMte3);
            DataCopyPad(gmY_[gmYBaseOffset], outValueFP16Local, {1, (uint16_t)(calHWElems * sizeof(T)), 0, 0});
        } else {
            SetFlag<HardEvent::V_MTE3>(eventVMte3);
            WaitFlag<HardEvent::V_MTE3>(eventVMte3);
            DataCopyPad(gmY_[gmYBaseOffset], outAddLocal, {1, (uint16_t)(calHWElems * sizeof(T)), 0, 0});
        }
    } else if constexpr (templateCNum == 2) {
        LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();

        ClipCoordinates(inputXWIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC32(nIdx, hwIdx, calHWElems, coordinatesLocal, nwWeightLocal, weightMaskUb, outValueLocal, false);
        ClipCoordinates(inputXEIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC32(nIdx, hwIdx, calHWElems, coordinatesLocal, neWeightLocal, weightMaskUb, outValueLocal, true);
        ClipCoordinates(inputXWIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC32(nIdx, hwIdx, calHWElems, coordinatesLocal, swWeightLocal, weightMaskUb, outValueLocal, true);
        ClipCoordinates(inputXEIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC32(nIdx, hwIdx, calHWElems, coordinatesLocal, seWeightLocal, weightMaskUb, outValueLocal, true);

        if constexpr (IsSameType<T, half>::value) {
            PipeBarrier<PIPE_ALL>();
            CopyOut(nIdx, hwIdx, calHWElems);
            PipeBarrier<PIPE_ALL>();
        }
    } else {
        LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();

        ClipCoordinates(inputXWIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, nwWeightLocal, weightMaskUb, outValueLocal, false);
        ClipCoordinates(inputXEIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, neWeightLocal, weightMaskUb, outValueLocal, true);
        ClipCoordinates(inputXWIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, swWeightLocal, weightMaskUb, outValueLocal, true);
        ClipCoordinates(inputXEIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, seWeightLocal, weightMaskUb, outValueLocal, true);

        if constexpr (IsSameType<T, half>::value) {
            PipeBarrier<PIPE_ALL>();
            CopyOut(nIdx, hwIdx, calHWElems);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <typename T, int templateCNum>
__aicore__ inline void GridSampler2DFullLoad<T, templateCNum>::Process()
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

    LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
    int32_t xElems = inputC_ * inputH_ * inputW_;

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {

        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = calHWBlock;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
        }

        if (nIdx != lastXNIdx_) {
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

            lastXNIdx_ = nIdx;
            int64_t xOffset = nIdx * inputC_ * inputH_ * inputW_;

            DataCopyExtParams paramsX;
            paramsX.blockCount = 1;
            paramsX.blockLen = xElems * sizeof(T);
            paramsX.srcStride = 0;
            paramsX.dstStride = 0;
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, gmX_[xOffset], paramsX, padParams);

            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }

        PerLoopCompute(nIdx, hwIdx, calHWElems);
    }
}

}  // namespace GridSample
#endif  // GIRD_SAMPLER_2D_FULLLOAD
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
 * \file grid_sampler_2d_fullLoad_310p.h
 * \brief
 */
#ifndef GIRD_SAMPLER_2D_FULLLOAD_310P
#define GIRD_SAMPLER_2D_FULLLOAD_310P

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler2DFullLoad310P {
public:
    __aicore__ inline GridSampler2DFullLoad310P(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ResetGMToZero();
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign);

    __aicore__ inline void ProcessingCoordinates(
        int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign, LocalTensor<float> tmpLocal);

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

    __aicore__ inline void MTE3ForNCHW(int32_t cIdx, int32_t calCElems, int32_t calHWElems, int32_t loopElems,
        int64_t outBaseOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd);

    __aicore__ inline void PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);

    __aicore__ inline void PointBilinearC1(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> addUb, LocalTensor<float> tmpLocal);

private:
    TPipe pipe;
    // 输入
    TBuf<QuePosition::VECIN> xBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    // 存储坐标
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;

    // 存储权重
    TBuf<QuePosition::VECCALC> weightBuf_;

    // 存放mask值
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;

    TBuf<QuePosition::VECCALC> bufferMaskBuf_;

    // 存放搬出数据
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> outAddBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;

    // FP16 场景使用的空间
    TBuf<QuePosition::VECCALC> gridFP16Buf_;
    TBuf<QuePosition::VECCALC> outValueFP16Buf_;

    // 软同步空间
    TBuf<QuePosition::VECCALC> mWorkBuf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<int32_t> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t B32_MASK = 64;
    const int64_t BATCH_BLOCK = 8;
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

    int64_t xUbSize = ORI_X_SIZE;
    int64_t calHWBlock = ORI_H_W_BLOCK;
    int64_t calHWBlockAlign = calHWBlock;
    int32_t mulWeightLoop = calHWBlockAlign / B32_MASK;

    int64_t calHWSize = calHWBlockAlign * sizeof(float);
    int64_t CalOutputSize = calHWSize * CHANNEL_BLOCK;
    int64_t maskUbSize = calHWBlockAlign / MASK_COUNT;  // 129

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

    int64_t lastXNIdx_ = -1;

    int64_t templateNum = 0;
    int64_t alignmentType = 0;

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;
};

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::ParseTilingData(const GridSampleTilingData *tilingData)
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
    templateNum = ((inputC_ == 1) && (inputH_ * inputW_ < C1_X_COUNT)) ? 1 : 0;
    if (templateNum == 1) {
        xUbSize = C1_X_SIZE;
        calHWBlock = C1_H_W_BLOCK;  // 每次处理2048个数，1024-》2048
        calHWBlockAlign = calHWBlock;
        mulWeightLoop = calHWBlockAlign / B32_MASK;
        calHWSize = calHWBlockAlign * sizeof(float);
        CalOutputSize = calHWSize * CHANNEL_BLOCK;
        maskUbSize = calHWBlock / MASK_COUNT;
    }

    gridHW_ = outputH_ * outputW_;
    preNUbLoop_ = (gridHW_ + calHWBlock - 1) / calHWBlock;   // 每个N、C有多少个block
    lastLoopHW_ = gridHW_ - calHWBlock * (preNUbLoop_ - 1);  // 尾块有多少数字
    // 考虑对齐拷贝场景，如果循环次数大于1，并且单次处理小于32字节，把最后一次处理合并到前一次处理中一起处理，避免最后因为尾数数量对齐导致结果出错
    if (preNUbLoop_ > 1 && lastLoopHW_ < BLOCK_NUM) {
        preNUbLoop_ = preNUbLoop_ - 1;
        lastLoopHW_ = calHWBlock + lastLoopHW_;
        calHWBlockAlign = calHWBlockAlign + BLOCK_NUM * 8;
        mulWeightLoop = calHWBlockAlign / B32_MASK;
        calHWSize = calHWBlockAlign * sizeof(float);
        CalOutputSize = calHWSize * CHANNEL_BLOCK;
        maskUbSize = (calHWBlock + 256) / MASK_COUNT;
    }

    lastLoopHWAlign_ = (lastLoopHW_ + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM;  // 32位对齐
    totalUbLoop_ = preNUbLoop_ * inputN_;                                      // 每个C有多少block数
    preCoreLoop_ = (totalUbLoop_ + needCoreNum_ - 1) / needCoreNum_;           // 每个核处理block次数
    needCoreNum_ = (totalUbLoop_ + preCoreLoop_ - 1) / preCoreLoop_;           // 更新需要的核数
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);          // 最后一核处理次数

    channelLoop_ = (inputC_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;  // 循环处理C的次数，每8个C同一批处理
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ int32_t *)workspace);
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

    pipe.InitBuffer(bufferMaskBuf_, BLOCK_SIZE * 2 * BATCH_BLOCK);

    // C不等于1场景，输出是带上了C通道，需要使用outValueBuf_进行计算 C通道当前使用的是8
    if (templateNum == 0) {
        pipe.InitBuffer(outValueBuf_, CalOutputSize);  //  8 * calHWSize
    } else {
        pipe.InitBuffer(outValueBuf_, calHWSize);  //  8 * calHWSize
    }

    pipe.InitBuffer(coorBuf_, calHWSize);  // calHWSize

    pipe.InitBuffer(mWorkBuf_, needCoreNum_ * BLOCK_SIZE);  // 软同步使用
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::ClipCoordinates(LocalTensor<int32_t> iXIntUb,
    LocalTensor<int32_t> iYIntUb, LocalTensor<float> tmpLocal, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> weightMaskUb)
{
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<float> iXFpUb = tmpLocal;
    LocalTensor<float> iYFpUb = tmpLocal[calHWBlockAlign];

    auto inputYIntTmpUb = tmpLocal[calHWBlockAlign * 2].ReinterpretCast<int32_t>();
    auto tmpIntLocal1 = tmpLocal[calHWBlockAlign * 3].ReinterpretCast<int32_t>();
    auto tmpIntLocal2 = tmpLocal[calHWBlockAlign * 4].ReinterpretCast<int32_t>();

    PipeBarrier<PIPE_V>();
    Adds(inputXIntTmpUb, iXIntUb, 0, calHWBlockAlign);
    Adds(inputYIntTmpUb, iYIntUb, 0, calHWBlockAlign);  // x，y值深拷贝到临时变量中
    PipeBarrier<PIPE_V>();

    Cast(iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, calHWBlockAlign);
    Cast(iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(maskUbSize * 3);
    LocalTensor<uint8_t> maskXUb = weightMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[maskUbSize];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[maskUbSize * 2];  // 2: iY temp mask

    CoordinatesGetMaskWithRange(
        iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);  // x坐标，y坐标，tmp*4, 只要坐标在0-size之间的数
    int32_t maskNum = (maskUbSize + 1) / 2;                         // 除2数据量按照uint16类型折半
    // 合法的X的mask
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    // 合法的Y的mask
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // maskXUbTmp：合法的点的mask
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    weightMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();

    // 重计算坐标，使坐标不超过边界
    Mins(tmpIntLocal1, inputXIntTmpUb, (int32_t)(inputW_ - 1), calHWBlockAlign);
    Mins(tmpIntLocal2, inputYIntTmpUb, (int32_t)(inputH_ - 1), calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Maxs(inputXIntTmpUb, tmpIntLocal1, 0, calHWBlockAlign);
    Maxs(inputYIntTmpUb, tmpIntLocal2, 0, calHWBlockAlign);
    PipeBarrier<PIPE_V>();

    // 这里是y* inputW + x，计算点坐标偏移
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::Clip(LocalTensor<float> tmpLocal)
{
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(tmpLocal);  // 取上下界为边界值
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(tmpLocal);  // 映射方式，比较复杂
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    CompareScalar(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, (calHWBlockAlign + 63) / 64 * 64);  // x > 0
    CompareScalar(maskXUb,
        iXFpUb,
        static_cast<float>(inputW_ - 1),
        CMPMODE::LE,
        (calHWBlockAlign + 63) / 64 * 64);                                                   // x < inputW - 1
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, (calHWBlockAlign + 63) / 64 * 64);  // y > 0
    CompareScalar(maskYUb,
        iYFpUb,
        static_cast<float>(inputH_ - 1),
        CMPMODE::LE,
        (calHWBlockAlign + 63) / 64 * 64);  // y < inputH - 1

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

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
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
__aicore__ inline void GridSampler2DFullLoad310P<T>::CoordinatesSelectTensor(
    LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (calHWBlockAlign + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::BorderClip(LocalTensor<float> tmpLocal)
{
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlockAlign];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlockAlign * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlockAlign * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlockAlign * 4];

    // tmpBuf[x,x,3,4,x,x]->tmpBuf[1,x,x,4,x,x]
    Mins(tmpLocal1, tmpLocal3, (float)(inputW_ - 1), calHWBlockAlign);
    // tmpBuf[1,x,x,4,x,x]->tmpBuf[1,2,x,x,x,x]
    Mins(tmpLocal2, tmpLocal4, (float)(inputH_ - 1), calHWBlockAlign);
    PipeBarrier<PIPE_V>();

    // tmpBuf[1,2,X,x,x,x]->tmpBuf[x,2,3,x,x,x]
    Maxs(tmpLocal3, tmpLocal1, (float)0, calHWBlockAlign);
    // tmpBuf[x,2,3,x,x,x]->tmpBuf[x,x,3,4,x,x]
    Maxs(tmpLocal4, tmpLocal2, (float)0, calHWBlockAlign);
    PipeBarrier<PIPE_V>();  // 上下界处理，border直接取边界值

    // weightMaskBuf_作tmpBuf用，和weight无关
    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(maskUbSize);
    // +INF/-INF/NAN 场景下，+INF/-INF * 0 = NAN，消INF
    Muls(tmpLocal5, tmpLocal3, (float)(0.0), calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    // NAN eq NAN = FALSE，maskUb是NAN的mask
    Compare(maskUb, tmpLocal5, tmpLocal5, CMPMODE::EQ, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    // 对上一步mask的位置置0，即+INF/-INF/NAN 全置0
    CoordinatesSelectScalar(tmpLocal3, tmpLocal3, maskUb, 0.0f, calHWBlockAlign);
    PipeBarrier<PIPE_V>();

    // Y同理
    Muls(tmpLocal5, tmpLocal4, (float)(0.0), calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpLocal5, tmpLocal5, CMPMODE::EQ, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(tmpLocal4, tmpLocal4, maskUb, 0.0f, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::ReflectClip(LocalTensor<float> tmpLocal)
{
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlockAlign];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlockAlign * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlockAlign * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlockAlign * 4];
    LocalTensor<float> tmpLocal6 = tmpLocal[calHWBlockAlign * 5];
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
    Muls(tmpLocal1, tmpLocal3, (float)(0.0), calHWBlockAlign);  // 去除inf值
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpLocal1, tmpLocal1, CMPMODE::EQ, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(tmpLocal3, tmpLocal3, maskUb, 0.0f, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Muls(tmpLocal1, tmpLocal4, (float)(0.0), calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpLocal1, tmpLocal1, CMPMODE::EQ, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(tmpLocal4, tmpLocal4, maskUb, 0.0f, calHWBlockAlign);
    PipeBarrier<PIPE_V>();

    Mins(tmpLocal3, tmpLocal3, (float)(inputW_ - 1), calHWBlockAlign);  // 边界值处理
    PipeBarrier<PIPE_V>();
    Maxs(tmpLocal3, tmpLocal3, (float)0, calHWBlockAlign);
    PipeBarrier<PIPE_V>();

    Mins(tmpLocal4, tmpLocal4, (float)(inputH_ - 1), calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Maxs(tmpLocal4, tmpLocal4, (float)0, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
    LocalTensor<float> coorSubUb, LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
    LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
    const int64_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        Duplicate(coorSubUb, (float)0.0, calHWBlockAlign);
        return;
    }

    float minS = static_cast<float>(twiceLow) / 2;               // -0.5 0
    float negMinS = static_cast<float>(-1.0) * minS;             // 0.5 0
    float spanS = static_cast<float>(twiceHigh - twiceLow) / 2;  // twice_span

    // new relative position
    Adds(coorSubUb, iFpUb, negMinS, calHWBlockAlign);  // x + 0.5
    PipeBarrier<PIPE_V>();
    Abs(coorSubUb, coorSubUb, calHWBlockAlign);  // |x + 0.5|
    PipeBarrier<PIPE_V>();

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanS), calHWBlockAlign);  // |x + 0.5| * 2 / size
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, calHWBlockAlign);  // 向下取整, double_flips
    PipeBarrier<PIPE_V>();
    Muls(extraFpUb, extraFpUb, spanS, calHWBlockAlign);  // extraFpUb * twice_span
    PipeBarrier<PIPE_V>();
    Sub(extraFpUb, coorSubUb, extraFpUb, calHWBlockAlign);  // extra
    PipeBarrier<PIPE_V>();

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanS), calHWBlockAlign);  // |x + 0.5| / twice_span
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, calHWBlockAlign);  // 向下取整, double_flips
    PipeBarrier<PIPE_V>();

    // coordinate
    /*
     S1: get two results for both possibilities, out1: extra + min, out2: muls(extra, -1.0) + span + min
     S2: get mod val, mods: flips % 2
     S3: get mask tensor, masks: CompareScalar(mods, 0.0)
     S4: select val from out1 and out2 by mask tensor, out: Select(out1, out2, mask)
    */
    LocalTensor<float> out1 =
        tmpFpUb;  // 下面逻辑超复杂，实际上就是torch的mininum(extra, twice_span_vec - extra) + low的逻辑
    LocalTensor<float> out2 = extraFpUb;
    LocalTensor<float> mods = fmodFpUb;

    Adds(out1, extraFpUb, minS, calHWBlockAlign);   // tmpLocal5 = extra - 0.5
    Muls(out2, extraFpUb, -1.0f, calHWBlockAlign);  // extraFpUb * -1
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, spanS, calHWBlockAlign);  // （extraFpUb * -1） + twice_span
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, minS, calHWBlockAlign);  // （extraFpUb * -1） + twice_span - 0.5
    PipeBarrier<PIPE_V>();

    Muls(mods, coorSubUb, static_cast<float>(1 / 2.0), calHWBlockAlign);  // double_flips / 2
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, calHWBlockAlign);
    PipeBarrier<PIPE_V>();
    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, calHWBlockAlign);  // 向下取整
    PipeBarrier<PIPE_V>();
    Muls(mods, mods, 2.0f, calHWBlockAlign);  // FLOOR(double_flips / 2) * 2
    PipeBarrier<PIPE_V>();
    Sub(mods, coorSubUb, mods, calHWBlockAlign);  // double_flips - FLOOR(double_flips / 2) * 2
    PipeBarrier<PIPE_V>();

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, calHWBlockAlign);
    PipeBarrier<PIPE_V>();

    CoordinatesSelectTensor(out1, out2, coorSubUb, maskUb);
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::MTE3ForNCHW(int32_t cIdx, int32_t calCElems, int32_t calHWElems,
    int32_t loopElems, int64_t outBaseOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
    bool isAutomicAdd)
{
    int64_t gmYBaseOffset = outBaseOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    LocalTensor<float> tmpLocal1 = tmpBuf_.Get<float>();
    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
    bufPattern.SetValue(0, 0b11111111);
    uint32_t mask = 16;
    uint64_t rsvdCnt = 0;

    if (calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb, calHWBlockAlign);  // 值 * 权重
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopy(gmY_[gmYBaseOffset], outValueUb, loopElems);
            PipeBarrier<PIPE_MTE3>();
            SetAtomicNone();
        } else {
            if (calHWElems >= BLOCK_NUM) {
                DataCopy(gmY_[gmYBaseOffset], outValueUb, calHWElems / BLOCK_NUM * BLOCK_NUM);
                PipeBarrier<PIPE_MTE3>();

                if (calHWElems != loopElems) {
                    GatherMask(
                        tmpLocal1, outValueUb[calHWElems - BLOCK_NUM], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
                    PipeBarrier<PIPE_V>();
                    DataCopy(gmY_[gmYBaseOffset + calHWElems - BLOCK_NUM], tmpLocal1, BLOCK_NUM);
                    PipeBarrier<PIPE_MTE3>();
                    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                }
            } else {
                // 如果数据量较小，使用atomicAdd方式累加
                SetAtomicAdd<float>();
                DataCopy(gmY_[gmYBaseOffset], outValueUb, BLOCK_NUM);
                SetAtomicNone();
            }
        }
    } else {
        uint8_t repStride = calHWBlockAlign / BLOCK_NUM;
        for (int32_t i = 0; i < mulWeightLoop; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, repStride, repStride, 0});
        }  // 值 * 权重

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 搬出，outValueUb里面是CHW，搬出也是CHW
        if (isAutomicAdd) {
            for (int64_t i = 0; i < calCElems; i++) {
                SetAtomicAdd<float>();
                DataCopy(gmY_[gmYBaseOffset + i * (outputH_ * outputW_)], outValueUb[calHWBlockAlign * i], loopElems);
                PipeBarrier<PIPE_MTE3>();
                SetAtomicNone();
            }
        } else {
            for (int64_t i = 0; i < calCElems; i++) {
                if (calHWElems >= BLOCK_NUM) {
                    // 不对齐场景，先拷贝对齐部分，不对齐部分重新设置偏移量拷贝
                    DataCopy(gmY_[gmYBaseOffset + i * (outputH_ * outputW_)],
                        outValueUb[calHWBlockAlign * i],
                        calHWElems / BLOCK_NUM * BLOCK_NUM);
                    PipeBarrier<PIPE_MTE3>();

                    if (calHWElems != loopElems) {
                        GatherMask(tmpLocal1,
                            outValueUb[calHWBlockAlign * i + calHWElems - BLOCK_NUM],
                            bufPattern,
                            true,
                            mask,
                            {1, 1, 8, 8},
                            rsvdCnt);
                        PipeBarrier<PIPE_V>();
                        DataCopy(gmY_[gmYBaseOffset + i * (outputH_ * outputW_) + calHWElems - BLOCK_NUM],
                            tmpLocal1,
                            BLOCK_NUM);
                        PipeBarrier<PIPE_MTE3>();
                        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                    }
                } else {
                    // 如果数据量较小，使用atomicAdd方式累加
                    SetAtomicAdd<float>();
                    DataCopy(
                        gmY_[gmYBaseOffset + i * (outputH_ * outputW_)], outValueUb[calHWBlockAlign * i], BLOCK_NUM);
                    SetAtomicNone();
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
    int32_t calHWElemsAlign, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, calHWBlockAlign);
    }

    PipeBarrier<PIPE_V>();
    Muls(coordinatesUb,
        coordinatesUb,
        (int32_t)(sizeof(T) * inputC_),
        calHWElems);  // 获取偏移值，Gather偏移量单位是字节
    auto coorUb = coordinatesUb.ReinterpretCast<uint32_t>();
    int64_t outBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * calHWBlock;
    if (alignmentType == 1) {
        outBaseOffset = nIdx * gridHW_ * inputC_;
    }
    int32_t loop_elems = calHWElemsAlign;
    int32_t ubOffset = 0;
    LocalTensor<T> xLocal = xBuf_.Get<T>();

    for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {  // C维度循环, 一次同时处理8个C
        Duplicate(outValueUb, (float)0.0, outValueUb.GetSize());
        int32_t calCElems = perLoopChannel_;  // 8
        if (cIdx == channelLoop_ - 1) {
            calCElems = lastLoopChannel_;
        }
        PipeBarrier<PIPE_V>();
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (int32_t c_idx = 0; c_idx < calCElems; c_idx++) {
            uint32_t srcBaseAddr = cIdx * perLoopChannel_ * sizeof(T) + (uint32_t)c_idx * sizeof(T);
            Gather(outValueUb[c_idx * calHWBlockAlign],
                xLocal,
                coorUb,
                srcBaseAddr,
                calHWElems);  // 从xLocal的coorUb地址，偏移srcBaseAddr，放到outValueUb中
        }                     // 一次拷贝8个calHWBlock数量

        PipeBarrier<PIPE_V>();
        for (size_t i = 0; i < calCElems; i++) {
            ubOffset = i * calHWBlockAlign;
            Select(outValueUb[ubOffset],
                weightMaskUb,
                outValueUb[ubOffset],
                0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                calHWBlockAlign);  // 判断值是否有效
        }
        PipeBarrier<PIPE_V>();
        MTE3ForNCHW(cIdx,
            calCElems,
            calHWElems,
            loop_elems,
            outBaseOffset,
            weightUb,
            outValueUb,
            isAutomicAdd);  // Cindex， 计算C的个数，一次计算数量，输出偏移量， 权重， 输出ub tensor， 是否累加
        event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::PointBilinearC1(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
    int32_t calHWElemsAlign, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
    LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> addUb, LocalTensor<float> tmpLocal)
{
    LocalTensor<float> tmpFPLocal = tmpLocal.ReinterpretCast<float>();
    LocalTensor<float> outValueUb = tmpFPLocal[calHWBlockAlign * 5];

    PipeBarrier<PIPE_V>();
    Muls(coordinatesUb,
        coordinatesUb,
        (int32_t)(sizeof(T) * inputC_),
        calHWElems);  // inputC_ = 1，C在最后一维，*C获取真实迁移
    PipeBarrier<PIPE_V>();

    auto coorUb = coordinatesUb.ReinterpretCast<uint32_t>();
    LocalTensor<T> xLocal = xBuf_.Get<T>();  // 一个D维度的输入
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    if (inputC_ == 1) {
        Gather(outValueUb, xLocal, coorUb, 0, calHWBlockAlign);  // 从xLocal的coorUb地址拷贝calHWBlock个数量
        PipeBarrier<PIPE_V>();
        Mul(outValueUb, outValueUb, weightUb, calHWElems);  // 值 * 权重
        PipeBarrier<PIPE_V>();
        Select(outValueUb,
            weightMaskUb,
            outValueUb,
            0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            calHWBlockAlign);  // 判断是否是有效值
        PipeBarrier<PIPE_V>();
        Add(addUb, addUb, outValueUb, calHWElems);  // 累加到UB的输出位置
        PipeBarrier<PIPE_V>();
    } else {  // 只有C *情况会走入
        for (int32_t c_idx = 0; c_idx < inputC_; c_idx++) {
            uint32_t srcBaseAddr = (uint32_t)c_idx * sizeof(T);
            Gather(outValueUb[c_idx * BLOCK_NUM],
                xLocal,
                coorUb,
                srcBaseAddr,
                BLOCK_NUM);  // if逻辑其实一样，inputC_ < 8, 只是方便好懂
        }
        PipeBarrier<PIPE_V>();
        for (int32_t c_idx = 0; c_idx < inputC_; c_idx++) {
            int32_t ubOffset = c_idx * BLOCK_NUM;
            Select(outValueUb[ubOffset],
                weightMaskUb,
                outValueUb[ubOffset],
                0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                BLOCK_NUM);  // if逻辑其实一样，inputC_ < 8, 只是方便好懂
        }
        PipeBarrier<PIPE_V>();
        for (int32_t c_idx = 0; c_idx < inputC_; c_idx++) {
            int32_t ubOffset = c_idx * BLOCK_NUM;
            Mul(outValueUb[ubOffset], outValueUb[ubOffset], weightUb, BLOCK_NUM);
        }  // 值 * 权重
        PipeBarrier<PIPE_V>();
        for (int32_t c_idx = 0; c_idx < inputC_; c_idx++) {
            int32_t ubOffset = c_idx * BLOCK_NUM;
            Add(addUb[ubOffset], addUb[ubOffset], outValueUb[ubOffset], calHWElems);  // 累加到UB的输出位置
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::ProcessingCoordinates(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign, LocalTensor<float> tmpLocal)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * calHWBlock * 2;  // GM偏移地址
    if (alignmentType == 1) {
        gridGmOffset = nIdx * gridHW_ * 2;
    }
    LocalTensor<float> tmpLocal1 = tmpLocal;                   // 要计算的grid数据，x
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlockAlign];  // 要计算的grid数据，y
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlockAlign * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlockAlign * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlockAlign * 4];
    LocalTensor<float> tmpLocal6 = tmpLocal[calHWBlockAlign * 5];

    Duplicate(tmpLocal, (float)0.0, calHWBlockAlign * 2);

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    // grid put in tmpBuf[1,2,x,x,x,x]
    //  搬入2份calHWBlock
    DataCopy(tmpLocal1, gmGrid_[gridGmOffset], calHWElemsAlign * 2);  // 把要算的grid搬运到tmpLocal1和tmpLocal2
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);

    // put in tmpBuf[x,x,3,4,x,x]
    Adds(tmpLocal3, tmpLocal1, (float)1.0, calHWElemsAlign * 2);  // 加1后，grid的datarange从-1~1到0~2
    PipeBarrier<PIPE_V>();

    uint32_t mask = calHWBlockAlign * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;  // 取偶数索引，x
    uint8_t yPattern = 2;  // 取奇数索引，y
    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;

    // weight put in tmpBuf[1,2,x,x,x,x]   1->x  2->y
    GatherMask(tmpLocal1,
        tmpLocal3,
        xPattern,
        true,
        mask,
        {1, 1, src0RepeatStride, src1RepeatStride},
        rsvdCnt);  // 连续，循环一次， tensor（x + 1.0）
    GatherMask(
        tmpLocal2, tmpLocal3, yPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);  // tensor（y
                                                                                                           // + 1.0）
    PipeBarrier<PIPE_V>();  // x值放到inputXFpLocal， y值放到inputYFpLocal
    // alignCorner流程
    if (alignCorners_ == 1) {
        // put in tmpBuf[x,x,3,4,x,x]
        Muls(tmpLocal3, tmpLocal1, (float)((float)0.5 * (inputW_ - (float)1.0)), calHWElemsAlign);
        Muls(tmpLocal4, tmpLocal2, (float)((float)0.5 * (inputH_ - (float)1.0)), calHWElemsAlign);
    } else {
        // put in tmpBuf[x,x,x,x,5,6]
        Muls(tmpLocal5, tmpLocal1, (float)((float)0.5 * inputW_), calHWElemsAlign);
        Muls(tmpLocal6, tmpLocal2, (float)((float)0.5 * inputH_), calHWElemsAlign);
        PipeBarrier<PIPE_V>();

        // put in tmpBuf[x,x,3,4,x,x]
        Adds(tmpLocal3, tmpLocal5, (float)(-0.5), calHWElemsAlign);
        Adds(tmpLocal4, tmpLocal6, (float)(-0.5), calHWElemsAlign);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::PerLoopCompute(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign)
{
    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
    // 处理坐标以及alignCorner流程，将坐标存储在tmpBuf[x,x,3,4,x,x]
    ProcessingCoordinates(nIdx, hwIdx, calHWElems, calHWElemsAlign, tmpLocal);  // 从grid中读取出x，y放到3,4位置

    // clip后的结果坐标存储在tmpBuf[x,x,3,4,x,x]
    // tmpBuf_3是X坐标,tmpBuf_4是Y坐标
    Clip(tmpLocal);  // 计算完成后，3,4位置的结果实际上已经是映射在input矩阵中的真实下标了

    // 存储X和Y的int型坐标
    LocalTensor<int32_t> inputXIntLocal = inputXIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> inputYIntLocal = inputYIntBuf_.Get<int32_t>();

    LocalTensor<int32_t> inputXWIntLocal = inputXIntLocal;
    LocalTensor<int32_t> inputXEIntLocal = inputXIntLocal[calHWBlockAlign];
    LocalTensor<int32_t> inputYWIntLocal = inputYIntLocal;
    LocalTensor<int32_t> inputYEIntLocal = inputYIntLocal[calHWBlockAlign];  // 存放四个点下标

    // 临时变量空间
    LocalTensor<float> tmpLocal1 = tmpLocal;
    LocalTensor<float> tmpLocal2 = tmpLocal[calHWBlockAlign];
    LocalTensor<float> tmpLocal3 = tmpLocal[calHWBlockAlign * 2];
    LocalTensor<float> tmpLocal4 = tmpLocal[calHWBlockAlign * 3];
    LocalTensor<float> tmpLocal5 = tmpLocal[calHWBlockAlign * 4];
    LocalTensor<float> tmpLocal6 = tmpLocal[calHWBlockAlign * 5];

    // 存储weight值
    LocalTensor<float> weightLocal = weightBuf_.Get<float>();
    LocalTensor<float> WeightTmpLocal1 = weightLocal;
    LocalTensor<float> WeightTmpLocal2 = weightLocal[calHWBlockAlign];
    LocalTensor<float> WeightTmpLocal3 = weightLocal[calHWBlockAlign * 2];
    LocalTensor<float> WeightTmpLocal4 = weightLocal[calHWBlockAlign * 3];

    Cast(inputXWIntLocal, tmpLocal3, RoundMode::CAST_FLOOR, calHWElemsAlign);
    Cast(inputYWIntLocal, tmpLocal4, RoundMode::CAST_FLOOR, calHWElemsAlign);
    PipeBarrier<PIPE_V>();
    Cast(WeightTmpLocal1, inputXWIntLocal, RoundMode::CAST_NONE, calHWElemsAlign);
    Cast(WeightTmpLocal2, inputYWIntLocal, RoundMode::CAST_NONE, calHWElemsAlign);

    // int型X坐标从左到右 inputXWIntLocal inputXEIntLocal
    // int型Y坐标从左到右 inputYWIntLocal inputYEIntLocal
    Adds(inputXEIntLocal, inputXWIntLocal, 1, calHWElemsAlign);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, calHWElemsAlign);
    PipeBarrier<PIPE_V>();

    // float型X坐标从左到右 WeightTmpLocal1 tmpLocal3 WeightTmpLocal3
    // float型Y坐标从左到右 WeightTmpLocal2 tmpLocal4 WeightTmpLocal4
    Adds(WeightTmpLocal3, WeightTmpLocal1, (float)1.0, calHWElemsAlign);
    Adds(WeightTmpLocal4, WeightTmpLocal2, (float)1.0, calHWElemsAlign);
    PipeBarrier<PIPE_V>();

    // tmpLocal1:ceilX   tmpLocal2:ceilY
    // tmpLocal5:floorX  tmpLocal6:floorY
    Sub(tmpLocal1, WeightTmpLocal3, tmpLocal3, calHWElemsAlign);  // 四个点单方向真实权重
    Sub(tmpLocal2, WeightTmpLocal4, tmpLocal4, calHWElemsAlign);
    Sub(tmpLocal5, tmpLocal3, WeightTmpLocal1, calHWElemsAlign);
    Sub(tmpLocal6, tmpLocal4, WeightTmpLocal2, calHWElemsAlign);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> nwWeightLocal = WeightTmpLocal1;
    LocalTensor<float> neWeightLocal = WeightTmpLocal2;
    LocalTensor<float> swWeightLocal = WeightTmpLocal3;
    LocalTensor<float> seWeightLocal = WeightTmpLocal4;

    // nwWeightLocal:ceilX * ceilY,  neWeightLocal:floorX * ceilY
    // swWeightLocal:ceilX * floorY, seWeightLocal:floorX * floorY
    Mul(nwWeightLocal, tmpLocal1, tmpLocal2, calHWElemsAlign);
    Mul(neWeightLocal, tmpLocal5, tmpLocal2, calHWElemsAlign);
    Mul(swWeightLocal, tmpLocal1, tmpLocal6, calHWElemsAlign);
    Mul(seWeightLocal, tmpLocal5, tmpLocal6, calHWElemsAlign);  // 四个点权重
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>();
    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>();
    if (templateNum == 1 || alignmentType == 1) {
        LocalTensor<float> outAddLocal = outValueBuf_.Get<float>();

        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

        if (alignmentType == 0) {
            Duplicate(outAddLocal, (float)0.0, calHWBlockAlign);
        }

        if (alignmentType == 1) {
            int64_t ubOffset = BLOCK_NUM * (hwIdx % BATCH_BLOCK) * inputC_;
            if (blockIDX != needCoreNum_ - 1 && hwIdx >= preCoreLoop_ / BATCH_BLOCK * BATCH_BLOCK) {
                ubOffset = BLOCK_NUM * (hwIdx - (preCoreLoop_ / BATCH_BLOCK - 1) * BATCH_BLOCK) * inputC_;
            } else if (blockIDX == needCoreNum_ - 1 && hwIdx < lastCoreLoop_ / BATCH_BLOCK * BATCH_BLOCK) {
                ubOffset = BLOCK_NUM * (hwIdx % BATCH_BLOCK) * inputC_;
            } else if (blockIDX == needCoreNum_ - 1 && hwIdx >= lastCoreLoop_ / BATCH_BLOCK * BATCH_BLOCK) {
                int64_t lastLenth = lastCoreLoop_ / BATCH_BLOCK - 1 > 0 ? lastCoreLoop_ / BATCH_BLOCK - 1 : 0;
                ubOffset = BLOCK_NUM * (hwIdx - lastLenth * BATCH_BLOCK) * inputC_;
            }
            outAddLocal = outAddLocal[ubOffset];
            Duplicate(outAddLocal, (float)0.0, BLOCK_NUM * inputC_);
        }
        ClipCoordinates(inputXWIntLocal,
            inputYWIntLocal,
            tmpLocal,
            coordinatesLocal,
            weightMaskUb);  // xw坐标，yw坐标，tmpLocal， 真实下标迁移量， 权重mask
        PointBilinearC1(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            nwWeightLocal,
            weightMaskUb,
            outAddLocal,
            tmpLocal);
        ClipCoordinates(inputXEIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            neWeightLocal,
            weightMaskUb,
            outAddLocal,
            tmpLocal);
        ClipCoordinates(inputXWIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            swWeightLocal,
            weightMaskUb,
            outAddLocal,
            tmpLocal);
        ClipCoordinates(inputXEIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinearC1(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            seWeightLocal,
            weightMaskUb,
            outAddLocal,
            tmpLocal);

        event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3);

        if (alignmentType == 1) {
            if ((hwIdx % BATCH_BLOCK == BATCH_BLOCK - 1) || hwIdx == preCoreLoop_ - 1 || nIdx == inputN_ - 1) {
                if ((blockIDX != needCoreNum_ - 1 &&
                        (preCoreLoop_ % BATCH_BLOCK == 0 || hwIdx != preCoreLoop_ / BATCH_BLOCK * BATCH_BLOCK - 1)) ||
                    (blockIDX == needCoreNum_ - 1 &&
                        (lastCoreLoop_ % BATCH_BLOCK == 0 || hwIdx != lastCoreLoop_ / BATCH_BLOCK * BATCH_BLOCK - 1))) {
                    outAddLocal = outValueBuf_.Get<float>();
                    int64_t nIdxLength = BATCH_BLOCK;
                    if (blockIDX != needCoreNum_ - 1 && hwIdx == preCoreLoop_ - 1) {
                        nIdxLength = preCoreLoop_ - (preCoreLoop_ / BATCH_BLOCK - 1) * BATCH_BLOCK;
                    }
                    if (blockIDX == needCoreNum_ - 1 && hwIdx == lastCoreLoop_ - 1) {
                        int64_t lastLenth = lastCoreLoop_ / BATCH_BLOCK - 1 > 0 ? lastCoreLoop_ / BATCH_BLOCK - 1 : 0;
                        nIdxLength = lastCoreLoop_ - lastLenth * BATCH_BLOCK;
                    }
                    nIdxLength = nIdxLength * inputC_;

                    // 通过GatherMask把中间的0去除
                    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
                    Duplicate(bufPattern, (uint32_t)0, bufPattern.GetSize());
                    for (int64_t i = 0; i < (nIdxLength + 3) / 4; i++) {
                        uint32_t bufValue = 0;
                        int8_t subLength = nIdxLength - i * 4 >= 4 ? 4 : nIdxLength - i * 4;
                        for (int8_t k = 0; k < subLength; k++) {
                            for (int8_t j = 0; j < gridHW_; j++) {
                                bufValue = bufValue + (1 << (k * BLOCK_NUM + j));
                            }
                        }
                        bufPattern.SetValue(i, bufValue);
                    }
                    uint32_t mask = nIdxLength * BLOCK_NUM;
                    uint64_t rsvdCnt = 0;
                    GatherMask(outAddLocal, outAddLocal, bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
                    PipeBarrier<PIPE_V>();
                    int64_t gmYBaseOffset = ((nIdx + 1) * inputC_ - nIdxLength) * gridHW_;
                    int64_t outputLength = nIdxLength * gridHW_;
                    if (outputLength % BATCH_BLOCK == 0) {
                        DataCopy(gmY_[gmYBaseOffset], outAddLocal, outputLength);
                    } else {
                        if (outputLength >= BLOCK_NUM) {
                            // 不对齐场景，先拷贝对齐部分，不对齐部分重新设置偏移量拷贝
                            DataCopy(gmY_[gmYBaseOffset], outAddLocal, outputLength);
                            PipeBarrier<PIPE_MTE3>();
                            // 使用GatherMask设置
                            LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
                            bufPattern.SetValue(0, 0b11111111);
                            uint32_t mask = 16;
                            uint64_t rsvdCnt = 0;
                            GatherMask(tmpLocal1,
                                outAddLocal[outputLength - BLOCK_NUM],
                                bufPattern,
                                true,
                                mask,
                                {1, 1, 8, 8},
                                rsvdCnt);
                            PipeBarrier<PIPE_V>();
                            DataCopy(gmY_[gmYBaseOffset + outputLength - BLOCK_NUM], tmpLocal1, BLOCK_NUM);
                            PipeBarrier<PIPE_MTE3>();
                        } else {
                            // 只有尾块会进来，数据量非常小
                            DataCopy(gmY_[gmYBaseOffset], outAddLocal, BLOCK_NUM);
                        }
                    }
                }
            }
        }

        if (alignmentType == 0) {
            int64_t gmYBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * calHWBlock;
            if (calHWElems != calHWElemsAlign) {
                if (calHWElems >= BLOCK_NUM) {
                    // 不对齐场景，先拷贝对齐部分，不对齐部分重新设置偏移量拷贝
                    DataCopy(gmY_[gmYBaseOffset], outAddLocal, calHWElemsAlign - BLOCK_NUM);
                    PipeBarrier<PIPE_MTE3>();
                    // 使用GatherMask设置
                    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
                    bufPattern.SetValue(0, 0b11111111);
                    uint32_t mask = 16;
                    uint64_t rsvdCnt = 0;
                    GatherMask(
                        tmpLocal1, outAddLocal[calHWElems - BLOCK_NUM], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
                    PipeBarrier<PIPE_V>();
                    DataCopy(gmY_[gmYBaseOffset + calHWElems - BLOCK_NUM], tmpLocal1, BLOCK_NUM);
                    PipeBarrier<PIPE_MTE3>();
                } else {
                    // 如果数据量较小，使用atomicAdd方式累加
                    SetAtomicAdd<float>();
                    DataCopy(gmY_[gmYBaseOffset], outAddLocal, BLOCK_NUM);
                    SetAtomicNone();
                }
            } else {
                DataCopy(gmY_[gmYBaseOffset], outAddLocal, calHWElemsAlign);
            }
            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        }

    } else {
        LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
        ClipCoordinates(inputXWIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            nwWeightLocal,
            weightMaskUb,
            outValueLocal,
            false);
        ClipCoordinates(inputXEIntLocal, inputYWIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            neWeightLocal,
            weightMaskUb,
            outValueLocal,
            true);
        ClipCoordinates(inputXWIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            swWeightLocal,
            weightMaskUb,
            outValueLocal,
            true);
        ClipCoordinates(inputXEIntLocal, inputYEIntLocal, tmpLocal, coordinatesLocal, weightMaskUb);
        PointBilinear(nIdx,
            hwIdx,
            calHWElems,
            calHWElemsAlign,
            coordinatesLocal,
            seWeightLocal,
            weightMaskUb,
            outValueLocal,
            true);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::ResetGMToZero()
{
    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    Duplicate(outValueLocal, (float)0.0, BLOCK_NUM);

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
                int64_t gmYBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * calHWBlock + cIdx * gridHW_;
                DataCopy(gmY_[gmYBaseOffset], outValueLocal, BLOCK_NUM);
            }
        }
    }
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    LocalTensor<int32_t> workLocal_ = mWorkBuf_.AllocTensor<int32_t>();
    SyncAll(gmWorkspace_, workLocal_, needCoreNum_);
}

template <typename T>
__aicore__ inline void GridSampler2DFullLoad310P<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;  // 每个核开始的block数
    int32_t calHWElems = 0;
    int32_t calHWElemsAlign = 0;

    if (gridHW_ < BLOCK_NUM && preCoreLoop_ < BATCH_BLOCK) {
        ResetGMToZero();
    }

    if (gridHW_ < BLOCK_NUM && preCoreLoop_ >= BATCH_BLOCK) {
        alignmentType = 1;
    }

    int64_t loopSize = preCoreLoop_;  // 要处理的block数量
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
    int32_t xElems = inputC_ * inputH_ * inputW_;

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;   // N维的index
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;  // h、w在block中位置
        calHWElems = calHWBlock;
        calHWElemsAlign = calHWBlock;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
            calHWElemsAlign = lastLoopHWAlign_;
        }

        if (alignmentType == 1) {
            hwIdx = loopIdx;
        }

        // 把block对应N的值，一次性从GM搬运到UB
        if (nIdx != lastXNIdx_) {
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

            lastXNIdx_ = nIdx;
            int64_t xOffset = nIdx * inputC_ * inputH_ * inputW_;
            DataCopy(xLocal, gmX_[xOffset], (xElems + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);

            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }
        PerLoopCompute(nIdx, hwIdx, calHWElems, calHWElemsAlign);
    }
}

}  // namespace GridSample
#endif  // GIRD_SAMPLER_2D_FULLLOAD_310P
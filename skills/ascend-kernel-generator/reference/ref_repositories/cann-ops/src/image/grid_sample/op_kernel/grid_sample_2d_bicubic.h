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
 * \file grid_sampler_2d_bicubic.h
 * \brief
 */
#ifndef GIRD_SAMPLER_BICUBIC_2D
#define GIRD_SAMPLER_BICUBIC_2D

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSamplerBicubic2D {
public:
    __aicore__ inline GridSamplerBicubic2D(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void CubicConvolution1(LocalTensor<float> coeff, LocalTensor<float> x);
    __aicore__ inline void CubicConvolution2(LocalTensor<float> coeff, LocalTensor<float> x);
    __aicore__ inline void GetCubicUpsampleCoefficients(LocalTensor<float> coeffTx0, LocalTensor<float> coeffTx1,
        LocalTensor<float> coeffTx2, LocalTensor<float> coeffTx3, LocalTensor<float> coeffTy0,
        LocalTensor<float> coeffTy1, LocalTensor<float> coeffTy2, LocalTensor<float> coeffTy3,
        LocalTensor<float> cubicTx, LocalTensor<float> cubicTy);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb, LocalTensor<uint8_t> maskTmpXUb,
        LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void CoordinatesSelectScalar(
        LocalTensor<float> iFpUb, LocalTensor<float> oFpUb, LocalTensor<uint8_t> maskUb, const float scalarVal);
    __aicore__ inline void ClipInfNan2Zero(LocalTensor<float> coordFpUb);
    __aicore__ inline void ZerosCoordinatesSelectScalar(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> inputXIntUb, LocalTensor<int32_t> inputYIntUb, LocalTensor<uint8_t> maskXUb,
        LocalTensor<uint8_t> maskYUb, const float scalarVal);
    __aicore__ inline void CoordinatesSelectTensor(
        LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);
    __aicore__ inline void ZerosCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb);
    __aicore__ inline void BorderCoordinates(
        LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<int32_t> coorUb);
    __aicore__ inline void ReflectCoordinates(
        LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<int32_t> coorUb);
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb, LocalTensor<uint8_t> maskUb,
        LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow, const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void OutTransposeFp16(
        int32_t channelAlign, LocalTensor<half> xLocal, LocalTensor<half> outValueUb);
    __aicore__ inline void OutTransposeFp32(
        int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);
    __aicore__ inline void ApplyCoeffTx(int32_t calCElems, int32_t loopOffset, LocalTensor<float> coeffTx,
        LocalTensor<float> outValueUb, LocalTensor<float> interp1dUb, int32_t interp1dIdx);
    __aicore__ inline void ApplyCoeffTy(
        int32_t calCElems, int32_t loopOffset, LocalTensor<float> coeffTy, LocalTensor<float> interp1dUb);
    __aicore__ inline void MTE3ForNCHW(int64_t gmYBaseOffset, int32_t calCElems, int64_t calHwNum, int32_t loopElems,
        LocalTensor<float> interp1dUb, GlobalTensor<float> dstGm, int32_t interp1dIdx);
    __aicore__ inline void CubicInterp1d(int32_t nIdx, int64_t outBaseOffset, int32_t loopIdx, int32_t loopOffset,
        int32_t loopElems, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> coeffTx, LocalTensor<float> coeffTy,
        LocalTensor<uint8_t> weightMaskUb, int32_t cIdx, int32_t calCElems, LocalTensor<float> interp1dUb,
        LocalTensor<float> outValueUb, int32_t interp1dIdx);
    __aicore__ inline void CopyOutFp16(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> gridFp32Buf_;
    TBuf<QuePosition::VECCALC> coeffBuf_;
    TBuf<QuePosition::VECCALC> coeffTmpBuf_;
    TBuf<QuePosition::VECCALC> inputXYFPBuf_;
    TBuf<QuePosition::VECCALC> inputXFpBuf_;
    TBuf<QuePosition::VECCALC> inputYFpBuf_;
    TBuf<QuePosition::VECCALC> infNanFpBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> coorTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> interp1dBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> infNanMaskBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;

    TBuf<QuePosition::VECCALC> gridFp16Buf_;
    TBuf<QuePosition::VECCALC> yFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;

    const int64_t X_UB_SIZE_4_GENERAL = 32768;    // 32KB
    const int64_t X_UB_SIZE_4_FP16 = 16384;       // 16KB
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;  //  4KB
    const int64_t GRID_UB_SIZE_4_FP16 = 2048;     //  2KB
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;     //  2KB
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
};

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ParseTilingData(const GridSampleTilingData *tilingData)
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
    preNUbLoop_ = (gridHW_ + CAL_H_W_BLOCK - 1) / CAL_H_W_BLOCK;
    lastLoopHW_ = gridHW_ - CAL_H_W_BLOCK * (preNUbLoop_ - 1);
    totalUbLoop_ = preNUbLoop_ * inputN_;
    preCoreLoop_ = (totalUbLoop_ + needCoreNum_ - 1) / needCoreNum_;
    needCoreNum_ = (totalUbLoop_ + preCoreLoop_ - 1) / preCoreLoop_;
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = (inputC_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();

    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer initialize
    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);                // 32KB
    pipe.InitBuffer(gridFp32Buf_, GRID_UB_SIZE_4_GENERAL);      //  4KB
    pipe.InitBuffer(coeffBuf_, GRID_UB_SIZE_4_GENERAL * 4);     // 16KB    512 * 4 * 8
    pipe.InitBuffer(coeffTmpBuf_, Y_UB_SIZE_4_GENERAL);         //  2KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);     //  4KB
    pipe.InitBuffer(inputXFpBuf_, GRID_UB_SIZE_4_GENERAL * 2);  //  8KB    512 * 4 * 4
    pipe.InitBuffer(inputYFpBuf_, GRID_UB_SIZE_4_GENERAL);      //  4KB
    pipe.InitBuffer(infNanFpBuf_, Y_UB_SIZE_4_GENERAL);         //  2KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);           //  2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);             //  2KB
    pipe.InitBuffer(coorTmpBuf_, Y_UB_SIZE_4_GENERAL);          //  2KB
    pipe.InitBuffer(interp1dBuf_, X_UB_SIZE_4_GENERAL);         // 32KB
    pipe.InitBuffer(outValueBuf_, X_UB_SIZE_4_GENERAL);         // 32KB
    pipe.InitBuffer(maskBuf_, 960);                             // 960B
    pipe.InitBuffer(weightMaskBuf_, 320);                       // 320B
    pipe.InitBuffer(infNanMaskBuf_, 320);                       // 320B
    pipe.InitBuffer(modBuf_, Y_UB_SIZE_4_GENERAL);              //  2KB
    pipe.InitBuffer(extraBuf_, Y_UB_SIZE_4_GENERAL);            //  2KB
    pipe.InitBuffer(outTmpBuf_, GRID_UB_SIZE_4_GENERAL);        //  4KB
    if constexpr (IsSameType<T, half>::value) {
        pipe.InitBuffer(gridFp16Buf_, GRID_UB_SIZE_4_FP16);  //  2KB
        pipe.InitBuffer(yFp16Buf_, X_UB_SIZE_4_FP16);        // 16KB
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CubicConvolution1(LocalTensor<float> coeff, LocalTensor<float> x)
{
    float alph = 2.0f - 0.75f;
    float beta = 0.75f - 3.0f;

    Muls(coeff, x, alph, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(coeff, coeff, beta, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(coeff, coeff, x, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(coeff, coeff, x, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(coeff, coeff, 1.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CubicConvolution2(LocalTensor<float> coeff, LocalTensor<float> x)
{
    float a = -0.75f;
    float alph = 0.75f * 5.0f;
    float beta = -0.75f * 8.0f;
    float gama = 0.75f * 4.0f;

    Muls(coeff, x, a, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(coeff, coeff, alph, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(coeff, coeff, x, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(coeff, coeff, beta, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(coeff, coeff, x, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(coeff, coeff, gama, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::GetCubicUpsampleCoefficients(LocalTensor<float> coeffTx0,
    LocalTensor<float> coeffTx1, LocalTensor<float> coeffTx2, LocalTensor<float> coeffTx3, LocalTensor<float> coeffTy0,
    LocalTensor<float> coeffTy1, LocalTensor<float> coeffTy2, LocalTensor<float> coeffTy3, LocalTensor<float> cubicTx,
    LocalTensor<float> cubicTy)
{
    LocalTensor<float> cubicTx1 = outValueBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> cubicTx2 = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> cubicTx3 = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 8);
    LocalTensor<float> cubicTxTmp = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 12);
    LocalTensor<float> cubicTy1 = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 16);
    LocalTensor<float> cubicTy2 = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 20);
    LocalTensor<float> cubicTy3 = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 24);
    LocalTensor<float> cubicTyTmp = outValueBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 28);

    Adds(cubicTx1, cubicTx, 1.0f, CAL_H_W_BLOCK);
    Adds(cubicTy1, cubicTy, 1.0f, CAL_H_W_BLOCK);
    Muls(cubicTxTmp, cubicTx, -1.0f, CAL_H_W_BLOCK);
    Muls(cubicTyTmp, cubicTy, -1.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(cubicTx2, cubicTxTmp, 1.0f, CAL_H_W_BLOCK);
    Adds(cubicTx3, cubicTxTmp, 2.0f, CAL_H_W_BLOCK);
    Adds(cubicTy2, cubicTyTmp, 1.0f, CAL_H_W_BLOCK);
    Adds(cubicTy3, cubicTyTmp, 2.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CubicConvolution2(coeffTx0, cubicTx1);
    CubicConvolution1(coeffTx1, cubicTx);
    CubicConvolution1(coeffTx2, cubicTx2);
    CubicConvolution2(coeffTx3, cubicTx3);
    CubicConvolution2(coeffTy0, cubicTy1);
    CubicConvolution1(coeffTy1, cubicTy);
    CubicConvolution1(coeffTy2, cubicTy2);
    CubicConvolution2(coeffTy3, cubicTy3);
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ClipCoordinates(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> wMaskUb)
{
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderCoordinates(iXFpUb, iYFpUb, coorUb);
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectCoordinates(iXFpUb, iYFpUb, coorUb);
    } else {  // default "zeros"
        ZerosCoordinates(iXFpUb, iYFpUb, coorUb, wMaskUb);
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iIntUb, iIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    CompareScalar(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    CompareScalar(maskXUb, iXFpUb, static_cast<float>(inputW_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    CompareScalar(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // map masks number to bit number of uint16 data
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;
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
__aicore__ inline void GridSamplerBicubic2D<T>::CoordinatesSelectScalar(
    LocalTensor<float> iFpUb, LocalTensor<float> oFpUb, LocalTensor<uint8_t> maskUb, const float scalarVal)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (CAL_H_W_BLOCK + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CoordinatesSelectTensor(
    LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (CAL_H_W_BLOCK + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ClipInfNan2Zero(LocalTensor<float> coordFpUb)
{
    LocalTensor<uint8_t> infNanMaskUb = infNanMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<float> infNanFpUb = infNanFpBuf_.Get<float>();
    Muls(infNanFpUb, coordFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(infNanMaskUb, infNanFpUb, infNanFpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(coordFpUb, coordFpUb, infNanMaskUb, 0.0f);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ZerosCoordinatesSelectScalar(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> inputXIntUb, LocalTensor<int32_t> inputYIntUb,
    LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb, const float scalarVal)
{
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK * 2);  // 2: temp buffer for iX and iY
    LocalTensor<float> tmpFpXUb = tmpFpUb;
    LocalTensor<float> tmpFpYUb = tmpFpUb[CAL_H_W_BLOCK];

    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (CAL_H_W_BLOCK + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;

    Select(tmpFpXUb, maskXUb, iXFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    Select(tmpFpYUb, maskYUb, iYFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();

    ClipInfNan2Zero(tmpFpXUb);
    ClipInfNan2Zero(tmpFpYUb);

    Cast(inputXIntUb, tmpFpXUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    Cast(inputYIntUb, tmpFpYUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ZerosCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
    LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb)
{
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;

    /*
      S1: check idx in range [0, IW], mask1
      S2: check idy in range [0, IH], mask2
      S3: merge mask with And func, mask = mask1 & mask2
      S4: select val beyond iY and 0 by mask
      S5: select val beyond iX and 0 by mask
      S6: calculate coor, coor = Y * inputW_ + X
    */
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);  // 3: three masks for iY and temp masks
    LocalTensor<uint8_t> maskXUb = weightMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);

    ZerosCoordinatesSelectScalar(iXFpUb, iYFpUb, inputXIntTmpUb, inputYIntTmpUb, maskXUb, maskYUb, 0.0f);

    // map masks number to bit number of uint16 data
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);

    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, inputXIntTmpUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::BorderCoordinates(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<int32_t> coorUb)
{
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK * 2);
    LocalTensor<float> tmpFpXUb = tmpFpUb;
    LocalTensor<float> tmpFpYUb = tmpFpUb[CAL_H_W_BLOCK];
    PipeBarrier<PIPE_V>();

    Adds(tmpFpXUb, iXFpUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    ClipInfNan2Zero(tmpFpXUb);
    Cast(inputXIntTmpUb, tmpFpXUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);

    Adds(tmpFpYUb, iYFpUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    ClipInfNan2Zero(tmpFpYUb);
    Cast(inputYIntTmpUb, tmpFpYUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ - 1));
    CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ReflectCoordinates(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<int32_t> coorUb)
{
    LocalTensor<float> coorSubUb = coorTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> extraFpUb = extraBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> fmodFpUb = modBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
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

    ReflectCoordinatesGeneral(iYFpUb, coorSubUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowY);
    ClipInfNan2Zero(coorSubUb);
    Cast(coorUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesFrameRange(coorUb, (int32_t)(inputH_ - 1));
    Muls(coorUb, coorUb, (int32_t)inputW_, CAL_H_W_BLOCK);

    ReflectCoordinatesGeneral(iXFpUb, coorSubUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowX);
    ClipInfNan2Zero(coorSubUb);
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesFrameRange(tmpIntUb, (int32_t)(inputW_ - 1));
    Add(coorUb, tmpIntUb, coorUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
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

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    for (int32_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation = nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int cIter = 0; cIter < channelAlign; cIter++) {
            int32_t xLocalOffset = i * channelAlign + cIter;
            if (cIter >= calCElems) {
                xLocal.SetValue(xLocalOffset, static_cast<T>(0.0));
                continue;
            }

            int64_t coordinate = baseLocation + cIter * inputH_ * inputW_;
            xLocal.SetValue(xLocalOffset, gmX_.GetValue(coordinate));
        }
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    int64_t base = nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;
    auto timeStep = loopElems / 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
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
__aicore__ inline void GridSamplerBicubic2D<T>::OutTransposeFp16(
    int32_t channelAlign, LocalTensor<half> xLocal, LocalTensor<half> outValueUb)
{
    LocalTensor<half> dstList[16];
    LocalTensor<half> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == B16_ALIGN_FACTOR) {
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
__aicore__ inline void GridSamplerBicubic2D<T>::OutTransposeFp32(
    int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb)
{
    LocalTensor<float> dstList[16];
    LocalTensor<float> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == 8) {
        transDataParams.repeatTimes = 8;
        transDataParams.dstRepStride = 2;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 8];
        }

        for (int32_t i = 0; i < 8; i++) {
            dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE];
            dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 8;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * channelAlign + j * 16 * channelAlign];
            }

            for (int32_t i = 0; i < 8; i++) {
                dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE + j * 16];
                dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8 + j * 16];
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
__aicore__ inline void GridSamplerBicubic2D<T>::ApplyCoeffTx(int32_t calCElems, int32_t loopOffset,
    LocalTensor<float> coeffTx, LocalTensor<float> outValueUb, LocalTensor<float> interp1dUb, int32_t interp1dIdx)
{
    if (calCElems == 1) {
        if (interp1dIdx % 4 == 0) {
            Mul(interp1dUb, outValueUb, coeffTx[loopOffset], TRANSE_REP_STRIDE);
            PipeBarrier<PIPE_V>();
        } else {
            Mul(outValueUb, outValueUb, coeffTx[loopOffset], TRANSE_REP_STRIDE);
            PipeBarrier<PIPE_V>();
            Add(interp1dUb, interp1dUb, outValueUb, TRANSE_REP_STRIDE);
            PipeBarrier<PIPE_V>();
        }
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = loopOffset + i * B32_MASK;
            if (interp1dIdx % 4 == 0) {
                Mul(interp1dUb[outOffset],
                    outValueUb[outOffset],
                    coeffTx[weightOffset],
                    B32_MASK,
                    calCElems,
                    {1, 1, 1, 16, 16, 0});
                PipeBarrier<PIPE_V>();
            } else {
                Mul(outValueUb[outOffset],
                    outValueUb[outOffset],
                    coeffTx[weightOffset],
                    B32_MASK,
                    calCElems,
                    {1, 1, 1, 16, 16, 0});
                PipeBarrier<PIPE_V>();
                Add(interp1dUb[outOffset],
                    interp1dUb[outOffset],
                    outValueUb[outOffset],
                    B32_MASK,
                    calCElems,
                    {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::ApplyCoeffTy(
    int32_t calCElems, int32_t loopOffset, LocalTensor<float> coeffTy, LocalTensor<float> interp1dUb)
{
    if (calCElems == 1) {
        Mul(interp1dUb, interp1dUb, coeffTy[loopOffset], TRANSE_REP_STRIDE);
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = loopOffset + i * B32_MASK;
            Mul(interp1dUb[outOffset],
                interp1dUb[outOffset],
                coeffTy[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, 16, 16, 0});
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::MTE3ForNCHW(int64_t gmYBaseOffset, int32_t calCElems, int64_t calHwNum,
    int32_t loopElems, LocalTensor<float> interp1dUb, GlobalTensor<float> dstGm, int32_t interp1dIdx)
{
    if (calCElems == 1) {
        if (interp1dIdx == 3) {
            DataCopyPad(dstGm[gmYBaseOffset], interp1dUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        } else {
            SetAtomicAdd<float>();
            DataCopyPad(dstGm[gmYBaseOffset], interp1dUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            SetAtomicNone();
        }
    } else {
        if (interp1dIdx == 3) {
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * calHwNum;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(dstGm[gmYOffset],
                    interp1dUb[i * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
        } else {
            SetAtomicAdd<float>();
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * calHwNum;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(dstGm[gmYOffset],
                    interp1dUb[i * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
            SetAtomicNone();
        }
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CubicInterp1d(int32_t nIdx, int64_t outBaseOffset, int32_t loopIdx,
    int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coordinatesUb, LocalTensor<float> coeffTx,
    LocalTensor<float> coeffTy, LocalTensor<uint8_t> weightMaskUb, int32_t cIdx, int32_t calCElems,
    LocalTensor<float> interp1dUb, LocalTensor<float> outValueUb, int32_t interp1dIdx)
{
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();

    LocalTensor<float> weightTx = coeffTx;
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        weightTx = coeffTmpBuf_.Get<float>(CAL_H_W_BLOCK);
        CoordinatesSelectScalar(coeffTx, weightTx, weightMaskUb, 0.0f);

        auto weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();
        int32_t maskOffset = loopIdx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));
    }

    LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
    int32_t channelAlign = Ceil(calCElems, B32_ALIGN_FACTOR) * B32_ALIGN_FACTOR;
    if constexpr (IsSameType<T, half>::value) {
        channelAlign = Ceil(calCElems, B16_ALIGN_FACTOR) * B16_ALIGN_FACTOR;
    }
    if (channelLast_ == LAYOUT_NHWC) {
        MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal);
    } else {
        MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal);
    }

    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);

    if constexpr (IsSameType<T, half>::value) {  // T: fp16
        LocalTensor<T> yFp16Ub = yFp16Buf_.Get<T>();
        OutTransposeFp16(channelAlign, xLocal, yFp16Ub);
        PipeBarrier<PIPE_V>();
        Cast(outValueUb, yFp16Ub, RoundMode::CAST_NONE, calCElems * TRANSE_REP_STRIDE);
    } else {  // T: fp32
        OutTransposeFp32(channelAlign, xLocal, outValueUb);
    }
    PipeBarrier<PIPE_V>();

    if (paddingMode_ == PADDING_MODE_ZEROS) {
        for (size_t i = 0; i < calCElems; i++) {
            int32_t ubOffset = i * TRANSE_REP_STRIDE;
            Select(outValueUb[ubOffset],
                maskUbTmp,
                outValueUb[ubOffset],
                0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                TRANSE_REP_STRIDE);
        }
        PipeBarrier<PIPE_V>();
    }

    ApplyCoeffTx(calCElems, loopOffset, weightTx, outValueUb, interp1dUb, interp1dIdx);

    if (interp1dIdx % 4 == 3) {
        ApplyCoeffTy(calCElems, loopOffset, coeffTy, interp1dUb);

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        if constexpr (IsSameType<T, half>::value) {
            int64_t gmYOffset = CAL_H_W_BLOCK * inputC_ * blockIDX + loopOffset + cIdx * CHANNEL_BLOCK * CAL_H_W_BLOCK;
            MTE3ForNCHW(gmYOffset, calCElems, CAL_H_W_BLOCK, loopElems, interp1dUb, gmWorkspace_, interp1dIdx);
        } else {
            int64_t gmYOffset = outBaseOffset + loopOffset + cIdx * CHANNEL_BLOCK * gridHW_;
            MTE3ForNCHW(gmYOffset, calCElems, gridHW_, loopElems, interp1dUb, gmY_, interp1dIdx);
        }

        event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V);
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::CopyOutFp16(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    LocalTensor<float> outLocal = xBuf_.AllocTensor<float>();
    LocalTensor<T> outLocalFp16 = yFp16Buf_.AllocTensor<T>();

    // compute ChannelAlign * 512 data per-loop
    int64_t loopTime = Ceil(inputC_, B16_ALIGN_FACTOR);
    int64_t lastC = inputC_ - B16_ALIGN_FACTOR * (loopTime - 1);
    int64_t dataCnt = CAL_H_W_BLOCK * B16_ALIGN_FACTOR;
    int64_t basegmWorkSpaceAddr = blockIDX * CAL_H_W_BLOCK * inputC_;

    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    for (size_t i = 0; i < loopTime - 1; i++) {
        DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + dataCnt * i], dataCnt);

        SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
        Cast(outLocalFp16, outLocal, RoundMode::CAST_NONE, dataCnt);

        SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
        DataCopyExtParams params;
        params.blockCount = B16_ALIGN_FACTOR;
        params.blockLen = calHWElems * sizeof(T);
        params.srcStride = CAL_H_W_BLOCK / B16_ALIGN_FACTOR - Ceil(calHWElems, B16_ALIGN_FACTOR);
        params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
        int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * CAL_H_W_BLOCK +
                            i * B16_ALIGN_FACTOR * outputH_ * outputW_;
        DataCopyPad(gmY_[gmYOffset], outLocalFp16, params);

        SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);

        SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    }

    dataCnt = CAL_H_W_BLOCK * lastC;
    DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + CAL_H_W_BLOCK * B16_ALIGN_FACTOR * (loopTime - 1)], dataCnt);

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    Cast(outLocalFp16, outLocal, RoundMode::CAST_NONE, dataCnt);

    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    DataCopyExtParams params;
    params.blockCount = lastC;
    params.blockLen = calHWElems * sizeof(T);
    params.srcStride = CAL_H_W_BLOCK / B16_ALIGN_FACTOR - Ceil(calHWElems, B16_ALIGN_FACTOR);
    params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
    int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * CAL_H_W_BLOCK +
                        (loopTime - 1) * B16_ALIGN_FACTOR * outputH_ * outputW_;
    DataCopyPad(gmY_[gmYOffset], outLocalFp16, params);
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;
    LocalTensor<float> gridFp32Local = gridFp32Buf_.Get<float>();
    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = calHWElems * 2 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    DataCopyPadExtParams<T> padParamsGrid{false, 0, 0, 0};
    if constexpr (IsSameType<T, half>::value) {  // T: fp16
        LocalTensor<T> gridFp16Local = gridFp16Buf_.Get<T>();
        DataCopyPad(gridFp16Local, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(gridFp32Local, gridFp16Local, RoundMode::CAST_NONE, CAL_H_W_BLOCK * 2);
        PipeBarrier<PIPE_V>();
    } else {  // T: fp32
        DataCopyPad(gridFp32Local, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    LocalTensor<float> inputXYUb = inputXYFPBuf_.Get<float>();
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
    GatherMask(inputXFpLocal, inputXYUb, xPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYUb, yPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();

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

    LocalTensor<float> inputXWFpLocal = inputXFpBuf_.Get<float>(CAL_H_W_BLOCK);  // x_nw
    LocalTensor<float> inputYWFpLocal = inputYFpBuf_.Get<float>(CAL_H_W_BLOCK);  // y_nw
    LocalTensor<float> cubicTx = inputXFpLocal;
    LocalTensor<float> cubicTy = inputYFpLocal;

    // calcu tx and ty
    Floor(inputXWFpLocal, inputXFpLocal, CAL_H_W_BLOCK);
    Floor(inputYWFpLocal, inputYFpLocal, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(cubicTx, inputXFpLocal, inputXWFpLocal, CAL_H_W_BLOCK);
    Sub(cubicTy, inputYFpLocal, inputYWFpLocal, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // calcu coefficients with tx and ty
    LocalTensor<float> coeffTx0 = coeffBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> coeffTx1 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> coeffTx2 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 8);
    LocalTensor<float> coeffTx3 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 12);

    LocalTensor<float> coeffTy0 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 16);
    LocalTensor<float> coeffTy1 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 20);
    LocalTensor<float> coeffTy2 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 24);
    LocalTensor<float> coeffTy3 = coeffBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 28);

    GetCubicUpsampleCoefficients(
        coeffTx0, coeffTx1, coeffTx2, coeffTx3, coeffTy0, coeffTy1, coeffTy2, coeffTy3, cubicTx, cubicTy);

    LocalTensor<float> xneFpLocal = inputXWFpLocal;
    LocalTensor<float> xnwFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> xseFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 8);
    LocalTensor<float> xswFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 12);
    Adds(xnwFpLocal, xneFpLocal, -1.0f, CAL_H_W_BLOCK);
    Adds(xswFpLocal, xneFpLocal, 1.0f, CAL_H_W_BLOCK);
    Adds(xseFpLocal, xneFpLocal, 2.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> yFpLocal = inputYFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<float> interp1dUb = interp1dBuf_.Get<float>();
    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);

    int32_t transLoop = (calHWElems + TRANSE_REP_STRIDE - 1) / TRANSE_REP_STRIDE;
    int32_t loopElems = TRANSE_REP_STRIDE;
    int32_t loopOffset = 0;
    int64_t outBaseOffset = nIdx * outputH_ * outputW_ * inputC_ + hwIdx * CAL_H_W_BLOCK;
    PipeBarrier<PIPE_ALL>();
    for (int32_t loopIdx = 0; loopIdx < transLoop; loopIdx++) {
        if (loopIdx == transLoop - 1) {
            loopElems = calHWElems - TRANSE_REP_STRIDE * (transLoop - 1);
        }
        loopOffset = loopIdx * TRANSE_REP_STRIDE;

        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }

            Adds(yFpLocal, inputYWFpLocal, -1.0f, CAL_H_W_BLOCK);
            PipeBarrier<PIPE_V>();

            ClipCoordinates(xnwFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx0,
                coeffTy0,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                0);
            ClipCoordinates(xneFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx1,
                coeffTy0,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                1);
            ClipCoordinates(xswFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx2,
                coeffTy0,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                2);
            ClipCoordinates(xseFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx3,
                coeffTy0,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                3);

            Adds(yFpLocal, inputYWFpLocal, 0.0f, CAL_H_W_BLOCK);
            PipeBarrier<PIPE_V>();

            ClipCoordinates(xnwFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx0,
                coeffTy1,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                4);
            ClipCoordinates(xneFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx1,
                coeffTy1,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                5);
            ClipCoordinates(xswFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx2,
                coeffTy1,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                6);
            ClipCoordinates(xseFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx3,
                coeffTy1,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                7);

            Adds(yFpLocal, inputYWFpLocal, 1.0f, CAL_H_W_BLOCK);
            PipeBarrier<PIPE_V>();

            ClipCoordinates(xnwFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx0,
                coeffTy2,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                8);
            ClipCoordinates(xneFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx1,
                coeffTy2,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                9);
            ClipCoordinates(xswFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx2,
                coeffTy2,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                10);
            ClipCoordinates(xseFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx3,
                coeffTy2,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                11);

            Adds(yFpLocal, inputYWFpLocal, 2.0f, CAL_H_W_BLOCK);
            PipeBarrier<PIPE_V>();

            ClipCoordinates(xnwFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx0,
                coeffTy3,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                12);
            ClipCoordinates(xneFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx1,
                coeffTy3,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                13);
            ClipCoordinates(xswFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx2,
                coeffTy3,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                14);
            ClipCoordinates(xseFpLocal, yFpLocal, coordinatesLocal, weightMaskUb);
            CubicInterp1d(nIdx,
                outBaseOffset,
                loopIdx,
                loopOffset,
                loopElems,
                coordinatesLocal,
                coeffTx3,
                coeffTy3,
                weightMaskUb,
                cIdx,
                calCElems,
                interp1dUb,
                outValueLocal,
                15);
        }
    }

    if constexpr (IsSameType<T, half>::value) {
        PipeBarrier<PIPE_ALL>();
        CopyOutFp16(nIdx, hwIdx, calHWElems);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void GridSamplerBicubic2D<T>::Process()
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
        int32_t nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = CAL_H_W_BLOCK;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
        }
        PerLoopCompute(nIdx, hwIdx, calHWElems);
    }
}

}  // namespace GridSample
#endif  // GIRD_SAMPLER_BICUBIC_2D

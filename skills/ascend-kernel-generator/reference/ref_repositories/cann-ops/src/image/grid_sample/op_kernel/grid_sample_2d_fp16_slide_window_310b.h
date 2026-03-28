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
 * \file grid_sampler_2d_fp16_slide_window_310b.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_FP16_SLIDE_WINDOW_310B
#define GRID_SAMPLER_2D_FP16_SLIDE_WINDOW_310B

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

struct Mte2ParamFp16 {
    int64_t coordVal_0;
    int64_t xLocation_0;
    int64_t coordVal_1;
    int64_t xLocation_1;

    __aicore__ inline Mte2ParamFp16()
    {}

    __aicore__ inline Mte2ParamFp16(int64_t coordVal_0, int64_t xLocation_0)
        : coordVal_0(coordVal_0), xLocation_0(xLocation_0)
    {
        coordVal_1 = 0;
        xLocation_1 = 0;
    }

    __aicore__ inline Mte2ParamFp16(int64_t coordVal_0, int64_t xLocation_0, int64_t coordVal_1, int64_t xLocation_1)
        : coordVal_0(coordVal_0), xLocation_0(xLocation_0), coordVal_1(coordVal_1), xLocation_1(xLocation_1)
    {}
};

template <typename T>
class GridSampler2DFP16SlideWindow310B {
public:
    __aicore__ inline GridSampler2DFP16SlideWindow310B(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void InitTensor();
    __aicore__ inline void PerLoopCompute(uint32_t nIdx, uint32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign);
    __aicore__ inline void CalculateXYLocal(
        LocalTensor<T> gridLocal, LocalTensor<float> &inputXFpLocal, LocalTensor<float> &inputYFpLocal);
    __aicore__ inline void CalculateWeight(LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal);
    __aicore__ inline void ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub, LocalTensor<float> x1Ub,
        LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<uint8_t> weightMaskUb, uint16_t id);
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
    __aicore__ inline void MTE2ForNCHW(uint32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int16_t loopElems, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWCType1(uint32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int16_t loopElems, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWCType1loop(int32_t loopOffset, LocalTensor<T> xLocal, int64_t base,
        int64_t doubleChannelAlign, int64_t forthChannelAlign, uint64_t wcLength, uint16_t blockLen, int16_t timeStep);
    __aicore__ inline Mte2ParamFp16 GetMte2ParamForType1(int64_t base, int64_t doubleChannelAlign,
        int64_t forthChannelAlign, uint64_t wcLength, int64_t offset, int64_t indexOffset, int16_t index);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);
    __aicore__ inline void calculateEachPointValue(uint32_t nIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueUbSum);
    __aicore__ inline void PointBilinear(uint32_t nIdx, uint32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign);
    __aicore__ inline void PointBilinearLoop(int16_t loop_idx, uint32_t nIdx, uint32_t hwIdx, int32_t calCElems,
        int16_t loopElems, LocalTensor<T> xLocal, LocalTensor<T> outValueFp16Ub, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueUbSum, int64_t outBaseOffset, int32_t trans_loop);
    __aicore__ inline void GetWeightMask();
    __aicore__ inline void SetMaskUbTensor(int16_t loop_idx);
    __aicore__ inline void SelectOutputValueMask(int32_t calCElems, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueUb2, LocalTensor<float> outValueUb3, LocalTensor<float> outValueUb4);
    __aicore__ inline void MTE3ForNCHW(int16_t cIdx, int32_t calCElems, int32_t loopOffset, int16_t loopElems,
        int64_t outBaseOffset, LocalTensor<T> outValueUbSum);
    __aicore__ inline void MTE3ForNCHWAlignmentType1(int32_t calCElems, int16_t loopElems, LocalTensor<T> outValueUbSum,
        int64_t gmYBaseOffset, event_t eventIdVToMte3, int16_t loopElemsAlign);
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

    TBuf<QuePosition::VECCALC> inputGridFp32Buf_;
    TBuf<QuePosition::VECCALC> outValueFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
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
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / 8;
    const int64_t MASK_SIZE = 960;
    const int64_t WEIGHT_MASK_SIZE = 320;
    const int64_t GRID_FP16_OFFSET = CAL_H_W_BLOCK;
    const int64_t OUT_FP16_OFFSET = TRANSE_REP_STRIDE * CHANNEL_BLOCK * 4 * sizeof(T);

    const int64_t OUT_UB_SIZE_4_GENERAL = 65536;
    const int64_t OUT_UB_SIZE_GENERAL = 16384;
    const int64_t X_UB_SIZE_4_GENERAL = 32768;
    const int64_t X_UB_SIZE_4_FP16 = 16384;  // 16k
    const int64_t GRID_UB_SIZE_4_FP16 = 2048;

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

    int64_t maxCoordVal = 0;

    LocalTensor<uint16_t> bufPattern;
    LocalTensor<T> bufTensor;
    LocalTensor<float> nwWeightLocal;
    LocalTensor<float> neWeightLocal;
    LocalTensor<float> swWeightLocal;
    LocalTensor<float> seWeightLocal;
    LocalTensor<int32_t> inputXWIntLocal;
    LocalTensor<int32_t> inputXEIntLocal;
    LocalTensor<int32_t> inputYWIntLocal;
    LocalTensor<int32_t> inputYEIntLocal;
    LocalTensor<float> inputXWFpLocal;
    LocalTensor<float> inputXEFpLocal;
    LocalTensor<float> inputYWFpLocal;
    LocalTensor<float> inputYEFpLocal;
    LocalTensor<uint8_t> weightMaskUb;
    LocalTensor<uint8_t> weightMaskUb2;
    LocalTensor<uint8_t> weightMaskUb3;
    LocalTensor<uint8_t> weightMaskUb4;
    LocalTensor<uint16_t> weightMaskUb5;
    LocalTensor<uint16_t> weightMaskUb6;
    LocalTensor<uint16_t> weightMaskUb7;
    LocalTensor<uint16_t> weightMaskUb8;
    LocalTensor<uint16_t> weightMaskUb9;
    LocalTensor<uint64_t> weightMaskUbTmp;
    LocalTensor<uint64_t> weightMaskUbTmp_3;
    LocalTensor<uint64_t> weightMaskUbTmp_4;
    LocalTensor<uint64_t> weightMaskUbTmp_6;
    LocalTensor<uint64_t> weightMaskUbTmp_8;
    LocalTensor<uint64_t> weightMaskUbTmp_9;

    LocalTensor<uint32_t> maskUb;
    LocalTensor<uint32_t> maskUb3;
    LocalTensor<uint32_t> maskUb4;
    LocalTensor<uint32_t> maskUb6;
    LocalTensor<uint32_t> maskUb8;
    LocalTensor<uint32_t> maskUb9;

    LocalTensor<int32_t> coorUb;

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
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::ParseTilingData(const GridSampleTilingData *tilingData)
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
    maxCoordVal = inputN_ * inputC_ * inputW_ * inputH_;
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer申请初始化 159KB
    pipe.InitBuffer(gridQueue_, 1, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(dupBuf_, 2048);                          // 2KB
    pipe.InitBuffer(dupBuf3_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf4_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf6_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf8_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf9_, 2048);                         // 2KB

    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);             // 32KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(inputXIntBuf_, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(inputYIntBuf_, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(inputXFpBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(inputYFpBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB

    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);     // 8KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);  // 8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);         // 2KB
    pipe.InitBuffer(intTmpBuf2_, Y_UB_SIZE_4_GENERAL);        // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);           // 2KB

    pipe.InitBuffer(outValueBuf_, OUT_UB_SIZE_4_GENERAL);  // 32KB
    pipe.InitBuffer(outValueBuf2_, OUT_UB_SIZE_GENERAL);   // 16KB

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

    pipe.InitBuffer(bufferMaskBuf_, BLOCK_SIZE * 2);          // 32B
    pipe.InitBuffer(bufferBuf_, BLOCK_SIZE * CHANNEL_BLOCK);  // 4K

    InitTensor();
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::InitTensor()
{
    bufTensor = bufferBuf_.Get<T>();
    nwWeightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    neWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    swWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    seWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    inputXWIntLocal = inputXIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    inputXEIntLocal = inputXIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    inputYWIntLocal = inputYIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    inputYEIntLocal = inputYIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    inputXWFpLocal = inputXFpBuf_.Get<float>(CAL_H_W_BLOCK);
    inputXEFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    inputYWFpLocal = inputYFpBuf_.Get<float>(CAL_H_W_BLOCK);
    inputYEFpLocal = inputYFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    coorUb = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb2 = weightMaskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb3 = weightMaskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb4 = weightMaskBuf4_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb5 = weightMaskBuf5_.Get<uint16_t>(MASK_UB_SIZE);
    weightMaskUb6 = weightMaskBuf6_.Get<uint16_t>(MASK_UB_SIZE);
    weightMaskUb7 = weightMaskBuf7_.Get<uint16_t>(MASK_UB_SIZE);
    weightMaskUb8 = weightMaskBuf8_.Get<uint16_t>(MASK_UB_SIZE);
    weightMaskUb9 = weightMaskBuf9_.Get<uint16_t>(MASK_UB_SIZE);
    weightMaskUbTmp = weightMaskUb7.ReinterpretCast<uint64_t>();
    weightMaskUbTmp_3 = weightMaskUb5.ReinterpretCast<uint64_t>();
    weightMaskUbTmp_4 = weightMaskUb4.ReinterpretCast<uint64_t>();
    weightMaskUbTmp_6 = weightMaskUb6.ReinterpretCast<uint64_t>();
    weightMaskUbTmp_8 = weightMaskUb8.ReinterpretCast<uint64_t>();
    weightMaskUbTmp_9 = weightMaskUb9.ReinterpretCast<uint64_t>();

    maskUb = maskBuf_.Get<uint32_t>(MASK_UB_SIZE);
    maskUb3 = maskBuf3_.Get<uint32_t>(MASK_UB_SIZE);
    maskUb4 = maskBuf4_.Get<uint32_t>(MASK_UB_SIZE);
    maskUb6 = maskBuf6_.Get<uint32_t>(MASK_UB_SIZE);
    maskUb8 = maskBuf8_.Get<uint32_t>(MASK_UB_SIZE);
    maskUb9 = maskBuf9_.Get<uint32_t>(MASK_UB_SIZE);

    int32_t trans_loop = Ceil(lastLoopHWAlign_, TRANSE_REP_STRIDE);
    int16_t loopElems = lastLoopHW_ - TRANSE_REP_STRIDE * (trans_loop - 1);
    int16_t loopElemsAlign = Ceil(loopElems, BLOCK_NUM) * BLOCK_NUM;

    bufPattern = bufferMaskBuf_.Get<uint16_t>();
    auto loopElemslength = loopElemsAlign - loopElems;
    auto offset1 = (1 << BLOCK_NUM) - (1 << (BLOCK_NUM - loopElemslength));
    auto offset2 = (1 << (BLOCK_NUM - loopElemslength)) - 1;
    bufPattern.SetValue(0, offset1);
    bufPattern.SetValue(1, offset2);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::ComputeWeightSub(LocalTensor<float> w1Ub,
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
 * @param {LocalTensor<uint8_t>} out wMaskUb
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::ClipCoordinates(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<uint8_t> wMaskUb,
    uint16_t id)
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
        Adds(inputXIntTmpUb, iXIntUb, 0, CAL_H_W_BLOCK);
        Adds(inputYIntTmpUb, iYIntUb, 0, CAL_H_W_BLOCK);

        // 重计算坐标，使坐标不超过边界
        CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ - 1));
        CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));

        Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);

        Add(coorUb, coorUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    }
}

/**
 * @description: 原坐标越界时计算新坐标
 * @param {LocalTensor<float>} X坐标
 * @param {LocalTensor<float>} Y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
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
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::CoordinatesFrameRange(
    LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_H_W_BLOCK);

    Maxs(iIntUb, iIntUb, 0, CAL_H_W_BLOCK);
}

/**
 * @description: 取出合法坐标点：maskXUb，maskYUb
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
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

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // 合并上面的两个结果，得到最终合法点
    And(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, MASK_UB_SIZE);
    And(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, MASK_UB_SIZE);

    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, MASK_UB_SIZE);

    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
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
    Select(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::CoordinatesSelectTensor(
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
}

/**
 * @description: PaddingMode：Border
 * @param {LocalTensor<float>} x坐标
 * @param {LocalTensor<float>} y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::BorderClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);

    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);

    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
}

/**
 * @description: PaddingMode：Reflection
 * @param {LocalTensor<float>} x坐标
 * @param {LocalTensor<float>} y坐标
 * @return {*}
 */
template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::ReflectClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    LocalTensor<float> extraFpUb = extraBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> fmodFpUb = modBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    int64_t twiceLow = (alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowY = REFLECT_RATIO * (inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (inputW_ - 1);
    if (alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowY = REFLECT_RATIO * inputH_ - 1;
        twiceLowX = REFLECT_RATIO * inputW_ - 1;
    }
    ReflectCoordinatesGeneral(iYFpUb, iYFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowY);

    ReflectCoordinatesGeneral(iXFpUb, iXFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowX);

    LocalTensor<float> tmpUb = inputXYFPBuf_.Get<float>();
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);

    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);

    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);

    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);

    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);

    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);

    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);

    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);

    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
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

    Abs(coorSubUb, coorSubUb, CAL_H_W_BLOCK);

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_H_W_BLOCK);

    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);

    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);

    Muls(extraFpUb, extraFpUb, spanS, CAL_H_W_BLOCK);

    Sub(extraFpUb, coorSubUb, extraFpUb, CAL_H_W_BLOCK);

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_H_W_BLOCK);

    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);

    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);

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

    Adds(out2, out2, spanS, CAL_H_W_BLOCK);

    Adds(out2, out2, minS, CAL_H_W_BLOCK);

    Muls(mods, coorSubUb, static_cast<float>(1 / 2.0), CAL_H_W_BLOCK);

    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);

    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);

    Muls(mods, mods, 2.0f, CAL_H_W_BLOCK);

    Sub(mods, coorSubUb, mods, CAL_H_W_BLOCK);

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, CAL_H_W_BLOCK);

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
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::MTE2ForNCHW(uint32_t nIdx, int16_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int16_t loopElems, LocalTensor<T> xLocal)
{
    for (int16_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation =
            (int64_t)nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int16_t cIter = 0; cIter < channelAlign; cIter++) {
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

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::MTE2ForNHWCType1(uint32_t nIdx, int16_t cIdx,
    int32_t calCElems, int32_t channelAlign, int32_t loopOffset, int16_t loopElems,

    LocalTensor<T> xLocal)
{
    int64_t base = (int64_t)nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;

    auto timeStep = loopElems / 8;

    int32_t calCElemsAlign = Ceil(calCElems, BLOCK_NUM) * BLOCK_NUM;
    int64_t doubleChannelAlign = static_cast<int64_t>(channelAlign) * 2;
    int64_t forthChannelAlign = static_cast<int64_t>(channelAlign) * 4;
    uint16_t blockLen = calCElemsAlign * 2;
    uint64_t wcLength = inputW_ * inputC_;

    MTE2ForNHWCType1loop(loopOffset, xLocal, base, doubleChannelAlign, forthChannelAlign, wcLength, blockLen, timeStep);

    for (int16_t i = loopElems / 8 * 8; i < loopElems; i++) {
        int64_t coordVal_0 = base + coorUb.GetValue(loopOffset + i);
        int64_t coordVal_0_1 = coordVal_0 + wcLength;
        if (coordVal_0_1 >= maxCoordVal) {
            coordVal_0_1 = coordVal_0;
        }
        DataCopy(xLocal[i * forthChannelAlign], gmX_[coordVal_0], blockLen);
        DataCopy(xLocal[i * forthChannelAlign + doubleChannelAlign], gmX_[coordVal_0_1], blockLen);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::MTE2ForNHWCType1loop(int32_t loopOffset,
    LocalTensor<T> xLocal, int64_t base, int64_t doubleChannelAlign, int64_t forthChannelAlign, uint64_t wcLength,
    uint16_t blockLen, int16_t timeStep)
{
    // 这边为了不打断流水，提高性能
    for (int16_t i = 0; i < timeStep; i++) {
        int64_t offset = loopOffset + i * 8;
        int64_t indexOffset = i * 8;
        Mte2ParamFp16 mte2Param0 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 0);
        Mte2ParamFp16 mte2Param1 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 1);
        Mte2ParamFp16 mte2Param2 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 2);
        Mte2ParamFp16 mte2Param3 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 3);
        Mte2ParamFp16 mte2Param4 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 4);
        Mte2ParamFp16 mte2Param5 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 5);
        Mte2ParamFp16 mte2Param6 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 6);
        Mte2ParamFp16 mte2Param7 =
            GetMte2ParamForType1(base, doubleChannelAlign, forthChannelAlign, wcLength, offset, indexOffset, 7);

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
}

template <typename T>
__aicore__ inline Mte2ParamFp16 GridSampler2DFP16SlideWindow310B<T>::GetMte2ParamForType1(int64_t base,
    int64_t doubleChannelAlign, int64_t forthChannelAlign, uint64_t wcLength, int64_t offset, int64_t indexOffset,
    int16_t index)
{
    int64_t coordVal_0 = base + coorUb.GetValue(offset + index);
    int64_t xLocation_0 = (indexOffset + index) * forthChannelAlign;
    int64_t coordVal_0_1 = coordVal_0 + wcLength;
    if (coordVal_0_1 >= maxCoordVal) {
        coordVal_0_1 = coordVal_0;
    }

    int64_t xLocation_0_1 = xLocation_0 + doubleChannelAlign;
    Mte2ParamFp16 mte2Param = Mte2ParamFp16(coordVal_0, xLocation_0, coordVal_0_1, xLocation_0_1);
    return mte2Param;
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::OutTranspose(
    int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb)
{
    uint64_t dstList[16];
    uint64_t srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = channelAlign * 4 / BLOCK_NUM;
    transDataParams.dstRepStride = TRANSE_REP_STRIDE;
    transDataParams.srcRepStride = 1;
    for (int32_t j = 0; j < 8; j++) {
        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = (uint64_t)(xLocal[i * channelAlign * 4 + j * 16 * channelAlign * 4].GetPhyAddr());
        }

        for (int32_t i = 0; i < 16; i++) {
            dstList[i] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + j * 16].GetPhyAddr());
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::calculateEachPointValue(uint32_t nIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, LocalTensor<float> weightUb, LocalTensor<float> outValueUb,
    LocalTensor<float> outValueUbSum)
{
    for (int16_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
        int32_t outOffset = i * B32_MASK;
        int32_t weightOffset = loopOffset + i * B32_MASK;
        Mul(outValueUb[outOffset],
            outValueUb[outOffset],
            weightUb[weightOffset],
            B32_MASK,
            calCElems,
            {1, 1, 1, 16, 16, 0});
    }

    Add(outValueUbSum, outValueUbSum, outValueUb, TRANSE_REP_STRIDE * channelAlign);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::PointBilinear(
    uint32_t nIdx, uint32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign)
{
    Muls(coorUb, coorUb, (int32_t)inputC_, CAL_H_W_BLOCK);

    if (paddingMode_ == PADDING_MODE_ZEROS) {
        // 非法的点的weight置0
        CoordinatesSelectScalar(nwWeightLocal, nwWeightLocal, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(neWeightLocal, neWeightLocal, weightMaskUb2, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(swWeightLocal, swWeightLocal, weightMaskUb3, 0.0f, CAL_H_W_BLOCK);
        CoordinatesSelectScalar(seWeightLocal, seWeightLocal, weightMaskUb4, 0.0f, CAL_H_W_BLOCK);
    }
    LocalTensor<float> outValueUb = outValueBuf_.Get<float>();
    LocalTensor<T> outValueFp16Ub =
        outValueBuf_.GetWithOffset<T>(perLoopChannel_ * (TRANSE_REP_STRIDE), OUT_FP16_OFFSET);
    LocalTensor<float> outValueUbSum = outValueBuf2_.Get<float>();
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半  32

    GetWeightMask();

    int32_t trans_loop = Ceil(calHWElemsAlign, TRANSE_REP_STRIDE);
    int16_t loopElems = TRANSE_REP_STRIDE;
    int64_t outBaseOffset = (int64_t)nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK;
    int32_t ubOffset = 0;

    LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();

    auto dupUbU32_4 = dupBuf4_.Get<uint32_t>();
    auto dupUbU32_6 = dupBuf6_.Get<uint32_t>();
    auto dupUbU32_8 = dupBuf8_.Get<uint32_t>();
    auto dupUbU32 = dupBuf_.Get<uint32_t>();
    auto dupUbU32_3 = dupBuf3_.Get<uint32_t>();
    auto dupUbU32_9 = dupBuf9_.Get<uint32_t>();

    // 按vmask(128)分块，循环处理
    for (int16_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            loopElems = calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        PointBilinearLoop(loop_idx,
            nIdx,
            hwIdx,
            CHANNEL_BLOCK,
            loopElems,
            xLocal,
            outValueFp16Ub,
            outValueUb,
            outValueUbSum,
            outBaseOffset,
            trans_loop);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::GetWeightMask()
{
    auto weightMaskUbTmp1 = weightMaskUb.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp2 = weightMaskUb2.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp3 = weightMaskUb3.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp4 = weightMaskUb4.ReinterpretCast<uint16_t>();
    And(weightMaskUb5, weightMaskUbTmp1, weightMaskUbTmp3, MASK_UB_SIZE);
    And(weightMaskUb7, weightMaskUbTmp1, weightMaskUbTmp2, MASK_UB_SIZE);
    And(weightMaskUb6, weightMaskUbTmp3, weightMaskUbTmp4, MASK_UB_SIZE);
    And(weightMaskUb9, weightMaskUbTmp2, weightMaskUbTmp4, MASK_UB_SIZE);
    Or(weightMaskUb8, weightMaskUbTmp2, weightMaskUbTmp3, MASK_UB_SIZE);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::SetMaskUbTensor(int16_t loop_idx)
{
    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();
    auto maskUbTmp3 = maskUb3.ReinterpretCast<uint64_t>();
    auto maskUbTmp4 = maskUb4.ReinterpretCast<uint64_t>();
    auto maskUbTmp6 = maskUb6.ReinterpretCast<uint64_t>();
    auto maskUbTmp8 = maskUb8.ReinterpretCast<uint64_t>();
    auto maskUbTmp9 = maskUb9.ReinterpretCast<uint64_t>();

    int32_t maskOffset = loop_idx * 2;
    maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
    maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));
    maskUbTmp.SetValue(2, weightMaskUbTmp.GetValue(maskOffset));
    maskUbTmp.SetValue(3, weightMaskUbTmp.GetValue(maskOffset + 1));

    maskUbTmp3.SetValue(0, weightMaskUbTmp_3.GetValue(maskOffset));
    maskUbTmp3.SetValue(1, weightMaskUbTmp_3.GetValue(maskOffset + 1));
    maskUbTmp3.SetValue(2, weightMaskUbTmp_3.GetValue(maskOffset));
    maskUbTmp3.SetValue(3, weightMaskUbTmp_3.GetValue(maskOffset + 1));

    maskUbTmp4.SetValue(0, weightMaskUbTmp_4.GetValue(maskOffset));
    maskUbTmp4.SetValue(1, weightMaskUbTmp_4.GetValue(maskOffset + 1));
    maskUbTmp4.SetValue(2, weightMaskUbTmp_4.GetValue(maskOffset));
    maskUbTmp4.SetValue(3, weightMaskUbTmp_4.GetValue(maskOffset + 1));

    maskUbTmp6.SetValue(0, weightMaskUbTmp_6.GetValue(maskOffset));
    maskUbTmp6.SetValue(1, weightMaskUbTmp_6.GetValue(maskOffset + 1));
    maskUbTmp6.SetValue(2, weightMaskUbTmp_6.GetValue(maskOffset));
    maskUbTmp6.SetValue(3, weightMaskUbTmp_6.GetValue(maskOffset + 1));

    maskUbTmp8.SetValue(0, weightMaskUbTmp_8.GetValue(maskOffset));
    maskUbTmp8.SetValue(1, weightMaskUbTmp_8.GetValue(maskOffset + 1));
    maskUbTmp8.SetValue(2, weightMaskUbTmp_8.GetValue(maskOffset));
    maskUbTmp8.SetValue(3, weightMaskUbTmp_8.GetValue(maskOffset + 1));

    maskUbTmp9.SetValue(0, weightMaskUbTmp_9.GetValue(maskOffset));
    maskUbTmp9.SetValue(1, weightMaskUbTmp_9.GetValue(maskOffset + 1));
    maskUbTmp9.SetValue(2, weightMaskUbTmp_9.GetValue(maskOffset));
    maskUbTmp9.SetValue(3, weightMaskUbTmp_9.GetValue(maskOffset + 1));
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::PointBilinearLoop(int16_t loop_idx, uint32_t nIdx,
    uint32_t hwIdx, int32_t calCElems, int16_t loopElems, LocalTensor<T> xLocal, LocalTensor<T> outValueFp16Ub,
    LocalTensor<float> outValueUb, LocalTensor<float> outValueUbSum, int64_t outBaseOffset, int32_t trans_loop)
{
    int32_t loopOffset = loop_idx * TRANSE_REP_STRIDE;
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetMaskUbTensor(loop_idx);
    for (int16_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
        int32_t calCElems = perLoopChannel_;
        if (cIdx == channelLoop_ - 1) {
            calCElems = lastLoopChannel_;
        }
        int32_t channelAlign = Ceil(calCElems, BLOCK_NUM) * BLOCK_NUM;
        if (channelLast_ == LAYOUT_NHWC && dataCopyType != DATA_COPY_TYPE_1) {
            // FP16，repeatTime > 1时，如果后面的地址也超过GM地址，会报AICore Error， FP32就不会。
            MTE2ForNHWCType1(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, xLocal);
        } else if (channelLast_ == LAYOUT_NHWC && dataCopyType == DATA_COPY_TYPE_1) {
            MTE2ForNHWCType1(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, xLocal);
        } else {
            MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, xLocal);
        }

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        OutTranspose(channelAlign, xLocal, outValueFp16Ub);
        Cast(outValueUb, outValueFp16Ub, RoundMode::CAST_NONE, calCElems * TRANSE_REP_STRIDE * 4);

        LocalTensor<float> outValueUb2 = outValueUb[channelAlign * (TRANSE_REP_STRIDE)];
        LocalTensor<float> outValueUb3 = outValueUb2[channelAlign * (TRANSE_REP_STRIDE)];
        LocalTensor<float> outValueUb4 = outValueUb3[channelAlign * (TRANSE_REP_STRIDE)];

        SelectOutputValueMask(calCElems, outValueUb, outValueUb2, outValueUb3, outValueUb4);

        Duplicate(outValueUbSum, (float)0.0, outValueUbSum.GetSize());

        calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, nwWeightLocal, outValueUb, outValueUbSum);
        calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, neWeightLocal, outValueUb2, outValueUbSum);
        calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, swWeightLocal, outValueUb3, outValueUbSum);
        calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, seWeightLocal, outValueUb4, outValueUbSum);
        auto outValueUbSumFp16 = outValueUbSum.ReinterpretCast<T>();
        Cast(outValueUbSumFp16, outValueUbSum, RoundMode::CAST_NONE, TRANSE_REP_STRIDE * CHANNEL_BLOCK);
        MTE3ForNCHW(cIdx, calCElems, loopOffset, loopElems, outBaseOffset, outValueUbSumFp16);
        SetFlag<HardEvent::MTE3_V>(eventMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::SelectOutputValueMask(int32_t calCElems,
    LocalTensor<float> outValueUb, LocalTensor<float> outValueUb2, LocalTensor<float> outValueUb3,
    LocalTensor<float> outValueUb4)
{
    BinaryRepeatParams repParams{1, 1, 1, 8, 8, 8};

    auto dupUbU32_4 = dupBuf4_.Get<uint32_t>();
    auto dupUbU32_6 = dupBuf6_.Get<uint32_t>();
    auto dupUbU32_8 = dupBuf8_.Get<uint32_t>();
    auto dupUbU32 = dupBuf_.Get<uint32_t>();
    auto dupUbU32_3 = dupBuf3_.Get<uint32_t>();
    auto dupUbU32_9 = dupBuf9_.Get<uint32_t>();

    Copy(dupUbU32_9, maskUb9, 8, 16, {1, 1, 1, 0});
    Select(outValueUb4,
        dupUbU32_9,
        outValueUb4,
        outValueUb2,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        64,
        calCElems * (TRANSE_REP_STRIDE / 64),
        repParams);

    Copy(dupUbU32_6, maskUb6, 8, 16, {1, 1, 1, 0});
    Select(outValueUb4,
        dupUbU32_6,
        outValueUb4,
        outValueUb3,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        64,
        calCElems * (TRANSE_REP_STRIDE / 64),
        repParams);

    Copy(dupUbU32_8, maskUb8, 8, 16, {1, 1, 1, 0});
    Select(outValueUb4,
        dupUbU32_8,
        outValueUb4,
        outValueUb,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        64,
        calCElems * (TRANSE_REP_STRIDE / 64),
        repParams);
    Copy(dupUbU32_4, maskUb4, 8, 16, {1, 1, 1, 0});

    Copy(dupUbU32, maskUb, 8, 16, {1, 1, 1, 0});
    Copy(dupUbU32_3, maskUb3, 8, 16, {1, 1, 1, 0});

    Select(outValueUb2,
        dupUbU32,
        outValueUb2,
        outValueUb,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        64,
        calCElems * (TRANSE_REP_STRIDE / 64),
        repParams);
    Select(outValueUb3,
        dupUbU32_3,
        outValueUb3,
        outValueUb,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        64,
        calCElems * (TRANSE_REP_STRIDE / 64),
        repParams);
    Select(outValueUb4,
        dupUbU32_4,
        outValueUb4,
        (float)0.0,
        SELMODE::VSEL_TENSOR_SCALAR_MODE,
        64,
        calCElems * (TRANSE_REP_STRIDE / 64),
        repParams);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::MTE3ForNCHW(int16_t cIdx, int32_t calCElems,
    int32_t loopOffset, int16_t loopElems, int64_t outBaseOffset, LocalTensor<T> outValueUbSum)
{
    int64_t gmYBaseOffset = outBaseOffset + loopOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    int16_t loopElemsAlign = Ceil(loopElems, BLOCK_NUM) * BLOCK_NUM;
    uint32_t mask = 32;
    uint64_t rsvdCnt = 0;
    bufPattern = bufferMaskBuf_.Get<uint16_t>();

    // 这个提出来时间优化略等于无
    if (alignmentType_ != ALIGNMENT_TYPE_1) {
        if (loopElemsAlign != loopElems) {
            for (int16_t i = 0; i < calCElems; i++) {
                int64_t outputOffset = i * TRANSE_REP_STRIDE;
                GatherMask(bufTensor[i * BLOCK_NUM],
                    outValueUbSum[outputOffset + loopElemsAlign - 2 * BLOCK_NUM],
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
        MTE3ForNCHWAlignmentType1(calCElems, loopElems, outValueUbSum, gmYBaseOffset, eventIdVToMte3, loopElemsAlign);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::MTE3ForNCHWAlignmentType1(int32_t calCElems,
    int16_t loopElems, LocalTensor<T> outValueUbSum, int64_t gmYBaseOffset, event_t eventIdVToMte3,
    int16_t loopElemsAlign)
{
    if (loopElemsAlign != loopElems) {
        for (int16_t i = 0; i < calCElems; i++) {
            int64_t outputOffset = i * TRANSE_REP_STRIDE;
            for (int16_t j = 0; j < loopElemsAlign - loopElems; j++) {
                outValueUbSum.SetValue(outputOffset + loopElems + j, 0.0f);
            }
        }
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    for (int16_t i = 0; i < calCElems; i++) {
        int64_t gmYOffset = gmYBaseOffset + i * gridHW_;
        int64_t outputOffset = i * TRANSE_REP_STRIDE;
        if (loopElemsAlign == loopElems) {
            DataCopy(gmY_[gmYOffset], outValueUbSum[outputOffset], loopElems);
        }

        if (loopElemsAlign != loopElems) {
            SetAtomicAdd<half>();
            DataCopy(gmY_[gmYOffset], outValueUbSum[outputOffset], loopElemsAlign);
            SetAtomicNone();
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::PerLoopCompute(
    uint32_t nIdx, uint32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;

    LocalTensor<T> gridLocal = gridQueue_.AllocTensor<T>();
    DataCopy(gridLocal[GRID_FP16_OFFSET], gmGrid_[gridGmOffset], calHWElemsAlign * 2);
    gridQueue_.EnQue(gridLocal);

    LocalTensor<float> inputXFpLocal;
    LocalTensor<float> inputYFpLocal;
    CalculateXYLocal(gridLocal, inputXFpLocal, inputYFpLocal);

    CalculateWeight(inputXFpLocal, inputYFpLocal);

    // 划窗条件不满足，走兜底分支
    ClipCoordinates(inputXWFpLocal, inputYWFpLocal, inputXWIntLocal, inputYWIntLocal, weightMaskUb, 1);
    ClipCoordinates(inputXEFpLocal, inputYWFpLocal, inputXEIntLocal, inputYWIntLocal, weightMaskUb2, 2);
    ClipCoordinates(inputXWFpLocal, inputYEFpLocal, inputXWIntLocal, inputYEIntLocal, weightMaskUb3, 3);
    ClipCoordinates(inputXEFpLocal, inputYEFpLocal, inputXEIntLocal, inputYEIntLocal, weightMaskUb4, 4);

    PointBilinear(nIdx, hwIdx, calHWElems, calHWElemsAlign);

    gridQueue_.FreeTensor(gridLocal);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::CalculateXYLocal(
    LocalTensor<T> gridLocal, LocalTensor<float> &inputXFpLocal, LocalTensor<float> &inputYFpLocal)
{
    gridQueue_.DeQue();
    LocalTensor<float> gridFp32Local = gridLocal.template ReinterpretCast<float>();
    Cast(gridFp32Local, gridLocal[GRID_FP16_OFFSET], RoundMode::CAST_NONE, CAL_H_W_BLOCK * 2);

    LocalTensor<float> inputXYUb = inputXYFPBuf_.Get<float>();
    // 加1后，grid的datarange从-1~1到0~2
    Adds(inputXYUb, gridFp32Local, (float)1.0, CAL_H_W_BLOCK * 2);

    uint32_t mask = 64;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;

    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;

    uint16_t repeatTime = CAL_H_W_BLOCK * 2 / 64;

    inputXFpLocal = gridFp32Local;
    inputYFpLocal = gridFp32Local[CAL_H_W_BLOCK];
    // 分别取x和y
    GatherMask(
        inputXFpLocal, inputXYUb, xPattern, true, mask, {1, repeatTime, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(
        inputYFpLocal, inputXYUb, yPattern, true, mask, {1, repeatTime, src0RepeatStride, src1RepeatStride}, rsvdCnt);

    // 不同alignCorners_的unnormlize处理
    if (alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (inputW_ - (float)1.0)), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * (inputH_ - (float)1.0)), CAL_H_W_BLOCK);
    } else {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * inputW_), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * inputH_), CAL_H_W_BLOCK);

        Adds(inputXFpLocal, inputXFpLocal, (float)(-0.5), CAL_H_W_BLOCK * 2);
    }

    // 处理越界坐标
    Clip(inputXFpLocal, inputYFpLocal);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::CalculateWeight(
    LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal)
{
    Cast(inputXWIntLocal, inputXFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    Cast(inputYWIntLocal, inputYFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);

    Cast(inputXWFpLocal, inputXWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputYWFpLocal, inputYWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    // 分别计算左上，右上，左下，右下的坐标
    Adds(inputXEIntLocal, inputXWIntLocal, 1, CAL_H_W_BLOCK);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, CAL_H_W_BLOCK);
    Adds(inputXEFpLocal, inputXWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    Adds(inputYEFpLocal, inputYWFpLocal, (float)1.0, CAL_H_W_BLOCK);

    LocalTensor<float> weightTmpLocal = weightTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> weightTmp1Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> weightTmp2Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> weightTmp3Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    Sub(weightTmpLocal, inputXEFpLocal, inputXFpLocal, CAL_H_W_BLOCK);
    Sub(weightTmp1Local, inputXFpLocal, inputXWFpLocal, CAL_H_W_BLOCK);
    Sub(weightTmp2Local, inputYEFpLocal, inputYFpLocal, CAL_H_W_BLOCK);
    Sub(weightTmp3Local, inputYFpLocal, inputYWFpLocal, CAL_H_W_BLOCK);

    Mul(nwWeightLocal, weightTmpLocal, weightTmp2Local, CAL_H_W_BLOCK);
    Mul(neWeightLocal, weightTmp1Local, weightTmp2Local, CAL_H_W_BLOCK);
    Mul(swWeightLocal, weightTmpLocal, weightTmp3Local, CAL_H_W_BLOCK);
    Mul(seWeightLocal, weightTmp1Local, weightTmp3Local, CAL_H_W_BLOCK);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::ResetGMToZero()
{
    LocalTensor<T> outValueLocal = outValueBuf2_.Get<T>();
    Duplicate(outValueLocal, (T)0.0, outValueLocal.GetSize());
    uint32_t nIdx = 0;
    uint32_t hwIdx = 0;
    uint32_t preLoopNum = blockIDX * preCoreLoop_;  // 每个核开始的block数

    int64_t loopSize = preCoreLoop_;  // 要处理的block数量
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    for (int64_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;   // N维的index
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;  // h、w在block中位置
        if (hwIdx == preNUbLoop_ - 1) {
            for (int64_t cIdx = 0; cIdx < inputC_; cIdx++) {
                int64_t gmYBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK + cIdx * gridHW_;
                DataCopy(gmY_[gmYBaseOffset], outValueLocal, lastLoopHWAlign_);
            }
        }
    }
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventMte3V);
}

template <typename T>
__aicore__ inline void GridSampler2DFP16SlideWindow310B<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    uint32_t nIdx = 0;
    uint32_t hwIdx = 0;
    uint32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calHWElems = 0;
    int32_t calHWElemsAlign = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    if (gridHW_ < BLOCK_NUM || alignmentType_ == ALIGNMENT_TYPE_1) {
        ResetGMToZero();
    }
    for (int64_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
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
#endif  // GRID_SAMPLER_2D_FP16_SLIDE_WINDOW_310B
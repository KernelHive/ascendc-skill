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
 * \file grid_sampler_3d_common.h
 * \brief
 */
#ifndef GIRD_SAMPLER_3D_COMMON
#define GIRD_SAMPLER_3D_COMMON

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grid_sample_common.h"

namespace GridSample {

using namespace AscendC;

struct InputTensorStruct {
    LocalTensor<float> iXFpUb;
    LocalTensor<float> iYFpUb;
    LocalTensor<float> iZFpUb;
    LocalTensor<int32_t> iXIntUb;
    LocalTensor<int32_t> iYIntUb;
    LocalTensor<int32_t> iZIntUb;

    __aicore__ inline InputTensorStruct()
    {}

    __aicore__ inline InputTensorStruct(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> iZIntUb)
        : iXFpUb(iXFpUb), iYFpUb(iYFpUb), iZFpUb(iZFpUb), iXIntUb(iXIntUb), iYIntUb(iYIntUb), iZIntUb(iZIntUb)
    {}
};

struct ProcessParam {
    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t calDHWElems = 0;

    __aicore__ inline ProcessParam()
    {}
};

struct PointParam {
    int32_t loopElems = 0;
    int32_t loopOffset = 0;
    int64_t outBaseOffset = 0;
    int32_t maskOffset = 0;
    int32_t cIdx = 0;
    int32_t calCElems = 0;
    int32_t channelAlign = 0;
};

struct GridSampleCommonParam {
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
    int64_t lastLoopDHW_ = 0;
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;
};

struct IndexBuffer {
    TBuf<TPosition::VECCALC> intTmpBuf_;
    TBuf<TPosition::VECCALC> coorTmpBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> weightMaskBuf_;
    TBuf<TPosition::VECCALC> inputXYZFPBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
};

__aicore__ inline void initBufTensor(TBuf<QuePosition::VECCALC> bufferMaskXBuf_,
    TBuf<QuePosition::VECCALC> bufferMaskYBuf_, TBuf<QuePosition::VECCALC> bufferMaskZBuf_)
{
    uint32_t gatherMask1 = 0b01001001001001001001001001001001;
    uint32_t gatherMask2 = 0b10010010010010010010010010010010;
    uint32_t gatherMask3 = 0b00100100100100100100100100100100;
    LocalTensor<uint32_t> bufXPattern = bufferMaskXBuf_.Get<uint32_t>();
    bufXPattern.SetValue(NUM_0, gatherMask1);
    bufXPattern.SetValue(NUM_1, gatherMask2);
    bufXPattern.SetValue(NUM_2, gatherMask3);
    bufXPattern.SetValue(NUM_3, gatherMask1);
    bufXPattern.SetValue(NUM_4, gatherMask2);
    bufXPattern.SetValue(NUM_5, gatherMask3);

    LocalTensor<uint32_t> bufYPattern = bufferMaskYBuf_.Get<uint32_t>();
    bufYPattern.SetValue(NUM_0, gatherMask2);
    bufYPattern.SetValue(NUM_1, gatherMask3);
    bufYPattern.SetValue(NUM_2, gatherMask1);
    bufYPattern.SetValue(NUM_3, gatherMask2);
    bufYPattern.SetValue(NUM_4, gatherMask3);
    bufYPattern.SetValue(NUM_5, gatherMask1);

    LocalTensor<uint32_t> bufZPattern = bufferMaskZBuf_.Get<uint32_t>();
    bufZPattern.SetValue(NUM_0, gatherMask3);
    bufZPattern.SetValue(NUM_1, gatherMask1);
    bufZPattern.SetValue(NUM_2, gatherMask2);
    bufZPattern.SetValue(NUM_3, gatherMask3);
    bufZPattern.SetValue(NUM_4, gatherMask1);
    bufZPattern.SetValue(NUM_5, gatherMask2);
}

__aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData, GridSampleCommonParam &commonParam)
{
    commonParam.coreNum_ = tilingData->coreNumVar;
    commonParam.inputN_ = tilingData->inN;
    commonParam.inputC_ = tilingData->inC;
    commonParam.inputD_ = tilingData->inD;
    commonParam.inputH_ = tilingData->inH;
    commonParam.inputW_ = tilingData->inW;
    commonParam.outputD_ = tilingData->outD;
    commonParam.outputH_ = tilingData->outH;
    commonParam.outputW_ = tilingData->outW;
    commonParam.interpolationMode_ = tilingData->interpolationMode;
    commonParam.paddingMode_ = tilingData->paddingMode;
    commonParam.alignCorners_ = tilingData->alignCorners;
    commonParam.channelLast_ = tilingData->channelLast;
    commonParam.needCoreNum_ = tilingData->needCoreNum;
    commonParam.gridDHW_ = commonParam.outputD_ * commonParam.outputH_ * commonParam.outputW_;
    commonParam.preNUbLoop_ = (commonParam.gridDHW_ + CAL_D_H_W_BLOCK - 1) / CAL_D_H_W_BLOCK;
    commonParam.lastLoopDHW_ = commonParam.gridDHW_ - CAL_D_H_W_BLOCK * (commonParam.preNUbLoop_ - 1);
    commonParam.totalUbLoop_ = commonParam.preNUbLoop_ * commonParam.inputN_;
    commonParam.preCoreLoop_ = (commonParam.totalUbLoop_ + commonParam.needCoreNum_ - 1) / commonParam.needCoreNum_;
    commonParam.needCoreNum_ = (commonParam.totalUbLoop_ + commonParam.preCoreLoop_ - 1) / commonParam.preCoreLoop_;
    commonParam.lastCoreLoop_ = commonParam.totalUbLoop_ - commonParam.preCoreLoop_ * (commonParam.needCoreNum_ - 1);

    commonParam.channelLoop_ = (commonParam.inputC_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    commonParam.perLoopChannel_ = CHANNEL_BLOCK;
    commonParam.lastLoopChannel_ = commonParam.inputC_ - commonParam.perLoopChannel_ * (commonParam.channelLoop_ - 1);
}

__aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iIntUb, iIntUb, 0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CoordinatesGetMaskWithRange(InputTensorStruct inputTensorStruct, LocalTensor<uint8_t> maskXUb,
    LocalTensor<uint8_t> maskUb, GridSampleCommonParam commonParam)
{
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskZUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE * NUM_2];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * NUM_3];
    LocalTensor<uint8_t> maskTmpZUb = maskUb[MASK_UB_SIZE * NUM_4];

    CompareScalar(maskTmpXUb, inputTensorStruct.iXFpUb, 0.0f, CMPMODE::GE, CAL_D_H_W_BLOCK);
    CompareScalar(
        maskXUb, inputTensorStruct.iXFpUb, static_cast<float>(commonParam.inputW_ - 1), CMPMODE::LE, CAL_D_H_W_BLOCK);
    CompareScalar(maskTmpYUb, inputTensorStruct.iYFpUb, 0.0f, CMPMODE::GE, CAL_D_H_W_BLOCK);
    CompareScalar(
        maskYUb, inputTensorStruct.iYFpUb, static_cast<float>(commonParam.inputH_ - 1), CMPMODE::LE, CAL_D_H_W_BLOCK);
    CompareScalar(maskTmpZUb, inputTensorStruct.iZFpUb, 0.0f, CMPMODE::GE, CAL_D_H_W_BLOCK);
    CompareScalar(
        maskZUb, inputTensorStruct.iZFpUb, static_cast<float>(commonParam.inputD_ - 1), CMPMODE::LE, CAL_D_H_W_BLOCK);

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

__aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
    LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum)
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

__aicore__ inline void CoordinatesSelectTensor(
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

__aicore__ inline void OutTransposeFp32(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb)
{
    uint64_t dstList[16];
    uint64_t srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == NUM_8) {
        transDataParams.repeatTimes = MIN_CHANNEL_ALIGN;
        transDataParams.dstRepStride = NUM_2;
        transDataParams.srcRepStride = NUM_16;

        for (int32_t i = 0; i < NUM_16; i++) {
            srcList[i] = (uint64_t)(xLocal[i * NUM_8].GetPhyAddr());
        }

        for (int32_t i = 0; i < NUM_8; i++) {
            dstList[i * NUM_2] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE].GetPhyAddr());
            dstList[i * NUM_2 + 1] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + NUM_8].GetPhyAddr());
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= NUM_64) {
        transDataParams.repeatTimes = channelAlign / NUM_8;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < NUM_8; j++) {
            for (int32_t i = 0; i < NUM_16; i++) {
                srcList[i] = (uint64_t)(xLocal[i * channelAlign + j * NUM_16 * channelAlign].GetPhyAddr());
            }

            for (int32_t i = 0; i < NUM_8; i++) {
                dstList[i * NUM_2] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + j * NUM_16].GetPhyAddr());
                dstList[i * NUM_2 + NUM_1] =
                    (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + NUM_8 + j * NUM_16].GetPhyAddr());
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<float>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

__aicore__ inline void ClipCoordinates(InputTensorStruct inputTensorStruct, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> wMaskUb, IndexBuffer &indexBuffer, GridSampleCommonParam commonParam)
{
    LocalTensor<int32_t> tmpYIntUb = indexBuffer.intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> tmpZIntUb = indexBuffer.coorTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpYIntUb;
    LocalTensor<int32_t> inputZIntTmpUb = tmpZIntUb;
    PipeBarrier<PIPE_V>();
    Adds(inputXIntTmpUb, inputTensorStruct.iXIntUb, 0, CAL_D_H_W_BLOCK);
    Adds(inputYIntTmpUb, inputTensorStruct.iYIntUb, 0, CAL_D_H_W_BLOCK);
    Adds(inputZIntTmpUb, inputTensorStruct.iZIntUb, 0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Cast(inputTensorStruct.iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(inputTensorStruct.iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(inputTensorStruct.iZFpUb, inputZIntTmpUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = indexBuffer.maskBuf_.Get<uint8_t>(MASK_UB_SIZE * NUM_5);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskZUb = maskUb[MASK_UB_SIZE];

    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE * NUM_2];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * NUM_3];
    LocalTensor<uint8_t> maskTmpZUb = maskUb[MASK_UB_SIZE * NUM_4];
    CoordinatesGetMaskWithRange(inputTensorStruct, maskXUb, maskUb, commonParam);
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    auto maskZUbTmp = maskZUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    And(maskXUbTmp, maskZUbTmp, maskXUbTmp, maskNum);
    wMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();

    CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(commonParam.inputW_ - 1));
    CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(commonParam.inputH_ - 1));
    CoordinatesFrameRange(inputZIntTmpUb, (int32_t)(commonParam.inputD_ - 1));

    PipeBarrier<PIPE_V>();

    int32_t tmpWH_ = commonParam.inputW_ * commonParam.inputH_;
    Muls(inputZIntTmpUb, inputZIntTmpUb, (int32_t)tmpWH_, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)commonParam.inputW_, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputYIntTmpUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, inputZIntTmpUb, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb,
    IndexBuffer &indexBuffer, GridSampleCommonParam commonParam)
{
    Mins(iXFpUb, iXFpUb, (float)(commonParam.inputW_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpUb, iYFpUb, (float)(commonParam.inputH_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iZFpUb, iZFpUb, (float)(commonParam.inputD_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iZFpUb, iZFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskUb = indexBuffer.weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<float> tmpUb = indexBuffer.inputXYZFPBuf_.Get<float>();
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

__aicore__ inline void ReflectClipFilterValue(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
    LocalTensor<float> iZFpUb, IndexBuffer &indexBuffer, GridSampleCommonParam commonParam)
{
    LocalTensor<float> tmpUb = indexBuffer.inputXYZFPBuf_.Get<float>();
    LocalTensor<uint8_t> maskUb = indexBuffer.maskBuf_.Get<uint8_t>(MASK_UB_SIZE * NUM_3);

    Muls(tmpUb, iZFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iZFpUb, iZFpUb, maskUb, 0.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iZFpUb, iZFpUb, (float)(commonParam.inputD_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iZFpUb, iZFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iXFpUb, iXFpUb, (float)(commonParam.inputW_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpUb, iYFpUb, (float)(commonParam.inputH_ - 1), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReflectCoordinatesGeneralSelect(
    LocalTensor<float> coorSubUb, float minS, float spanS, IndexBuffer &indexBuffer)
{
    LocalTensor<float> fmodFpUb = indexBuffer.modBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = indexBuffer.maskBuf_.Get<uint8_t>(MASK_UB_SIZE * NUM_3);
    LocalTensor<float> tmpFpUb = indexBuffer.outTmpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<float> extraFpUb = indexBuffer.extraBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = indexBuffer.intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);

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

    Adds(out1, extraFpUb, minS, CAL_D_H_W_BLOCK);
    Muls(out2, extraFpUb, -1.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, spanS, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2, out2, minS, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(mods, coorSubUb, static_cast<float>(1 / FLOAT_2), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(mods, mods, FLOAT_2, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(mods, coorSubUb, mods, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CoordinatesSelectTensor(out1, out2, coorSubUb, maskUb);
}

__aicore__ inline void ReflectCoordinatesGeneral(
    LocalTensor<float> coorSubUb, const int64_t twiceLow, const int64_t twiceHigh, IndexBuffer &indexBuffer)
{
    if (twiceLow == twiceHigh) {
        Duplicate(coorSubUb, (float)0.0, CAL_D_H_W_BLOCK);
        return;
    }
    LocalTensor<float> extraFpUb = indexBuffer.extraBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = indexBuffer.intTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);

    float minS = static_cast<float>(twiceLow) / 2;
    float negMinS = static_cast<float>(-1.0) * minS;
    float spanS = static_cast<float>(twiceHigh - twiceLow) / 2;

    // new relative position
    Adds(coorSubUb, coorSubUb, negMinS, CAL_D_H_W_BLOCK);
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

    ReflectCoordinatesGeneralSelect(coorSubUb, minS, spanS, indexBuffer);
}

__aicore__ inline void ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb,
    IndexBuffer &indexBuffer, GridSampleCommonParam commonParam)
{
    LocalTensor<float> coorSubUb = indexBuffer.coorTmpBuf_.Get<float>(CAL_D_H_W_BLOCK);

    // coorUb = Z * inputW_ * inputH_ + Y * inputW_ + X
    int64_t twiceLow = (commonParam.alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowZ = REFLECT_RATIO * (commonParam.inputD_ - 1);
    int64_t twiceLowY = REFLECT_RATIO * (commonParam.inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (commonParam.inputW_ - 1);
    if (commonParam.alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowZ = REFLECT_RATIO * commonParam.inputD_ - 1;
        twiceLowY = REFLECT_RATIO * commonParam.inputH_ - 1;
        twiceLowX = REFLECT_RATIO * commonParam.inputW_ - 1;
    }

    ReflectCoordinatesGeneral(iZFpUb, twiceLow, twiceLowZ, indexBuffer);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(iYFpUb, twiceLow, twiceLowY, indexBuffer);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(iXFpUb, twiceLow, twiceLowX, indexBuffer);
    PipeBarrier<PIPE_V>();

    ReflectClipFilterValue(iXFpUb, iYFpUb, iZFpUb, indexBuffer, commonParam);
}

}  // namespace GridSample
#endif  //  GIRD_SAMPLER_3D_COMMON
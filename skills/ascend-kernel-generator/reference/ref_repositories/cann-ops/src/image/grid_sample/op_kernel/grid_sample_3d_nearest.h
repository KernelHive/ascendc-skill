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
 * \file grid_sampler_3d_nearest.h
 * \brief
 */
#ifndef GIRD_SAMPLER_3D_NEAREST
#define GIRD_SAMPLER_3D_NEAREST

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grid_sample_3d_common.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler3DNearest {
public:
    __aicore__ inline GridSampler3DNearest(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PerLoopCompute(ProcessParam processParam);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void ZeroClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void MTE2ForNCHW(
        int32_t nIdx, PointParam pointNearestParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWC(
        int32_t nIdx, PointParam pointNearestParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void OutTransposeFp16(int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);

    __aicore__ inline void MTE3ForNCHWFp16(ProcessParam processParam, PointParam pointNearestParam,
        LocalTensor<float> weightUb, LocalTensor<float> outValueUb);

    __aicore__ inline void PointNearestEachChannel(ProcessParam processParam, LocalTensor<uint64_t> maskUbTmp,
        PointParam pointNearestParam, LocalTensor<T> xLocal);

    __aicore__ inline void MTE3ForNCHWFp32(ProcessParam processParam, PointParam pointNearestParam,
        LocalTensor<float> weightUb, LocalTensor<float> outValueU);

    __aicore__ inline void PointNearest(ProcessParam processParam);

    __aicore__ inline void CalculateGrid(ProcessParam processParam, LocalTensor<float> inputXFpLocal,
        LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf_;

    TBuf<QuePosition::VECCALC> gridFp32Buf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputZIntBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskXBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskYBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskZBuf_;

    TBuf<QuePosition::VECCALC> gridFp16Buf_;
    TBuf<QuePosition::VECCALC> yFp16Buf_;
    TBuf<QuePosition::VECCALC> outValueFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    LocalTensor<int32_t> coordinatesLocal;
    LocalTensor<float> weightLocal;
    LocalTensor<float> outValueLocal;
    LocalTensor<uint8_t> weightMaskUb;

    const int64_t X_UB_SIZE_4_GENERAL = 32768;    // 32KB
    const int64_t X_UB_SIZE_4_FP16 = 16384;       // 16KB
    const int64_t GRID_UB_SIZE_4_GENERAL = 6144;  //  6KB
    const int64_t GRID_UB_SIZE_4_FP16 = 3072;     //  3KB
    const int64_t XYZ_UB_SIZE_4_GENERAL = 4096;   //  4KB
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;     //  2KB

    int64_t blockIDX = 0;
    uint64_t rsvdCnt = 0;
    uint32_t mask = 192;
    uint16_t repeatTime = CAL_D_H_W_BLOCK * 3 / 192;

    GridSampleCommonParam commonParam{};
    IndexBuffer indexBuffer{};
};

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData, commonParam);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer initialize
    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);                          // 32KB
    pipe.InitBuffer(gridFp32Buf_, GRID_UB_SIZE_4_GENERAL);                //  6KB
    pipe.InitBuffer(indexBuffer.inputXYZFPBuf_, GRID_UB_SIZE_4_GENERAL);  //  6KB
    pipe.InitBuffer(inputXIntBuf_, XYZ_UB_SIZE_4_GENERAL * 2);            //  8KB
    pipe.InitBuffer(inputYIntBuf_, XYZ_UB_SIZE_4_GENERAL);                //  4KB
    pipe.InitBuffer(inputZIntBuf_, XYZ_UB_SIZE_4_GENERAL);                //  4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);                 //  8KB
    pipe.InitBuffer(indexBuffer.intTmpBuf_, Y_UB_SIZE_4_GENERAL);         //  2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);                       //  2KB
    pipe.InitBuffer(indexBuffer.coorTmpBuf_, Y_UB_SIZE_4_GENERAL);        //  2KB
    pipe.InitBuffer(outValueBuf_, X_UB_SIZE_4_GENERAL);                   // 32KB
    pipe.InitBuffer(indexBuffer.maskBuf_, 2048);                          // 2KB
    pipe.InitBuffer(indexBuffer.weightMaskBuf_, 320);                     // 320B
    pipe.InitBuffer(indexBuffer.modBuf_, Y_UB_SIZE_4_GENERAL);            //  2KB
    pipe.InitBuffer(indexBuffer.extraBuf_, Y_UB_SIZE_4_GENERAL);          //  2KB
    pipe.InitBuffer(indexBuffer.outTmpBuf_, XYZ_UB_SIZE_4_GENERAL);       //  4KB
    pipe.InitBuffer(bufferMaskXBuf_, BLOCK_SIZE * 6);                     // 64B
    pipe.InitBuffer(bufferMaskYBuf_, BLOCK_SIZE * 4);                     // 64B
    pipe.InitBuffer(bufferMaskZBuf_, BLOCK_SIZE * 4);                     // 64B

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        pipe.InitBuffer(gridFp16Buf_, GRID_UB_SIZE_4_FP16);   // 3KB
        pipe.InitBuffer(yFp16Buf_, X_UB_SIZE_4_FP16);         // 16KB
        pipe.InitBuffer(outValueFp16Buf_, X_UB_SIZE_4_FP16);  // 16KB
    }

    initBufTensor(bufferMaskXBuf_, bufferMaskYBuf_, bufferMaskZBuf_);
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::Clip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb)
{
    if (commonParam.paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(iXFpUb, iYFpUb, iZFpUb, indexBuffer, commonParam);
    } else if (commonParam.paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(iXFpUb, iYFpUb, iZFpUb, indexBuffer, commonParam);
    } else if (commonParam.paddingMode_ == PADDING_MODE_ZEROS) {
        ZeroClip(iXFpUb, iYFpUb, iZFpUb);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::ZeroClip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb)
{
    LocalTensor<uint8_t> maskUb = indexBuffer.weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<float> tmpUb = indexBuffer.inputXYZFPBuf_.Get<float>();
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, -100.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, -100.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(tmpUb, iZFpUb, (float)(0.0), CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iZFpUb, iZFpUb, maskUb, -100.0f, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::MTE2ForNCHW(
    int32_t nIdx, PointParam pointNearestParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    for (int32_t i = 0; i < pointNearestParam.loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(pointNearestParam.loopOffset + i);
        int64_t baseLocation = nIdx * commonParam.inputC_ * commonParam.inputH_ * commonParam.inputW_ + coordVal +
                               pointNearestParam.cIdx * CHANNEL_BLOCK * commonParam.inputH_ * commonParam.inputW_;
        for (int cIter = 0; cIter < pointNearestParam.channelAlign; cIter++) {
            int32_t xLocalOffset = i * pointNearestParam.channelAlign + cIter;
            if (cIter >= pointNearestParam.calCElems) {
                if constexpr (IsSameType<T, bfloat16_t>::value) {
                    xLocal.SetValue(xLocalOffset, ToBfloat16(0.0));
                } else {
                    xLocal.SetValue(xLocalOffset, static_cast<T>(0.0));
                }
                continue;
            }

            int64_t coordinate = baseLocation + cIter * commonParam.inputH_ * commonParam.inputW_;
            xLocal.SetValue(xLocalOffset, gmX_.GetValue(coordinate));
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::MTE2ForNHWC(
    int32_t nIdx, PointParam pointNearestParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    int64_t base = nIdx * commonParam.inputH_ * commonParam.inputW_ * commonParam.inputD_ * commonParam.inputC_ +
                   pointNearestParam.cIdx * CHANNEL_BLOCK;
    auto timeStep = pointNearestParam.loopElems / 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = pointNearestParam.calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int32_t i = 0; i < timeStep; i++) {
        int64_t coordVal_0 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8) * commonParam.inputC_;
        int64_t coordVal_1 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 1) * commonParam.inputC_;
        int64_t coordVal_2 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 2) * commonParam.inputC_;
        int64_t coordVal_3 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 3) * commonParam.inputC_;
        int64_t coordVal_4 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 4) * commonParam.inputC_;
        int64_t coordVal_5 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 5) * commonParam.inputC_;
        int64_t coordVal_6 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 6) * commonParam.inputC_;
        int64_t coordVal_7 = coorUb.GetValue(pointNearestParam.loopOffset + i * 8 + 7) * commonParam.inputC_;
        int64_t location_0 = base + coordVal_0;
        int64_t location_1 = base + coordVal_1;
        int64_t location_2 = base + coordVal_2;
        int64_t location_3 = base + coordVal_3;
        int64_t location_4 = base + coordVal_4;
        int64_t location_5 = base + coordVal_5;
        int64_t location_6 = base + coordVal_6;
        int64_t location_7 = base + coordVal_7;

        DataCopyPad(xLocal[(i * 8) * pointNearestParam.channelAlign], gmX_[location_0], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 1) * pointNearestParam.channelAlign], gmX_[location_1], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 2) * pointNearestParam.channelAlign], gmX_[location_2], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 3) * pointNearestParam.channelAlign], gmX_[location_3], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 4) * pointNearestParam.channelAlign], gmX_[location_4], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 5) * pointNearestParam.channelAlign], gmX_[location_5], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 6) * pointNearestParam.channelAlign], gmX_[location_6], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 7) * pointNearestParam.channelAlign], gmX_[location_7], params, padParams);
    }
    for (auto i = pointNearestParam.loopElems / 8 * 8; i < pointNearestParam.loopElems; i++) {
        int64_t coordVal_0 = coorUb.GetValue(pointNearestParam.loopOffset + i) * commonParam.inputC_;
        int64_t location_0 = base + coordVal_0;
        DataCopyPad(xLocal[i * pointNearestParam.channelAlign], gmX_[location_0], params, padParams);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::OutTransposeFp16(
    int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb)
{
    uint64_t rstList[16];
    uint64_t srcList[16];

    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == B16_ALIGN_FACTOR) {
        transDataParams.repeatTimes = 8;
        transDataParams.dstRepStride = 1;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            rstList[i] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE].GetPhyAddr());
        }

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = (uint64_t)(xLocal[i * 16].GetPhyAddr());
        }
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<T>(rstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 16;
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                rstList[i] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE + j * 16].GetPhyAddr());
            }

            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = (uint64_t)(xLocal[i * channelAlign + j * 16 * channelAlign].GetPhyAddr());
            }
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<T>(rstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::MTE3ForNCHWFp32(
    ProcessParam processParam, PointParam pointNearestParam, LocalTensor<float> weightUb, LocalTensor<float> outValueUb)
{
    // 512 * inputC_ * blockIDX 每个核的地址
    // loopOffset 偏移的是几个128
    int64_t gmYBaseOffset = pointNearestParam.outBaseOffset + pointNearestParam.loopOffset +
                            pointNearestParam.cIdx * CHANNEL_BLOCK * commonParam.gridDHW_;

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    uint32_t blockLength = pointNearestParam.loopElems * sizeof(T);
    if (pointNearestParam.calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb[pointNearestParam.loopOffset], TRANSE_REP_STRIDE);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, blockLength, 0, 0, 0});
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = pointNearestParam.loopOffset + i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                pointNearestParam.calCElems,
                {1, 1, 1, 16, 16, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        uint32_t srcStride = TRANSE_REP_STRIDE * sizeof(T) / BLOCK_SIZE -
                             ((pointNearestParam.loopElems * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE);
        uint32_t dstStride = commonParam.gridDHW_ * sizeof(T) - pointNearestParam.loopElems * sizeof(T);
        DataCopyPad(gmY_[gmYBaseOffset],
            outValueUb,
            {(uint16_t)pointNearestParam.calCElems, blockLength, srcStride, dstStride, 0});
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::MTE3ForNCHWFp16(
    ProcessParam processParam, PointParam pointNearestParam, LocalTensor<float> weightUb, LocalTensor<float> outValueUb)
{
    // 512 * inputC_ * blockIDX 每个核的地址
    // loopOffset 偏移的是几个128
    int64_t gmYBaseOffset2 = CAL_D_H_W_BLOCK * commonParam.inputC_ * blockIDX + pointNearestParam.loopOffset +
                             pointNearestParam.cIdx * CHANNEL_BLOCK * CAL_D_H_W_BLOCK;
    int64_t gmYBaseOffset = pointNearestParam.outBaseOffset + pointNearestParam.loopOffset +
                            pointNearestParam.cIdx * CHANNEL_BLOCK * commonParam.gridDHW_;

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    LocalTensor<T> outLocalFP16 = yFp16Buf_.AllocTensor<T>();
    uint32_t blockLength = pointNearestParam.loopElems * sizeof(T);

    if (pointNearestParam.calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb[pointNearestParam.loopOffset], TRANSE_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        Cast(outLocalFP16, outValueUb, RoundMode::CAST_RINT, TRANSE_REP_STRIDE);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        PipeBarrier<PIPE_MTE3>();
        DataCopyPad(gmY_[gmYBaseOffset], outLocalFP16, {1, blockLength, 0, 0, 0});
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = pointNearestParam.loopOffset + i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffset],
                B32_MASK,
                pointNearestParam.calCElems,
                {1, 1, 1, 16, 16, 0});
        }

        Cast(outLocalFP16, outValueUb, RoundMode::CAST_RINT, TRANSE_REP_STRIDE * CHANNEL_BLOCK);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        uint32_t srcStride = TRANSE_REP_STRIDE * sizeof(T) / BLOCK_SIZE -
                             ((pointNearestParam.loopElems * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE);
        uint32_t dstStride = commonParam.gridDHW_ * sizeof(T) - pointNearestParam.loopElems * sizeof(T);
        DataCopyPad(gmY_[gmYBaseOffset],
            outLocalFP16,
            {(uint16_t)pointNearestParam.calCElems, blockLength, srcStride, dstStride, 0});
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::PointNearestEachChannel(
    ProcessParam processParam, LocalTensor<uint64_t> maskUbTmp, PointParam pointNearestParam, LocalTensor<T> xLocal)
{
    if (commonParam.channelLast_ == LAYOUT_NHWC) {
        MTE2ForNHWC(processParam.nIdx, pointNearestParam, coordinatesLocal, xLocal);
    } else {
        MTE2ForNCHW(processParam.nIdx, pointNearestParam, coordinatesLocal, xLocal);
    }

    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);

    if constexpr (IsSameType<T, half>::value) {  // T: fp16
        LocalTensor<T> xLocalFp16 = outValueFp16Buf_.Get<T>();
        OutTransposeFp16(pointNearestParam.channelAlign, xLocal, xLocalFp16);
        PipeBarrier<PIPE_V>();
        Cast(outValueLocal, xLocalFp16, RoundMode::CAST_NONE, pointNearestParam.calCElems * TRANSE_REP_STRIDE);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        LocalTensor xLocalFp32 = xBuf_.Get<float>();
        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, pointNearestParam.channelAlign * TRANSE_REP_STRIDE);
        OutTransposeFp32(pointNearestParam.channelAlign, xLocalFp32, outValueLocal);
    } else {  // T: fp32
        OutTransposeFp32(pointNearestParam.channelAlign, xLocal, outValueLocal);
    }
    PipeBarrier<PIPE_V>();

    for (size_t i = 0; i < pointNearestParam.calCElems; i++) {
        int32_t ubOffset = i * TRANSE_REP_STRIDE;
        Select(outValueLocal[ubOffset],
            maskUbTmp,
            outValueLocal[ubOffset],
            0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            TRANSE_REP_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        MTE3ForNCHWFp16(processParam, pointNearestParam, weightLocal, outValueLocal);
    } else {
        MTE3ForNCHWFp32(processParam, pointNearestParam, weightLocal, outValueLocal);
    }
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventMte3V);
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::PointNearest(ProcessParam processParam)
{
    if (commonParam.paddingMode_ == PADDING_MODE_ZEROS) {
        CoordinatesSelectScalar(weightLocal, weightLocal, weightMaskUb, 0.0f, CAL_D_H_W_BLOCK);
    }

    LocalTensor<uint8_t> maskUb = indexBuffer.maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();
    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();

    int32_t trans_loop = (processParam.calDHWElems + TRANSE_REP_STRIDE - 1) / TRANSE_REP_STRIDE;
    PointParam pointNearestParam;
    pointNearestParam.loopElems = TRANSE_REP_STRIDE;
    pointNearestParam.outBaseOffset =
        processParam.nIdx * commonParam.gridDHW_ * commonParam.inputC_ + processParam.hwIdx * CAL_D_H_W_BLOCK;
    PipeBarrier<PIPE_ALL>();
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            pointNearestParam.loopElems = processParam.calDHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        pointNearestParam.loopOffset = loop_idx * TRANSE_REP_STRIDE;
        pointNearestParam.maskOffset = loop_idx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(pointNearestParam.maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(pointNearestParam.maskOffset + 1));

        LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
        if (IsSameType<T, bfloat16_t>::value) {
            xLocal = xLocal[TRANSE_REP_STRIDE * CHANNEL_BLOCK];
        }
        for (pointNearestParam.cIdx = 0; pointNearestParam.cIdx < commonParam.channelLoop_; pointNearestParam.cIdx++) {
            pointNearestParam.calCElems = commonParam.perLoopChannel_;
            if (pointNearestParam.cIdx == commonParam.channelLoop_ - 1) {
                pointNearestParam.calCElems = commonParam.lastLoopChannel_;
            }
            pointNearestParam.channelAlign = Ceil(pointNearestParam.calCElems, B32_ALIGN_FACTOR) * B32_ALIGN_FACTOR;
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                pointNearestParam.channelAlign = Ceil(pointNearestParam.calCElems, B16_ALIGN_FACTOR) * B16_ALIGN_FACTOR;
            }

            PointNearestEachChannel(processParam, maskUbTmp, pointNearestParam, xLocal);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::CalculateGrid(ProcessParam processParam,
    LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal)
{
    int64_t gridGmOffset = processParam.nIdx * commonParam.gridDHW_ * 3 + processParam.hwIdx * CAL_D_H_W_BLOCK * 3;

    LocalTensor<float> gridFp32Local = gridFp32Buf_.Get<float>();
    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = processParam.calDHWElems * 3 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    DataCopyPadExtParams<T> padParamsGrid{false, 0, 0, 0};
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {  // T: fp16
        LocalTensor<T> gridFp16Local = gridFp16Buf_.Get<T>();
        DataCopyPad(gridFp16Local, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(gridFp32Local, gridFp16Local, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK * 3);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyPad(gridFp32Local, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    LocalTensor<float> inputXYZUb = indexBuffer.inputXYZFPBuf_.Get<float>();
    Adds(inputXYZUb, gridFp32Local, (float)1.0, CAL_D_H_W_BLOCK * 3);

    LocalTensor<uint32_t> bufXPattern = bufferMaskXBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> bufYPattern = bufferMaskYBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> bufZPattern = bufferMaskZBuf_.Get<uint32_t>();

    PipeBarrier<PIPE_V>();

    // 分别取x和y(inputXFpLocal, inputXYZUb, xPattern, true, mask,
    GatherMask(inputXFpLocal, inputXYZUb, bufXPattern, true, 192, {1, repeatTime, 24, 0}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYZUb, bufYPattern, true, 192, {1, repeatTime, 24, 0}, rsvdCnt);
    GatherMask(inputZFpLocal, inputXYZUb, bufZPattern, true, 192, {1, repeatTime, 24, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (commonParam.alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (commonParam.inputW_ - (float)1.0)), CAL_D_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * (commonParam.inputH_ - (float)1.0)), CAL_D_H_W_BLOCK);
        Muls(inputZFpLocal, inputZFpLocal, (float)((float)0.5 * (commonParam.inputD_ - (float)1.0)), CAL_D_H_W_BLOCK);
    } else {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * commonParam.inputW_), CAL_D_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * commonParam.inputH_), CAL_D_H_W_BLOCK);
        Muls(inputZFpLocal, inputZFpLocal, (float)((float)0.5 * commonParam.inputD_), CAL_D_H_W_BLOCK);
        PipeBarrier<PIPE_V>();
        Adds(inputXFpLocal, inputXFpLocal, (float)(-0.5), CAL_D_H_W_BLOCK * 3);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::PerLoopCompute(ProcessParam processParam)
{
    LocalTensor<float> gridFp32Local = gridFp32Buf_.Get<float>();
    LocalTensor<float> inputXFpLocal = gridFp32Local;
    LocalTensor<float> inputYFpLocal = gridFp32Local[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputZFpLocal = gridFp32Local[CAL_D_H_W_BLOCK * 2];
    LocalTensor<int32_t> inputXIntLocal = inputXIntBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> inputYIntLocal = inputYIntBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    LocalTensor<int32_t> inputZIntLocal = inputZIntBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);

    CalculateGrid(processParam, inputXFpLocal, inputYFpLocal, inputZFpLocal);

    Clip(inputXFpLocal, inputYFpLocal, inputZFpLocal);

    Cast(inputXIntLocal, inputXFpLocal, RoundMode::CAST_RINT, CAL_D_H_W_BLOCK);
    Cast(inputYIntLocal, inputYFpLocal, RoundMode::CAST_RINT, CAL_D_H_W_BLOCK);
    Cast(inputZIntLocal, inputZFpLocal, RoundMode::CAST_RINT, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(inputXFpLocal, inputXIntLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(inputYFpLocal, inputYIntLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(inputZFpLocal, inputZIntLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    weightLocal = weightBuf_.Get<float>(CAL_D_H_W_BLOCK);
    coordinatesLocal = coorBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    outValueLocal = outValueBuf_.Get<float>();
    weightMaskUb = indexBuffer.weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    Duplicate(weightLocal, (float)1.0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    InputTensorStruct inputTensorStruct{
        inputXFpLocal, inputYFpLocal, inputZFpLocal, inputXIntLocal, inputYIntLocal, inputZIntLocal};
    ClipCoordinates(inputTensorStruct, coordinatesLocal, weightMaskUb, indexBuffer, commonParam);

    PointNearest(processParam);
}

template <typename T>
__aicore__ inline void GridSampler3DNearest<T>::Process()
{
    if (blockIDX >= commonParam.needCoreNum_) {
        return;
    }
    int32_t preLoopNum = blockIDX * commonParam.preCoreLoop_;

    int64_t loopSize = commonParam.preCoreLoop_;
    if (blockIDX == commonParam.needCoreNum_ - 1) {
        loopSize = commonParam.lastCoreLoop_;
    }

    ProcessParam processNearestParam{};
    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        processNearestParam.nIdx = (preLoopNum + loopIdx) / commonParam.preNUbLoop_;
        processNearestParam.hwIdx = (preLoopNum + loopIdx) % commonParam.preNUbLoop_;
        processNearestParam.calDHWElems = CAL_D_H_W_BLOCK;
        if (processNearestParam.hwIdx == commonParam.preNUbLoop_ - 1) {
            processNearestParam.calDHWElems = commonParam.lastLoopDHW_;
        }

        PerLoopCompute(processNearestParam);
    }
}
}  // namespace GridSample
#endif  // GIRD_SAMPLER_3D_NEAREST
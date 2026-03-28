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
 * \file grid_sampler_3d.h
 * \brief
 */
#ifndef GIRD_SAMPLER_3D
#define GIRD_SAMPLER_3D

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grid_sample_3d_common.h"

namespace GridSample {

using namespace AscendC;

struct CoordinatesStruct {
    LocalTensor<int32_t> tnwCoordinates;
    LocalTensor<int32_t> tneCoordinates;
    LocalTensor<int32_t> tswCoordinates;
    LocalTensor<int32_t> tsecoordinates;
    LocalTensor<int32_t> bnwCoordinates;
    LocalTensor<int32_t> bneCoordinates;
    LocalTensor<int32_t> bswCoordinates;
    LocalTensor<int32_t> bsecoordinates;
};

struct WeightStruct {
    LocalTensor<float> tnwWeightLocal;
    LocalTensor<float> tneWeightLocal;
    LocalTensor<float> tswWeightLocal;
    LocalTensor<float> tseWeightLocal;
    LocalTensor<float> bnwWeightLocal;
    LocalTensor<float> bneWeightLocal;
    LocalTensor<float> bswWeightLocal;
    LocalTensor<float> bseWeightLocal;
};

struct CalculatePointBilinearParam {
    LocalTensor<int32_t> coordinatesUb;
    LocalTensor<float> outValueUb;
    LocalTensor<float> outValueTotalLocal;
    LocalTensor<float> weightUb;
    LocalTensor<uint64_t> maskUbTmp;
    bool isAtomicAdd;

    __aicore__ inline CalculatePointBilinearParam()
    {}

    __aicore__ inline CalculatePointBilinearParam(LocalTensor<int32_t> coordinatesUb, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueTotalLocal, LocalTensor<float> weightUb, LocalTensor<uint64_t> maskUbTmp,
        bool isAtomicAdd)
        : coordinatesUb(coordinatesUb), outValueUb(outValueUb), outValueTotalLocal(outValueTotalLocal),
          weightUb(weightUb), maskUbTmp(maskUbTmp), isAtomicAdd(isAtomicAdd)
    {}
};

template <typename T>
class GridSampler3D {
public:
    __aicore__ inline GridSampler3D(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData);
    __aicore__ inline void Process();

private:
    //   __aicore__ inline void ParseTilingData(const GridSampleTilingData* tilingData);

    __aicore__ inline void InitBuffer();

    __aicore__ inline void PerLoopCompute(ProcessParam processParam);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void MTE2ForNCHW(
        int32_t nIdx, PointParam pointBilinearParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWC(
        int32_t nIdx, PointParam pointBilinearParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void OutTransposeFp16(int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);

    __aicore__ inline void calculatePointBilinear(ProcessParam processParam,
        CalculatePointBilinearParam calculatePointBilinearParam, PointParam pointBilinearParam, LocalTensor<T> xLocal);
    __aicore__ inline void MTE3ForNCHW(
        ProcessParam processParam, PointParam pointBilinearParam, LocalTensor<T> outValueUb);
    __aicore__ inline void PointBilinear(ProcessParam processParam, LocalTensor<float> outValueUb, bool isAutomicAdd);

    __aicore__ inline void GetXLocal(ProcessParam processParam, CalculatePointBilinearParam calculatePointBilinearParam,
        PointParam pointBilinearParam, LocalTensor<T> xLocal, LocalTensor<float> outValueLocal);

    __aicore__ inline void CalculateGrid(ProcessParam processParam, LocalTensor<float> gridFp32Local,
        LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal);
    __aicore__ inline void calculateGridWeight(
        LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal);

    __aicore__ inline void clipAllCoordinate(LocalTensor<float> outValueLocal);
    __aicore__ inline CoordinatesStruct GetCoordinatesStruct();
    __aicore__ inline void GetWeightMaskStruct();
    __aicore__ inline WeightStruct GetWeightStruct();
    __aicore__ inline void GetMaskStruct();
    __aicore__ inline void GetInputTensor();

    __aicore__ inline void PointBilinearEachChannel(ProcessParam processParam, LocalTensor<float> outValueUb,
        LocalTensor<float> outValueTotalLocal, PointParam pointBilinearParam, LocalTensor<T> xLocal);

    __aicore__ inline void PointBilinearSetMask(PointParam pointBilinearParam);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gridQueue_;

    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputZIntBuf_;
    TBuf<QuePosition::VECCALC> inputXFpBuf_;
    TBuf<QuePosition::VECCALC> inputYFpBuf_;
    TBuf<QuePosition::VECCALC> inputZFpBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> weightTmpBuf_;
    TBuf<QuePosition::VECCALC> weightTmp1Buf_;
    TBuf<QuePosition::VECCALC> weightTmp2Buf_;
    TBuf<QuePosition::VECCALC> weightTmp3Buf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> outValueSumBuf_;
    TBuf<QuePosition::VECCALC> maskBuf2_;
    TBuf<QuePosition::VECCALC> maskBuf3_;
    TBuf<QuePosition::VECCALC> maskBuf4_;
    TBuf<QuePosition::VECCALC> maskBuf5_;
    TBuf<QuePosition::VECCALC> maskBuf6_;
    TBuf<QuePosition::VECCALC> maskBuf7_;
    TBuf<QuePosition::VECCALC> maskBuf8_;
    TBuf<QuePosition::VECCALC> weightMaskBuf2_;
    TBuf<QuePosition::VECCALC> weightMaskBuf3_;
    TBuf<QuePosition::VECCALC> weightMaskBuf4_;
    TBuf<QuePosition::VECCALC> weightMaskBuf5_;
    TBuf<QuePosition::VECCALC> weightMaskBuf6_;
    TBuf<QuePosition::VECCALC> weightMaskBuf7_;
    TBuf<QuePosition::VECCALC> weightMaskBuf8_;

    TBuf<QuePosition::VECCALC> bufferMaskXBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskYBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskZBuf_;

    TBuf<QuePosition::VECCALC> gridFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    LocalTensor<uint8_t> weightMaskUb;
    LocalTensor<uint8_t> weightMaskUb2;
    LocalTensor<uint8_t> weightMaskUb3;
    LocalTensor<uint8_t> weightMaskUb4;
    LocalTensor<uint8_t> weightMaskUb5;
    LocalTensor<uint8_t> weightMaskUb6;
    LocalTensor<uint8_t> weightMaskUb7;
    LocalTensor<uint8_t> weightMaskUb8;
    LocalTensor<uint64_t> weightMaskUbTmp;
    LocalTensor<uint64_t> weightMaskUbTmp2;
    LocalTensor<uint64_t> weightMaskUbTmp3;
    LocalTensor<uint64_t> weightMaskUbTmp4;
    LocalTensor<uint64_t> weightMaskUbTmp5;
    LocalTensor<uint64_t> weightMaskUbTmp6;
    LocalTensor<uint64_t> weightMaskUbTmp7;
    LocalTensor<uint64_t> weightMaskUbTmp8;

    LocalTensor<uint8_t> maskUb;
    LocalTensor<uint64_t> maskUbTmp;
    LocalTensor<uint8_t> maskUb2;
    LocalTensor<uint64_t> maskUbTmp2;
    LocalTensor<uint8_t> maskUb3;
    LocalTensor<uint64_t> maskUbTmp3;
    LocalTensor<uint8_t> maskUb4;
    LocalTensor<uint64_t> maskUbTmp4;
    LocalTensor<uint8_t> maskUb5;
    LocalTensor<uint64_t> maskUbTmp5;
    LocalTensor<uint8_t> maskUb6;
    LocalTensor<uint64_t> maskUbTmp6;
    LocalTensor<uint8_t> maskUb7;
    LocalTensor<uint64_t> maskUbTmp7;
    LocalTensor<uint8_t> maskUb8;
    LocalTensor<uint64_t> maskUbTmp8;

    LocalTensor<int32_t> inputXWIntLocal;
    LocalTensor<int32_t> inputXEIntLocal;
    LocalTensor<int32_t> inputYWIntLocal;
    LocalTensor<int32_t> inputYEIntLocal;
    LocalTensor<int32_t> inputZWIntLocal;
    LocalTensor<int32_t> inputZEIntLocal;
    LocalTensor<float> inputXWFpLocal;
    LocalTensor<float> inputXEFpLocal;
    LocalTensor<float> inputYWFpLocal;
    LocalTensor<float> inputYEFpLocal;
    LocalTensor<float> inputZWFpLocal;
    LocalTensor<float> inputZEFpLocal;

    const int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);

    const int64_t X_UB_SIZE_4_GENERAL = 32768;
    const int64_t GRID_UB_SIZE_4_GENERAL = 6144;
    const int64_t GRID_UB_SIZE_4_FP16 = 3072;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t XYZ_UB_SIZE_4_GENERAL = 4096;  //  4KB

    const int64_t OUT_VAL_NUM = 4096;
    const int64_t X_UB_OFFSET = 512;

    const int64_t OUT_FP16_OFFSET = TRANSE_REP_STRIDE * CHANNEL_BLOCK * sizeof(T);

    int64_t blockIDX = 0;

    GridSampleCommonParam commonParam{};
    IndexBuffer indexBuffer{};

    uint64_t rsvdCnt = 0;
    uint32_t mask = 192;
};

template <typename T>
__aicore__ inline void GridSampler3D<T>::InitBuffer()
{
    // buffer申请初始化 119KB
    pipe.InitBuffer(gridQueue_, 1, GRID_UB_SIZE_4_GENERAL);  // 6KB

    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);                          // 32KB
    pipe.InitBuffer(indexBuffer.inputXYZFPBuf_, GRID_UB_SIZE_4_GENERAL);  // 6KB
    pipe.InitBuffer(inputXIntBuf_, XYZ_UB_SIZE_4_GENERAL);                // 4KB
    pipe.InitBuffer(inputYIntBuf_, XYZ_UB_SIZE_4_GENERAL);                // 4KB
    pipe.InitBuffer(inputZIntBuf_, XYZ_UB_SIZE_4_GENERAL);                // 4KB
    pipe.InitBuffer(inputXFpBuf_, XYZ_UB_SIZE_4_GENERAL);                 // 4KB
    pipe.InitBuffer(inputYFpBuf_, XYZ_UB_SIZE_4_GENERAL);                 // 4KB
    pipe.InitBuffer(inputZFpBuf_, XYZ_UB_SIZE_4_GENERAL);                 // 4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 8);                 // 16KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);              // 8KB
    pipe.InitBuffer(indexBuffer.intTmpBuf_, Y_UB_SIZE_4_GENERAL);         // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);                       // 2KB
    pipe.InitBuffer(indexBuffer.coorTmpBuf_, Y_UB_SIZE_4_GENERAL);        // 2KB
    pipe.InitBuffer(outValueBuf_, X_UB_SIZE_4_GENERAL);                   // 32KB
    pipe.InitBuffer(outValueSumBuf_, X_UB_SIZE_4_GENERAL);                // 32KB

    pipe.InitBuffer(indexBuffer.maskBuf_, 960);  // 960B
    pipe.InitBuffer(maskBuf2_, 960);             // 960B
    pipe.InitBuffer(maskBuf3_, 960);             // 960B
    pipe.InitBuffer(maskBuf4_, 960);             // 960B
    pipe.InitBuffer(maskBuf5_, 960);             // 960B
    pipe.InitBuffer(maskBuf6_, 960);             // 960B
    pipe.InitBuffer(maskBuf7_, 960);             // 960B
    pipe.InitBuffer(maskBuf8_, 960);             // 960B

    pipe.InitBuffer(indexBuffer.weightMaskBuf_, 320);  // 320B
    pipe.InitBuffer(weightMaskBuf2_, 320);             // 320B
    pipe.InitBuffer(weightMaskBuf3_, 320);             // 320B
    pipe.InitBuffer(weightMaskBuf4_, 320);             // 320B
    pipe.InitBuffer(weightMaskBuf5_, 320);             // 320B
    pipe.InitBuffer(weightMaskBuf6_, 320);             // 320B
    pipe.InitBuffer(weightMaskBuf7_, 320);             // 320B
    pipe.InitBuffer(weightMaskBuf8_, 320);             // 320B

    pipe.InitBuffer(indexBuffer.modBuf_, Y_UB_SIZE_4_GENERAL);       // 2KB
    pipe.InitBuffer(indexBuffer.extraBuf_, Y_UB_SIZE_4_GENERAL);     // 2KB
    pipe.InitBuffer(indexBuffer.outTmpBuf_, XYZ_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(bufferMaskXBuf_, BLOCK_SIZE * 6);                // 64B
    pipe.InitBuffer(bufferMaskYBuf_, BLOCK_SIZE * 6);                // 64B
    pipe.InitBuffer(bufferMaskZBuf_, BLOCK_SIZE * 6);                // 64B

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        pipe.InitBuffer(gridFp16Buf_, GRID_UB_SIZE_4_FP16);  // 3KB
    }
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData, commonParam);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    InitBuffer();

    initBufTensor(bufferMaskXBuf_, bufferMaskYBuf_, bufferMaskZBuf_);
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::Clip(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb)
{
    if (commonParam.paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(iXFpUb, iYFpUb, iZFpUb, indexBuffer, commonParam);
    } else if (commonParam.paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(iXFpUb, iYFpUb, iZFpUb, indexBuffer, commonParam);
    }
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::MTE2ForNCHW(
    int32_t nIdx, PointParam pointBilinearParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    for (int32_t i = 0; i < pointBilinearParam.loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(pointBilinearParam.loopOffset + i);
        int64_t baseLocation = nIdx * commonParam.inputC_ * commonParam.inputH_ * commonParam.inputW_ + coordVal +
                               pointBilinearParam.cIdx * CHANNEL_BLOCK * commonParam.inputH_ * commonParam.inputW_;
        for (int cIter = 0; cIter < pointBilinearParam.channelAlign; cIter++) {
            int32_t xLocalOffset = i * pointBilinearParam.channelAlign + cIter;
            if (cIter >= pointBilinearParam.calCElems) {
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
__aicore__ inline void GridSampler3D<T>::MTE2ForNHWC(
    int32_t nIdx, PointParam pointBilinearParam, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    int64_t base = nIdx * commonParam.inputH_ * commonParam.inputW_ * commonParam.inputD_ * commonParam.inputC_ +
                   pointBilinearParam.cIdx * CHANNEL_BLOCK;
    auto timeStep = pointBilinearParam.loopElems / 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = pointBilinearParam.calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    for (int32_t i = 0; i < timeStep; i++) {
        int64_t coordVal_0 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8) * commonParam.inputC_;
        int64_t coordVal_1 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 1) * commonParam.inputC_;
        int64_t coordVal_2 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 2) * commonParam.inputC_;
        int64_t coordVal_3 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 3) * commonParam.inputC_;
        int64_t coordVal_4 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 4) * commonParam.inputC_;
        int64_t coordVal_5 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 5) * commonParam.inputC_;
        int64_t coordVal_6 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 6) * commonParam.inputC_;
        int64_t coordVal_7 = coorUb.GetValue(pointBilinearParam.loopOffset + i * 8 + 7) * commonParam.inputC_;
        int64_t location_0 = base + coordVal_0;
        int64_t location_1 = base + coordVal_1;
        int64_t location_2 = base + coordVal_2;
        int64_t location_3 = base + coordVal_3;
        int64_t location_4 = base + coordVal_4;
        int64_t location_5 = base + coordVal_5;
        int64_t location_6 = base + coordVal_6;
        int64_t location_7 = base + coordVal_7;

        DataCopyPad(xLocal[(i * 8) * pointBilinearParam.channelAlign], gmX_[location_0], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 1) * pointBilinearParam.channelAlign], gmX_[location_1], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 2) * pointBilinearParam.channelAlign], gmX_[location_2], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 3) * pointBilinearParam.channelAlign], gmX_[location_3], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 4) * pointBilinearParam.channelAlign], gmX_[location_4], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 5) * pointBilinearParam.channelAlign], gmX_[location_5], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 6) * pointBilinearParam.channelAlign], gmX_[location_6], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 7) * pointBilinearParam.channelAlign], gmX_[location_7], params, padParams);
    }
    for (auto i = pointBilinearParam.loopElems / 8 * 8; i < pointBilinearParam.loopElems; i++) {
        int64_t coordVal_0 = coorUb.GetValue(pointBilinearParam.loopOffset + i) * commonParam.inputC_;
        int64_t location_0 = base + coordVal_0;
        DataCopyPad(xLocal[i * pointBilinearParam.channelAlign], gmX_[location_0], params, padParams);
    }
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::OutTransposeFp16(
    int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb)
{
    uint64_t dstList[16];
    uint64_t srcList[16];

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
            srcList[i] = (uint64_t)(xLocal[i * 16].GetPhyAddr());
        }

        for (int32_t i = 0; i < 16; i++) {
            dstList[i] = (uint64_t)(outValueUb[i * TRANSE_REP_STRIDE].GetPhyAddr());
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 16;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = (uint64_t)(xLocal[i * channelAlign + j * 16 * channelAlign].GetPhyAddr());
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
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::MTE3ForNCHW(
    ProcessParam processParam, PointParam pointBilinearParam, LocalTensor<T> outValueUb)
{
    int64_t gmYBaseOffset = pointBilinearParam.outBaseOffset + pointBilinearParam.loopOffset +
                            pointBilinearParam.cIdx * CHANNEL_BLOCK * commonParam.gridDHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    uint32_t blockLength = pointBilinearParam.loopElems * sizeof(T);
    if (pointBilinearParam.calCElems == 1) {
        DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, blockLength, 0, 0, 0});

    } else {
        uint32_t srcStride = TRANSE_REP_STRIDE * sizeof(T) / BLOCK_SIZE -
                             ((pointBilinearParam.loopElems * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE);
        uint32_t dstStride = commonParam.gridDHW_ * sizeof(T) - pointBilinearParam.loopElems * sizeof(T);

        DataCopyPad(gmY_[gmYBaseOffset],
            outValueUb,
            {(uint16_t)pointBilinearParam.calCElems, blockLength, srcStride, dstStride, 0});
    }
}

template <typename T>
__aicore__ inline CoordinatesStruct GridSampler3D<T>::GetCoordinatesStruct()
{
    CoordinatesStruct coordinatesStruct{};
    coordinatesStruct.tnwCoordinates = coorBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    coordinatesStruct.tneCoordinates = indexBuffer.inputXYZFPBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    coordinatesStruct.tswCoordinates =
        indexBuffer.inputXYZFPBuf_.GetWithOffset<int32_t>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    coordinatesStruct.tsecoordinates =
        indexBuffer.inputXYZFPBuf_.GetWithOffset<int32_t>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 8);
    coordinatesStruct.bnwCoordinates = indexBuffer.outTmpBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    coordinatesStruct.bneCoordinates =
        indexBuffer.outTmpBuf_.GetWithOffset<int32_t>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    coordinatesStruct.bswCoordinates = indexBuffer.modBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    coordinatesStruct.bsecoordinates = indexBuffer.extraBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    return coordinatesStruct;
}

template <typename T>
__aicore__ inline WeightStruct GridSampler3D<T>::GetWeightStruct()
{
    WeightStruct weightStruct{};
    weightStruct.tnwWeightLocal = weightBuf_.Get<float>(CAL_D_H_W_BLOCK);
    weightStruct.tneWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    weightStruct.tswWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 2 * 4);
    weightStruct.tseWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 3 * 4);
    weightStruct.bnwWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4 * 4);
    weightStruct.bneWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 5 * 4);
    weightStruct.bswWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 6 * 4);
    weightStruct.bseWeightLocal = weightBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 7 * 4);
    return weightStruct;
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::GetWeightMaskStruct()
{
    weightMaskUb = indexBuffer.weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb2 = weightMaskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb3 = weightMaskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb4 = weightMaskBuf4_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb5 = weightMaskBuf5_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb6 = weightMaskBuf6_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb7 = weightMaskBuf7_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUb8 = weightMaskBuf8_.Get<uint8_t>(MASK_UB_SIZE);
    weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();
    weightMaskUbTmp2 = weightMaskUb2.ReinterpretCast<uint64_t>();
    weightMaskUbTmp3 = weightMaskUb3.ReinterpretCast<uint64_t>();
    weightMaskUbTmp4 = weightMaskUb4.ReinterpretCast<uint64_t>();
    weightMaskUbTmp5 = weightMaskUb5.ReinterpretCast<uint64_t>();
    weightMaskUbTmp6 = weightMaskUb6.ReinterpretCast<uint64_t>();
    weightMaskUbTmp7 = weightMaskUb7.ReinterpretCast<uint64_t>();
    weightMaskUbTmp8 = weightMaskUb8.ReinterpretCast<uint64_t>();
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::GetMaskStruct()
{
    maskUb = indexBuffer.maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp = maskUb.ReinterpretCast<uint64_t>();
    maskUb2 = maskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp2 = maskUb2.ReinterpretCast<uint64_t>();
    maskUb3 = maskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp3 = maskUb3.ReinterpretCast<uint64_t>();
    maskUb4 = maskBuf4_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp4 = maskUb4.ReinterpretCast<uint64_t>();
    maskUb5 = maskBuf5_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp5 = maskUb5.ReinterpretCast<uint64_t>();
    maskUb6 = maskBuf6_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp6 = maskUb6.ReinterpretCast<uint64_t>();
    maskUb7 = maskBuf7_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp7 = maskUb7.ReinterpretCast<uint64_t>();
    maskUb8 = maskBuf8_.Get<uint8_t>(MASK_UB_SIZE);
    maskUbTmp8 = maskUb8.ReinterpretCast<uint64_t>();
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::PointBilinearEachChannel(ProcessParam processParam,
    LocalTensor<float> outValueUb, LocalTensor<float> outValueTotalLocal, PointParam pointBilinearParam,
    LocalTensor<T> xLocal)
{
    CoordinatesStruct coordinatesStruct = GetCoordinatesStruct();
    WeightStruct weightStruct = GetWeightStruct();
    GetMaskStruct();
    pointBilinearParam.calCElems = commonParam.perLoopChannel_;
    if (pointBilinearParam.cIdx == commonParam.channelLoop_ - 1) {
        pointBilinearParam.calCElems = commonParam.lastLoopChannel_;
    }
    pointBilinearParam.channelAlign = Ceil(pointBilinearParam.calCElems, B32_ALIGN_FACTOR) * B32_ALIGN_FACTOR;
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        pointBilinearParam.channelAlign = Ceil(pointBilinearParam.calCElems, B16_ALIGN_FACTOR) * B16_ALIGN_FACTOR;
    }

    CalculatePointBilinearParam calculatePointBilinearParam{coordinatesStruct.tnwCoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.tnwWeightLocal,
        maskUbTmp,
        false};
    CalculatePointBilinearParam calculatePointBilinearParam2{coordinatesStruct.tneCoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.tneWeightLocal,
        maskUbTmp2,
        true};
    CalculatePointBilinearParam calculatePointBilinearParam3{coordinatesStruct.tswCoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.tswWeightLocal,
        maskUbTmp3,
        true};
    CalculatePointBilinearParam calculatePointBilinearParam4{coordinatesStruct.tsecoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.tseWeightLocal,
        maskUbTmp4,
        true};
    CalculatePointBilinearParam calculatePointBilinearParam5{coordinatesStruct.bnwCoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.bnwWeightLocal,
        maskUbTmp5,
        true};
    CalculatePointBilinearParam calculatePointBilinearParam6{coordinatesStruct.bneCoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.bneWeightLocal,
        maskUbTmp6,
        true};
    CalculatePointBilinearParam calculatePointBilinearParam7{coordinatesStruct.bswCoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.bswWeightLocal,
        maskUbTmp7,
        true};
    CalculatePointBilinearParam calculatePointBilinearParam8{coordinatesStruct.bsecoordinates,
        outValueUb,
        outValueTotalLocal,
        weightStruct.bseWeightLocal,
        maskUbTmp8,
        true};
    calculatePointBilinear(processParam, calculatePointBilinearParam, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam2, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam3, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam4, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam5, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam6, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam7, pointBilinearParam, xLocal);
    calculatePointBilinear(processParam, calculatePointBilinearParam8, pointBilinearParam, xLocal);

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        auto outValueUbSumFp16 = outValueTotalLocal.ReinterpretCast<T>();
        Cast(outValueUbSumFp16, outValueTotalLocal, RoundMode::CAST_RINT, TRANSE_REP_STRIDE * CHANNEL_BLOCK);
        MTE3ForNCHW(processParam, pointBilinearParam, outValueUbSumFp16);
    } else {
        MTE3ForNCHW(processParam, pointBilinearParam, outValueTotalLocal);
    }

    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventMte3V);
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::PointBilinearSetMask(PointParam pointBilinearParam)
{
    maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp2.SetValue(0, weightMaskUbTmp2.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp2.SetValue(1, weightMaskUbTmp2.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp3.SetValue(0, weightMaskUbTmp3.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp3.SetValue(1, weightMaskUbTmp3.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp4.SetValue(0, weightMaskUbTmp4.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp4.SetValue(1, weightMaskUbTmp4.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp5.SetValue(0, weightMaskUbTmp5.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp5.SetValue(1, weightMaskUbTmp5.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp6.SetValue(0, weightMaskUbTmp6.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp6.SetValue(1, weightMaskUbTmp6.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp7.SetValue(0, weightMaskUbTmp7.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp7.SetValue(1, weightMaskUbTmp7.GetValue(pointBilinearParam.maskOffset + 1));

    maskUbTmp8.SetValue(0, weightMaskUbTmp8.GetValue(pointBilinearParam.maskOffset));
    maskUbTmp8.SetValue(1, weightMaskUbTmp8.GetValue(pointBilinearParam.maskOffset + 1));
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::PointBilinear(
    ProcessParam processParam, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    CoordinatesStruct coordinatesStruct = GetCoordinatesStruct();
    GetWeightMaskStruct();
    WeightStruct weightStruct = GetWeightStruct();
    GetMaskStruct();

    LocalTensor<float> outValueTotalLocal = outValueSumBuf_.Get<float>();

    if (commonParam.paddingMode_ == PADDING_MODE_ZEROS) {
        CoordinatesSelectScalar(
            weightStruct.tnwWeightLocal, weightStruct.tnwWeightLocal, weightMaskUb, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.tneWeightLocal, weightStruct.tneWeightLocal, weightMaskUb2, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.tswWeightLocal, weightStruct.tswWeightLocal, weightMaskUb3, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.tseWeightLocal, weightStruct.tseWeightLocal, weightMaskUb4, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.bnwWeightLocal, weightStruct.bnwWeightLocal, weightMaskUb5, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.bneWeightLocal, weightStruct.bneWeightLocal, weightMaskUb6, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.bswWeightLocal, weightStruct.bswWeightLocal, weightMaskUb7, 0.0f, CAL_D_H_W_BLOCK);
        CoordinatesSelectScalar(
            weightStruct.bseWeightLocal, weightStruct.bseWeightLocal, weightMaskUb8, 0.0f, CAL_D_H_W_BLOCK);
    }

    PointParam pointBilinearParam{};
    int32_t trans_loop = (processParam.calDHWElems + TRANSE_REP_STRIDE - 1) / TRANSE_REP_STRIDE;
    pointBilinearParam.loopElems = TRANSE_REP_STRIDE;
    pointBilinearParam.outBaseOffset =
        processParam.nIdx * commonParam.gridDHW_ * commonParam.inputC_ + processParam.hwIdx * CAL_D_H_W_BLOCK;
    PipeBarrier<PIPE_ALL>();
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            pointBilinearParam.loopElems = processParam.calDHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        pointBilinearParam.loopOffset = loop_idx * TRANSE_REP_STRIDE;
        pointBilinearParam.maskOffset = loop_idx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        PointBilinearSetMask(pointBilinearParam);

        LocalTensor<T> xLocal = xBuf_.Get<T>();
        if (IsSameType<T, bfloat16_t>::value) {
            xLocal = xLocal[TRANSE_REP_STRIDE * CHANNEL_BLOCK];
        }
        for (pointBilinearParam.cIdx = 0; pointBilinearParam.cIdx < commonParam.channelLoop_;
             pointBilinearParam.cIdx++) {
            PointBilinearEachChannel(processParam, outValueUb, outValueTotalLocal, pointBilinearParam, xLocal);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::GetXLocal(ProcessParam processParam,
    CalculatePointBilinearParam calculatePointBilinearParam, PointParam pointBilinearParam, LocalTensor<T> xLocal,
    LocalTensor<float> outValueLocal)
{
    if (commonParam.channelLast_ == LAYOUT_NHWC) {
        MTE2ForNHWC(processParam.nIdx, pointBilinearParam, calculatePointBilinearParam.coordinatesUb, xLocal);
    } else {
        MTE2ForNCHW(processParam.nIdx, pointBilinearParam, calculatePointBilinearParam.coordinatesUb, xLocal);
    }
    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);

    if constexpr (IsSameType<T, half>::value) {  // T: fp16
        LocalTensor<T> outValueFp16Ub =
            outValueBuf_.GetWithOffset<T>(commonParam.perLoopChannel_ * (TRANSE_REP_STRIDE), OUT_FP16_OFFSET);
        OutTransposeFp16(pointBilinearParam.channelAlign, xLocal, outValueFp16Ub);
        PipeBarrier<PIPE_V>();
        Cast(outValueLocal, outValueFp16Ub, RoundMode::CAST_NONE, pointBilinearParam.calCElems * TRANSE_REP_STRIDE);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        LocalTensor xLocalFp32 = xBuf_.Get<float>();
        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, pointBilinearParam.channelAlign * TRANSE_REP_STRIDE);
        OutTransposeFp32(pointBilinearParam.channelAlign, xLocalFp32, outValueLocal);
    } else {  // T: fp32
        OutTransposeFp32(pointBilinearParam.channelAlign, xLocal, outValueLocal);
    }

    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::calculatePointBilinear(ProcessParam processParam,
    CalculatePointBilinearParam calculatePointBilinearParam, PointParam pointBilinearParam, LocalTensor<T> xLocal)
{
    auto outValueLocal = calculatePointBilinearParam.outValueTotalLocal;
    if (calculatePointBilinearParam.isAtomicAdd) {
        outValueLocal = calculatePointBilinearParam.outValueUb;
    }

    GetXLocal(processParam, calculatePointBilinearParam, pointBilinearParam, xLocal, outValueLocal);
    for (size_t i = 0; i < pointBilinearParam.calCElems; i++) {
        int32_t ubOffset = i * TRANSE_REP_STRIDE;
        Select(outValueLocal[ubOffset],
            calculatePointBilinearParam.maskUbTmp,
            outValueLocal[ubOffset],
            0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            TRANSE_REP_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    if (pointBilinearParam.calCElems == 1) {
        // 乘以权重
        Mul(outValueLocal,
            outValueLocal,
            calculatePointBilinearParam.weightUb[pointBilinearParam.loopOffset],
            TRANSE_REP_STRIDE);
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = pointBilinearParam.loopOffset + i * B32_MASK;

            Mul(outValueLocal[outOffset],
                outValueLocal[outOffset],
                calculatePointBilinearParam.weightUb[weightOffset],
                B32_MASK,
                pointBilinearParam.calCElems,
                {1, 1, 1, 16, 16, 0});
        }
    }
    if (calculatePointBilinearParam.isAtomicAdd) {
        Add(calculatePointBilinearParam.outValueTotalLocal,
            calculatePointBilinearParam.outValueTotalLocal,
            outValueLocal,
            pointBilinearParam.calCElems * TRANSE_REP_STRIDE);
    }
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::CalculateGrid(ProcessParam processParam, LocalTensor<float> gridFp32Local,
    LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal)
{
    int64_t gridGmOffset = processParam.nIdx * commonParam.gridDHW_ * 3 + processParam.hwIdx * CAL_D_H_W_BLOCK * 3;
    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = processParam.calDHWElems * 3 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    DataCopyPadExtParams<T> padParamsGrid{false, 0, 0, 0};
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {  // T: fp16
        LocalTensor<T> gridHalfLocal = gridFp16Buf_.Get<T>();
        DataCopyPad(gridHalfLocal, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(gridFp32Local, gridHalfLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK * 3);
        PipeBarrier<PIPE_V>();
    } else {  // T: fp32
        DataCopyPad(gridFp32Local, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    LocalTensor<float> inputXYZUb = indexBuffer.inputXYZFPBuf_.Get<float>();
    Adds(inputXYZUb, gridFp32Local, (float)1.0, CAL_D_H_W_BLOCK * 3);

    uint16_t repeatTime = CAL_D_H_W_BLOCK * 3 / 192;
    LocalTensor<uint32_t> bufZPattern = bufferMaskZBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> bufXPattern = bufferMaskXBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> bufYPattern = bufferMaskYBuf_.Get<uint32_t>();

    PipeBarrier<PIPE_V>();

    // 分别取x和y(inputXFpLocal, inputXYZUb, xPattern, true, mask,
    GatherMask(inputZFpLocal, inputXYZUb, bufZPattern, true, 192, {1, repeatTime, 24, 0}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYZUb, bufYPattern, true, 192, {1, repeatTime, 24, 0}, rsvdCnt);
    GatherMask(inputXFpLocal, inputXYZUb, bufXPattern, true, 192, {1, repeatTime, 24, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (commonParam.alignCorners_ == 1) {
        Muls(inputZFpLocal, inputZFpLocal, (float)((float)0.5 * (commonParam.inputD_ - (float)1.0)), CAL_D_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * (commonParam.inputH_ - (float)1.0)), CAL_D_H_W_BLOCK);
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (commonParam.inputW_ - (float)1.0)), CAL_D_H_W_BLOCK);
    } else {
        Muls(inputZFpLocal, inputZFpLocal, (float)((float)0.5 * commonParam.inputD_), CAL_D_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * commonParam.inputH_), CAL_D_H_W_BLOCK);
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * commonParam.inputW_), CAL_D_H_W_BLOCK);
        PipeBarrier<PIPE_V>();
        Adds(inputXFpLocal, inputXFpLocal, (float)(-0.5), CAL_D_H_W_BLOCK * 3);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::GetInputTensor()
{
    inputXWIntLocal = inputXIntBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    inputXEIntLocal = inputXIntBuf_.GetWithOffset<int32_t>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    inputYWIntLocal = inputYIntBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    inputYEIntLocal = inputYIntBuf_.GetWithOffset<int32_t>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    inputZWIntLocal = inputZIntBuf_.Get<int32_t>(CAL_D_H_W_BLOCK);
    inputZEIntLocal = inputZIntBuf_.GetWithOffset<int32_t>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    inputXWFpLocal = inputXFpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    inputXEFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    inputYWFpLocal = inputYFpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    inputYEFpLocal = inputYFpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    inputZWFpLocal = inputZFpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    inputZEFpLocal = inputZFpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::calculateGridWeight(
    LocalTensor<float> inputXFpLocal, LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal)
{
    WeightStruct weightStruct = GetWeightStruct();
    GetInputTensor();

    LocalTensor<float> weightTmpLocal = weightTmpBuf_.Get<float>(CAL_D_H_W_BLOCK);
    LocalTensor<float> weightTmp1Local = weightTmpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 4);
    LocalTensor<float> weightTmp2Local = weightTmpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 2 * 4);
    LocalTensor<float> weightTmp3Local = weightTmpBuf_.GetWithOffset<float>(CAL_D_H_W_BLOCK, CAL_D_H_W_BLOCK * 3 * 4);

    Sub(weightStruct.tnwWeightLocal, inputXEFpLocal, inputXFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.tneWeightLocal, inputXFpLocal, inputXWFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.tswWeightLocal, inputXEFpLocal, inputXFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.tseWeightLocal, inputXFpLocal, inputXWFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.bnwWeightLocal, inputXEFpLocal, inputXFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.bneWeightLocal, inputXFpLocal, inputXWFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.bswWeightLocal, inputXEFpLocal, inputXFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightStruct.bseWeightLocal, inputXFpLocal, inputXWFpLocal, CAL_D_H_W_BLOCK);

    Sub(weightTmpLocal, inputYEFpLocal, inputYFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightTmp1Local, inputYFpLocal, inputYWFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightTmp2Local, inputZEFpLocal, inputZFpLocal, CAL_D_H_W_BLOCK);
    Sub(weightTmp3Local, inputZFpLocal, inputZWFpLocal, CAL_D_H_W_BLOCK);

    PipeBarrier<PIPE_V>();
    Mul(weightStruct.tnwWeightLocal, weightStruct.tnwWeightLocal, weightTmpLocal, CAL_D_H_W_BLOCK);
    Mul(weightStruct.tneWeightLocal, weightStruct.tneWeightLocal, weightTmpLocal, CAL_D_H_W_BLOCK);
    Mul(weightStruct.tswWeightLocal, weightStruct.tswWeightLocal, weightTmp1Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.tseWeightLocal, weightStruct.tseWeightLocal, weightTmp1Local, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(weightStruct.tnwWeightLocal, weightStruct.tnwWeightLocal, weightTmp2Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.tneWeightLocal, weightStruct.tneWeightLocal, weightTmp2Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.tswWeightLocal, weightStruct.tswWeightLocal, weightTmp2Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.tseWeightLocal, weightStruct.tseWeightLocal, weightTmp2Local, CAL_D_H_W_BLOCK);

    Mul(weightStruct.bnwWeightLocal, weightStruct.bnwWeightLocal, weightTmpLocal, CAL_D_H_W_BLOCK);
    Mul(weightStruct.bneWeightLocal, weightStruct.bneWeightLocal, weightTmpLocal, CAL_D_H_W_BLOCK);
    Mul(weightStruct.bswWeightLocal, weightStruct.bswWeightLocal, weightTmp1Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.bseWeightLocal, weightStruct.bseWeightLocal, weightTmp1Local, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(weightStruct.bnwWeightLocal, weightStruct.bnwWeightLocal, weightTmp3Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.bneWeightLocal, weightStruct.bneWeightLocal, weightTmp3Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.bswWeightLocal, weightStruct.bswWeightLocal, weightTmp3Local, CAL_D_H_W_BLOCK);
    Mul(weightStruct.bseWeightLocal, weightStruct.bseWeightLocal, weightTmp3Local, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::clipAllCoordinate(LocalTensor<float> outValueLocal)
{
    GetInputTensor();
    CoordinatesStruct coordinatesStruct = GetCoordinatesStruct();
    GetWeightMaskStruct();

    InputTensorStruct inputTensorStruct{
        inputXWFpLocal, inputYWFpLocal, inputZWFpLocal, inputXWIntLocal, inputYWIntLocal, inputZWIntLocal};
    InputTensorStruct inputTensorStruct2{
        inputXEFpLocal, inputYWFpLocal, inputZWFpLocal, inputXEIntLocal, inputYWIntLocal, inputZWIntLocal};
    InputTensorStruct inputTensorStruct3{
        inputXWFpLocal, inputYEFpLocal, inputZWFpLocal, inputXWIntLocal, inputYEIntLocal, inputZWIntLocal};
    InputTensorStruct inputTensorStruct4{
        inputXEFpLocal, inputYEFpLocal, inputZWFpLocal, inputXEIntLocal, inputYEIntLocal, inputZWIntLocal};
    InputTensorStruct inputTensorStruct5{
        inputXWFpLocal, inputYWFpLocal, inputZEFpLocal, inputXWIntLocal, inputYWIntLocal, inputZEIntLocal};
    InputTensorStruct inputTensorStruct6{
        inputXEFpLocal, inputYWFpLocal, inputZEFpLocal, inputXEIntLocal, inputYWIntLocal, inputZEIntLocal};
    InputTensorStruct inputTensorStruct7{
        inputXWFpLocal, inputYEFpLocal, inputZEFpLocal, inputXWIntLocal, inputYEIntLocal, inputZEIntLocal};
    InputTensorStruct inputTensorStruct8{
        inputXEFpLocal, inputYEFpLocal, inputZEFpLocal, inputXEIntLocal, inputYEIntLocal, inputZEIntLocal};

    ClipCoordinates(inputTensorStruct, coordinatesStruct.tnwCoordinates, weightMaskUb, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct2, coordinatesStruct.tneCoordinates, weightMaskUb2, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct3, coordinatesStruct.tswCoordinates, weightMaskUb3, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct4, coordinatesStruct.tsecoordinates, weightMaskUb4, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct5, coordinatesStruct.bnwCoordinates, weightMaskUb5, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct6, coordinatesStruct.bneCoordinates, weightMaskUb6, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct7, coordinatesStruct.bswCoordinates, weightMaskUb7, indexBuffer, commonParam);
    ClipCoordinates(inputTensorStruct8, coordinatesStruct.bsecoordinates, weightMaskUb8, indexBuffer, commonParam);
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::PerLoopCompute(ProcessParam processParam)
{
    LocalTensor<float> gridFp32Local = gridQueue_.AllocTensor<float>();
    LocalTensor<float> inputXFpLocal = gridFp32Local;
    LocalTensor<float> inputYFpLocal = gridFp32Local[CAL_D_H_W_BLOCK];
    LocalTensor<float> inputZFpLocal = gridFp32Local[CAL_D_H_W_BLOCK * 2];
    CalculateGrid(processParam, gridFp32Local, inputXFpLocal, inputYFpLocal, inputZFpLocal);
    Clip(inputXFpLocal, inputYFpLocal, inputZFpLocal);

    GetInputTensor();

    Cast(inputXWIntLocal, inputXFpLocal, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    Cast(inputYWIntLocal, inputYFpLocal, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    Cast(inputZWIntLocal, inputZFpLocal, RoundMode::CAST_FLOOR, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(inputXWFpLocal, inputXWIntLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(inputYWFpLocal, inputYWIntLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Cast(inputZWFpLocal, inputZWIntLocal, RoundMode::CAST_NONE, CAL_D_H_W_BLOCK);
    Adds(inputXEIntLocal, inputXWIntLocal, 1, CAL_D_H_W_BLOCK);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, CAL_D_H_W_BLOCK);
    Adds(inputZEIntLocal, inputZWIntLocal, 1, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Adds(inputXEFpLocal, inputXWFpLocal, (float)1.0, CAL_D_H_W_BLOCK);
    Adds(inputYEFpLocal, inputYWFpLocal, (float)1.0, CAL_D_H_W_BLOCK);
    Adds(inputZEFpLocal, inputZWFpLocal, (float)1.0, CAL_D_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    calculateGridWeight(inputXFpLocal, inputYFpLocal, inputZFpLocal);

    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    clipAllCoordinate(outValueLocal);

    PointBilinear(processParam, outValueLocal, true);

    gridQueue_.FreeTensor(gridFp32Local);
}

template <typename T>
__aicore__ inline void GridSampler3D<T>::Process()
{
    if (blockIDX >= commonParam.needCoreNum_) {
        return;
    }
    ProcessParam processParam;
    int32_t preLoopNum = blockIDX * commonParam.preCoreLoop_;

    int64_t loopSize = commonParam.preCoreLoop_;
    if (blockIDX == commonParam.needCoreNum_ - 1) {
        loopSize = commonParam.lastCoreLoop_;
    }

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        processParam.nIdx = (preLoopNum + loopIdx) / commonParam.preNUbLoop_;
        processParam.hwIdx = (preLoopNum + loopIdx) % commonParam.preNUbLoop_;
        processParam.calDHWElems = CAL_D_H_W_BLOCK;
        if (processParam.hwIdx == commonParam.preNUbLoop_ - 1) {
            processParam.calDHWElems = commonParam.lastLoopDHW_;
        }
        PerLoopCompute(processParam);
    }
}

}  // namespace GridSample
#endif  // GIRD_SAMPLER_3D
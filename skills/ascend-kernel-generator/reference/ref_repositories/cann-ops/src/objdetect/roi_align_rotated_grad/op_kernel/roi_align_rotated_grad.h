/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file roi_align_rotated_grad.h
 */
#ifndef _ROI_ALIGN_ROTATED_GRAD_H_
#define _ROI_ALIGN_ROTATED_GRAD_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

static constexpr uint32_t repeatTime32 = 32;
static constexpr uint32_t fourTimes = 4;
static constexpr uint32_t roisDim0 = 0;
static constexpr uint32_t roisDim1 = 1;
static constexpr uint32_t roisDim2 = 2;
static constexpr uint32_t roisDim3 = 3;
static constexpr uint32_t roisDim4 = 4;
static constexpr uint32_t roisDim5 = 5;
static constexpr uint32_t dim0 = 0;
static constexpr uint32_t dim1 = 1;
static constexpr uint32_t dim2 = 2;
static constexpr uint32_t dim4 = 4;
static constexpr float alignForRois = 0.5;

using namespace AscendC;

class KernelRoiAlignRotatedGrad {
public:
    __aicore__ inline KernelRoiAlignRotatedGrad() {}

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR rois, GM_ADDR grad_input,
                                const RoiAlignRotatedGradTilingData *__restrict tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        coreRoisNums = tiling_data->coreRoisNums;
        coreRoisTail = tiling_data->coreRoisTail;
        boxSize = tiling_data->boxSize;
        pooledWidth = tiling_data->pooledWidth;
        pooledHeight = tiling_data->pooledHeight;
        batchSize = tiling_data->batchSize;
        channelNum = tiling_data->channelNum;
        width = tiling_data->width;
        height = tiling_data->height;
        aligned = tiling_data->aligned;
        clockwise = tiling_data->clockwise;
        samplingRatio = tiling_data->samplingRatio;
        spatialScale = tiling_data->spatialScale;
        dataSize = repeatTime32 / sizeof(float);
        alignChannelNum = (channelNum + dataSize - 1) / dataSize;
        alignChannelNum = alignChannelNum * dataSize;

        uint32_t coreId = GetBlockIdx();
        if (coreId < coreRoisTail) {
            coreRoisNums += 1;
            startOffset = coreRoisNums * coreId;
        } else {
            startOffset = coreRoisNums * coreId + coreRoisTail;
        }

        eventIdMte2ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());

        copyParams = {2, static_cast<uint16_t>(channelNum * 2 / dataSize), 0, static_cast<uint16_t>((width - 2) * channelNum / dataSize)};

        roisGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rois), boxLength * boxSize);
        gradOutputsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grad_output), boxLength * channelNum * pooledHeight * pooledWidth);
        gradInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grad_input), batchSize * channelNum * height * width);
        InitBuffer();
    }

    __aicore__ inline void Process()
    {
        GetLocalTensor();
        uint32_t computeBatchSize = constComputeBatchSize;
        uint32_t computeBatchNum = (coreRoisNums + constComputeBatchSize - 1) / constComputeBatchSize;
        for (uint32_t taskBatchIdx = 0; taskBatchIdx < computeBatchNum; taskBatchIdx++) {
            uint32_t offset = startOffset + taskBatchIdx * constComputeBatchSize;
            if (taskBatchIdx == computeBatchNum - 1) {
                computeBatchSize = coreRoisNums - taskBatchIdx * computeBatchSize;
            }
            uint32_t alignComputeBatchNum = (computeBatchSize + dataSize - 1) / dataSize;
            alignComputeBatchNum = alignComputeBatchNum * dataSize;
            CopyIn(offset, alignComputeBatchNum);
            for (uint32_t taskIdx = 0; taskIdx < computeBatchSize; taskIdx++) {
                Compute(taskIdx, offset + taskIdx);
            }
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe.InitBuffer(idxUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(xUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(yUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(hUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(wUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(angleUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(cosUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(sinUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(binSizeHUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(binSizeWUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(binGridWUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(binGridHUb, constComputeBatchSize * sizeof(int32_t));

        pipe.InitBuffer(binGridSizeWUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(binGridSizeHUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(deltaStartWUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(deltaStartHUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(countTmpUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(countUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(tmpUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(gradBinUb, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW1Ub, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW2Ub, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW3Ub, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW4Ub, alignChannelNum * sizeof(float));

        pipe.InitBuffer(gradOutUb, fourTimes * alignChannelNum * sizeof(float));

        pipe.InitBuffer(tmpChannelUb, alignChannelNum * sizeof(float));
    }

    __aicore__ inline void GetLocalTensor()
    {
        idxLocal = idxUb.Get<int32_t>();
        xLocal = xUb.Get<float>();
        yLocal = yUb.Get<float>();
        hLocal = hUb.Get<float>();
        wLocal = wUb.Get<float>();
        angleLocal = angleUb.Get<float>();

        cosLocal = cosUb.Get<float>();
        sinLocal = sinUb.Get<float>();

        binSizeHLocal = binSizeHUb.Get<float>();
        binSizeWLocal = binSizeWUb.Get<float>();

        binGridWLocal = binGridWUb.Get<int32_t>();
        binGridHLocal = binGridHUb.Get<int32_t>();

        binGridSizeWLocal = binGridSizeWUb.Get<float>();
        binGridSizeHLocal = binGridSizeHUb.Get<float>();

        deltaStartWLocal = deltaStartWUb.Get<float>();
        deltaStartHLocal = deltaStartHUb.Get<float>();

        countTmpLocal = countTmpUb.Get<int32_t>();
        countLocal = countUb.Get<float>();
        tmpLocal = tmpUb.Get<float>();

        gradBinLocal = gradBinUb.Get<float>();
        gradW1Local = gradW1Ub.Get<float>();
        gradW2Local = gradW2Ub.Get<float>();
        gradW3Local = gradW3Ub.Get<float>();
        gradW4Local = gradW4Ub.Get<float>();

        gradOutLocal = gradOutUb.Get<float>();

        tmpChannelLocal = tmpChannelUb.Get<float>();
        Duplicate(tmpChannelLocal, (float)0.0, alignChannelNum);
        Duplicate(tmpChannelLocal, (float)1.0, channelNum);
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t computeBatchSize)
    {
        DataCopy(tmpLocal, roisGM[offset + boxSize * roisDim0], computeBatchSize);
        DataCopy(xLocal, roisGM[offset + boxSize * roisDim1], computeBatchSize);
        DataCopy(yLocal, roisGM[offset + boxSize * roisDim2], computeBatchSize);
        DataCopy(wLocal, roisGM[offset + boxSize * roisDim3], computeBatchSize);
        DataCopy(hLocal, roisGM[offset + boxSize * roisDim4], computeBatchSize);
        DataCopy(angleLocal, roisGM[offset + boxSize * roisDim5], computeBatchSize);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (clockwise) {
            Muls(angleLocal, angleLocal, (float)-1.0, computeBatchSize);
        }

        Cast(idxLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, computeBatchSize);
        Muls(xLocal, xLocal, (float)spatialScale, computeBatchSize);
        Muls(yLocal, yLocal, (float)spatialScale, computeBatchSize);
        Muls(hLocal, hLocal, (float)spatialScale, computeBatchSize);
        Muls(wLocal, wLocal, (float)spatialScale, computeBatchSize);

        if (aligned) {
            Adds(xLocal, xLocal, -alignForRois, computeBatchSize);
            Adds(yLocal, yLocal, -alignForRois, computeBatchSize);
        } else {
            Maxs(hLocal, hLocal, (float)1.0, computeBatchSize);
            Maxs(wLocal, wLocal, (float)1.0, computeBatchSize);
        }

        Cos(cosLocal, angleLocal, computeBatchSize);
        Sin(sinLocal, angleLocal, computeBatchSize);

        Duplicate(tmpLocal, (float)pooledHeight, computeBatchSize);
        Div(binSizeHLocal, hLocal, tmpLocal, computeBatchSize);
        Duplicate(tmpLocal, (float)pooledWidth, computeBatchSize);
        Div(binSizeWLocal, wLocal, tmpLocal, computeBatchSize);

        if (samplingRatio > 0) {
            Duplicate(binGridHLocal, samplingRatio, computeBatchSize);
            Duplicate(binGridWLocal, samplingRatio, computeBatchSize);
        } else {
            Cast(binGridHLocal, binSizeHLocal, AscendC::RoundMode::CAST_CEIL, computeBatchSize);
            Cast(binGridWLocal, binSizeWLocal, AscendC::RoundMode::CAST_CEIL, computeBatchSize);
        }

        Cast(tmpLocal, binGridHLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Div(binGridSizeHLocal, binSizeHLocal, tmpLocal, computeBatchSize);

        Cast(tmpLocal, binGridWLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Div(binGridSizeWLocal, binSizeWLocal, tmpLocal, computeBatchSize);

        Muls(deltaStartWLocal, wLocal, -alignForRois, computeBatchSize);
        Muls(deltaStartHLocal, hLocal, -alignForRois, computeBatchSize);

        Mul(countTmpLocal, binGridWLocal, binGridHLocal, computeBatchSize);
        Cast(countLocal, countTmpLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Maxs(countLocal, countLocal, (float)1.0, computeBatchSize);
    }

    __aicore__ inline void Compute(uint32_t taskIdx, uint32_t offset)
    {
        pIdx = idxLocal.GetValue(taskIdx);
        pX = xLocal.GetValue(taskIdx);
        pY = yLocal.GetValue(taskIdx);
        pCos = cosLocal.GetValue(taskIdx);
        pSin = sinLocal.GetValue(taskIdx);

        pBinSizeW = binSizeWLocal.GetValue(taskIdx);
        pBinSizeH = binSizeHLocal.GetValue(taskIdx);
        pBinGridW = binGridWLocal.GetValue(taskIdx);
        pBinGridH = binGridHLocal.GetValue(taskIdx);

        pDeltaStartW = deltaStartWLocal.GetValue(taskIdx);
        pDeltaStartH = deltaStartHLocal.GetValue(taskIdx);

        pBinGridSizeH = binGridSizeHLocal.GetValue(taskIdx);
        pBinGridSizeW = binGridSizeWLocal.GetValue(taskIdx);

        pCount = countLocal.GetValue(taskIdx);

        for (index = 0; index < pooledHeight * pooledWidth; index++) {
            pH = index / pooledWidth;
            pW = index - pH * pooledWidth;
            baseOffset = ((offset * pooledHeight + pH) * pooledWidth + pW) * channelNum;

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            DataCopy(gradBinLocal, gradOutputsGm[baseOffset], alignChannelNum);

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if (alignChannelNum != channelNum) {
                Mul(gradBinLocal, gradBinLocal, tmpChannelLocal, alignChannelNum);
            }
            Muls(gradBinLocal, gradBinLocal, (float)1.0 / (float)pCount, alignChannelNum);

            for (iy = 0; iy < pBinGridH; iy++) {
                yy = pDeltaStartH + pH * pBinSizeH + (iy + alignForRois) * pBinGridSizeH;
                for (ix = 0; ix < pBinGridW; ix++) {
                    xx = pDeltaStartW + pW * pBinSizeW + (ix + alignForRois) * pBinGridSizeW;

                    x = yy * pSin + xx * pCos + pX;
                    y = yy * pCos - xx * pSin + pY;

                    bilinearInterpolate();

                    if (xl >= 0 && xh >= 0 && yl >= 0 && yh >= 0) {
                        if (channelNum == alignChannelNum && xh > xl && yh > yl) {
                            CopyOutTogether();
                        } else {
                            CopyOut();
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void bilinearInterpolate()
    {
        if (y < -1 || y > height || x < -1 || x > width) {
            xl = -1;
            return ;
        }
        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        yl = static_cast<int32_t>(y);
        xl = static_cast<int32_t>(x);

        if (yl >= height - 1) {
            yl = yh = height - 1;
            y = float(yl);
        } else {
            yh = yl + 1;
        }

        if (xl >= width - 1) {
            xl = xh = width - 1;
            x = float(xl);
        } else {
            xh = xl + 1;
        }

        ly = y - yl;
        lx = x - xl;

        w4 = ly * lx;
        w1 = w4 + 1 - ly - lx;
        w2 = lx - w4;
        w3 = ly - w4;
    }

    __aicore__ inline void CopyOut()
    {
        w1Offset = ((pIdx * height+ yl)* width + xl) * channelNum;
        w2Offset = ((pIdx * height+ yl)* width + xh) * channelNum;
        w3Offset = ((pIdx * height+ yh)* width + xl) * channelNum;
        w4Offset = ((pIdx * height+ yh)* width + xh) * channelNum;

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(gradW1Local, gradBinLocal, w1, alignChannelNum);
        Muls(gradW2Local, gradBinLocal, w2, alignChannelNum);
        Muls(gradW3Local, gradBinLocal, w3, alignChannelNum);
        Muls(gradW4Local, gradBinLocal, w4, alignChannelNum);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();

        DataCopy(gradInputGm[w1Offset], gradW1Local, alignChannelNum);
        DataCopy(gradInputGm[w2Offset], gradW2Local, alignChannelNum);
        DataCopy(gradInputGm[w3Offset], gradW3Local, alignChannelNum);
        DataCopy(gradInputGm[w4Offset], gradW4Local, alignChannelNum);

        SetAtomicNone();
    }

    __aicore__ inline void CopyOutTogether()
    {
        w1Offset = ((pIdx * height+ yl)* width + xl) * channelNum;

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(gradOutLocal, gradBinLocal, w1, channelNum);
        Muls(gradOutLocal[channelNum], gradBinLocal, w2, channelNum);
        Muls(gradOutLocal[channelNum * dim2], gradBinLocal, w3, channelNum);
        Muls(gradOutLocal[channelNum * roisDim3], gradBinLocal, w4, channelNum);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();

        DataCopy(gradInputGm[w1Offset], gradOutLocal, copyParams);

        SetAtomicNone();
    }
private:
    TPipe pipe;
    GlobalTensor<float> roisGM;
    GlobalTensor<float> gradOutputsGm;
    GlobalTensor<float> gradInputGm;

    TBuf <TPosition::VECCALC> idxUb;
    TBuf <TPosition::VECCALC> xUb;
    TBuf <TPosition::VECCALC> yUb;
    TBuf <TPosition::VECCALC> hUb;
    TBuf <TPosition::VECCALC> wUb;
    TBuf <TPosition::VECCALC> angleUb;
    TBuf <TPosition::VECCALC> cosUb;
    TBuf <TPosition::VECCALC> sinUb;
    TBuf <TPosition::VECCALC> binSizeHUb;
    TBuf <TPosition::VECCALC> binSizeWUb;
    TBuf <TPosition::VECCALC> binGridWUb;
    TBuf <TPosition::VECCALC> binGridHUb;
    TBuf <TPosition::VECCALC> binGridSizeWUb;
    TBuf <TPosition::VECCALC> binGridSizeHUb;
    TBuf <TPosition::VECCALC> deltaStartWUb;
    TBuf <TPosition::VECCALC> deltaStartHUb;
    TBuf <TPosition::VECCALC> countTmpUb;
    TBuf <TPosition::VECCALC> countUb;
    TBuf <TPosition::VECCALC> tmpUb;

    TBuf <TPosition::VECCALC> gradBinUb;
    TBuf <TPosition::VECCALC> gradW1Ub;
    TBuf <TPosition::VECCALC> gradW2Ub;
    TBuf <TPosition::VECCALC> gradW3Ub;
    TBuf <TPosition::VECCALC> gradW4Ub;
    TBuf <TPosition::VECCALC> tmpChannelUb;
    TBuf <TPosition::VECCALC> gradOutUb;

    LocalTensor<int32_t> idxLocal;
    LocalTensor<float> xLocal;
    LocalTensor<float> yLocal;
    LocalTensor<float> hLocal;
    LocalTensor<float> wLocal;
    LocalTensor<float> angleLocal;
    LocalTensor<float> cosLocal;
    LocalTensor<float> sinLocal;
    LocalTensor<float> binSizeHLocal;
    LocalTensor<float> binSizeWLocal;
    LocalTensor<int32_t> binGridWLocal;
    LocalTensor<int32_t> binGridHLocal;
    LocalTensor<float> binGridSizeWLocal;
    LocalTensor<float> binGridSizeHLocal;
    LocalTensor<float> deltaStartWLocal;
    LocalTensor<float> deltaStartHLocal;
    LocalTensor<int32_t> countTmpLocal;
    LocalTensor<float> countLocal;
    LocalTensor<float> tmpLocal;

    LocalTensor<float> gradBinLocal;
    LocalTensor<float> gradW1Local;
    LocalTensor<float> gradW2Local;
    LocalTensor<float> gradW3Local;
    LocalTensor<float> gradW4Local;
    LocalTensor<float> tmpChannelLocal;
    LocalTensor<float> gradOutLocal;

    uint32_t coreRoisNums;
    uint32_t coreRoisTail;
    uint32_t boxSize;
    uint32_t boxLength = 6;
    uint32_t batchSize;
    uint32_t channelNum;
    uint32_t width;
    uint32_t height;
    int32_t pooledWidth;
    int32_t pooledHeight;
    bool aligned;
    bool clockwise;
    int32_t samplingRatio;
    float spatialScale;

    uint32_t dataSize;
    uint32_t alignChannelNum;

    uint32_t startOffset;
    uint32_t baseOffset;
    uint32_t w1Offset;
    uint32_t w2Offset;
    uint32_t w3Offset;
    uint32_t w4Offset;
    uint32_t constComputeBatchSize = 256;

    int32_t pIdx;
    int32_t index;
    float pX;
    float pY;
    int32_t pH;
    int32_t pW;
    float pCos;
    float pSin;

    float pBinSizeW;
    float pBinSizeH;
    int32_t pBinGridW;
    int32_t pBinGridH;
    float pDeltaStartW;
    float pDeltaStartH;
    float pBinGridSizeH;
    float pBinGridSizeW;

    float pCount;

    float tmpH;
    float tmpW;
    float tmpPH;
    float tmpPW;

    int32_t ix;
    int32_t iy;
    float xx;
    float yy;
    float x;
    float y;
    int32_t xl;
    int32_t xh;
    int32_t yl;
    int32_t yh;

    float lx;
    float hx;
    float ly;
    float hy;

    float w1;
    float w2;
    float w3;
    float w4;

    event_t eventIdVToMte2;
    event_t eventIdVToMte3;
    event_t eventIdMte2ToV;
    event_t eventIdMte3ToV;
    event_t eventIdMte3ToMte2;
    AscendC::DataCopyParams copyParams;
};
#endif // ROI_ALIGN_ROTATED_GRAD_H
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ms_deform_attn_generic.h
 * \brief
 */
#ifndef MS_DEFORM_ATTN_GENERIC_H
#define MS_DEFORM_ATTN_GENERIC_H

#include "kernel_operator.h"
using namespace AscendC;

class KernelMultiScaleDeformableAttn {
public:

    __aicore__ inline KernelMultiScaleDeformableAttn() {}
    __aicore__ inline void Init(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valuLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnFunctionTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        coreNum = tiling_data->coreNum;

        numLevelsAlign = AlignUp(numLevels, dataAlign);

        maxUbNum = (useUbSize / sizeof(DTYPE_VALUE) - threeBuffer * numLevelsAlign
                                     - fourBuffer * embedDims) / (numQuerieBuffer + numEmbedBuffer * embedDims);
        maxUbNum = maxUbNum / dataAlign * dataAlign;
        numQueriesper = DivCeil(numQueries, maxUbNum);
        numQueriestail = numQueries - (numQueriesper - 1) * maxUbNum;
        numQueriestail = AlignUp(numQueriestail, dataAlign);
        numQueriesAlign = numQueries <= maxUbNum ? numQueriestail : maxUbNum;

        taskNum = batchSize * numHeads * numLevels * numPoints * numQueriesper;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        weightStride0 = numQueries;
        weightStride1 = numPoints * weightStride0;
        weightStride2 = numLevels * weightStride1;
        weightStride3 = numHeads * weightStride2;
        
        valueStride0 = embedDims;
        valueStride1 = numKeys * valueStride0;
        valueStride2 = numHeads * valueStride1;
        wStride = embedDims;

        outputStride0 = embedDims;
        outputStride1 = numHeads * outputStride0;
        outputStride2 = numQueries * outputStride1;

        copyOutParams = {1, (uint32_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0, 0};
        copyInParamsV3 = {1, (uint32_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0, 0};

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value),
                                batchSize * numKeys * numHeads * embedDims);
        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES *>(valueSpatialShapes),
                                             numLevels * TWO);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES *>(valuLevelStartIndex),
                                               numLevels);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(samplingLocations),
                                   batchSize * numQueries * numHeads * numLevels * numPoints * TWO);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attentionWeights),
                                           batchSize * numQueries * numHeads * numLevels * numPoints);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(output),
                                    batchSize * numQueries * numHeads * embedDims);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, twoBuffer * numLevelsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(zerosUb, fourBuffer * numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightsUb, numQueriesAlign * sizeof(DTYPE_VALUE));
   
        pipe->InitBuffer(locWUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locHUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(imUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(lowUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(lowFloatUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(distLowUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4Ub, numQueriesAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(wv1Ub, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv2Ub, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv3Ub, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv4Ub, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(outputUb, numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void GetLocalTensor()
    {
        attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();
        shapesLocal = shapeUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        offsetLocal = offsetUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();

        locWLocal = locWUb.Get<DTYPE_VALUE>();
        locHLocal = locHUb.Get<DTYPE_VALUE>();

        imLocal = imUb.Get<DTYPE_VALUE>();
        lowLocal = lowUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        lowFloatLocal = lowFloatUb.Get<DTYPE_VALUE>();
        zerosLocal = zerosUb.Get<DTYPE_VALUE>();
        distLowLocal = distLowUb.Get<DTYPE_VALUE>();

        w4Local = w4Ub.Get<DTYPE_VALUE>();
        wv1Local = wv1Ub.Get<DTYPE_VALUE>();
        wv2Local = wv2Ub.Get<DTYPE_VALUE>();
        wv3Local = wv3Ub.Get<DTYPE_VALUE>();
        wv4Local = wv4Ub.Get<DTYPE_VALUE>();

        outputLocal = outputUb.Get<DTYPE_VALUE>();
    }
    
    __aicore__ inline void ClearOutput()
    {
        InitOutput<DTYPE_VALUE>(outputGm, batchSize * numQueries * numHeads * embedDims, 0);
        if ASCEND_IS_AIV {
            SyncAll();
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t startIdx = startOffset;
        nqloop = startIdx % numQueriesper;
        startIdx = startIdx / numQueriesper;
        point = startIdx % numPoints;
        startIdx = startIdx / numPoints;
        level = startIdx % numLevels;
        startIdx = startIdx / numLevels;
        head = startIdx % numHeads;
        batch = startIdx / numHeads;

        DataCopyPad(shapesLocal, valueSpatialShapesGm, {1, (uint32_t)(TWO * numLevels * sizeof(DTYPE_VALUE_SPATIAL_SHAPES)), 0, 0, 0}, padParamsInt);
        DataCopyPad(offsetLocal, valueLevelStartIndexGm, {1, (uint32_t)(numLevels * sizeof(DTYPE_VALUE_SPATIAL_SHAPES)), 0, 0, 0}, padParamsInt);
        
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        for (uint64_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    }

private:
    __aicore__ inline void ComputeGradSeparate(DTYPE_VALUE distHH, DTYPE_VALUE distHW,
                                               DTYPE_VALUE distLH, DTYPE_VALUE distLW,
                                               DTYPE_VALUE w1, DTYPE_VALUE w2,
                                               DTYPE_VALUE w3, DTYPE_VALUE w4,
                                               DTYPE_VALUE attentionWeight)
    {
        if (hLow >= 0 && wLow >= 0) {
            DataCopyPad(zerosLocal[v1Id * embedDims + queryOffsetv], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], 
                        copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv], w1, embedDims);
        }
        if (hLow >= 0 && wLow < w - 1) {
            DataCopyPad(zerosLocal[v2Id * embedDims + queryOffsetv], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], 
            copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv], w2, embedDims);
        }
        if (hLow < h - 1 && wLow >= 0) {
            DataCopyPad(zerosLocal[v3Id * embedDims + queryOffsetv], valueGm[offsetValue + hLowPtrOffset + hStride + wLowPtrOffset], 
            copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv], w3, embedDims);
        }
        if (hLow < h - 1 && wLow < w - 1) {
            DataCopyPad(zerosLocal[v4Id * embedDims + queryOffsetv], valueGm[offsetValue + hLowPtrOffset + hStride + wLowPtrOffset + wStride], 
            copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv], w4, embedDims);
        }

        Add(wv1Local, wv1Local, wv2Local, embedDims);
        Add(wv3Local, wv3Local, wv4Local, embedDims);
        Add(wv1Local, wv1Local, wv3Local, embedDims);
        Muls(outputLocal[queryOffset], wv1Local, attentionWeight, embedDims);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        DataCopyPad(outputGm[batch * outputStride2 + (nqloop * maxUbNum + query) * outputStride1 + head * outputStride0], outputLocal[queryOffset], copyOutParams);
    }

    __aicore__ inline void ComputeGradTogether(DTYPE_VALUE distLH, DTYPE_VALUE distLW,
                                               DTYPE_VALUE w1, DTYPE_VALUE w2, DTYPE_VALUE w3, DTYPE_VALUE w4,
                                               DTYPE_VALUE attentionWeight)
    {
        DataCopyPad(zerosLocal[queryOffsetv], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], 
            copyInParamsV1, padParamsFloat);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv], w1, embedDims);
        Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv], w2, embedDims);
        Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv], w3, embedDims);
        Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv], w4, embedDims);

        Add(wv1Local, wv1Local, wv2Local, embedDims);
        Add(wv3Local, wv3Local, wv4Local, embedDims);
        Add(wv1Local, wv1Local, wv3Local, embedDims);
        Muls(outputLocal[queryOffset], wv1Local, attentionWeight, embedDims);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(outputGm[batch * outputStride2 + (nqloop * maxUbNum + query) * outputStride1 + head * outputStride0], outputLocal[queryOffset], copyOutParams);
    }

    __aicore__ inline void GridSampleCompute()
    {
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetAtomicAdd<DTYPE_VALUE>();
        for (query = 0; query < thisCycleNum; query++) {
            queryOffset = query * embedDims;
            queryOffsetv = query * fourBuffer * embedDims;
            hIm = imLocal.GetValue(thisCycleNumAlign + query);
            wIm = imLocal.GetValue(query);
            if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                hLow = lowLocal.GetValue(thisCycleNumAlign + query);
                wLow = lowLocal.GetValue(query);
                hLowPtrOffset = hLow * hStride;
                wLowPtrOffset = wLow * wStride;
                DTYPE_VALUE distH = distLowLocal.GetValue(thisCycleNumAlign + query);
                DTYPE_VALUE distW = distLowLocal.GetValue(query);
                DTYPE_VALUE attenWeight = attentionWeightLocal.GetValue(query);
                w4 = w4Local.GetValue(query);
                w1 = w4 + 1 - distH - distW;
                w2 = distW - w4;
                w3 = distH - w4;
                    
                if (hLow >= 0 && wLow >= 0 && hLow < h - 1 && wLow < w - 1) {
                    ComputeGradTogether(distH, distW, w1, w2, w3, w4, attenWeight);
                } else {
                    Duplicate(wv1Local, (DTYPE_VALUE)0, embedDims);
                    Duplicate(wv2Local, (DTYPE_VALUE)0, embedDims);
                    Duplicate(wv3Local, (DTYPE_VALUE)0, embedDims);
                    Duplicate(wv4Local, (DTYPE_VALUE)0, embedDims);
                    ComputeGradSeparate(1 - distH, 1 - distW, distH, distW, w1, w2, w3, w4, attenWeight);
                }
            }
        }
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    __aicore__ inline void ScalarUpdate(uint64_t taskIdx)
    {
        if (taskIdx > startOffset) {
            nqloop += 1;
            point += (nqloop / numQueriesper);
            nqloop %= numQueriesper;
            level += (point / numPoints);
            point %= numPoints;
            head += (level / numLevels);
            level %= numLevels;
            batch += (head / numHeads);
            head %= numHeads;
        }
        levelStartId = offsetLocal.GetValue(level);
        h = shapesLocal.GetValue(level * TWO);
        w = shapesLocal.GetValue(level * TWO + 1);

        offsetWeight = batch * weightStride3 + head * weightStride2 + level * weightStride1 + point  * weightStride0;
        offsetLocation = TWO * offsetWeight;
        thisCycleNumAlign = (nqloop == numQueriesper -1) ? numQueriestail : maxUbNum;
        thisCycleNum = (nqloop == numQueriesper -1) ? (numQueries - (numQueriesper - 1) * maxUbNum) : maxUbNum;
        offsetValue = batch * valueStride2 + head * valueStride1 + levelStartId * valueStride0;
        hStride = w * wStride;
        copyInParamsV1 = {2, (uint32_t)(2 * embedDims * sizeof(DTYPE_VALUE)), (uint32_t)((hStride - 2 * embedDims) * sizeof(DTYPE_VALUE)), 0, 0};
        copyInParamsV2 = {1, (uint32_t)(thisCycleNum * sizeof(DTYPE_VALUE)), 0, 0, 0};
    }

    __aicore__ inline void Compute(uint64_t taskIdx)
    {
        ScalarUpdate(taskIdx);
        
        Duplicate(zerosLocal, (DTYPE_VALUE)0, fourBuffer * numQueriesAlign * embedDims);
        DataCopyPad(locWLocal, locationGm[offsetLocation + nqloop * maxUbNum],  copyInParamsV2, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(imLocal, locWLocal, (DTYPE_VALUE)w, thisCycleNumAlign);

        DataCopyPad(locHLocal, locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyInParamsV2, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(imLocal[thisCycleNumAlign], locHLocal, (DTYPE_VALUE)h, thisCycleNumAlign);

        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        DataCopyPad(attentionWeightLocal, attentionWeightsGm[offsetWeight + nqloop * maxUbNum], copyInParamsV2, padParamsFloat);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV); 
        Adds(imLocal, imLocal, CONV_CONSTANT, TWO * thisCycleNumAlign); 
        Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, TWO * thisCycleNumAlign); 
        Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, TWO * thisCycleNumAlign);
        Sub(distLowLocal, imLocal, lowFloatLocal, TWO * thisCycleNumAlign);
        Mul(w4Local, distLowLocal[thisCycleNumAlign], distLowLocal, thisCycleNumAlign);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        GridSampleCompute();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<DTYPE_VALUE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> attentionWeightsUb, shapeUb, offsetUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, lowFloatUb;
    TBuf<TPosition::VECCALC> zerosUb, distLowUb, w4Ub;
    TBuf<TPosition::VECCALC> wv1Ub, wv2Ub, wv3Ub, wv4Ub, outputUb;

    uint32_t coreNum;
    uint32_t curBlockIdx;
    uint32_t taskNum, taskNumPerCore;
    uint32_t dataAlign, blockBytes = 32;
    uint32_t queryOffset, queryOffsetv;
    uint32_t v1Id = 0, v2Id = 1, v3Id = 2, v4Id = 3;
    uint32_t thisCycleNum, thisCycleNumAlign, maxUbNum;
    uint32_t twoBuffer = 2, TWO = 2, threeBuffer = 3, fourBuffer = 4, eightBuffer = 8;
    uint32_t numQuerieBuffer= 12, numEmbedBuffer = 5;
    uint32_t useUbSize = 190 * 1024;
    uint64_t batchSize, numKeys, numHeads, embedDims, numLevels, numQueries, numPoints;
    uint64_t numQueriesAlign, numQueriesper, numQueriestail, numLevelsAlign;
    uint64_t batch, query, head, level, point, nqloop;
    uint64_t startOffset, endOffset;
    uint64_t weightStride0, weightStride1, weightStride2, weightStride3;
    uint64_t valueStride0, valueStride1, valueStride2;
    uint64_t outputStride0, outputStride1, outputStride2;
    uint64_t levelStartId;
    uint64_t offsetValue, offsetWeight, offsetLocation, wStride, hStride;
    uint64_t hLowPtrOffset, wLowPtrOffset;
    int64_t h, w, hLow, wLow;
    
    DTYPE_VALUE hIm, wIm;
    DTYPE_VALUE w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    DTYPE_VALUE CONV_CONSTANT = -0.5;

    LocalTensor<DTYPE_VALUE> lowFloatLocal;
    LocalTensor<DTYPE_VALUE> distLowLocal;
    LocalTensor<DTYPE_VALUE> locWLocal, locHLocal;
    LocalTensor<DTYPE_VALUE> imLocal;
    LocalTensor<DTYPE_VALUE> zerosLocal;
    LocalTensor<DTYPE_VALUE> attentionWeightLocal;
    LocalTensor<DTYPE_VALUE> wv1Local, wv2Local, wv3Local, wv4Local, w4Local;
    LocalTensor<DTYPE_VALUE> outputLocal;
    LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal, offsetLocal;
    LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> lowLocal;

    DataCopyExtParams copyInParamsV1, copyInParamsV2, copyInParamsV3, copyOutParams;
    DataCopyPadExtParams<DTYPE_VALUE_SPATIAL_SHAPES> padParamsInt {false, 0, 0, 0};
    DataCopyPadExtParams<DTYPE_VALUE> padParamsFloat {false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
};

#endif // MS_DEFORM_ATTN_GENERIC_H
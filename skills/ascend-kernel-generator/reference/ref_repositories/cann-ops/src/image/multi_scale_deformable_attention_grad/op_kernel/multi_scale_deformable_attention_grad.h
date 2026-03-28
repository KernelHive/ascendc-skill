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
 * \file multi_scale_deformable_attention_grad.h
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

class MultiScaleDeformableAttentionGrad {
public:
    __aicore__ inline MultiScaleDeformableAttentionGrad(){};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                            GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm,
                            const MultiScaleDeformableAttentionGradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        maxUbNum = tiling_data->maxUbNum;
        coreNum = tiling_data->coreNum;

        numLevelsAlign = AlignUp(numLevels, dataAlign);
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

        gradOutStride0 = embedDims;
        gradOutStride1 = numHeads * gradOutStride0;
        gradOutStride2 = numQueries * gradOutStride1;

        weightStride0 = numQueries;
        weightStride1 = numPoints * weightStride0;
        weightStride2 = numLevels * weightStride1;
        weightStride3 = numHeads * weightStride2;
        
        valueStride0 = embedDims;
        valueStride1 = numHeads * valueStride0;
        valueStride2 = numKeys * valueStride1;
        wStride = numHeads * embedDims;

        baseOffsetUb = numQueriesAlign * embedDims;
        copyOutParams = {1, (uint32_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0, 0};
        copyInParams = {1, (uint32_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0, 0};

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMteWeight = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3X = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3Y = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_S>());

        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value_gm),
                                batchSize * numKeys * numHeads * embedDims);
        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(spatial_shapes_gm),
                                             numLevels * TWO);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(level_start_index_gm),
                                               numLevels);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(sampling_loc_gm),
                                   batchSize * numQueries * numHeads * numLevels * numPoints * TWO);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attn_weight_gm),
                                           batchSize * numQueries * numHeads * numLevels * numPoints);
        gradOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_output_gm),
                                     batchSize * numQueries * numHeads * embedDims);

        gradValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_value_gm),
                                    batchSize * numKeys * numHeads * embedDims);
        gradLocationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_sampling_loc_gm),
                                       batchSize * numQueries * numHeads * numLevels * TWO * numPoints);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_attn_weight_gm),
                                     batchSize * numQueries * numHeads * numLevels * numPoints);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, twoBuffer * numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(zerosUb, eightBuffer * numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradUb, numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid1Ub, numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid2Ub, numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid3Ub, numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid4Ub, numQueriesAlign * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(attentionWeightsUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpXUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightSumUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locWUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locHUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(imUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(lowUb, twoBuffer * numQueriesAlign * sizeof(int32_t));
        pipe->InitBuffer(lowFloatUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(distLowUb, twoBuffer * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4Ub, numQueriesAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpAUb, twoBuffer * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpBUb, twoBuffer * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv1Ub, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv2Ub, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv3Ub, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv4Ub, embedDims * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void GetLocalTensor()
    {
        attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();
        shapesLocal = shapeUb.Get<int32_t>();
        offsetLocal = offsetUb.Get<int32_t>();
        xLocal = tmpXUb.Get<DTYPE_VALUE>();
        yLocal = tmpYUb.Get<DTYPE_VALUE>();
        weightSumLocal = weightSumUb.Get<DTYPE_VALUE>();
        topGradLocal = topGradUb.Get<DTYPE_VALUE>();
        locWLocal = locWUb.Get<DTYPE_VALUE>();
        locHLocal = locHUb.Get<DTYPE_VALUE>();

        imLocal = imUb.Get<DTYPE_VALUE>();
        lowLocal = lowUb.Get<int32_t>();
        lowFloatLocal = lowFloatUb.Get<DTYPE_VALUE>();
        zerosLocal = zerosUb.Get<DTYPE_VALUE>();
        distLowLocal = distLowUb.Get<DTYPE_VALUE>();

        tmpALocal = tmpAUb.Get<DTYPE_VALUE>();
        tmpBLocal = tmpBUb.Get<DTYPE_VALUE>();

        w4Local = w4Ub.Get<DTYPE_VALUE>();
        wv1Local = wv1Ub.Get<DTYPE_VALUE>();
        wv2Local = wv2Ub.Get<DTYPE_VALUE>();
        wv3Local = wv3Ub.Get<DTYPE_VALUE>();
        wv4Local = wv4Ub.Get<DTYPE_VALUE>();

        mid1Local = mid1Ub.Get<DTYPE_VALUE>();
        mid2Local = mid2Ub.Get<DTYPE_VALUE>();
        mid3Local = mid3Ub.Get<DTYPE_VALUE>();
        mid4Local = mid4Ub.Get<DTYPE_VALUE>();
    }
    
    __aicore__ inline void ClearOutput()
    {
        switch (curBlockIdx) {
            case 0:
                InitOutput<DTYPE_VALUE>(gradValueGm,
                                        batchSize * numKeys * numHeads * embedDims, 0);
                break;
            case 1:
                InitOutput<DTYPE_VALUE>(gradLocationGm,
                                        batchSize * numQueries * numHeads * numLevels * TWO * numPoints, 0);
                break;
            case 2:
                InitOutput<DTYPE_VALUE>(gradWeightGm,
                                        batchSize * numQueries * numHeads * numLevels * numPoints, 0);
                break;
            default:
                break;
        }
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

        DataCopyPad(shapesLocal, valueSpatialShapesGm, {1, (uint32_t)(TWO * numLevels * sizeof(int32_t)), 0, 0, 0}, padParamsInt);
        DataCopyPad(offsetLocal, valueLevelStartIndexGm, {1, (uint32_t)(numLevels * sizeof(int32_t)), 0, 0, 0}, padParamsInt);
        
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        for (uint64_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMteWeight);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3X);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3Y);
        pipe->ReleaseEventID<HardEvent::MTE3_S>(eventIdMte3ToS);
    }

private:
    template <bool AddH, bool AddW>
    __aicore__ inline void ComputeGrad(const LocalTensor<DTYPE_VALUE> &wvLocal, const LocalTensor<DTYPE_VALUE> &mid,
                                       uint32_t vId, DTYPE_VALUE distH, DTYPE_VALUE distW,
                                       uint64_t hPtrOffset, uint64_t wPtrOffset, DTYPE_VALUE w)
    {
        DataCopyPad(zerosLocal[vId * embedDims + queryOffsetv + 4 * baseOffsetUb],
                    valueGm[offsetValue + hPtrOffset + wPtrOffset], 
                    copyInParams, padParamsFloat);
                    
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Muls(mid[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w, embedDims);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(wvLocal, zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], w, embedDims);
        Muls(tmpALocal, zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], distW, embedDims);
        Muls(tmpBLocal, zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], distH, embedDims);
        if (AddH) {
            Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDims);
        } else {
            Sub(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDims);
        }
        if (AddW) {
            Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDims);
        } else {
            Sub(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDims);
        }
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(gradValueGm[offsetValue + hPtrOffset + wPtrOffset], mid[queryOffset], copyOutParams);
    }

    __aicore__ inline void ComputeGradSeparate(DTYPE_VALUE distHH, DTYPE_VALUE distHW,
                                               DTYPE_VALUE distLH, DTYPE_VALUE distLW,
                                               DTYPE_VALUE w1, DTYPE_VALUE w2,
                                               DTYPE_VALUE w3, DTYPE_VALUE w4,
                                               DTYPE_VALUE attentionWeight)
    {
        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb],
             topGradLocal[query * embedDims], attentionWeight, embedDims);
        if (hLow >= 0 && wLow >= 0) {
            ComputeGrad<false, false>(wv1Local, mid1Local, v1Id, distHH, distHW,
                                      hLowPtrOffset, wLowPtrOffset, w1);
        }
        if (hLow >= 0 && wLow < w - 1) {
            ComputeGrad<false, true>(wv2Local, mid2Local, v2Id, distHH, distLW,
                                     hLowPtrOffset, wLowPtrOffset + wStride, w2);
        }
        if (hLow < h - 1 && wLow >= 0) {
            ComputeGrad<true, false>(wv3Local, mid3Local, v3Id, distLH, distHW,
                                     hLowPtrOffset + hStride, wLowPtrOffset,  w3);
        }
        if (hLow < h - 1 && wLow < w - 1) {
            ComputeGrad<true, true>(wv4Local, mid4Local, v4Id, distLH, distLW,
                                    hLowPtrOffset + hStride, wLowPtrOffset + wStride, w4);
        }
        Add(wv1Local, wv1Local, wv2Local, embedDims);
        Add(wv3Local, wv3Local, wv4Local, embedDims);
        Add(wv1Local, wv1Local, wv3Local, embedDims);
        Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDims], wv1Local, embedDims);
    }

    __aicore__ inline void ComputeGradTogether(DTYPE_VALUE distLH, DTYPE_VALUE distLW,
                                               DTYPE_VALUE w1, DTYPE_VALUE w2, DTYPE_VALUE w3, DTYPE_VALUE w4,
                                               DTYPE_VALUE attentionWeight)
    {
        DataCopyPad(zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                    valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], 
                    copyInParams, padParamsFloat);
        DataCopyPad(zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                    valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], 
                    copyInParams, padParamsFloat);
        DataCopyPad(zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                    valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], 
                    copyInParams, padParamsFloat);
        DataCopyPad(zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                    valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride], 
                    copyInParams, padParamsFloat);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb],
             topGradLocal[query * embedDims], attentionWeight, embedDims);

        Muls(mid1Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w1, embedDims);
        Muls(mid2Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w2, embedDims);
        Muls(mid3Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w3, embedDims);
        Muls(mid4Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w4, embedDims);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w1, embedDims);
        Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w2, embedDims);
        Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w3, embedDims);
        Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w4, embedDims);

        // V3 - V1
        Sub(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
            zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
            zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], embedDims);
        // V4 - V2
        Sub(tmpALocal, zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
            zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], embedDims);
        // V2 -V1
        Sub(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
            zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
            zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], embedDims);
        // V4 + V1 - V3 - V2
        Sub(tmpALocal, tmpALocal, zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], embedDims);
        // (V4 + V1 - V3 - V2) * distLH
        Muls(tmpBLocal, tmpALocal, distLH, embedDims);
        // (V4 + V1 - V3 - V2) * distLW
        Muls(tmpALocal, tmpALocal, distLW, embedDims);
        // (V2 - V1) + (V4 + V1 - V3 - V2) * distLH
        Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
            zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDims);
        // (V3 - V1) + (V4 + V1 - V3 - V2) * distLW
        Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
            zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDims);

        Add(wv1Local, wv1Local, wv2Local, embedDims);
        Add(wv3Local, wv3Local, wv4Local, embedDims);
        Add(wv1Local, wv1Local, wv3Local, embedDims);
        Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDims], wv1Local, embedDims);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], mid1Local[queryOffset], copyOutParams);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride],
                    mid2Local[queryOffset], copyOutParams);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride],
                    mid3Local[queryOffset], copyOutParams);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride],
                    mid4Local[queryOffset], copyOutParams);
    }

    __aicore__ inline void GridSampleCompute()
    {
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetAtomicAdd<DTYPE_VALUE>();
        for (query = 0; query < thisCycleNum; query++) {
            queryOffset = query * embedDims;
            queryOffsetv = query * FOUR * embedDims;
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
        offsetGrad = batch * gradOutStride2 + nqloop * maxUbNum * gradOutStride1 + head * gradOutStride0;
        offsetWeight = batch * weightStride3 + head * weightStride2 + level * weightStride1 + point  * weightStride0;
        offsetLocation = TWO * offsetWeight;
        thisCycleNumAlign = (nqloop == numQueriesper -1) ? numQueriestail : maxUbNum;
        thisCycleNum = (nqloop == numQueriesper -1) ? (numQueries - (numQueriesper - 1) * maxUbNum) : maxUbNum;
        offsetValue = batch * valueStride2 + levelStartId * valueStride1 + head * valueStride0;
        hStride = w * wStride;
        copyParams = {1, (uint32_t)(thisCycleNum * sizeof(DTYPE_VALUE)), 0, 0, 0};
        sumParams = {thisCycleNum, embedDims, embedDims};
    }

    __aicore__ inline void Compute(uint64_t taskIdx)
    {
        ScalarUpdate(taskIdx);
        
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        Duplicate(zerosLocal, (DTYPE_VALUE)0, eightBuffer * numQueriesAlign * embedDims);
        DataCopyPad(locWLocal, locationGm[offsetLocation + nqloop * maxUbNum],  copyParams, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(imLocal, locWLocal, (DTYPE_VALUE)w, thisCycleNumAlign);

        DataCopyPad(locHLocal, locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyParams, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(imLocal[thisCycleNumAlign], locHLocal, (DTYPE_VALUE)h, thisCycleNumAlign);

        DataCopyPad(attentionWeightLocal, attentionWeightsGm[offsetWeight + nqloop * maxUbNum], copyParams, padParamsFloat);
        for (query = 0; query < thisCycleNum; query++) {
            DataCopyPad(topGradLocal[query * embedDims], gradOutputGm[offsetGrad + query * gradOutStride1], copyInParams, padParamsFloat);
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        
        Adds(imLocal, imLocal, CONV_CONSTANT, TWO * thisCycleNumAlign);
        Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, TWO * thisCycleNumAlign);
        Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, TWO * thisCycleNumAlign);
        Sub(distLowLocal, imLocal, lowFloatLocal, TWO * thisCycleNumAlign);
        Mul(w4Local, distLowLocal[thisCycleNumAlign], distLowLocal, thisCycleNumAlign);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        GridSampleCompute();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    
        Mul(zerosLocal[gradWWeightId * baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
            zerosLocal[gradWWeightId * baseOffsetUb], thisCycleNum * embedDims);
        Mul(zerosLocal[gradHWeightId * baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
            zerosLocal[gradHWeightId * baseOffsetUb], thisCycleNum * embedDims);
        Muls(zerosLocal[gradWWeightId * baseOffsetUb], zerosLocal[gradWWeightId * baseOffsetUb],
            (DTYPE_VALUE)w, thisCycleNum * embedDims);
        Muls(zerosLocal[gradHWeightId * baseOffsetUb], zerosLocal[gradHWeightId * baseOffsetUb],
            (DTYPE_VALUE)h, thisCycleNum * embedDims);
        
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);

        Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
        Sum(xLocal, zerosLocal[gradWWeightId * baseOffsetUb], sumParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
        Sum(yLocal, zerosLocal[gradHWeightId * baseOffsetUb], sumParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
        DataCopyPad(gradWeightGm[offsetWeight + nqloop * maxUbNum], weightSumLocal, copyParams);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
        DataCopyPad(gradLocationGm[offsetLocation + nqloop * maxUbNum], xLocal, copyParams);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);
        DataCopyPad(gradLocationGm[offsetLocation + numQueries + nqloop * maxUbNum], yLocal, copyParams);
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, gradOutputGm, gradValueGm, gradLocationGm,
        gradWeightGm;
    GlobalTensor<int32_t> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> attentionWeightsUb, shapeUb, offsetUb, topGradUb;
    TBuf<TPosition::VECCALC> tmpXUb, tmpYUb, weightSumUb;
    TBuf<TPosition::VECCALC> zerosUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, lowFloatUb;
    TBuf<TPosition::VECCALC> distLowUb, w4Ub;
    TBuf<TPosition::VECCALC> tmpAUb, tmpBUb;

    uint32_t coreNum;
    uint32_t embedDims;
    uint32_t curBlockIdx;
    uint32_t dataAlign, blockBytes;
    uint32_t hOffsetUb, baseOffsetUb, queryOffset, queryOffsetv;
    uint32_t gradHWeightId = 0, gradWWeightId = 1, topGradValueId = 2, gradWeightId = 3;
    uint32_t v1Id = 0, v2Id = 1, v3Id = 2, v4Id = 3;
    uint32_t thisCycleNum, thisCycleNumAlign, maxUbNum;
    uint32_t twoBuffer = 2, TWO = 2, FOUR = 4, eightBuffer = 8;
    uint64_t batchSize, numKeys, numHeads, numLevels, numQueries, numPoints;
    uint64_t numQueriesAlign, numQueriesper, numQueriestail, numLevelsAlign;
    uint64_t batch, query, head, level, point, nqloop;
    uint64_t taskNum, taskNumPerCore;
    uint64_t startOffset, endOffset;
    uint64_t gradOutStride0, gradOutStride1, gradOutStride2;
    uint64_t weightStride0, weightStride1, weightStride2, weightStride3;
    uint64_t valueStride0, valueStride1, valueStride2;
    uint64_t levelStartId;
    uint64_t offsetValue, offsetWeight, offsetLocation, offsetGrad, wStride, hStride;
    uint64_t hLowPtrOffset, wLowPtrOffset;
    int64_t h, w, hLow, wLow;

    DTYPE_VALUE hIm, wIm;
    DTYPE_VALUE w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    DTYPE_VALUE CONV_CONSTANT = -0.5;

    LocalTensor<DTYPE_VALUE> lowFloatLocal;
    LocalTensor<DTYPE_VALUE> xLocal, yLocal;
    LocalTensor<DTYPE_VALUE> distLowLocal;
    LocalTensor<DTYPE_VALUE> locWLocal, locHLocal;
    LocalTensor<DTYPE_VALUE> imLocal;
    LocalTensor<DTYPE_VALUE> zerosLocal;
    LocalTensor<DTYPE_VALUE> weightSumLocal, tmpALocal, tmpBLocal;
    LocalTensor<DTYPE_VALUE> topGradLocal, attentionWeightLocal;
    LocalTensor<DTYPE_VALUE> wv1Local, wv2Local, wv3Local, wv4Local, w4Local;
    LocalTensor<DTYPE_VALUE> mid1Local, mid2Local, mid3Local, mid4Local;
    LocalTensor<int32_t> shapesLocal, offsetLocal;
    LocalTensor<int32_t> lowLocal;

    SumParams sumParams;
    // DataCopyExtParams copyParams, copyParamsV2;
    DataCopyExtParams copyParams, copyInParams, copyOutParams;
    DataCopyPadExtParams<int32_t> padParamsInt {false, 0, 0, 0};
    DataCopyPadExtParams<DTYPE_VALUE> padParamsFloat {false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV,
            eventIdVToMteWeight, eventIdVToMte3X, eventIdVToMte3Y, eventIdMte3ToS;
    
    TBuf<TPosition::VECCALC> wv1Ub, wv2Ub, wv3Ub, wv4Ub, mid1Ub, mid2Ub, mid3Ub, mid4Ub;
};
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
 * \file ctc_loss_v3_grad.h
 * \brief
 */
#ifndef CTC_LOSS_V3_GRAD_H_
#define CTC_LOSS_V3_GRAD_H_

#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
namespace CTCLossV3GradNS {
using namespace AscendC;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t FLOAT_SIZE = 4;
constexpr float GATHER_FLOAT_SIZE = 4.0f;
constexpr int64_t FLOAT_NUM_PER_BLOCK = 8;
constexpr int64_t NUM_PER_REPEAT = 64;
constexpr int64_t HALF_NUM_PER_REPEAT = 32;
constexpr int64_t MASK_NUM_PER_BYTES = 8;
constexpr int64_t DOUBLE = 2;
constexpr int64_t LARGE_BATCH = 1500;

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    return b == 0 ? 0 : (a + b - 1) / b * b;
};

template <typename TGrad, typename TIndices>
class CTCLossV3Grad {
 private:
    TPipe *pipe;
    GlobalTensor<TGrad> gradOutGm;
    GlobalTensor<TGrad> logProbsGm;
    GlobalTensor<TIndices> targetsGm;
    GlobalTensor<TIndices> inputLengthsGm;
    GlobalTensor<TIndices> targetLengthsGm;
    GlobalTensor<TGrad> negLogLikelihoodGm;
    GlobalTensor<TGrad> logAlphaGm;
    GlobalTensor<TGrad> gradGm;
    GlobalTensor<float> logBetaGm;
    GlobalTensor<float> tempBetaGm;
    GlobalTensor<int32_t> tempTargetsGm;
    GlobalTensor<int64_t> targetOffsetGm; 

    // input
    TQue<QuePosition::VECIN, 1> gradOutQue;
    TQue<QuePosition::VECIN, 1> logProbSliceQue;
    TQue<QuePosition::VECIN, 1> targetLengthsQue;
    TQue<QuePosition::VECIN, 1> targetLengthsBlockQue;
    TQue<QuePosition::VECIN, 1> inputLengthsQue;
    TQue<QuePosition::VECIN, 1> negLogLikelihoodQue;
    TQue<QuePosition::VECIN, 1> logAlphaQue;

    // output
    TQue<QuePosition::VECOUT, 1> gradSliceQue;

    TBuf<TPosition::VECCALC> logBeta1Buf;
    TBuf<TPosition::VECCALC> logBeta2Buf;
    TBuf<TPosition::VECCALC> logBeta3Buf;
    TBuf<TPosition::VECCALC> logBetaBuf;
    TBuf<TPosition::VECCALC> logProbBlankBuf;
    TBuf<TPosition::VECCALC> logProbLastBuf;

    TBuf<TPosition::VECCALC> maxBetaBuf;
    TBuf<TPosition::VECCALC> tempMaxBetaBuf;
    TBuf<TPosition::VECCALC> gtBeta1MaskBuf;
    TBuf<TPosition::VECCALC> gtBeta2MaskBuf;
    TBuf<TPosition::VECCALC> ifHaveThreeMaskBuf;
    TBuf<TPosition::VECCALC> ifNeginfMaskBuf;
    TBuf<TPosition::VECCALC> sumTwoBuf;
    TBuf<TPosition::VECCALC> sumThreeBuf;
    TBuf<TPosition::VECCALC> castTargetBuf;
    TBuf<TPosition::VECCALC> castFloatTargetBuf;
    TBuf<TPosition::VECCALC> resPartProbBuf;
    TBuf<TPosition::VECCALC> resProbBuf;
    TBuf<TPosition::VECCALC> downMaskBuf;
    TBuf<TPosition::VECCALC> upMaskBuf;
    TBuf<TPosition::VECCALC> allNeginfBuf;
    TBuf<TPosition::VECCALC> allZeroBuf;
    TBuf<TPosition::VECCALC> allOneBuf;
    TBuf<TPosition::VECCALC> scalarLcabBuf;
    TBuf<TPosition::VECCALC> logScalarLcabBuf;
    TBuf<TPosition::VECCALC> targetOffsetBuf;
    TBuf<TPosition::VECCALC> targetsBuf;
    TBuf<TPosition::VECCALC> changedTargetsBuf;
    TBuf<TPosition::VECCALC> targetsShiftBuf;
    TBuf<TPosition::VECCALC> intResProbBuf;
    TBuf<TPosition::VECCALC> evenMaskBuf;
    TBuf<TPosition::VECCALC> targetsDoubleBuf;

    int64_t coreNum = 0;
    int64_t curBlockIdx = 0;
    int64_t cLoop = 0;
    int64_t gradSliceLoopTail = 0;
    int64_t cLoopTail = 0;
    int64_t gradSliceTailLoopTail = 0;
    int64_t targetDsize = 0;
    int64_t gradDsize = 0;
    int64_t zeroInfinity = 0;
    int64_t targetNumPerBlock = 0;
    int64_t targetNumPerRepeat = 0;
    int64_t gradNumPerBlock = 0;
    int64_t gradNumPerRepeat = 0;
    int64_t doubleSi = 0;
    int64_t doubleSiAlign = 0;
    int64_t doubleSiInt32Align = 0;
    int64_t siAlign = 0;
    int64_t siBlockAlign = 0;
    int64_t sliceLengthAlign = 0;
    int64_t sDimRange = 1;
    int64_t targetsDimNum = 1;
    int64_t batchSize = 0;
    int64_t symbolSet = 0;
    int64_t BLANK = 0;
    int64_t maxTargetLength = 0;
    int64_t sliceLength = 0;
    int64_t sliceLengthTail = 0;
    int64_t probSliceNum = 1;
    int64_t alphaLength = 0;
    int64_t maxInputLength = 0;
    int64_t taskPerCore = 0;
    int64_t taskTailCore = 0;
    int64_t maskNumBeta = 0;
    int64_t maskNumTarget = 0;
    int64_t curCoreTaskNum = 0;
    int64_t startTaskId = 0;
    int64_t targetsNum = 0;
    int64_t maxTargetLengthBlockAlign = 0;
    int64_t maxTargetLengthAlign = 0;
    int64_t alphaLengthBlockAlign = 0;
    int64_t alphaLengthAlign = 0;
    // params per loop
    float alphaBlankChar = 0.0f;
    int64_t lastChar = -1;
    float logProbLastChar = 0.0f;
    float alphaLastChar = 0.0f;
    float logProbBlank = 0.0f;
    int64_t targetLength = -1;
    int64_t inputLength = -1;
    float gradOutCurBatch = 0.0f;
    float nll = 0.0f;

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams copyParamsOneC = {1, sizeof(TGrad), 0, 0};
    DataCopyParams copyParamsOneI = {1, sizeof(TIndices), 0, 0};
    DataCopyParams copyParamsOneInt64 = {1, 8, 0, 0};

    BinaryRepeatParams repeatBetaMaxSelect = {1, 1, 1, 8, 8, 8};
    DataCopyPadExtParams<TGrad> padParamsProb = {false, 0, 0, 0};
    DataCopyPadExtParams<TIndices> padParamsIndices = {false, 0, 0, 0};
    DataCopyPadExtParams<TIndices> padParamsOffset = {false, 0, 0, 0};

    LocalTensor<int64_t> targetsTensor;
    LocalTensor<float> changedTargetsTensor;
    LocalTensor<int64_t> targetsShiftTensor;
    LocalTensor<int32_t> targetsDoubleTensor;

    LocalTensor<uint8_t> gtBeta1MaskTensor;
    LocalTensor<uint8_t> gtBeta2MaskTensor;
    LocalTensor<uint8_t> ifHaveThreeMaskTensor;

    LocalTensor<float> logBetaTensor;
    LocalTensor<float> logBeta1Tensor;
    LocalTensor<float> logBeta2Tensor;
    LocalTensor<float> logBeta3Tensor;
    LocalTensor<float> maxBetaTensor;
    LocalTensor<float> tempMaxBetaTensor;
    LocalTensor<uint8_t> ifNeginfMaskTensor;

    LocalTensor<float> sumTwoTensor;
    LocalTensor<float> sumThreeTensor;
    LocalTensor<int32_t> intTargetsTensor;
    LocalTensor<float> castFloatTargetsTensor;
    LocalTensor<float> resPartProbTensor;
    LocalTensor<float> resProbTensor;
    LocalTensor<int64_t> intResProbTensor;
    LocalTensor<uint8_t> upMaskTensor;
    LocalTensor<uint8_t> downMaskTensor;
    LocalTensor<uint8_t> evenMaskTensor;

    LocalTensor<float> allNeginfTensor;
    LocalTensor<float> allZeroTensor;
    LocalTensor<int64_t> allOneTensor;
    LocalTensor<float> scalarLcabTensor;
    LocalTensor<float> logScalarLcabTensor;

 public:
__aicore__ inline CTCLossV3Grad(){};
__aicore__ inline void Init(const CTCLossV3GradTilingData* __restrict tilingData,
                            GM_ADDR gradOut, GM_ADDR logProbs, GM_ADDR targets, GM_ADDR inputLengths,
                            GM_ADDR targetLengths, GM_ADDR negLogLikelihood, GM_ADDR logAlpha, GM_ADDR grad,
                            GM_ADDR workspace, TPipe* inputPipe) {
    pipe = inputPipe;
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    coreNum = GetBlockNum();
    curBlockIdx = GetBlockIdx();
    sliceLength = tilingData->sliceLength;
    sliceLengthTail = tilingData->sliceLengthTail;
    probSliceNum = tilingData->probSliceNum;
    alphaLength = tilingData->alphaLength;
    maxInputLength = tilingData->maxInputLength;
    symbolSet = tilingData->symbolSet;
    batchSize = tilingData->batchSize;
    BLANK = tilingData->BLANK;
    targetsDimNum = tilingData->targetsDimNum;
    sDimRange = tilingData->sDimRange;
    targetsNum = tilingData->targetsNum;
    taskPerCore = tilingData->taskPerCore;
    taskTailCore = tilingData->taskTailCore;
    targetDsize = sizeof(TIndices);
    gradDsize = sizeof(TGrad);
    zeroInfinity = tilingData->zeroInfinity;
    sliceLengthAlign = CeilAlign(sliceLength, NUM_PER_REPEAT);
    gradNumPerBlock = BLOCK_BYTES / gradDsize;
    alphaLengthBlockAlign = CeilAlign(alphaLength, gradNumPerBlock);
    alphaLengthAlign = CeilAlign(alphaLength, NUM_PER_REPEAT);
    maskNumBeta = CeilAlign(alphaLengthAlign / MASK_NUM_PER_BYTES, BLOCK_BYTES);
    curCoreTaskNum = curBlockIdx < taskTailCore ? (taskPerCore + 1) : taskPerCore;
    startTaskId = curBlockIdx < taskTailCore ? (curCoreTaskNum * curBlockIdx) : (curCoreTaskNum * curBlockIdx + taskTailCore);

    gradOutGm.SetGlobalBuffer((__gm__ TGrad*)gradOut, batchSize);
    logProbsGm.SetGlobalBuffer((__gm__ TGrad*)logProbs, symbolSet * batchSize * maxInputLength);
    targetsGm.SetGlobalBuffer((__gm__ TIndices*)targets, targetsNum);
    inputLengthsGm.SetGlobalBuffer((__gm__ TIndices*)inputLengths, batchSize);
    targetLengthsGm.SetGlobalBuffer((__gm__ TIndices*)targetLengths, batchSize);
    negLogLikelihoodGm.SetGlobalBuffer((__gm__ TGrad*)negLogLikelihood, batchSize);
    logAlphaGm.SetGlobalBuffer((__gm__ TGrad*)logAlpha, batchSize * alphaLength * maxInputLength);
    gradGm.SetGlobalBuffer((__gm__ TGrad*)grad, symbolSet * batchSize * maxInputLength);
    logBetaGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize * alphaLength);
    tempBetaGm.SetGlobalBuffer((__gm__ float*)workspace + batchSize * alphaLength,
                                batchSize * alphaLength);
    tempTargetsGm.SetGlobalBuffer((__gm__ int32_t*)workspace + batchSize * alphaLength * DOUBLE,
                                  alphaLength * DOUBLE * batchSize);
    if (batchSize > LARGE_BATCH) {
        targetOffsetGm.SetGlobalBuffer((__gm__ int64_t*)workspace + batchSize * alphaLength * DOUBLE, batchSize);
    }
}

__aicore__ inline int64_t CalcuBatchOffset(int64_t batchId, const LocalTensor<int64_t>& targetOffsetTensor) {
    int64_t batchOffsetTargets;
    if (targetsDimNum == DOUBLE) {
        batchOffsetTargets = batchId * sDimRange;
    } else if (batchSize <= LARGE_BATCH) {
        batchOffsetTargets = targetOffsetTensor.GetValue(batchId);
    } else {
        MTE3ToMTE2Sync();
        DataCopyPad(targetOffsetTensor, targetOffsetGm[batchId], copyParamsOneInt64, padParams);
        MTE2ToSSync();
        batchOffsetTargets = targetOffsetTensor.GetValue(0);
    }
    return batchOffsetTargets;
}

__aicore__ inline void InitNewTargetsBuffer(int64_t batchSize, LocalTensor<int64_t>& targetOffsetTensor,
                                            LocalTensor<TIndices>& targetLengthsTensor) {
    if (batchSize > LARGE_BATCH) {
        DataCopyExtParams copyParamsOffset = {1, static_cast<uint32_t>(batchSize * INT64_SIZE), 0, 0, 0};
        SToMTE3Sync();
        DataCopyPad(targetOffsetGm, targetOffsetTensor, copyParamsOffset);
        targetLengthsQue.FreeTensor(targetLengthsTensor);
        pipe->Reset();
        pipe->InitBuffer(targetLengthsQue, 1, BLOCK_BYTES);
        pipe->InitBuffer(targetOffsetBuf, BLOCK_BYTES);
        targetOffsetTensor = targetOffsetBuf.Get<int64_t>();
        targetLengthsTensor = targetLengthsQue.AllocTensor<TIndices>();
    }
}

__aicore__ inline void Process() {
    pipe->InitBuffer(targetLengthsQue, 1, CeilAlign(batchSize * targetDsize, BLOCK_BYTES));
    pipe->InitBuffer(targetOffsetBuf, CeilAlign(batchSize * INT64_SIZE, BLOCK_BYTES));  
    LocalTensor<TIndices> targetLengthsTensor = targetLengthsQue.AllocTensor<TIndices>();
    LocalTensor<int64_t> targetOffsetTensor = targetOffsetBuf.Get<int64_t>();
    DataCopyExtParams copyParamsBatch = {1, static_cast<uint32_t>(batchSize * targetDsize), 0, 0, 0};
    DataCopyPad(targetLengthsTensor, targetLengthsGm, copyParamsBatch, padParamsOffset);
    MTE2ToSSync();
    GetMaxTargetLength(targetLengthsTensor, targetOffsetTensor);
    if (maxTargetLength <= 0 || maxTargetLength > (alphaLength - 1) / DOUBLE) {
        return;
    }
    InitNewTargetsBuffer(batchSize, targetOffsetTensor, targetLengthsTensor);
    InitChangelessBuffer();
    InitTargetLengthBuffer();
    InitTensor();
    CreateDuplicateTensor(); // Create fixed value UB
    for (int32_t taskLoopId = 0; taskLoopId < curCoreTaskNum; taskLoopId++) {
        int64_t batchId = startTaskId + taskLoopId;
        int64_t batchOffsetProb = batchId * symbolSet;
        // Calculate the params which are different per batch
        CopyInParamsPerBatch(batchId, targetLengthsTensor); 
        if (zeroInfinity && nll == INFINITY) { // Special scene: all the grad values of the currrent batch will be 0
            for (int32_t t = 0; t < maxInputLength; t++) {
                DuplicateGrad(batchOffsetProb, batchId, t);
            }
            continue;
        }
        int64_t batchOffsetTargets = CalcuBatchOffset(batchId, targetOffsetTensor);
        SToMTE2Sync();
        if (taskLoopId > 0) {
            VToMTE2Sync();
        }
        if (targetLength > 0) {
            CopyInTargetsTensor(batchOffsetTargets, batchId);
        }
        if (targetLength >= 0) {
            for (int64_t t = inputLength - 1; t >= 0; t--) {
                Compute(batchOffsetProb, batchId, t);
            }
        } // When targetLength < 0, the grad will be invalid
        for (int32_t t = inputLength < 0 ? 0 : inputLength; t < maxInputLength; t++) {
            DuplicateGrad(batchOffsetProb, batchId, t);
        }
    }
    targetLengthsQue.FreeTensor(targetLengthsTensor);
}

private:
__aicore__ inline void InitTargetLengthBuffer() {
    pipe->InitBuffer(targetsBuf, maxTargetLengthAlign * INT64_SIZE);
    pipe->InitBuffer(changedTargetsBuf, maxTargetLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(castTargetBuf, maxTargetLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(castFloatTargetBuf, maxTargetLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(resPartProbBuf, maxTargetLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(resProbBuf, maxTargetLengthAlign * FLOAT_SIZE);

    pipe->InitBuffer(targetsShiftBuf, CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT) * INT64_SIZE);
    pipe->InitBuffer(targetsDoubleBuf, CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT) * INT64_SIZE);
    pipe->InitBuffer(allOneBuf, CeilAlign(maxTargetLength + 1, HALF_NUM_PER_REPEAT) * INT64_SIZE);
    pipe->InitBuffer(intResProbBuf, CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT) * INT64_SIZE);
    pipe->InitBuffer(downMaskBuf, maskNumTarget);
    pipe->InitBuffer(upMaskBuf, maskNumTarget);
}

__aicore__ inline void InitChangelessBuffer() {
    pipe->InitBuffer(negLogLikelihoodQue, 1, BLOCK_BYTES);
    pipe->InitBuffer(inputLengthsQue, 1, BLOCK_BYTES);
    pipe->InitBuffer(gradOutQue, 1, BLOCK_BYTES);
    pipe->InitBuffer(logProbSliceQue, 1, sliceLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(gradSliceQue, 1, sliceLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(logAlphaQue, 1, alphaLengthBlockAlign * FLOAT_SIZE);

    pipe->InitBuffer(logBeta1Buf, alphaLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(logBeta2Buf, alphaLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(logBeta3Buf, alphaLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(logProbBlankBuf, BLOCK_BYTES);
    pipe->InitBuffer(logProbLastBuf, BLOCK_BYTES);
    pipe->InitBuffer(logBetaBuf, alphaLength * FLOAT_SIZE);
    pipe->InitBuffer(tempMaxBetaBuf, alphaLengthAlign * FLOAT_SIZE);

    pipe->InitBuffer(maxBetaBuf, alphaLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(sumTwoBuf, alphaLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(sumThreeBuf, alphaLengthAlign * FLOAT_SIZE);
    pipe->InitBuffer(allNeginfBuf, BLOCK_BYTES);
    pipe->InitBuffer(allZeroBuf, BLOCK_BYTES);

    pipe->InitBuffer(scalarLcabBuf, BLOCK_BYTES);
    pipe->InitBuffer(logScalarLcabBuf, BLOCK_BYTES);
    pipe->InitBuffer(gtBeta1MaskBuf, maskNumBeta);
    pipe->InitBuffer(gtBeta2MaskBuf, maskNumBeta);
    pipe->InitBuffer(evenMaskBuf, maskNumBeta);
    pipe->InitBuffer(ifHaveThreeMaskBuf, maskNumBeta);
    pipe->InitBuffer(ifNeginfMaskBuf, maskNumBeta);
}

__aicore__ inline void SToMTE2Sync() {
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}

__aicore__ inline void MTE2ToSSync() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

__aicore__ inline void SToMTE3Sync() {
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

__aicore__ inline void MTE3ToSSync() {
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

__aicore__ inline void SToVSync() {
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

__aicore__ inline void VToSSync() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

__aicore__ inline void MTE3ToVSync() {
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

__aicore__ inline void VToMTE3Sync() {
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}
__aicore__ inline void MTE3ToMTE2Sync() {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

__aicore__ inline void MTE2ToVSync() {
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

__aicore__ inline void VToMTE2Sync() {
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
}

__aicore__ inline void MTE2ToMTE3Sync() {
    event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
}

__aicore__ inline void InitTensor() {
    targetsTensor = targetsBuf.Get<int64_t>();
    changedTargetsTensor = changedTargetsBuf.Get<float>();
    targetsShiftTensor = targetsShiftBuf.Get<int64_t>();
    logBetaTensor = logBetaBuf.Get<float>();
    logBeta1Tensor = logBeta1Buf.Get<float>();
    logBeta2Tensor = logBeta2Buf.Get<float>();
    logBeta3Tensor = logBeta3Buf.Get<float>();
    maxBetaTensor = maxBetaBuf.Get<float>();
    tempMaxBetaTensor = tempMaxBetaBuf.Get<float>();
    sumTwoTensor = sumTwoBuf.Get<float>();
    sumThreeTensor = sumThreeBuf.Get<float>();
    intTargetsTensor = castTargetBuf.Get<int32_t>();
    castFloatTargetsTensor = castFloatTargetBuf.Get<float>();
    resPartProbTensor = resPartProbBuf.Get<float>();
    resProbTensor = resProbBuf.Get<float>();
    intResProbTensor = intResProbBuf.Get<int64_t>();
    allNeginfTensor = allNeginfBuf.Get<float>();
    allZeroTensor = allZeroBuf.Get<float>();
    allOneTensor = allOneBuf.Get<int64_t>();
    gtBeta1MaskTensor = gtBeta1MaskBuf.Get<uint8_t>();
    gtBeta2MaskTensor = gtBeta2MaskBuf.Get<uint8_t>();
    ifHaveThreeMaskTensor = ifHaveThreeMaskBuf.Get<uint8_t>();
    ifNeginfMaskTensor = ifNeginfMaskBuf.Get<uint8_t>();
    upMaskTensor = upMaskBuf.Get<uint8_t>();
    downMaskTensor = downMaskBuf.Get<uint8_t>();
    evenMaskTensor = evenMaskBuf.Get<uint8_t>();

    // Init LcabTensor
    scalarLcabTensor = scalarLcabBuf.Get<float>();
    logScalarLcabTensor = logScalarLcabBuf.Get<float>();
}

__aicore__ inline void DuplicateGrad(int64_t batchOffsetProb, int64_t batchId, int64_t t)
{
    LocalTensor<TGrad> gradSliceTensor = gradSliceQue.AllocTensor<TGrad>();
    int64_t sliceLengthCur = sliceLength;
    for (int32_t sliceId = 0; sliceId < probSliceNum; sliceId++) {
        if (sliceId > (probSliceNum - DOUBLE)) {
            sliceLengthCur = sliceLengthTail;
        }
        Duplicate(gradSliceTensor, static_cast<TGrad>(0), sliceLength);
        VToMTE3Sync();
        DataCopyExtParams copyParamsProb = {1, static_cast<uint32_t>(sliceLengthCur * gradDsize), 0, 0, 0};
        DataCopyPad(gradGm[batchOffsetProb + t * symbolSet * batchSize + sliceLength * sliceId],
                    gradSliceTensor, copyParamsProb);
        MTE3ToVSync();
    }
    gradSliceQue.FreeTensor(gradSliceTensor);
}

__aicore__ inline void updateLcab(int64_t t, int64_t sliceId, int64_t sliceLengthCur,
                                  const LocalTensor<float>& logAlphaTensor,
                                  const LocalTensor<float>& gradSliceTensor)
{
    if (t < inputLength - 1) {
        Duplicate(gradSliceTensor, -INFINITY, sliceLength);
        VToSSync();
        // Process non-blank positions
        for (int32_t targetId = targetLength - 1; targetId >= 0; targetId--) {
            int64_t curTarget = targetsTensor.GetValue(targetId);
            if (sliceLength * sliceId <= curTarget && curTarget < (sliceLength * sliceId + sliceLengthCur)) {
                int64_t curGradOffset = curTarget - sliceLength * sliceId;
                float alphaBetaCur = logAlphaTensor.GetValue(DOUBLE * targetId + 1);
                float lcab = gradSliceTensor.GetValue(curGradOffset);
                updateLcabOneTime(curGradOffset, alphaBetaCur, lcab, gradSliceTensor);
            }
        }
        // Process blank positions
        if (BLANK >= sliceLength * sliceId && BLANK < (sliceLength * sliceId + sliceLengthCur)) {
            for (int targetId = targetLength; targetId >= 0; targetId--) { 
                int64_t curGradOffset = BLANK - sliceLength * sliceId;
                float alphaBetaCur = logAlphaTensor.GetValue(DOUBLE * targetId);
                float lcab = gradSliceTensor.GetValue(curGradOffset);
 
                updateLcabOneTime(curGradOffset, alphaBetaCur, lcab, gradSliceTensor);
            }
        }
    } else {
        Duplicate(gradSliceTensor, -INFINITY, sliceLength);
        VToSSync();
        if (sliceLength * sliceId <= BLANK && BLANK < (sliceLength * sliceId + sliceLengthCur)) {
            gradSliceTensor.SetValue(BLANK - sliceLength * sliceId, logProbBlank + alphaBlankChar);
        }
        if (targetLength > 0 && (sliceLength * sliceId <= lastChar && lastChar < (sliceLength * sliceId + sliceLengthCur))) {
            gradSliceTensor.SetValue(lastChar - sliceLength * sliceId, logProbLastChar + alphaLastChar);
        }
    }
    SToVSync();
}

__aicore__ inline void updateLcabOneTime(int64_t curGradOffset, float alphaBetaCur, float lcab,
                                         const LocalTensor<float>& gradSliceTensor)
{
    if (lcab == -INFINITY) {
        gradSliceTensor.SetValue(curGradOffset, alphaBetaCur);
    } else {
        float maxTmp;
        if (lcab > alphaBetaCur) {
            maxTmp = lcab;
            scalarLcabTensor.SetValue(0, 0);
            scalarLcabTensor.SetValue(1, alphaBetaCur - lcab);
        } else {
            maxTmp = alphaBetaCur;
            scalarLcabTensor.SetValue(1, 0);
            scalarLcabTensor.SetValue(0, lcab - alphaBetaCur);
        }
        SToVSync();
        Exp(scalarLcabTensor, scalarLcabTensor, DOUBLE);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(scalarLcabTensor, scalarLcabTensor, scalarLcabTensor, DOUBLE);
        PipeBarrier<PIPE_V>();
        Log(logScalarLcabTensor, scalarLcabTensor, 1);
        VToSSync();
        float lcabNew = logScalarLcabTensor.GetValue(0) + maxTmp;
        gradSliceTensor.SetValue(curGradOffset, lcabNew);
    }
}

__aicore__ inline void CalcLogBeta(int64_t batchId)
{   
    // Calcu: log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax)+std::exp(lb3-lbmax))+lbmax
    DataCopyParams copyParamsDoubleSi0 = {1, static_cast<uint16_t>((doubleSi - 0) * FLOAT_SIZE), 0, 0};
    DataCopyParams copyParamsDoubleSi1 = {1, static_cast<uint16_t>((doubleSi - 1) * FLOAT_SIZE), 0, 0};
    DataCopyParams copyParamsDoubleSi2 = {1, static_cast<uint16_t>((doubleSi - DOUBLE) * FLOAT_SIZE), 0, 0};
    DataCopyPad(logBeta1Tensor, logBetaGm[batchId * alphaLength + 0], copyParamsDoubleSi0, padParams);
    DataCopyPad(logBeta2Tensor, logBetaGm[batchId * alphaLength + 1], copyParamsDoubleSi1, padParams);
    DataCopyPad(logBeta3Tensor, logBetaGm[batchId * alphaLength + DOUBLE], copyParamsDoubleSi2, padParams);
    MTE2ToSSync();
    logBeta2Tensor.SetValue(DOUBLE * targetLength, -INFINITY);
    logBeta3Tensor.SetValue(DOUBLE * targetLength - 1, -INFINITY);
    logBeta3Tensor.SetValue(DOUBLE * targetLength, -INFINITY);
    SToVSync();
    Compare(gtBeta1MaskTensor, logBeta2Tensor, logBeta1Tensor, CMPMODE::GT, doubleSiAlign);
    PipeBarrier<PIPE_V>();
    Select(maxBetaTensor, gtBeta1MaskTensor, logBeta2Tensor, logBeta1Tensor,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, doubleSiAlign); // Take the maximum value of logbeta1 and logbeta2
    BinaryRepeatParams repeatNeginfSelect = {1, 0, 1, 8, 0, 8};
    Select(logBeta3Tensor, ifHaveThreeMaskTensor, allNeginfTensor, logBeta3Tensor,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, NUM_PER_REPEAT, doubleSiAlign / NUM_PER_REPEAT, repeatNeginfSelect);
     // Correct the position of -inf in logbeta3
    PipeBarrier<PIPE_V>();
    Compare(gtBeta2MaskTensor, logBeta3Tensor, maxBetaTensor, CMPMODE::GT, doubleSiAlign);
    PipeBarrier<PIPE_V>();
    Select(maxBetaTensor, gtBeta2MaskTensor, logBeta3Tensor, maxBetaTensor, // Correct maxBetaTensor
           SELMODE::VSEL_TENSOR_TENSOR_MODE, doubleSiAlign);
    PipeBarrier<PIPE_V>();
    CompareScalar(ifNeginfMaskTensor, maxBetaTensor, -INFINITY, CMPMODE::EQ, doubleSiAlign);
    PipeBarrier<PIPE_V>();
    Select(maxBetaTensor, ifNeginfMaskTensor, allZeroTensor, maxBetaTensor,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, NUM_PER_REPEAT, doubleSiAlign / NUM_PER_REPEAT, repeatNeginfSelect);

    PipeBarrier<PIPE_V>();
    Sub(logBeta1Tensor, logBeta1Tensor, maxBetaTensor, doubleSi);
    Sub(logBeta2Tensor, logBeta2Tensor, maxBetaTensor, doubleSi);
    Sub(logBeta3Tensor, logBeta3Tensor, maxBetaTensor, doubleSi);
    PipeBarrier<PIPE_V>();
    Exp(logBeta1Tensor, logBeta1Tensor, doubleSi);
    Exp(logBeta2Tensor, logBeta2Tensor, doubleSi);
    Exp(logBeta3Tensor, logBeta3Tensor, doubleSi);
    PipeBarrier<PIPE_V>();
    Add(logBeta2Tensor, logBeta1Tensor, logBeta2Tensor, doubleSi);
    PipeBarrier<PIPE_V>();
    Add(logBeta3Tensor, logBeta2Tensor, logBeta3Tensor, doubleSi);
    PipeBarrier<PIPE_V>();
    Log(logBetaTensor, logBeta3Tensor, doubleSi);
    PipeBarrier<PIPE_V>();
    Add(logBetaTensor, logBetaTensor, maxBetaTensor, doubleSi);
}

__aicore__ inline void CopyInParamsPerBatch(int64_t batchId, const LocalTensor<TIndices>& targetLengthsTensor)
{
    LocalTensor<TGrad> gradOutTensor = gradOutQue.AllocTensor<TGrad>();
    LocalTensor<TGrad> negLogLikelihoodTensor = negLogLikelihoodQue.AllocTensor<TGrad>();
    LocalTensor<TIndices> inputLengthsTensor = inputLengthsQue.AllocTensor<TIndices>();
    DataCopyPad(negLogLikelihoodTensor, negLogLikelihoodGm[batchId], copyParamsOneC, padParams);
    DataCopyPad(gradOutTensor, gradOutGm[batchId], copyParamsOneC, padParams);
    DataCopyPad(inputLengthsTensor, inputLengthsGm[batchId], copyParamsOneI, padParams);
    if (batchSize > LARGE_BATCH) {
        DataCopyPad(targetLengthsTensor, targetLengthsGm[batchId], copyParamsOneI, padParams);
    }
    MTE2ToSSync();
    targetLength = batchSize > LARGE_BATCH ? targetLengthsTensor.GetValue(0) : targetLengthsTensor.GetValue(batchId);
    inputLength = inputLengthsTensor.GetValue(0);

    doubleSi = DOUBLE * targetLength + 1;
    doubleSiAlign = CeilAlign(doubleSi, NUM_PER_REPEAT);
    doubleSiInt32Align = CeilAlign(doubleSi, NUM_PER_REPEAT);
    siAlign = CeilAlign(targetLength, NUM_PER_REPEAT);
    if constexpr (std::is_same<TGrad, bfloat16_t>::value) {
        gradOutCurBatch = ToFloat(gradOutTensor.GetValue(0));
        nll = ToFloat(negLogLikelihoodTensor.GetValue(0));
    } else {
        gradOutCurBatch = gradOutTensor.GetValue(0);
        nll = negLogLikelihoodTensor.GetValue(0);
    }
    negLogLikelihoodQue.FreeTensor(negLogLikelihoodTensor);
    gradOutQue.FreeTensor(gradOutTensor);
    inputLengthsQue.FreeTensor(inputLengthsTensor);
}

__aicore__ inline void Compute(int64_t batchOffsetProb, int64_t batchId, int64_t t)
{
    LocalTensor<float> logProbSliceTensor = logProbSliceQue.AllocTensor<float>();
    LocalTensor<float> logAlphaTensor = logAlphaQue.AllocTensor<float>();
    LocalTensor<float> gradSliceTensor = gradSliceQue.AllocTensor<float>();
    LocalTensor<TGrad> logProbLastTensor = logProbLastBuf.Get<TGrad>();
    LocalTensor<TGrad> logProbBlankTensor = logProbBlankBuf.Get<TGrad>();
    CopyInCast(batchId * alphaLength * maxInputLength + t * alphaLength, alphaLengthBlockAlign,
               alphaLength, logAlphaTensor, logAlphaGm);
    DataCopyPad(logProbBlankTensor, logProbsGm[batchOffsetProb + t * symbolSet * batchSize + BLANK], copyParamsOneC,
                padParams); // Load the probability of blank.
    MTE2ToSSync();
    if constexpr (std::is_same<TGrad, bfloat16_t>::value) {
        logProbBlank = ToFloat(logProbBlankTensor.GetValue(0));
    } else {
        logProbBlank = logProbBlankTensor.GetValue(0);
    }
    lastChar = targetLength >= 1 ? targetsTensor.GetValue(targetLength - 1) : 0; 
    SToMTE2Sync();
    DataCopyPad(logProbLastTensor, logProbsGm[batchOffsetProb + t * symbolSet * batchSize + lastChar], copyParamsOneC,
                padParams);
    MTE2ToSSync();
    if constexpr (std::is_same<TGrad, bfloat16_t>::value) {
        logProbLastChar = ToFloat(logProbLastTensor.GetValue(0));
    } else {
        logProbLastChar = logProbLastTensor.GetValue(0);
    }
    Duplicate<float>(logBetaTensor, -INFINITY, alphaLength);
    if (t == inputLength - 1) {
        VToSSync();
        if (targetLength > 0) {
            logBetaTensor.SetValue(DOUBLE * targetLength - 1, logProbLastChar);
            alphaLastChar = logAlphaTensor.GetValue(DOUBLE * targetLength - 1);
        }
        logBetaTensor.SetValue(DOUBLE * targetLength, logProbBlank);
        alphaBlankChar = logAlphaTensor.GetValue(DOUBLE * targetLength);
        SToMTE3Sync();
    } else if ((t < inputLength - 1) && (targetLength > 0)) {
        CalcLogBeta(batchId);
        GatherLogBeta(t, batchOffsetProb, logProbSliceTensor);
        VToMTE3Sync();
        DataCopyParams copyParamsResProbOut = {1, static_cast<uint16_t>(DOUBLE * targetLength * FLOAT_SIZE), 0, 0};
        DataCopyPad(tempBetaGm[1 + batchId * alphaLength], intResProbTensor.ReinterpretCast<float>(),
                    copyParamsResProbOut);
        MTE3ToMTE2Sync();
        DataCopyParams copyParamsResProbIn = {1, static_cast<uint16_t>((DOUBLE * targetLength + 1) * FLOAT_SIZE), 0, 0};
        DataCopyPad(intResProbTensor.ReinterpretCast<float>(), tempBetaGm[0 + batchId * alphaLength],
                    copyParamsResProbIn, padParams);
        MTE2ToSSync();
        intResProbTensor.ReinterpretCast<float>().SetValue(0, logProbBlank);
        // Assign the log_prob values for all even positions using cast and mask.
        SToVSync();
        // logbeta + log_prob
        Add(logBetaTensor, logBetaTensor, intResProbTensor.ReinterpretCast<float>(), doubleSi);
        PipeBarrier<PIPE_V>();
        VToMTE3Sync();
    } else if ((t < inputLength - 1) && (targetLength == 0)) {
        DataCopyParams copyParamsOneBeta = {1, static_cast<uint16_t>(FLOAT_SIZE), 0, 0};
        DataCopyPad(logBeta1Tensor, logBetaGm[batchId * alphaLength + 0], copyParamsOneBeta, padParams);
        MTE2ToSSync();
        float formalBlankBeta = logBeta1Tensor.GetValue(0);
        VToSSync();
        logBetaTensor.SetValue(0, formalBlankBeta + logProbBlank);
        SToMTE3Sync();
    }
    DataCopyParams copyParamsDouble = {1, static_cast<uint16_t>(alphaLength * FLOAT_SIZE), 0, 0};
    DataCopyPad(logBetaGm[batchId * alphaLength], logBetaTensor, copyParamsDouble);
    
    // Compute the gradient, first calculate lcab, then update the gradient.
    MTE2ToVSync();
    Add(logAlphaTensor, logAlphaTensor, logBetaTensor, doubleSi);
    
    int64_t sliceLengthCur = sliceLength;
    for (int64_t sliceId = 0; sliceId < probSliceNum; sliceId++) {
        if (sliceId == probSliceNum - 1) {
            sliceLengthCur = sliceLengthTail;
        }
        updateLcab(t, sliceId, sliceLengthCur, logAlphaTensor, gradSliceTensor);
        CalcGrad(t, sliceId, sliceLengthCur, batchOffsetProb, logProbSliceTensor, gradSliceTensor);
    }

    logProbSliceQue.FreeTensor(logProbSliceTensor);
    logAlphaQue.FreeTensor(logAlphaTensor);
    gradSliceQue.FreeTensor(gradSliceTensor);
}

__aicore__ inline void CopyInCast(int64_t gmOffset, int64_t ubOffset, int64_t castLength,
                                const LocalTensor<float>& dstTensor,
                                const GlobalTensor<TGrad>& srcGm)
{
    DataCopyExtParams copyParamsProb = {1, static_cast<uint32_t>(castLength * gradDsize), 0, 0, 0};
    if constexpr (std::is_same<TGrad, float>::value) {
        DataCopyPad(dstTensor, srcGm[gmOffset], copyParamsProb, padParamsProb);
        MTE2ToVSync();
    } else {
        DataCopyPad(dstTensor.ReinterpretCast<TGrad>()[ubOffset],
                    srcGm[gmOffset],
                    copyParamsProb, padParamsProb);
        MTE2ToVSync();
        Cast(dstTensor, dstTensor.ReinterpretCast<TGrad>()[ubOffset], RoundMode::CAST_NONE, castLength);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void GatherLogBeta(int64_t t, int64_t batchOffsetProb, const LocalTensor<float>& logProbSliceTensor)
{   
    int64_t sliceLengthCur = sliceLength;
    Cast(castFloatTargetsTensor, targetsTensor, RoundMode::CAST_ROUND, targetLength);
    PipeBarrier<PIPE_V>();
    Muls(castFloatTargetsTensor, castFloatTargetsTensor, GATHER_FLOAT_SIZE, targetLength);
    for (int64_t sliceId = 0; sliceId < probSliceNum; sliceId++) {
        float upLimit = sliceId >= (probSliceNum - 1) ? symbolSet * FLOAT_SIZE : sliceLength * (1 + sliceId) *
                        FLOAT_SIZE;
        float negDownLimit = -sliceLength * sliceId * FLOAT_SIZE;
        float downLimit = sliceLength * sliceId * FLOAT_SIZE;
        if (sliceId >= probSliceNum - 1) {
            sliceLengthCur = sliceLengthTail;
        }
        PipeBarrier<PIPE_V>();
        CompareScalar(downMaskTensor, castFloatTargetsTensor, downLimit, CMPMODE::GE, siAlign);
        CompareScalar(upMaskTensor, castFloatTargetsTensor, upLimit, CMPMODE::LT, siAlign);
        // Determine the positions of valid data through comparison.
        Adds(changedTargetsTensor, castFloatTargetsTensor, negDownLimit, targetLength);
        // Subtract the threshold from uintTargetsTensor
        PipeBarrier<PIPE_V>();
        Select(changedTargetsTensor, downMaskTensor, changedTargetsTensor, 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, siAlign);
        PipeBarrier<PIPE_V>();
        Select(changedTargetsTensor, upMaskTensor, changedTargetsTensor, 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, siAlign);
        // Correct invalid data
        PipeBarrier<PIPE_V>();
        Cast(intTargetsTensor, changedTargetsTensor, RoundMode::CAST_ROUND, targetLength);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> uintTargetsTensor = intTargetsTensor.ReinterpretCast<uint32_t>();
        CopyInCast(batchOffsetProb + t * symbolSet * batchSize + sliceLength * sliceId,
                   sliceLengthAlign, sliceLengthCur, logProbSliceTensor, logProbsGm);
        Gather(resPartProbTensor, logProbSliceTensor, uintTargetsTensor, 0, targetLength);
        VToMTE2Sync();
        PipeBarrier<PIPE_V>();
        Select(resPartProbTensor, downMaskTensor, resPartProbTensor, resProbTensor,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, siAlign);
        PipeBarrier<PIPE_V>();
        Select(resProbTensor, upMaskTensor, resPartProbTensor, resProbTensor,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, siAlign);
        // The output results will retain the original values from resProbTensor,
        // while the rest are updated with the newly computed values.
        // This ensures all the correct values are obtained in the end.
    }

    Cast(intResProbTensor, resProbTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, targetLength);
    PipeBarrier<PIPE_V>();
    Select(intResProbTensor.ReinterpretCast<float>(), evenMaskTensor, intResProbTensor.ReinterpretCast<float>(),
           logProbBlank, SELMODE::VSEL_TENSOR_SCALAR_MODE, doubleSiAlign);
}

__aicore__ inline void CalcGrad(int64_t t, int64_t sliceId, int64_t sliceLengthCur, int64_t batchOffsetProb,
                                const LocalTensor<float>& logProbSliceTensor, const LocalTensor<float>& gradSliceTensor)
{
    // res = (exp(lp) - exp(res + nll - lp)) * gr
    CopyInCast(batchOffsetProb + t * symbolSet * batchSize + sliceLength * sliceId,  sliceLengthAlign, 
               sliceLengthCur,  logProbSliceTensor, logProbsGm);
    MTE2ToVSync();
    PipeBarrier<PIPE_V>();
    Adds(gradSliceTensor, gradSliceTensor, nll, sliceLengthCur);
    PipeBarrier<PIPE_V>();
    Sub(gradSliceTensor, gradSliceTensor, logProbSliceTensor, sliceLengthCur);
    Exp(logProbSliceTensor, logProbSliceTensor, sliceLengthCur);
    PipeBarrier<PIPE_V>();
    Exp(gradSliceTensor, gradSliceTensor, sliceLengthCur);
    PipeBarrier<PIPE_V>();
    Sub(gradSliceTensor, logProbSliceTensor, gradSliceTensor, sliceLengthCur);
    PipeBarrier<PIPE_V>();
    Muls(gradSliceTensor, gradSliceTensor, gradOutCurBatch, sliceLengthCur);
    if constexpr (std::is_same<TGrad, half>::value) {
        PipeBarrier<PIPE_V>();
        Cast(gradSliceTensor.ReinterpretCast<TGrad>(), gradSliceTensor, RoundMode::CAST_NONE, sliceLength);
    } else if constexpr (std::is_same<TGrad, bfloat16_t>::value) {
        PipeBarrier<PIPE_V>();
        Cast(gradSliceTensor.ReinterpretCast<TGrad>(), gradSliceTensor, RoundMode::CAST_RINT, sliceLength);
    }
    VToMTE3Sync();
    DataCopyExtParams copyParamsProb = {1, static_cast<uint32_t>(sliceLengthCur * gradDsize), 0, 0, 0};
    if constexpr (std::is_same<TGrad, float>::value) {
        DataCopyPad(gradGm[batchOffsetProb + t * symbolSet * batchSize + sliceLength * sliceId],
                    gradSliceTensor, copyParamsProb);
    } else {
        DataCopyPad(gradGm[batchOffsetProb + t * symbolSet * batchSize + sliceLength * sliceId],
                    gradSliceTensor.ReinterpretCast<TGrad>(), copyParamsProb);
    }
    MTE3ToVSync();
}

__aicore__ inline void GetMaxTargetLength(const LocalTensor<TIndices>& targetLengthsTensor,
                                          const  LocalTensor<int64_t>& targetOffsetTensor)
{
    // Calcu the real max target length.
    int64_t offsetTmp = 0;
    targetOffsetTensor.SetValue(0, 0);
    for (int32_t batchId = 0; batchId < batchSize; batchId++) {
        int64_t curTargetLength = targetLengthsTensor.GetValue(batchId);
        offsetTmp += curTargetLength;
        if (targetsDimNum == 1 && batchId < (batchSize - 1)) {
            targetOffsetTensor.SetValue(batchId + 1, offsetTmp);
        }
        maxTargetLength = maxTargetLength > curTargetLength ? maxTargetLength : curTargetLength;
    }
    maxTargetLengthBlockAlign = CeilAlign(maxTargetLength + 1, FLOAT_NUM_PER_BLOCK);
    maxTargetLengthAlign = CeilAlign(maxTargetLength, NUM_PER_REPEAT);
    maskNumTarget = CeilAlign(maxTargetLengthAlign / MASK_NUM_PER_BYTES, BLOCK_BYTES);
}

__aicore__ inline void CreateDuplicateTensor()
{
    Duplicate<float>(allNeginfTensor, -INFINITY, FLOAT_NUM_PER_BLOCK);
    Duplicate<float>(allZeroTensor, 0.0f, FLOAT_NUM_PER_BLOCK);
    Duplicate<int32_t>(allOneTensor.ReinterpretCast<int32_t>()[CeilAlign(maxTargetLength + 1, HALF_NUM_PER_REPEAT)],
                       1, maxTargetLength + 1);
    PipeBarrier<PIPE_V>();
    Cast(allOneTensor, allOneTensor.ReinterpretCast<int32_t>()[CeilAlign(maxTargetLength + 1, HALF_NUM_PER_REPEAT)],
         RoundMode::CAST_NONE, maxTargetLength + 1);
    PipeBarrier<PIPE_V>();
    CompareScalar(evenMaskTensor, allOneTensor.ReinterpretCast<int32_t>(), 1, CMPMODE::EQ,
                  CeilAlign(maxTargetLength + 1, HALF_NUM_PER_REPEAT) * DOUBLE);
}

__aicore__ inline void CopyInTargetsTensor(int64_t batchOffsetTargets, int64_t batchId)
{
    // Load the targets and generate the mask for ifhavethree.
    DataCopyParams copyParamsSi = {1, static_cast<uint16_t>(targetLength * targetDsize), 0, 0};
    DataCopyParams copyParamsSiShift = {1, static_cast<uint16_t>((targetLength - 1) * targetDsize), 0, 0};
    if constexpr (std::is_same<TIndices, int32_t>::value) {
        if (targetLength > 1) {
            DataCopyPad(targetsTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                        targetsGm[batchOffsetTargets], copyParamsSi, padParams);
            DataCopyPad(targetsShiftTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                        targetsGm[batchOffsetTargets + 1], copyParamsSiShift, padParams);
            MTE2ToVSync();
            Cast(targetsTensor, targetsTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                    RoundMode::CAST_NONE, targetLength);
            Cast(targetsShiftTensor, targetsShiftTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                    RoundMode::CAST_NONE, targetLength);
        } else {
            DataCopyPad(targetsTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                        targetsGm[batchOffsetTargets], copyParamsSi, padParams);
            MTE2ToVSync();
            Cast(targetsTensor, targetsTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                    RoundMode::CAST_NONE, targetLength);
        }
        VToMTE3Sync();
    } else {
        if (targetLength > 1) {
            DataCopyPad(targetsTensor, targetsGm[batchOffsetTargets], copyParamsSi, padParams);
            DataCopyPad(targetsShiftTensor, targetsGm[batchOffsetTargets + 1], copyParamsSiShift, padParams);
        } else {
            DataCopyPad(targetsTensor, targetsGm[batchOffsetTargets], copyParamsSi, padParams);
        }
        MTE2ToMTE3Sync();
    }
    DataCopyParams copyParamsDoubleSiOut = {1, static_cast<uint16_t>((doubleSi - 1) * FLOAT_SIZE), 0, 0};
    DataCopyParams copyParamsDoubleSiIn = {1, static_cast<uint16_t>((doubleSi) * FLOAT_SIZE), 0, 0};
    targetsDoubleTensor = targetsDoubleBuf.Get<int32_t>();
    DataCopyPad(tempTargetsGm[batchId * DOUBLE * alphaLength + 1], targetsTensor.ReinterpretCast<int32_t>(),
                copyParamsDoubleSiOut);
    DataCopyPad(tempTargetsGm[batchId * DOUBLE * alphaLength + alphaLength + 1],
                targetsShiftTensor.ReinterpretCast<int32_t>(), copyParamsDoubleSiOut);
    MTE3ToMTE2Sync();
    DataCopyPad(targetsDoubleTensor, tempTargetsGm[batchId * DOUBLE * alphaLength + 0], copyParamsDoubleSiIn, padParams);
    DataCopyPad(targetsShiftTensor.ReinterpretCast<int32_t>(),
                tempTargetsGm[batchId * DOUBLE * alphaLength + alphaLength + 0], copyParamsDoubleSiIn, padParams);
    MTE2ToSSync();
    targetsDoubleTensor.SetValue(0, 0);
    targetsShiftTensor.ReinterpretCast<int32_t>().SetValue(0, 0);
    targetsShiftTensor.ReinterpretCast<int32_t>().SetValue(DOUBLE * targetLength - 1,
                                                           targetsDoubleTensor.GetValue(DOUBLE * targetLength - 1));
    targetsShiftTensor.ReinterpretCast<int32_t>().SetValue(DOUBLE * targetLength, 0);
    SToVSync();
    Compare(ifHaveThreeMaskTensor, targetsDoubleTensor,
            targetsShiftTensor.template ReinterpretCast<int32_t>(), CMPMODE::EQ, doubleSiAlign);
    PipeBarrier<PIPE_V>();
}
};
}
#endif  // CTC_LOSS_V3_GRAD_H_
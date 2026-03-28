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
 * \file batch_norm_v3_full_reduce.h
 * \brief
 */

#ifndef BATCH_NORM_V3_FULL_REDUCE_H
#define BATCH_NORM_V3_FULL_REDUCE_H

#include "batch_norm_v3_base.h"

namespace BatchNormV3Ops {
using namespace AscendC;

#define BRC_IMPL(FUNC_IMPL, D_T, S0_T, S1_T, LINE_LENGTH, I_LOOP, I_REMAIN, J_LOOP, J_REMAIN, REP_STRIDE) \
    do {                                                                                                  \
        /* 切分保证I_LOOP和J_LOOP不会同时大于0 */                                              \
        for (int64_t i = 0; i < (I_LOOP); i++) {                                                            \
            FUNC_IMPL((D_T)[i * UINT8_MAX_NUM * (LINE_LENGTH)],                                               \
                (S0_T)[i * UINT8_MAX_NUM * (LINE_LENGTH)],                                                    \
                S1_T,                                                                                     \
                J_REMAIN,                                                                                 \
                UINT8_MAX_NUM,                                                                            \
                {1, 1, 1, REP_STRIDE, REP_STRIDE, 0});                                                    \
        }                                                                                                 \
        if (I_REMAIN) {                                                                                   \
            for (int64_t j = 0; j < (J_LOOP); j++) {                                                        \
                FUNC_IMPL((D_T)[j * ELEM_PER_REP_FP32],                                                     \
                    (S0_T)[j * ELEM_PER_REP_FP32],                                                          \
                    (S1_T)[j * ELEM_PER_REP_FP32],                                                          \
                    ELEM_PER_REP_FP32,                                                                    \
                    I_REMAIN,                                                                             \
                    {1, 1, 1, REP_STRIDE, REP_STRIDE, 0});                                                \
            }                                                                                             \
            if (J_REMAIN) {                                                                               \
                FUNC_IMPL((D_T)[(I_LOOP) * UINT8_MAX_NUM * (LINE_LENGTH) + (J_LOOP) * ELEM_PER_REP_FP32],         \
                    (S0_T)[(I_LOOP) * UINT8_MAX_NUM * (LINE_LENGTH) + (J_LOOP) * ELEM_PER_REP_FP32],              \
                    (S1_T)[(J_LOOP) * ELEM_PER_REP_FP32],                                                     \
                    J_REMAIN,                                                                             \
                    I_REMAIN,                                                                             \
                    {1, 1, 1, REP_STRIDE, REP_STRIDE, 0});                                                \
            }                                                                                             \
        }                                                                                                 \
    } while (0)

template <typename T1, typename T2, int PARALLEL_MODE>
class BatchNormV3FullReduce : public BatchNormV3Base<T1, T2> {
public:
    __aicore__ inline BatchNormV3FullReduce(TPipe *pipe)
    {
        this->pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
        GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR save_mean, GM_ADDR save_var,
        const BatchNormV3FullReduceTilingData *__restrict tilingData)
    {
        patternR1 = tilingData->patternR1;
        patternA = tilingData->patternA;
        patternR0 = tilingData->patternR0;
        patternR0Align = tilingData->patternR0Align;
        aUbFactor = tilingData->aUbFactor;
        forLoopNum = patternR0 / ELEM_PER_REP_FP32;
        forRemainNum = patternR0 % ELEM_PER_REP_FP32;
        repeatForLoopNum = patternR1 / UINT8_MAX_NUM;
        repeatForRemainNum = patternR1 % UINT8_MAX_NUM;
        repStride = patternR0Align / B32_BLOCK_ALIGN_NUM;
        if (this->blockIdx == this->useCoreNum - 1) {
            aUbLoop = tilingData->tailCoreAUbLoop;
            aUbTail = tilingData->tailCoreAUbTail;
        } else {
            aUbLoop = tilingData->aUbLoop;
            aUbTail = tilingData->aUbTail;
        }
        aUbSize = tilingData->aUbSize;
        rUbSize = tilingData->rUbSize;

        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;
        this->momentumReverse = tilingData->momentumReverse;
        // R = 1场景, 0除0生成Nan，running_var输出为Nan
        this->batchVarScale = (patternR1 * patternR0 == 1) ? static_cast<float>(0.0 / 0.0) : tilingData->batchVarScale;
        dichotomizeAddDiffSize = tilingData->dichotomizeAddDiffSize;
        coefficient0 = tilingData->coefficient0;
        coefficient1 = tilingData->coefficient1;
        uint64_t aGmBlockOffset =
            static_cast<uint64_t>(this->blockIdx) * static_cast<uint64_t>(tilingData->blockFactor);
        uint64_t aR0GmBlockOffset = aGmBlockOffset * patternR0;
        this->xGm.SetGlobalBuffer((__gm__ T1 *)x + aR0GmBlockOffset);
        this->weightGm.SetGlobalBuffer((__gm__ T2 *)weight + aGmBlockOffset);
        this->biasGm.SetGlobalBuffer((__gm__ T2 *)bias + aGmBlockOffset);
        this->runningMeanGm.SetGlobalBuffer((__gm__ float *)mean + aGmBlockOffset);
        this->runningVarGm.SetGlobalBuffer((__gm__ float *)var + aGmBlockOffset);

        this->yGm.SetGlobalBuffer((__gm__ T1 *)y + aR0GmBlockOffset);
        this->saveMeanGm.SetGlobalBuffer((__gm__ float *)save_mean + aGmBlockOffset);
        this->saveVarGm.SetGlobalBuffer((__gm__ float *)save_var + aGmBlockOffset);
        this->runningMeanOutGm.SetGlobalBuffer((__gm__ float *)mean_out + aGmBlockOffset);
        this->runningVarOutGm.SetGlobalBuffer((__gm__ float *)var_out + aGmBlockOffset);

        this->pipe_->InitBuffer(tmpBuf0, rUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(tmpBuf1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(tmpBuf2, aUbSize * FLOAT_SIZE);
        // 输入que
        this->pipe_->InitBuffer(xQueue, DOUBLE_BUFFER, rUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningMeanInQueue, 1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningVarInQueue, 1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(weightQueue, 1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(biasQueue, 1, aUbSize * FLOAT_SIZE);
        // 输出que
        this->pipe_->InitBuffer(yQueue, 1, rUbSize * sizeof(T1));
        this->pipe_->InitBuffer(runningMeanOutQueue, 1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningVarOutQueue, 1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(saveMeanQueue, 1, aUbSize * FLOAT_SIZE);
        this->pipe_->InitBuffer(saveVarQueue, 1, aUbSize * FLOAT_SIZE);
    }

    __aicore__ inline void Process()
    {
        int64_t aGmOffset;
        int64_t aProcNum;
        for (int64_t i = 0; i < aUbLoop; i++) {
            aGmOffset = i * aUbFactor;
            if (unlikely(i == aUbLoop - 1)) {
                aProcNum = aUbTail;
            } else {
                aProcNum = aUbFactor;
            }
            CopyInXPhase(aProcNum, aGmOffset);
            CopyInWeightBiasPhase(aProcNum, aGmOffset);
            CopyInRunningMeanVarPhase(aProcNum, aGmOffset);
            ComputeMeanPhase(aProcNum);
            ComputeVarAndYPhase(aProcNum);
            CopyOutYPhase(aProcNum, aGmOffset);
            CopyOutSaveMeanVarPhase(aProcNum, aGmOffset);
            ComputeRunningMeanVarPhase(aProcNum);
            CopyOutRunningMeanVarPhase(aProcNum, aGmOffset);
        }
    }

private:
    __aicore__ inline void CopyInXPhase(int64_t aProcNum, int64_t aGmOffset)
    {
        // -----------------------搬入x并Cast---------------------------
        xTensor = xQueue.AllocTensor<float>();
        if constexpr (!IsSameType<T1, float>::value) {
            LocalTensor<T1> xTensorHalf = xTensor.template ReinterpretCast<T1>();
            if constexpr (PARALLEL_MODE == FULL_REDUCE_NORMAL_MODE) {
                CopyInX(xTensorHalf[rUbSize], aProcNum, aGmOffset);
            } else {
                int64_t aR0Align = (aProcNum * patternR0 + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
                CopyInXAParallel(xTensorHalf[rUbSize], aProcNum * patternR0, aR0Align, aGmOffset);
            }
            xQueue.EnQue(xTensorHalf);
            xTensorHalf = xQueue.DeQue<T1>();
            if constexpr (PARALLEL_MODE == FULL_REDUCE_A_PARALLEL_MODE) {
                int64_t calcXNum =
                    (aProcNum * patternR0 + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK * patternR1;
                Cast(xTensor, xTensorHalf[rUbSize], RoundMode::CAST_NONE, calcXNum);
            } else {
                Cast(xTensor, xTensorHalf[rUbSize], RoundMode::CAST_NONE, aProcNum * patternR1 * patternR0Align);
            }
        } else {
            if constexpr (PARALLEL_MODE == FULL_REDUCE_NORMAL_MODE) {
                CopyInX(xTensor, aProcNum, aGmOffset);
            } else {
                int64_t aR0Align = (aProcNum * patternR0 + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
                CopyInXAParallel(xTensor, aProcNum * patternR0, aR0Align, aGmOffset);
            }
            xQueue.EnQue(xTensor);
        }
    }

    __aicore__ inline void CopyInWeightBiasPhase(int64_t aProcNum, int64_t aGmOffset)
    {
        // -----------------------搬入WeightBias并Cast------------------
        weightTensor = weightQueue.AllocTensor<float>();
        biasTensor = biasQueue.AllocTensor<float>();
        if constexpr (!IsSameType<T2, float>::value) {
            LocalTensor<T2> weightTensorHalf = weightTensor.template ReinterpretCast<T2>();
            LocalTensor<T2> biasTensorHalf = biasTensor.template ReinterpretCast<T2>();
            CopyInWeightOrBias(weightTensorHalf[aUbSize], this->weightGm, aProcNum, aGmOffset);
            weightQueue.EnQue(weightTensorHalf);
            weightTensorHalf = weightQueue.DeQue<T2>();
            Cast(weightTensor, weightTensorHalf[aUbSize], RoundMode::CAST_NONE, aProcNum);
            CopyInWeightOrBias(biasTensorHalf[aUbSize], this->biasGm, aProcNum, aGmOffset);
            biasQueue.EnQue(biasTensorHalf);
            biasTensorHalf = biasQueue.DeQue<T2>();
            Cast(biasTensor, biasTensorHalf[aUbSize], RoundMode::CAST_NONE, aProcNum);
        } else {
            CopyInWeightOrBias(weightTensor, this->weightGm, aProcNum, aGmOffset);
            CopyInWeightOrBias(biasTensor, this->biasGm, aProcNum, aGmOffset);
            if constexpr (PARALLEL_MODE == FULL_REDUCE_NORMAL_MODE) {
                eventIdMte2toS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
                SetFlag<HardEvent::MTE2_S>(eventIdMte2toS);
            } else {
                weightQueue.EnQue(weightTensor);
                biasQueue.EnQue(biasTensor);
            }
        }
    }

    __aicore__ inline void CopyInRunningMeanVarPhase(int64_t aProcNum, int64_t aGmOffset)
    {
        // -----------------------搬入runningMeanVar------------------
        runningMeanInTensor = runningMeanInQueue.AllocTensor<float>();
        CopyInMeanOrVar(runningMeanInTensor, this->runningMeanGm, aProcNum, aGmOffset);
        runningMeanInQueue.EnQue(runningMeanInTensor);
        runningVarInTensor = runningVarInQueue.AllocTensor<float>();
        CopyInMeanOrVar(runningVarInTensor, this->runningVarGm, aProcNum, aGmOffset);
        runningVarInQueue.EnQue(runningVarInTensor);
    }

    __aicore__ inline void ComputeMeanPhase(int64_t aProcNum)
    {
        // -----------------------计算Mean------------------
        saveMeanTensor = saveMeanQueue.AllocTensor<float>();
        reduceTensor = tmpBuf0.Get<float>();
        if constexpr (IsSameType<T1, float>::value) {
            xTensor = xQueue.DeQue<float>();
        } else {
            PipeBarrier<PIPE_V>();  // 依赖输入数据cast完成
        }
        if constexpr (PARALLEL_MODE == FULL_REDUCE_NORMAL_MODE) {
            int64_t calLineLength = patternR1 * patternR0Align;
            Muls(reduceTensor, xTensor, coefficient0, aProcNum * patternR1 * patternR0Align);
            PipeBarrier<PIPE_V>();
            DoNormalReduce(saveMeanTensor, reduceTensor, aProcNum);
            PipeBarrier<PIPE_V>();
            Muls(saveMeanTensor, saveMeanTensor, coefficient1, aProcNum);
            // 插入S等V的同步，saveMeanTensor.GetValue依赖saveMeanTensor计算完成
            TEventID eventIdVtoS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventIdVtoS);
            WaitFlag<HardEvent::V_S>(eventIdVtoS);
            for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
                finalMean = saveMeanTensor.GetValue(aNum);
                // 插入V等S的同步，SkipPadSubMean中Adds(-finalMean)依赖saveMeanTensor.GetValue完成
                TEventID eventIdStoV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(eventIdStoV);
                WaitFlag<HardEvent::S_V>(eventIdStoV);
                SkipPadSubMean(xTensor[aNum * calLineLength], -finalMean);
            }
        } else {
            int64_t aR0Align = (aProcNum * patternR0 + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
            int64_t calLineLength = aProcNum * patternR0;
            Muls(reduceTensor, xTensor, coefficient0, patternR1 * aR0Align);
            PipeBarrier<PIPE_V>();
            DoAParallelReduce(saveMeanTensor, reduceTensor, aR0Align);
            PipeBarrier<PIPE_V>();  // 依赖saveMeanTensor计算完成
            Muls(saveMeanTensor, saveMeanTensor, coefficient1, aR0Align);
            PipeBarrier<PIPE_V>();
            int64_t jLoop = calLineLength / ELEM_PER_REP_FP32;
            int64_t jRemain = calLineLength % ELEM_PER_REP_FP32;
            uint8_t repStride = static_cast<uint8_t>(aR0Align / B32_BLOCK_ALIGN_NUM);
            if (aR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM) && (jLoop < patternR1)) {
                BRC_IMPL(Sub,
                    xTensor,
                    xTensor,
                    saveMeanTensor,
                    aR0Align,
                    repeatForLoopNum,
                    repeatForRemainNum,
                    jLoop,
                    jRemain,
                    repStride);
            } else {
                for (int64_t r1Idx = 0; r1Idx < patternR1; r1Idx++) {
                    Sub(xTensor[r1Idx * aR0Align], xTensor[r1Idx * aR0Align], saveMeanTensor, calLineLength);
                }
            }
        }
    }

    __aicore__ inline void ComputeVarAndYPhase(int64_t aProcNum)
    {
        // -----------------------计算Var和Y------------------
        saveVarTensor = saveVarQueue.AllocTensor<float>();
        yTensor = yQueue.AllocTensor<T1>();
        PipeBarrier<PIPE_V>();  // 保证xTensor的vector计算完成
        RoundMode b16RoundMode = IsSameType<T1, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
        if constexpr (PARALLEL_MODE == FULL_REDUCE_NORMAL_MODE) {
            int64_t calcXNum = aProcNum * patternR1 * patternR0Align;
            int64_t calLineLength = patternR1 * patternR0Align;
            Mul(reduceTensor, xTensor, xTensor, calcXNum);
            PipeBarrier<PIPE_V>();
            Muls(reduceTensor, reduceTensor, coefficient0, calcXNum);
            PipeBarrier<PIPE_V>();
            DoNormalReduce(saveVarTensor, reduceTensor, aProcNum);
            PipeBarrier<PIPE_V>();
            Muls(saveVarTensor, saveVarTensor, coefficient1, aProcNum);
            TEventID eventIdVtoS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventIdVtoS);
            WaitFlag<HardEvent::V_S>(eventIdVtoS);
            if constexpr (IsSameType<T2, float>::value) {
                WaitFlag<HardEvent::MTE2_S>(eventIdMte2toS);
            }
            for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
                weightValue = weightTensor.GetValue(aNum);
                biasValue = biasTensor.GetValue(aNum);
                finalVar = saveVarTensor.GetValue(aNum);
                weightMulInvstd = static_cast<float>(weightValue) / sqrt(finalVar + static_cast<float>(this->epsilon));
                TEventID eventIdStoV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(eventIdStoV);
                WaitFlag<HardEvent::S_V>(eventIdStoV);
                Muls(xTensor[aNum * calLineLength], xTensor[aNum * calLineLength], weightMulInvstd, calLineLength);
                PipeBarrier<PIPE_V>();
                if constexpr (!IsSameType<T1, float>::value) {
                    Adds(xTensor[aNum * calLineLength], xTensor[aNum * calLineLength], biasValue, calLineLength);
                    PipeBarrier<PIPE_V>();
                    Cast(yTensor[aNum * calLineLength], xTensor[aNum * calLineLength], b16RoundMode, calLineLength);
                } else {
                    Adds(yTensor[aNum * calLineLength], xTensor[aNum * calLineLength], biasValue, calLineLength);
                }
            }
        } else {
            int64_t aR0Align = (aProcNum * patternR0 + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
            int64_t calLineLength = aProcNum * patternR0;
            int64_t calcXNum = patternR1 * aR0Align;
            Mul(reduceTensor, xTensor, xTensor, calcXNum);
            PipeBarrier<PIPE_V>();
            Muls(reduceTensor, reduceTensor, coefficient0, calcXNum);
            PipeBarrier<PIPE_V>();
            DoAParallelReduce(saveVarTensor, reduceTensor, aR0Align);
            PipeBarrier<PIPE_V>();
            Muls(saveVarTensor, saveVarTensor, coefficient1, aR0Align);
            PipeBarrier<PIPE_V>();
            stdTensor = runningMeanOutQueue.AllocTensor<float>();
            Adds(stdTensor, saveVarTensor, this->epsilon, calLineLength);
            PipeBarrier<PIPE_V>();
            Sqrt(stdTensor, stdTensor, calLineLength);
            PipeBarrier<PIPE_V>();
            if constexpr (IsSameType<T2, float>::value) {
                weightTensor = weightQueue.DeQue<float>();
                biasTensor = biasQueue.DeQue<float>();
            }
            int64_t jLoop = calLineLength / ELEM_PER_REP_FP32;
            int64_t jRemain = calLineLength % ELEM_PER_REP_FP32;
            uint8_t repStride = static_cast<uint8_t>(aR0Align / B32_BLOCK_ALIGN_NUM);
            if (aR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM) && (jLoop < patternR1)) {
                BRC_IMPL(Div,
                    xTensor,
                    xTensor,
                    stdTensor,
                    aR0Align,
                    repeatForLoopNum,
                    repeatForRemainNum,
                    jLoop,
                    jRemain,
                    repStride);
                PipeBarrier<PIPE_V>();
                BRC_IMPL(Mul,
                    xTensor,
                    xTensor,
                    weightTensor,
                    aR0Align,
                    repeatForLoopNum,
                    repeatForRemainNum,
                    jLoop,
                    jRemain,
                    repStride);
                PipeBarrier<PIPE_V>();
                if constexpr (!IsSameType<T1, float>::value) {
                    BRC_IMPL(Add,
                        xTensor,
                        xTensor,
                        biasTensor,
                        aR0Align,
                        repeatForLoopNum,
                        repeatForRemainNum,
                        jLoop,
                        jRemain,
                        repStride);
                    PipeBarrier<PIPE_V>();
                    Cast(yTensor, xTensor, b16RoundMode, calcXNum);
                } else {
                    BRC_IMPL(Add,
                        yTensor,
                        xTensor,
                        biasTensor,
                        aR0Align,
                        repeatForLoopNum,
                        repeatForRemainNum,
                        jLoop,
                        jRemain,
                        repStride);
                }
            } else {
                for (int64_t r1Idx = 0; r1Idx < patternR1; r1Idx++) {
                    Div(xTensor[r1Idx * aR0Align], xTensor[r1Idx * aR0Align], stdTensor, calLineLength);
                    PipeBarrier<PIPE_V>();
                    Mul(xTensor[r1Idx * aR0Align], xTensor[r1Idx * aR0Align], weightTensor, calLineLength);
                    PipeBarrier<PIPE_V>();
                    if constexpr (!IsSameType<T1, float>::value) {
                        Add(xTensor[r1Idx * aR0Align], xTensor[r1Idx * aR0Align], biasTensor, calLineLength);
                        PipeBarrier<PIPE_V>();
                        Cast(yTensor[r1Idx * aR0Align], xTensor[r1Idx * aR0Align], b16RoundMode, calLineLength);
                    } else {
                        Add(yTensor[r1Idx * aR0Align], xTensor[r1Idx * aR0Align], biasTensor, calLineLength);
                    }
                }
            }
            runningMeanOutQueue.FreeTensor(stdTensor);
        }
        xQueue.FreeTensor(xTensor);
        weightQueue.FreeTensor(weightTensor);
        biasQueue.FreeTensor(biasTensor);
    }

    __aicore__ inline void ComputeRunningMeanVarPhase(int64_t aProcNum)
    {
        // -----------------------计算RunningMeanVar-----------------------
        runningMeanInTensor = runningMeanInQueue.DeQue<float>();
        Muls(runningMeanInTensor, runningMeanInTensor, this->momentumReverse, aProcNum);
        runningVarInTensor = runningVarInQueue.DeQue<float>();
        Muls(runningVarInTensor, runningVarInTensor, this->momentumReverse, aProcNum);
        momentumMeanTensor = tmpBuf1.Get<float>();
        PipeBarrier<PIPE_V>();  // 依赖saveMeanTensor，saveVarTensor计算完成
        Muls(momentumMeanTensor, saveMeanTensor, this->momentum, aProcNum);
        saveMeanQueue.FreeTensor(saveMeanTensor);
        momentumVarTensor = tmpBuf2.Get<float>();
        Muls(momentumVarTensor, saveVarTensor, this->batchVarScale * this->momentum, aProcNum);
        saveVarQueue.FreeTensor(saveVarTensor);
        PipeBarrier<PIPE_V>();
        runningMeanOutTensor = runningMeanOutQueue.AllocTensor<float>();
        Add(runningMeanOutTensor, runningMeanInTensor, momentumMeanTensor, aProcNum);
        runningMeanInQueue.FreeTensor(runningMeanInTensor);
        runningVarOutTensor = runningVarOutQueue.AllocTensor<float>();
        Add(runningVarOutTensor, runningVarInTensor, momentumVarTensor, aProcNum);
        runningVarInQueue.FreeTensor(runningVarInTensor);
    }

    __aicore__ inline void CopyOutYPhase(int64_t aProcNum, int64_t aGmOffset)
    {
        // -----------------------搬出Y----------------------------
        yQueue.EnQue(yTensor);
        yTensor = yQueue.DeQue<T1>();
        if constexpr (PARALLEL_MODE == FULL_REDUCE_NORMAL_MODE) {
            CopyOutY(yTensor, aProcNum, aGmOffset);
        } else {
            CopyOutYAParallel(yTensor, aProcNum * patternR0, aGmOffset);
        }
        yQueue.FreeTensor(yTensor);
    }

    __aicore__ inline void CopyOutSaveMeanVarPhase(int64_t aProcNum, int64_t aGmOffset)
    {
        // -----------------------搬出SaveMeanVar-----------------------
        saveMeanQueue.EnQue(saveMeanTensor);
        saveMeanTensor = saveMeanQueue.DeQue<float>();
        CopyOutMeanOrVar(this->saveMeanGm, saveMeanTensor, aProcNum, aGmOffset);
        saveVarQueue.EnQue(saveVarTensor);
        saveVarTensor = saveVarQueue.DeQue<float>();
        CopyOutMeanOrVar(this->saveVarGm, saveVarTensor, aProcNum, aGmOffset);
    }

    __aicore__ inline void CopyOutRunningMeanVarPhase(int64_t aProcNum, int64_t aGmOffset)
    {
        // -----------------------搬出RunningMeanVar-----------------------
        runningMeanOutQueue.EnQue(runningMeanOutTensor);
        runningMeanOutTensor = runningMeanOutQueue.DeQue<float>();
        CopyOutMeanOrVar(this->runningMeanOutGm, runningMeanOutTensor, aProcNum, aGmOffset);
        runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
        runningVarOutQueue.EnQue(runningVarOutTensor);
        runningVarOutTensor = runningVarOutQueue.DeQue<float>();
        CopyOutMeanOrVar(this->runningVarOutGm, runningVarOutTensor, aProcNum, aGmOffset);
        runningVarOutQueue.FreeTensor(runningVarOutTensor);
    }

    __aicore__ inline void CopyInX(const LocalTensor<T1> &inTensor, int64_t aProcNum, int64_t aGmOffset)
    {
        DataCopyPadExtParams<T1> padParams = {true, 0, static_cast<uint8_t>(patternR0Align - patternR0), 0};
        DataCopyExtParams intriParams;
        if (aProcNum <= patternR1) {
            intriParams.blockCount = static_cast<uint16_t>(patternR1);
            intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
            intriParams.srcStride = static_cast<uint32_t>((patternA - 1) * patternR0 * sizeof(T1));
            intriParams.dstStride = 0;
            for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
                DataCopyPad(inTensor[patternR1 * patternR0Align * aNum],
                    this->xGm[(aGmOffset + aNum) * patternR0],
                    intriParams,
                    padParams);
            }
        } else {
            intriParams.blockCount = static_cast<uint16_t>(aProcNum);
            intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
            intriParams.srcStride = 0;
            intriParams.dstStride = static_cast<uint32_t>((patternR1 - 1) * patternR0Align / X_NUM_PER_BLOCK);
            for (int64_t r1Idx = 0; r1Idx < patternR1; r1Idx++) {
                DataCopyPad(inTensor[r1Idx * patternR0Align],
                    this->xGm[(r1Idx * patternA + aGmOffset) * patternR0],
                    intriParams,
                    padParams);
            }
        }
    }

    __aicore__ inline void CopyInXAParallel(
        const LocalTensor<T1> &inTensor, int64_t eleNum, int64_t eleNumAlign, int64_t aGmOffset)
    {
        DataCopyPadExtParams<T1> padParams = {true, 0, static_cast<uint8_t>(eleNumAlign - eleNum), 0};
        DataCopyExtParams intriParams;
        // 全载模板 patternR1不会大于65535
        intriParams.blockCount = static_cast<uint16_t>(patternR1);
        intriParams.blockLen = static_cast<uint32_t>(eleNum * sizeof(T1));
        intriParams.srcStride = static_cast<uint32_t>((patternA * patternR0 - eleNum) * sizeof(T1));
        intriParams.dstStride = 0;
        DataCopyPad(inTensor, this->xGm[aGmOffset * patternR0], intriParams, padParams);
    }

    __aicore__ inline void CopyOutY(LocalTensor<T1> &outTensor, int64_t aProcNum, int64_t aGmOffset)
    {
        DataCopyExtParams intriParams;
        if (aProcNum <= patternR1) {
            intriParams.blockCount = static_cast<uint16_t>(patternR1);
            intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
            intriParams.srcStride = 0;
            intriParams.dstStride = static_cast<uint32_t>((patternA - 1) * patternR0 * sizeof(T1));
            for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
                DataCopyPad(this->yGm[(aGmOffset + aNum) * patternR0],
                    outTensor[patternR1 * patternR0Align * aNum],
                    intriParams);
            }
        } else {
            intriParams.blockCount = static_cast<uint16_t>(aProcNum);
            intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
            intriParams.srcStride = static_cast<uint32_t>((patternR1 - 1) * patternR0Align / X_NUM_PER_BLOCK);
            intriParams.dstStride = 0;
            for (int64_t r1Idx = 0; r1Idx < patternR1; r1Idx++) {
                DataCopyPad(this->yGm[(r1Idx * patternA + aGmOffset) * patternR0],
                    outTensor[r1Idx * patternR0Align],
                    intriParams);
            }
        }
    }

    __aicore__ inline void CopyOutYAParallel(LocalTensor<T1> &outTensor, int64_t eleNum, int64_t aGmOffset)
    {
        DataCopyExtParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(patternR1);
        intriParams.blockLen = static_cast<uint32_t>(eleNum * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = static_cast<uint32_t>((patternA * patternR0 - eleNum) * sizeof(T1));
        DataCopyPad(this->yGm[aGmOffset * patternR0], outTensor, intriParams);
    }

    __aicore__ inline void CopyInWeightOrBias(
        const LocalTensor<T2> &inTensor, GlobalTensor<T2> inGm, int64_t eleNum, int64_t gmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(eleNum * sizeof(T2));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(inTensor, inGm[gmOffset], intriParams, padParams);
    }

    __aicore__ inline void CopyOutMeanOrVar(
        GlobalTensor<float> outGm, LocalTensor<float> &outTensor, int64_t eleNum, int64_t gmOffset)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(eleNum * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(outGm[gmOffset], outTensor, intriParams);
    }

    __aicore__ inline void CopyInMeanOrVar(
        LocalTensor<float> &inTensor, GlobalTensor<float> inGm, int64_t eleNum, int64_t gmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(eleNum * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(inTensor, inGm[gmOffset], intriParams, padParams);
    }

    __aicore__ inline void SkipPadSubMean(const LocalTensor<float> &calcTensor, float negMean)
    {
        /*
        函数实现对patternR1行个patternR0数据加negMean的计算（即减均值）
        计算需要跳过patternR0对齐到patternR0Align的补0数值部分，否则影响后续计算方差
        calcTensor: patternR1 * patternR0Align
        */
        if (patternR0 == patternR0Align) {
            Adds(calcTensor, calcTensor, negMean, patternR1 * patternR0);
        } else {
            if ((forLoopNum < patternR1) && (patternR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {
                // patternR1 不会大于255, 循环走进去的条件forLoopNum >= 1,则patternR0 >= 64, 64*255 > rUbSize
                for (int64_t i = 0; i < forLoopNum; i++) {
                    Adds(calcTensor[i * ELEM_PER_REP_FP32],
                        calcTensor[i * ELEM_PER_REP_FP32],
                        negMean,
                        ELEM_PER_REP_FP32,
                        patternR1,
                        {1, 1, repStride, repStride});
                }
                if (forRemainNum > 0) {
                    for (int64_t i = 0; i < repeatForLoopNum; i++) {
                        Adds(calcTensor[forLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                            calcTensor[forLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                            negMean,
                            forRemainNum,
                            UINT8_MAX_NUM,
                            {1, 1, repStride, repStride});
                    }
                    if (repeatForRemainNum > 0) {
                        Adds(calcTensor[forLoopNum * ELEM_PER_REP_FP32 +
                                        repeatForLoopNum * UINT8_MAX_NUM * patternR0Align],
                            calcTensor[forLoopNum * ELEM_PER_REP_FP32 +
                                       repeatForLoopNum * UINT8_MAX_NUM * patternR0Align],
                            negMean,
                            forRemainNum,
                            repeatForRemainNum,
                            {1, 1, repStride, repStride});
                    }
                }
            } else {
                for (int64_t i = 0; i < patternR1; i++) {
                    Adds(calcTensor[patternR0Align * i], calcTensor[patternR0Align * i], negMean, patternR0);
                }
            }
        }
    }

    __aicore__ inline void DoNormalReduce(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int64_t lineNum)
    {
        /*
        函数实现同时lineNum行的行内二分累加，累加结果为lineNum个数，每行一个最终累加结果
        srcTensor为reduce之前的Tensor: lineNum * lineLength
        dstTensor为存放reduce结果的tensor: lineNum
        dichotomizeAddDiffSize为lineLength与比他小的最邻近二次幂的差值
        */
        int64_t lineLength = patternR1 * patternR0Align;
        int64_t sumNum = lineLength;
        if (dichotomizeAddDiffSize != 0) {
            for (int64_t i = 0; i < lineNum; i++) {
                Add(srcTensor[i * lineLength],
                    srcTensor[i * lineLength],
                    srcTensor[i * lineLength + sumNum - dichotomizeAddDiffSize],
                    dichotomizeAddDiffSize);
            }
            PipeBarrier<PIPE_V>();
            sumNum = sumNum - dichotomizeAddDiffSize;
        }
        while (sumNum > ELEM_PER_REP_FP32) {
            sumNum = sumNum / TWO_NUM;
            for (int64_t i = 0; i < lineNum; i++) {
                Add(srcTensor[i * lineLength], srcTensor[i * lineLength], srcTensor[i * lineLength + sumNum], sumNum);
            }
            PipeBarrier<PIPE_V>();
        }
        int64_t repeatForLoop = lineNum / UINT8_MAX_NUM;
        int64_t repeatForRemain = lineNum % UINT8_MAX_NUM;
        for (int64_t i = 0; i < repeatForLoop; i++) {
            WholeReduceSum<float>(dstTensor[i * UINT8_MAX_NUM],
                srcTensor[i * UINT8_MAX_NUM * lineLength],
                sumNum,
                UINT8_MAX_NUM,
                1,
                1,
                lineLength / B32_BLOCK_ALIGN_NUM);
        }
        if (repeatForRemain) {
            WholeReduceSum<float>(dstTensor[repeatForLoop * UINT8_MAX_NUM],
                srcTensor[repeatForLoop * UINT8_MAX_NUM * lineLength],
                sumNum,
                repeatForRemain,
                1,
                1,
                lineLength / B32_BLOCK_ALIGN_NUM);
        }
    }

    __aicore__ inline void DoAParallelReduce(
        LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int64_t lineLength)
    {
        /*
        函数实现patternR1行的二分累加，累加结果为一行
        srcTensor为reduce之前的Tensor: patternR1 * lineLength
        dstTensor为存放reduce结果的tensor: lineLength
        */
        int64_t nowRows = patternR1;
        if (nowRows == 1) {
            Adds<float>(dstTensor, srcTensor, 0, lineLength);
            return;
        }
        // row为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            Add(srcTensor,
                srcTensor,
                srcTensor[(nowRows - dichotomizeAddDiffSize) * lineLength],
                dichotomizeAddDiffSize * lineLength);
            PipeBarrier<PIPE_V>();
            nowRows = nowRows - dichotomizeAddDiffSize;
        }
        while (nowRows > 1) {
            nowRows = nowRows / TWO_NUM;
            if (nowRows == 1) {
                Add(dstTensor, srcTensor, srcTensor[lineLength], lineLength);
            } else {
                Add(srcTensor, srcTensor, srcTensor[nowRows * lineLength], nowRows * lineLength);
                PipeBarrier<PIPE_V>();
            }
        }
    }

private:
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t X_NUM_PER_BLOCK = BLOCK_SIZE / sizeof(T1);
    constexpr static uint32_t FLOAT_SIZE = 4;
    constexpr static uint32_t TWO_NUM = 2;
    constexpr static uint32_t DOUBLE_BUFFER = 2;
    constexpr static uint32_t ELEM_PER_REP_FP32 = 64;
    constexpr static uint32_t BLOCK_NUM_PER_REP = 8;
    constexpr static uint32_t B32_BLOCK_ALIGN_NUM = 8;
    constexpr static uint32_t UINT8_MAX_NUM = 255;
    constexpr static int FULL_REDUCE_NORMAL_MODE = 0;
    constexpr static int FULL_REDUCE_A_PARALLEL_MODE = 1;

    /* shape variable */
    int64_t patternR1;
    int64_t patternA;
    int64_t patternR0;
    int64_t patternR0Align;
    /* spilt variable */
    int64_t aUbFactor;
    int64_t aUbLoop;
    int64_t aUbTail;
    int64_t aUbSize;
    int64_t rUbSize;

    int64_t forLoopNum;
    int64_t forRemainNum;
    int64_t repeatForLoopNum;
    int64_t repeatForRemainNum;
    uint8_t repStride;
    /* offset variable */
    uint64_t aUbLoopNowStartIdx;
    uint64_t xGmOffset;

    /* calculate variable */
    int64_t dichotomizeAddDiffSize;
    uint64_t acc_val = 0;
    float finalMean = 0.0;
    float finalVar = 0.0;
    float weightValue = 0.0;
    float biasValue = 0.0;
    float coefficient0 = 0.0;
    float coefficient1 = 0.0;
    float weightMulInvstd = 0.0;
    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> weightQueue;
    TQue<QuePosition::VECIN, 1> biasQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> saveMeanQueue;
    TQue<QuePosition::VECOUT, 1> saveVarQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;
    TBuf<TPosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;

    LocalTensor<float> runningMeanInTensor;
    LocalTensor<float> runningVarInTensor;
    LocalTensor<float> runningMeanOutTensor;
    LocalTensor<float> runningVarOutTensor;
    LocalTensor<float> saveMeanTensor;
    LocalTensor<float> saveVarTensor;
    LocalTensor<float> momentumMeanTensor;
    LocalTensor<float> momentumVarTensor;
    LocalTensor<float> weightTensor;
    LocalTensor<float> biasTensor;

    LocalTensor<float> xTensor;
    LocalTensor<float> reduceTensor;
    LocalTensor<float> stdTensor;
    LocalTensor<T1> yTensor;
    TEventID eventIdMte2toS;
};
}  // namespace BatchNormV3Ops
#endif  // BATCH_NORM_V3_FULL_REDUCE_H

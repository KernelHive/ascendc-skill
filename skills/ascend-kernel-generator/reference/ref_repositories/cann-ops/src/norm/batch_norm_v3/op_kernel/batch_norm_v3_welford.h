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
 * \file batch_norm_v3_welford.h
 * \brief
 */

#ifndef BATCH_NORM_V3_WELFORD_H
#define BATCH_NORM_V3_WELFORD_H

#include "batch_norm_v3_base.h"

namespace BatchNormV3Ops {
using namespace AscendC;

template <typename T1, typename T2, int SPLIT_MODE, int R0_ALIGN_MODE>
class BatchNormV3Welford : public BatchNormV3Base<T1, T2> {
public:
    __aicore__ inline BatchNormV3Welford(TPipe *pipe)
    {
        this->pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
        GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR save_mean, GM_ADDR save_var,
        const BatchNormV3WelfordTilingData *__restrict tilingData)
    {
        patternR1 = tilingData->patternR1;
        patternA = tilingData->patternA;
        patternR0 = tilingData->patternR0;
        patternR0Align = tilingData->patternR0Align;
        aUbFactor = tilingData->aUbFactor;
        if (this->blockIdx == this->useCoreNum - 1) {
            aUbLoop = tilingData->tailCoreAUbLoop;
            aUbTail = tilingData->tailCoreAUbTail;
        } else {
            aUbLoop = tilingData->aUbLoop;
            aUbTail = tilingData->aUbTail;
        }
        r0UbFactor = tilingData->r0UbFactor;
        r0UbLoop = tilingData->r0UbLoop;
        r0UbTail = tilingData->r0UbTail;
        procNR0 = tilingData->procNR0;
        nR0Loop = tilingData->nR0Loop;
        lastLoopNR0 = tilingData->lastLoopNR0;
        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;
        this->momentumReverse = tilingData->momentumReverse;
        this->batchVarScale = tilingData->batchVarScale;
        dichotomizeAddDiffSize = tilingData->dichotomizeAddDiffSize;

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

        this->pipe_->InitBuffer(tmpBuf0, r0UbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(tmpBuf3, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(tmpBuf4, aUbFactor * FLOAT_SIZE);
        // 输入que
        this->pipe_->InitBuffer(xQueue, DOUBLE_BUFFER, r0UbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningMeanInQueue, 1, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningVarInQueue, 1, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(weightQueue, 1, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(biasQueue, 1, aUbFactor * FLOAT_SIZE);
        // 输出que
        this->pipe_->InitBuffer(yQueue, DOUBLE_BUFFER, r0UbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningMeanOutQueue, 1, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(runningVarOutQueue, 1, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(saveMeanQueue, 1, aUbFactor * FLOAT_SIZE);
        this->pipe_->InitBuffer(saveVarQueue, 1, aUbFactor * FLOAT_SIZE);
    }

    __aicore__ inline void Process()
    {
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < aUbLoop; aUbLoopIdx++) {
            aUbLoopNowStartIdx = aUbLoopIdx * aUbFactor;
            if (unlikely(aUbLoopIdx == aUbLoop - 1)) {
                aProcNum = aUbTail;
            } else {
                aProcNum = aUbFactor;
            }
            saveMeanTensor = saveMeanQueue.AllocTensor<float>();
            saveVarTensor = saveVarQueue.AllocTensor<float>();
            meanTensor = tmpBuf0.Get<float>();
            m2Tensor = yQueue.AllocTensor<float>();
            CalcMeanVar();
            yQueue.FreeTensor(m2Tensor);
            CopyOutSaveMeanVar(saveMeanTensor, saveVarTensor, aProcNum, aUbLoopNowStartIdx);
            // 计算runningmean
            runningMeanInTensor = runningMeanInQueue.AllocTensor<float>();
            runningVarInTensor = runningVarInQueue.AllocTensor<float>();
            runningMeanOutTensor = runningMeanOutQueue.AllocTensor<float>();
            runningVarOutTensor = runningVarOutQueue.AllocTensor<float>();
            momentumMeanTensor = tmpBuf3.Get<float>();
            momentumVarTensor = tmpBuf4.Get<float>();
            weightTensor = weightQueue.AllocTensor<float>();
            biasTensor = biasQueue.AllocTensor<float>();
            CopyInRunningMeanVar(runningMeanInTensor, runningVarInTensor, aProcNum, aUbLoopNowStartIdx);
            Muls(runningMeanInTensor, runningMeanInTensor, this->momentumReverse, aProcNum);
            Muls(runningVarInTensor, runningVarInTensor, this->momentumReverse, aProcNum);
            Muls(momentumMeanTensor, saveMeanTensor, this->momentum, aProcNum);
            Muls(momentumVarTensor, saveVarTensor, this->batchVarScale * this->momentum, aProcNum);
            PipeBarrier<PIPE_V>();
            Add(runningMeanOutTensor, runningMeanInTensor, momentumMeanTensor, aProcNum);
            runningMeanInQueue.FreeTensor(runningMeanInTensor);
            Add(runningVarOutTensor, runningVarInTensor, momentumVarTensor, aProcNum);
            runningVarInQueue.FreeTensor(runningVarInTensor);
            CopyOutRunningMeanVar(runningMeanOutTensor, runningVarOutTensor, aProcNum, aUbLoopNowStartIdx);
            runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
            runningVarOutQueue.FreeTensor(runningVarOutTensor);
            // 计算y
            CopyInWeightBiasAndCast(weightTensor, biasTensor, aProcNum, aUbLoopNowStartIdx);
            CalcYAndCopyOut();
            saveMeanQueue.FreeTensor(saveMeanTensor);
            saveVarQueue.FreeTensor(saveVarTensor);
            weightQueue.FreeTensor(weightTensor);
            biasQueue.FreeTensor(biasTensor);
        }
    }

    __aicore__ inline void CalcMeanVar()
    {
        for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
            count = 0;
            LocalTensor<float> deltaTensor = yQueue.AllocTensor<float>();
            Duplicate<float>(meanTensor, float(0.0), r0UbFactor);
            Duplicate<float>(m2Tensor, float(0.0), r0UbFactor);
            PipeBarrier<PIPE_V>();
            if constexpr (SPLIT_MODE == R0_SPLIT_NOT_ALIGN_MODE) {
                // R0切分整块
                for (int64_t r1LoopIdx = 0; r1LoopIdx < patternR1; r1LoopIdx++) {
                    for (int64_t r0LoopIdx = 0; r0LoopIdx < r0UbLoop - 1; r0LoopIdx++) {
                        xGmOffset = r1LoopIdx * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0 +
                                    r0LoopIdx * r0UbFactor;
                        xTensor = xQueue.AllocTensor<float>();
                        CopyInXAndCast(xTensor, r0UbFactor, xGmOffset);
                        WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, r0UbFactor);
                    }
                }
                // R0切分尾块
                for (int64_t r1LoopIdx = 0; r1LoopIdx < patternR1; r1LoopIdx++) {
                    xGmOffset = r1LoopIdx * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0 +
                                (r0UbLoop - 1) * r0UbFactor;
                    xTensor = xQueue.AllocTensor<float>();
                    CopyInXAndCast(xTensor, r0UbTail, xGmOffset);
                    WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, r0UbTail);
                }
                yQueue.FreeTensor(deltaTensor);
                WelfordParallelFinalizeR0NotAlign(count, meanTensor, m2Tensor, finalMean, finalVar);
                saveMeanTensor.SetValue(aNum, finalMean);
                saveVarTensor.SetValue(aNum, finalVar);
            } else if constexpr (SPLIT_MODE == R1_SPLIT_NOT_ALIGN_MODE) {
                // R1循环整块
                r0ProcNum = procNR0 * patternR0Align;
                for (int64_t nR0LoopIdx = 0; nR0LoopIdx < nR0Loop - 1; nR0LoopIdx++) {
                    xGmOffset = nR0LoopIdx * procNR0 * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0;
                    xTensor = xQueue.AllocTensor<float>();
                    CopyInNR0AndCast(xTensor, procNR0, xGmOffset);
                    WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, r0ProcNum);
                }
                // R1循环尾块
                r0ProcNum = lastLoopNR0 * patternR0Align;
                xGmOffset = (nR0Loop - 1) * procNR0 * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0;
                xTensor = xQueue.AllocTensor<float>();
                CopyInNR0AndCast(xTensor, lastLoopNR0, xGmOffset);
                WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, r0ProcNum);
                yQueue.FreeTensor(deltaTensor);
                WelfordParallelFinalizeR1NotAlign(count, meanTensor, m2Tensor, finalMean, finalVar);
                saveMeanTensor.SetValue(aNum, finalMean);
                saveVarTensor.SetValue(aNum, finalVar);
            } else if constexpr (SPLIT_MODE == R0_SPLIT_ALIGN_MODE) {
                // 对齐场景：r0UbTail = r0UbFactor，或者r0UbLoop==1,始终使用r0UbTail
                for (int64_t r1LoopIdx = 0; r1LoopIdx < patternR1; r1LoopIdx++) {
                    for (int64_t r0LoopIdx = 0; r0LoopIdx < r0UbLoop; r0LoopIdx++) {
                        xGmOffset = r1LoopIdx * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0 +
                                    r0LoopIdx * r0UbTail;
                        xTensor = xQueue.AllocTensor<float>();
                        CopyInXAndCast(xTensor, r0UbTail, xGmOffset);
                        WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, r0UbTail);
                    }
                }
                yQueue.FreeTensor(deltaTensor);
                WelfordParallelFinalizeR0Align(count, meanTensor, m2Tensor, finalMean, finalVar);
                saveMeanTensor.SetValue(aNum, finalMean);
                saveVarTensor.SetValue(aNum, finalVar);
            } else if constexpr (SPLIT_MODE == R1_SPLIT_ALIGN_MODE) {
                r0ProcNum = lastLoopNR0 * patternR0Align;
                for (int64_t nR0LoopIdx = 0; nR0LoopIdx < nR0Loop; nR0LoopIdx++) {
                    xGmOffset =
                        nR0LoopIdx * lastLoopNR0 * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0;
                    xTensor = xQueue.AllocTensor<float>();
                    CopyInNR0AndCast(xTensor, lastLoopNR0, xGmOffset);
                    WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, r0ProcNum);
                }
                yQueue.FreeTensor(deltaTensor);
                WelfordParallelFinalizeR1Align(count, meanTensor, m2Tensor, finalMean, finalVar);
                saveMeanTensor.SetValue(aNum, finalMean);
                saveVarTensor.SetValue(aNum, finalVar);
            } else {
                return;
            }
        }
    }

    __aicore__ inline void CalcYAndCopyOut()
    {
        PipeBarrier<PIPE_ALL>();
        for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
            weightValue = weightTensor.GetValue(aNum);
            biasValue = biasTensor.GetValue(aNum);
            finalMean = saveMeanTensor.GetValue(aNum);
            finalVar = saveVarTensor.GetValue(aNum);
            float weightMulInvstd =
                static_cast<float>(weightValue) / sqrt(finalVar + static_cast<float>(this->epsilon));
            if constexpr (SPLIT_MODE == R0_SPLIT_NOT_ALIGN_MODE || SPLIT_MODE == R0_SPLIT_ALIGN_MODE) {
                for (int64_t r1LoopIdx = 0; r1LoopIdx < patternR1; r1LoopIdx++) {
                    r0ProcNum = r0UbFactor;
                    for (int64_t r0LoopIdx = 0; r0LoopIdx < r0UbLoop; r0LoopIdx++) {
                        xGmOffset = r1LoopIdx * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0 +
                                    r0LoopIdx * r0UbFactor;
                        if (unlikely(r0LoopIdx == r0UbLoop - 1)) {
                            r0ProcNum = r0UbTail;
                        }
                        xTensor = xQueue.AllocTensor<float>();
                        CopyInXAndCast(xTensor, r0ProcNum, xGmOffset);
                        Adds(xTensor, xTensor, -finalMean, r0ProcNum);
                        PipeBarrier<PIPE_V>();
                        Muls(xTensor, xTensor, weightMulInvstd, r0ProcNum);
                        PipeBarrier<PIPE_V>();
                        if constexpr (!IsSameType<T1, float>::value) {
                            RoundMode b16RoundMode =
                                IsSameType<T1, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
                            Adds(xTensor, xTensor, biasValue, r0ProcNum);
                            PipeBarrier<PIPE_V>();
                            yTensor = yQueue.AllocTensor<T1>();
                            Cast(yTensor, xTensor, b16RoundMode, r0ProcNum);
                            xQueue.FreeTensor(xTensor);
                        } else {
                            yTensor = yQueue.AllocTensor<T1>();
                            Adds(yTensor, xTensor, biasValue, r0ProcNum);
                            xQueue.FreeTensor(xTensor);
                        }
                        CopyOutY(yTensor, r0ProcNum, xGmOffset);
                        yQueue.FreeTensor(yTensor);
                    }
                }
            } else {
                for (int64_t nR0LoopIdx = 0; nR0LoopIdx < nR0Loop; nR0LoopIdx++) {
                    r0ProcNum = procNR0 * patternR0Align;
                    int64_t prcoR0LineNum = procNR0;
                    xGmOffset = nR0LoopIdx * procNR0 * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0;
                    if (unlikely(nR0LoopIdx == nR0Loop - 1)) {
                        r0ProcNum = lastLoopNR0 * patternR0Align;
                        prcoR0LineNum = lastLoopNR0;
                    }
                    xTensor = xQueue.AllocTensor<float>();
                    CopyInNR0AndCast(xTensor, prcoR0LineNum, xGmOffset);
                    Adds(xTensor, xTensor, -finalMean, r0ProcNum);
                    PipeBarrier<PIPE_V>();
                    Muls(xTensor, xTensor, weightMulInvstd, r0ProcNum);
                    PipeBarrier<PIPE_V>();
                    if constexpr (!IsSameType<T1, float>::value) {
                        RoundMode b16RoundMode =
                            IsSameType<T1, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
                        Adds(xTensor, xTensor, biasValue, r0ProcNum);
                        PipeBarrier<PIPE_V>();
                        yTensor = yQueue.AllocTensor<T1>();
                        Cast(yTensor, xTensor, b16RoundMode, r0ProcNum);
                        xQueue.FreeTensor(xTensor);
                    } else {
                        yTensor = yQueue.AllocTensor<T1>();
                        Adds(yTensor, xTensor, biasValue, r0ProcNum);
                        xQueue.FreeTensor(xTensor);
                    }
                    CopyOutNR0Y(yTensor, prcoR0LineNum, xGmOffset);
                    yQueue.FreeTensor(yTensor);
                }
            }
        }
    }

private:
    __aicore__ inline void CopyInXAndCast(
        LocalTensor<float> &xTensor, const int64_t copyInSize, const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (!IsSameType<T1, float>::value) {
            LocalTensor<T1> xTensorHalf = xTensor.template ReinterpretCast<T1>();
            DataCopyPad(xTensorHalf[r0UbFactor], this->xGm[copyInGmOffset], intriParams, padParams);
            xQueue.EnQue(xTensorHalf);
            xTensorHalf = xQueue.DeQue<T1>();
            Cast(xTensor, xTensorHalf[r0UbFactor], RoundMode::CAST_NONE, copyInSize);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(xTensor, this->xGm[copyInGmOffset], intriParams, padParams);
            xQueue.EnQue(xTensor);
            xTensor = xQueue.DeQue<T1>();
        }
    }

    __aicore__ inline void CopyInNR0AndCast(
        LocalTensor<float> &xTensor, const int64_t copyInR0LineNum, const uint64_t copyInGmOffset)
    {
        DataCopyPadExtParams<T1> padParams = {true, 0, static_cast<uint8_t>(patternR0Align - patternR0), 0};
        if constexpr (R0_ALIGN_MODE == R0_ALIGN) {
            padParams = {false, 0, 0, 0};
        }
        DataCopyExtParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(copyInR0LineNum);
        intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
        intriParams.srcStride = static_cast<uint32_t>((patternA - 1) * patternR0 * sizeof(T1));
        intriParams.dstStride = 0;
        if constexpr (!IsSameType<T1, float>::value) {
            LocalTensor<T1> xTensorHalf = xTensor.template ReinterpretCast<T1>();
            DataCopyPad(xTensorHalf[r0UbFactor], this->xGm[copyInGmOffset], intriParams, padParams);
            xQueue.EnQue(xTensorHalf);
            xTensorHalf = xQueue.DeQue<T1>();
            Cast(xTensor, xTensorHalf[r0UbFactor], RoundMode::CAST_NONE, copyInR0LineNum * patternR0Align);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(xTensor, this->xGm[copyInGmOffset], intriParams, padParams);
            xQueue.EnQue(xTensor);
            xTensor = xQueue.DeQue<T1>();
        }
    }

    __aicore__ inline void CopyOutY(LocalTensor<T1> &yTensor, const int64_t copyOutSize, const uint64_t copyOutGmOffset)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyOutSize * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        yQueue.EnQue(yTensor);
        yTensor = yQueue.DeQue<T1>();
        DataCopyPad(this->yGm[copyOutGmOffset], yTensor, intriParams);
    }

    __aicore__ inline void CopyOutNR0Y(
        LocalTensor<T1> &yTensor, const int64_t copyOutLineNum, const uint64_t copyOutGmOffset)
    {
        DataCopyExtParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(copyOutLineNum);
        intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = static_cast<uint32_t>((patternA - 1) * patternR0 * sizeof(T1));
        yQueue.EnQue(yTensor);
        yTensor = yQueue.DeQue<T1>();
        DataCopyPad(this->yGm[copyOutGmOffset], yTensor, intriParams);
    }

    __aicore__ inline void CopyInWeightBiasAndCast(LocalTensor<float> &weightTensor, LocalTensor<float> &biasTensor,
        const int64_t copyInSize, const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(T2));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (!IsSameType<T2, float>::value) {
            LocalTensor<T2> weightTensorHalf = weightTensor.template ReinterpretCast<T2>();
            DataCopyPad(weightTensorHalf[aUbFactor], this->weightGm[copyInGmOffset], intriParams, padParams);
            weightQueue.EnQue(weightTensorHalf);
            weightTensorHalf = weightQueue.DeQue<T2>();
            Cast(weightTensor, weightTensorHalf[aUbFactor], RoundMode::CAST_NONE, copyInSize);
            LocalTensor<T2> biasTensorHalf = biasTensor.template ReinterpretCast<T2>();
            DataCopyPad(biasTensorHalf[aUbFactor], this->biasGm[copyInGmOffset], intriParams, padParams);
            biasQueue.EnQue(biasTensorHalf);
            biasTensorHalf = biasQueue.DeQue<T2>();
            Cast(biasTensor, biasTensorHalf[aUbFactor], RoundMode::CAST_NONE, copyInSize);
        } else {
            DataCopyPad(weightTensor, this->weightGm[copyInGmOffset], intriParams, padParams);
            DataCopyPad(biasTensor, this->biasGm[copyInGmOffset], intriParams, padParams);
        }
    }

    __aicore__ inline void CopyInRunningMeanVar(LocalTensor<float> &runningMeanInTensor,
        LocalTensor<float> &runningVarInTensor, const int64_t copyInSize, const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(runningMeanInTensor, this->runningMeanGm[copyInGmOffset], intriParams, padParams);
        runningMeanInQueue.EnQue(runningMeanInTensor);
        runningMeanInTensor = runningMeanInQueue.DeQue<float>();
        DataCopyPad(runningVarInTensor, this->runningVarGm[copyInGmOffset], intriParams, padParams);
        runningVarInQueue.EnQue(runningVarInTensor);
        runningVarInTensor = runningVarInQueue.DeQue<float>();
    }

    __aicore__ inline void CopyOutRunningMeanVar(LocalTensor<float> &runningMeanOutTensor,
        LocalTensor<float> &runningVarOutTensor, const int64_t copyOutSize, const uint64_t copyOutGmOffset)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyOutSize * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        runningMeanOutQueue.EnQue(runningMeanOutTensor);
        runningMeanOutTensor = runningMeanOutQueue.DeQue<float>();
        DataCopyPad(this->runningMeanOutGm[copyOutGmOffset], runningMeanOutTensor, intriParams);
        runningVarOutQueue.EnQue(runningVarOutTensor);
        runningVarOutTensor = runningVarOutQueue.DeQue<float>();
        DataCopyPad(this->runningVarOutGm[copyOutGmOffset], runningVarOutTensor, intriParams);
    }

    __aicore__ inline void CopyOutSaveMeanVar(LocalTensor<float> &saveMeanTensor, LocalTensor<float> &saveVarTensor,
        const int64_t copyOutSize, const uint64_t copyOutGmOffset)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyOutSize * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        saveMeanQueue.EnQue(saveMeanTensor);
        saveMeanTensor = saveMeanQueue.DeQue<float>();
        DataCopyPad(this->saveMeanGm[copyOutGmOffset], saveMeanTensor, intriParams);
        saveVarQueue.EnQue(saveVarTensor);
        saveVarTensor = saveVarQueue.DeQue<float>();
        DataCopyPad(this->saveVarGm[copyOutGmOffset], saveVarTensor, intriParams);
    }

    __aicore__ inline void WelfordParallelUpdate(float &count, LocalTensor<float> &meanTensor,
        LocalTensor<float> &m2Tensor, LocalTensor<float> &xTensor, LocalTensor<float> &deltaTensor,
        const uint32_t &calcMask)
    {
        count += 1;
        Sub(deltaTensor, xTensor, meanTensor, calcMask);
        PipeBarrier<PIPE_V>();
        Muls(xTensor, deltaTensor, 1 / count, calcMask);
        PipeBarrier<PIPE_V>();
        Add(meanTensor, meanTensor, xTensor, calcMask);
        xQueue.FreeTensor(xTensor);
        Mul(deltaTensor, deltaTensor, deltaTensor, calcMask);
        PipeBarrier<PIPE_V>();
        Muls(deltaTensor, deltaTensor, (count - 1) / count, calcMask);
        PipeBarrier<PIPE_V>();
        Add(m2Tensor, m2Tensor, deltaTensor, calcMask);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WelfordParallelFinalizeR0NotAlign(const float &count, LocalTensor<float> &meanTensor,
        LocalTensor<float> &m2Tensor, float &finalMean, float &finalVar)
    {
        float r0MulR1 = static_cast<float>(patternR0 * patternR1);
        LocalTensor<float> meanReduceTensor = yQueue.AllocTensor<float>();
        Muls(meanReduceTensor, meanTensor, count - patternR1, r0UbFactor);
        PipeBarrier<PIPE_V>();
        Muls(meanReduceTensor, meanReduceTensor, count / (count - patternR1), r0UbTail);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(meanReduceTensor, r0UbFactor, finalMean);
        yQueue.FreeTensor(meanReduceTensor);
        finalMean = finalMean / r0MulR1;
        Adds(meanTensor, meanTensor, -finalMean, r0UbFactor);
        PipeBarrier<PIPE_V>();
        Mul(meanTensor, meanTensor, meanTensor, r0UbFactor);
        PipeBarrier<PIPE_V>();
        Muls(meanTensor, meanTensor, count - patternR1, r0UbFactor);
        PipeBarrier<PIPE_V>();
        Muls(meanTensor, meanTensor, count / (count - patternR1), r0UbTail);
        PipeBarrier<PIPE_V>();
        Add(m2Tensor, m2Tensor, meanTensor, r0UbFactor);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(m2Tensor, r0UbFactor, finalVar);
        finalVar = finalVar / r0MulR1;
    }

    __aicore__ inline void WelfordParallelFinalizeR0Align(const float &count, LocalTensor<float> &meanTensor,
        LocalTensor<float> &m2Tensor, float &finalMean, float &finalVar)
    {
        LocalTensor<float> meanReduceTensor = yQueue.AllocTensor<float>();
        Adds(meanReduceTensor, meanTensor, float(0.0), r0UbTail);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(meanReduceTensor, r0UbTail, finalMean);
        yQueue.FreeTensor(meanReduceTensor);
        finalMean = finalMean / static_cast<float>(r0UbTail);
        Adds(meanTensor, meanTensor, -finalMean, r0UbTail);
        PipeBarrier<PIPE_V>();
        Mul(meanTensor, meanTensor, meanTensor, r0UbTail);
        PipeBarrier<PIPE_V>();
        Muls(meanTensor, meanTensor, count, r0UbTail);
        PipeBarrier<PIPE_V>();
        Add(m2Tensor, m2Tensor, meanTensor, r0UbTail);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(m2Tensor, r0UbTail, finalVar);
        finalVar = finalVar / float(r0UbTail * count);
    }

    __aicore__ inline void WelfordParallelFinalizeR1NotAlign(const float &count, LocalTensor<float> &meanTensor,
        LocalTensor<float> &m2Tensor, float &finalMean, float &finalVar)
    {
        r0ProcNum = procNR0 * patternR0Align;
        float r0MulR1 = static_cast<float>(patternR0 * patternR1);
        LocalTensor<float> meanReduceTensor = yQueue.AllocTensor<float>();
        Muls(meanReduceTensor, meanTensor, count - 1, r0ProcNum);
        PipeBarrier<PIPE_V>();
        Muls(meanReduceTensor, meanReduceTensor, count / (count - 1), lastLoopNR0 * patternR0Align);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(meanReduceTensor, r0ProcNum, finalMean);
        yQueue.FreeTensor(meanReduceTensor);
        finalMean = finalMean / r0MulR1;
        if constexpr (R0_ALIGN_MODE == R0_NOT_ALIGN) {
            SkipPadSubMean(meanTensor, procNR0);
        } else {
            Adds(meanTensor, meanTensor, -finalMean, r0ProcNum);
        }
        PipeBarrier<PIPE_V>();
        Mul(meanTensor, meanTensor, meanTensor, r0ProcNum);
        PipeBarrier<PIPE_V>();
        Muls(meanTensor, meanTensor, count - 1, r0ProcNum);
        PipeBarrier<PIPE_V>();
        Muls(meanTensor, meanTensor, count / (count - 1), lastLoopNR0 * patternR0Align);
        PipeBarrier<PIPE_V>();
        Add(m2Tensor, m2Tensor, meanTensor, r0ProcNum);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(m2Tensor, r0ProcNum, finalVar);
        finalVar = finalVar / r0MulR1;
    }

    __aicore__ inline void WelfordParallelFinalizeR1Align(const float &count, LocalTensor<float> &meanTensor,
        LocalTensor<float> &m2Tensor, float &finalMean, float &finalVar)
    {
        r0ProcNum = lastLoopNR0 * patternR0Align;
        LocalTensor<float> meanReduceTensor = yQueue.AllocTensor<float>();
        Adds(meanReduceTensor, meanTensor, float(0.0), r0ProcNum);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(meanReduceTensor, r0ProcNum, finalMean);
        yQueue.FreeTensor(meanReduceTensor);
        finalMean = finalMean / static_cast<float>(lastLoopNR0 * patternR0);
        if constexpr (R0_ALIGN_MODE == R0_NOT_ALIGN) {
            SkipPadSubMean(meanTensor, lastLoopNR0);
        } else {
            Adds(meanTensor, meanTensor, -finalMean, r0ProcNum);
        }
        PipeBarrier<PIPE_V>();
        Mul(meanTensor, meanTensor, meanTensor, r0ProcNum);
        PipeBarrier<PIPE_V>();
        Muls(meanTensor, meanTensor, count, r0ProcNum);
        PipeBarrier<PIPE_V>();
        Add(m2Tensor, m2Tensor, meanTensor, r0ProcNum);
        PipeBarrier<PIPE_V>();
        FullAichotomizeAdd(m2Tensor, r0ProcNum, finalVar);
        finalVar = finalVar / float(lastLoopNR0 * patternR0 * count);
    }

    __aicore__ inline void SkipPadSubMean(LocalTensor<float> &calcTensor, int64_t lineNum)
    {
        int64_t r0ForLoopNum = patternR0 / ELEM_PER_REP_FP32;
        int64_t r0ForRemainNum = patternR0 % ELEM_PER_REP_FP32;
        if ((r0ForLoopNum < lineNum) && (patternR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {
            uint8_t repStride = patternR0Align / B32_BLOCK_ALIGN_NUM;
            for (int64_t i = 0; i < r0ForLoopNum; i++) {
                Adds(calcTensor[i * ELEM_PER_REP_FP32],
                    calcTensor[i * ELEM_PER_REP_FP32],
                    -finalMean,
                    ELEM_PER_REP_FP32,
                    lineNum,
                    {1, 1, repStride, repStride});
            }
            if (r0ForRemainNum > 0) {
                int64_t repeatForLoopNum = lineNum / UINT8_MAX_NUM;
                for (int64_t i = 0; i < repeatForLoopNum; i++) {
                    Adds(calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                        calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                        -finalMean,
                        r0ForRemainNum,
                        UINT8_MAX_NUM,
                        {1, 1, repStride, repStride});
                }
                if ((lineNum % UINT8_MAX_NUM) > 0) {
                    Adds(calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 +
                                    repeatForLoopNum * UINT8_MAX_NUM * patternR0Align],
                        calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 +
                                   repeatForLoopNum * UINT8_MAX_NUM * patternR0Align],
                        -finalMean,
                        r0ForRemainNum,
                        (lineNum % UINT8_MAX_NUM),
                        {1, 1, repStride, repStride});
                }
            }
        } else {
            for (int64_t i = 0; i < lineNum; i++) {
                Adds(calcTensor[patternR0Align * i], calcTensor[patternR0Align * i], -finalMean, patternR0);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FullAichotomizeAdd(LocalTensor<float> &calcTensor, int64_t sumNum, float &sumValue)
    {
        // sumNum为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            Add(calcTensor, calcTensor, calcTensor[sumNum - dichotomizeAddDiffSize], dichotomizeAddDiffSize);
            PipeBarrier<PIPE_V>();
            sumNum = sumNum - dichotomizeAddDiffSize;
        }
        while (sumNum > ELEM_PER_REP_FP32) {
            sumNum = sumNum / TWO_NUM;
            Add(calcTensor, calcTensor, calcTensor[sumNum], sumNum);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMaskCount<float>();
        SetVectorMask<float>(0x0, sumNum);
        ReduceSum(calcTensor, calcTensor, calcTensor, 1);
        PipeBarrier<PIPE_V>();
        acc_val = GetAccVal();
        sumValue = *reinterpret_cast<float *>(&acc_val);
        SetMaskNorm();
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
    constexpr static int R0_SPLIT_NOT_ALIGN_MODE = 0;
    constexpr static int R0_SPLIT_ALIGN_MODE = 1;
    constexpr static int R1_SPLIT_NOT_ALIGN_MODE = 2;
    constexpr static int R1_SPLIT_ALIGN_MODE = 3;
    constexpr static int R0_NOT_ALIGN = 0;
    constexpr static int R0_ALIGN = 1;

    /* shape variable */
    int64_t patternR1;
    int64_t patternA;
    int64_t patternR0;
    int64_t patternR0Align;
    /* spilt variable */
    int64_t aUbFactor;
    int64_t aUbLoop;
    int64_t aUbTail;
    int64_t r0UbFactor;
    int64_t r0UbLoop;
    int64_t r0UbTail;
    int64_t procNR0;
    int64_t nR0Loop;
    int64_t lastLoopNR0;

    int64_t aProcNum;
    int64_t r0ProcNum;
    /* offset variable */
    uint64_t aUbLoopNowStartIdx;
    uint64_t xGmOffset;

    /* calculate variable */
    float count;
    int64_t dichotomizeAddDiffSize;
    uint64_t acc_val = 0;
    float finalMean = 0.0;
    float finalVar = 0.0;
    float weightValue = 0.0;
    float biasValue = 0.0;
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
    TBuf<TPosition::VECCALC> tmpBuf0, tmpBuf3, tmpBuf4;

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

    LocalTensor<float> meanTensor;
    LocalTensor<float> m2Tensor;
    LocalTensor<float> xTensor;
    LocalTensor<T1> yTensor;
};
}  // namespace BatchNormV3Ops
#endif  // BATCH_NORM_V3_WELFORD_H
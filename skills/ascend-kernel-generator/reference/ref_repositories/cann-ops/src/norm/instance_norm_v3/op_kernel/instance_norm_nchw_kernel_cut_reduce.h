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
 * \file instance_norm_nchw_kernel_cut_reduce.h
 * \brief
 */

#ifndef INSTANCE_NORM_NCHW_CUT_REDUCE_KERNEL_H_
#define INSTANCE_NORM_NCHW_CUT_REDUCE_KERNEL_H_

#include "instance_norm_nchw_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelInstanceNormNCHWCutReduce : public KernelInstanceNormNCHWBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelInstanceNormNCHWCutReduce(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance,
        GM_ADDR workspace, const InstanceNormV3TilingData *tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x, gamma, beta);
        this->InitOutGlobalTensors(y, mean, variance);

        Ppipe->InitBuffer(inDataQue, BUFFER_NUM, this->ubFactor * sizeof(T));   // D * 2
        Ppipe->InitBuffer(outDataQue, BUFFER_NUM, this->ubFactor * sizeof(T));  // D * 2

        Ppipe->InitBuffer(outReduceBuf, 2 * this->cAxisFactor * sizeof(float));  // D * 2
        Ppipe->InitBuffer(xBufFp32, this->ubFactor * sizeof(float));             // D * 4
        Ppipe->InitBuffer(yBufFp32, this->ubFactor * sizeof(float));             // D * 4
        Ppipe->InitBuffer(zBufFp32, this->ubFactor * sizeof(float));             // D * 4
        Ppipe->InitBuffer(gammaBuf, this->cAxisFactor * sizeof(float));          // D * 4
        Ppipe->InitBuffer(betaBuf, this->cAxisFactor * sizeof(float));           // D * 4
    }

    __aicore__ inline void Process()
    {
        uint32_t cLoops = this->cAxis / this->cAxisFactor;
        uint32_t cTails = this->cAxis % this->cAxisFactor;
        uint32_t cGmOffset = 0;
        for (uint32_t cBlockIdx = 0; cBlockIdx < cLoops; ++cBlockIdx) {
            this->ProcessCBlock(cGmOffset, this->cAxisFactor);
            cGmOffset += this->cAxisFactor;
        }
        if (likely(cTails > 0)) {
            this->ProcessCBlock(cGmOffset, cTails);
        }
    }

private:
    __aicore__ inline void ProcessCBlock(uint32_t cGmOffset, uint32_t cNums)
    {
        LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        LocalTensor<float> betaLocal = betaBuf.Get<float>();

        uint32_t gmOffset = 0;
        uint32_t reduceGmOffset = cGmOffset;

        CopyInGammaBeta(gammaLocal, betaLocal, cGmOffset, cNums);

        float gammaScalar;
        float betaScalar;

        for (uint32_t nIdx = 0; nIdx < this->nLoops; ++nIdx) {
            gmOffset = nIdx * this->cAxis * this->reduceNums + cGmOffset;
            for (uint32_t cIdx = 0; cIdx < cNums; ++cIdx) {
                gammaScalar = gammaLocal.GetValue(cIdx);
                betaScalar = betaLocal.GetValue(cIdx);
                ComputeNormSliceHW(gmOffset, cIdx, gammaScalar, betaScalar);
                gmOffset += this->reduceNums;
            }
            CopyOutMeanVar(reduceGmOffset, cNums);
            reduceGmOffset += this->cAxis;
        }
    }

    __aicore__ inline void ComputeNormSliceHW(uint32_t gmOffset, uint32_t cIdx, float gammaScalar, float betaScalar)
    {
        uint32_t hwLoops = this->reduceNums / this->ubFactor;
        uint32_t hwTails = this->reduceNums % this->ubFactor;
        uint32_t hwGmOffset = gmOffset;

        LocalTensor<float> meanVarTensor = outReduceBuf.Get<float>();
        LocalTensor<float> meanTensor = meanVarTensor[0];
        LocalTensor<float> varTensor = meanVarTensor[this->cAxisFactor];

        LocalTensor<float> xSumTensor = yBufFp32.Get<float>();
        LocalTensor<float> xSquareSumTensor = zBufFp32.Get<float>();
        Duplicate(xSumTensor, (float)0.0, this->ubFactor);
        Duplicate(xSquareSumTensor, (float)0.0, this->ubFactor);

        for (uint32_t hwIdx = 0; hwIdx < hwLoops; ++hwIdx) {
            CopyInHW(hwGmOffset, this->ubFactor);
            ComputeSum(xSumTensor, xSquareSumTensor, this->ubFactor);
            hwGmOffset += this->ubFactor;
        }
        if (likely(hwTails > 0)) {
            CopyInHW(hwGmOffset, hwTails);
            ComputeSum(xSumTensor, xSquareSumTensor, hwTails);
        }

        ReduceSum(xSumTensor, xSumTensor, xSumTensor, this->ubFactor);
        ReduceSum(xSquareSumTensor, xSquareSumTensor, xSquareSumTensor, this->ubFactor);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float meanScalar = xSumTensor.GetValue(0);
        float xSquareMeanScalar = xSquareSumTensor.GetValue(0);
        float varScalar = xSquareMeanScalar - meanScalar * meanScalar;  // Var(x) = E(x**2) - E(x)**2
        meanTensor.SetValue(cIdx, meanScalar);
        varTensor.SetValue(cIdx, varScalar);
        float rstdScalar = 1 / sqrt(varScalar + this->eps);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        PipeBarrier<PIPE_V>();

        hwGmOffset = gmOffset;
        for (uint32_t hwIdx = 0; hwIdx < hwLoops; ++hwIdx) {
            CopyInHW(hwGmOffset, this->ubFactor);
            ComputeNorm(gammaScalar, betaScalar, meanScalar, rstdScalar, this->ubFactor);
            CopyOut(hwGmOffset, this->ubFactor);
            hwGmOffset += this->ubFactor;
        }
        if (likely(hwTails > 0)) {
            CopyInHW(hwGmOffset, hwTails);
            ComputeNorm(gammaScalar, betaScalar, meanScalar, rstdScalar, hwTails);
            CopyOut(hwGmOffset, hwTails);
        }
    }

    __aicore__ inline void ComputeSum(
        LocalTensor<float> &xSumTensor, LocalTensor<float> &xSquareSumTensor, uint32_t size)
    {
        LocalTensor<float> xFp32Tensor = xBufFp32.Get<float>();
        Axpy(xSumTensor, xFp32Tensor, this->avgFactor, size);  // xSumTensor <- xSumTensor + x / N
        PipeBarrier<PIPE_V>();
        Mul(xFp32Tensor, xFp32Tensor, xFp32Tensor, size);  // xFp32Tensor <- x**2
        PipeBarrier<PIPE_V>();
        Axpy(xSquareSumTensor, xFp32Tensor, this->avgFactor, size);  // xSquareSumTensor <- xSquareSumTensor + x**2 / N
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutMeanVar(uint32_t gmOffset, uint32_t size)
    {
        LocalTensor<float> meanVarTensor = outReduceBuf.Get<float>();
        LocalTensor<T> meanVarTensorOutType = meanVarTensor.ReinterpretCast<T>();
        LocalTensor<T> meanOut = meanVarTensorOutType[0];
        LocalTensor<T> varOut = meanVarTensorOutType[this->cAxisFactor];

        LocalTensor<T> fakeTensor = outDataQue.template AllocTensor<T>();

        if constexpr (!IsSame<T, float>::value) {
            // total size of meanvar is 2 cfactor
            Cast(meanVarTensorOutType, meanVarTensor, RoundMode::CAST_NONE, this->cAxisFactor * 2);
            PipeBarrier<PIPE_V>();
        } else {
        }
        outDataQue.EnQue(fakeTensor);
        LocalTensor<T> fakeOut = outDataQue.template DeQue<T>();

        DataCopyCustomUB2GM(this->meanGm[gmOffset], meanOut, size);
        DataCopyCustomUB2GM(this->varianceGm[gmOffset], varOut, size);

        outDataQue.FreeTensor(fakeOut);
    }

    __aicore__ inline void ComputeNorm(
        float gammaScalar, float betaScalar, float meanScalar, float rstdScalar, uint32_t size)
    {
        LocalTensor<float> xFp32Tensor = xBufFp32.Get<float>();
        LocalTensor<float> yFp32Tensor = yBufFp32.Get<float>();
        LocalTensor<float> zFp32Tensor = zBufFp32.Get<float>();

        Adds(yFp32Tensor, xFp32Tensor, meanScalar * -1, size);  // yFp32Tensor <- x - E(x)
        PipeBarrier<PIPE_V>();
        Muls(xFp32Tensor, yFp32Tensor, rstdScalar, size);  // yFp32Tensor <- (x - E(x)) * rstd
        PipeBarrier<PIPE_V>();
        Muls(yFp32Tensor, xFp32Tensor, gammaScalar, size);  // yFp32Tensor <- (x - E(x)) * rstd * gamma
        PipeBarrier<PIPE_V>();

        LocalTensor<T> yLocal = outDataQue.template AllocTensor<T>();
        if constexpr (IsSame<T, float>::value) {
            Adds(yLocal, yFp32Tensor, betaScalar, size);  // yLocal <- (x - E(x)) * rstd * gamma + beta
        } else {
            Adds(xFp32Tensor, yFp32Tensor, betaScalar, size);  // xFp32Tensor <- (x - E(x)) * rstd * gamma + beta
            PipeBarrier<PIPE_V>();
            Cast(yLocal, xFp32Tensor, RoundMode::CAST_NONE, size);
        }
        PipeBarrier<PIPE_V>();
        outDataQue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t size)
    {
        LocalTensor<T> yOut = outDataQue.template DeQue<T>();
        DataCopyCustomUB2GM(this->yGm[gmOffset], yOut, size);
        outDataQue.FreeTensor(yOut);
    }

    __aicore__ inline void CopyInGammaBeta(
        LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal, uint32_t cGmOffset, uint32_t size)
    {
        if constexpr (IsSame<T, float>::value) {
            DataCopyCustomGM2UB(gammaLocal, this->gammaGm[cGmOffset], size);
            DataCopyCustomGM2UB(betaLocal, this->betaGm[cGmOffset], size);
            event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventMTE2S);
            WaitFlag<HardEvent::MTE2_S>(eventMTE2S);
        } else {
            auto gammaLocalHalf = gammaLocal.ReinterpretCast<half>()[this->cAxisFactor];
            auto betaLocalHalf = betaLocal.ReinterpretCast<half>()[this->cAxisFactor];
            DataCopyCustomGM2UB(gammaLocalHalf, this->gammaGm[cGmOffset], size);
            DataCopyCustomGM2UB(betaLocalHalf, this->betaGm[cGmOffset], size);

            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            Cast(gammaLocal, gammaLocalHalf, RoundMode::CAST_NONE, size);
            Cast(betaLocal, betaLocalHalf, RoundMode::CAST_NONE, size);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInHW(uint32_t gmOffset, uint32_t size)
    {
        LocalTensor<T> xIn = inDataQue.template AllocTensor<T>();
        DataCopyCustomGM2UB(xIn, this->xGm[gmOffset], size);
        inDataQue.EnQue(xIn);
        LocalTensor<T> xLocal = inDataQue.template DeQue<T>();
        LocalTensor<float> xFp32Tensor = xBufFp32.Get<float>();
        if constexpr (IsSame<T, float>::value) {
            Adds(xFp32Tensor, xLocal, (float)0.0, size);
        } else {
            Cast(xFp32Tensor, xLocal, RoundMode::CAST_NONE, size);
        }
        PipeBarrier<PIPE_V>();
        inDataQue.FreeTensor(xLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inDataQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outDataQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;
    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> outReduceBuf;
};

#endif  // INSTANCE_NORM_NCHW_CUT_REDUCE_KERNEL_H_

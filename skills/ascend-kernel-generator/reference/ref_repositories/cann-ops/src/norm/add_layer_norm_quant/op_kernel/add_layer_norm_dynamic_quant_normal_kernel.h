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
 * \file add_layer_norm_dynamic_quant_normal_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
#define ADD_LAYER_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_

#include "add_layer_norm_quant_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormDynamicQuantNormal : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormDynamicQuantNormal(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *smooth1, __gm__ uint8_t *smooth2, __gm__ uint8_t *fake1,
        __gm__ uint8_t *fake2, __gm__ uint8_t *y1, __gm__ uint8_t *y2, __gm__ uint8_t *x, __gm__ uint8_t *outScale1,
        __gm__ uint8_t *outScale2, __gm__ uint8_t *workspace, const AddLayerNormQuantTilingData *tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y1, y2, x);

        this->smooth1Exist = tiling->scaleOffsetMode >= 100;
        this->smooth2Exist = tiling->scaleOffsetMode >= 200;

        smooth1Gm.SetGlobalBuffer((__gm__ T *)smooth1);
        smooth2Gm.SetGlobalBuffer((__gm__ T *)smooth2);
        scales1Gm.SetGlobalBuffer((__gm__ float *)outScale1 + block_idx * this->firstDimPerCore);
        scales2Gm.SetGlobalBuffer((__gm__ float *)outScale2 + block_idx * this->firstDimPerCore);

        /*
          UB = 3 * this->rowStep * alignedCol * sizeof(T)
              + 2 * this->rowStep * alignedCol * sizeof(float)
              + Count(gamma,beta,bias) * alignedCol * sizeof(T)
              + 512Bytes(256 + reduceOut)
        */

        Ppipe->InitBuffer(
            inRowsQue, BUFFER_NUM, ROUND_UP32(2 * this->rowStep * this->numLastDimAligned * sizeof(T)));  // 2 * D * 2
        Ppipe->InitBuffer(
            outRowQue, BUFFER_NUM, ROUND_UP32(this->rowStep * this->numLastDimAligned * sizeof(T)));  // D * 2

        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(this->rowStep * this->numLastDimAligned * sizeof(float)));  // D * 4
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(this->rowStep * this->numLastDimAligned * sizeof(float)));  // D * 4
        Ppipe->InitBuffer(gammaBuf, ROUND_UP32(this->numLastDimAligned * sizeof(T)));                      // D * 2
        Ppipe->InitBuffer(betaBuf, ROUND_UP32(this->numLastDimAligned * sizeof(T)));                       // D * 2

        if constexpr (IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(biasBuf, ROUND_UP32(this->numLastDimAligned * sizeof(T)));  // D * 2 // 9792
        }

        Ppipe->InitBuffer(scales1Que, BUFFER_NUM, ROUND_UP32(this->rowStep * sizeof(float)));
        Ppipe->InitBuffer(scales2Que, BUFFER_NUM, ROUND_UP32(this->rowStep * sizeof(float)));
        Ppipe->InitBuffer(divisorBuf, 8 * sizeof(float));  // D * 2
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(this->rowWork, this->rowStep);
        CopyInGammaBeta();

        LocalTensor<T> gammaLocal = gammaBuf.template Get<T>();
        LocalTensor<T> betaLocal = betaBuf.template Get<T>();

        DataCopyPadParams padParams;
        if (this->lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = this->numLastDimAligned - this->numLastDim;
        }

        int32_t gmOffset = 0;
        int32_t gmOffsetScale = 0;
        int32_t elementCount = this->numLastDimAligned * this->rowStep;

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            CopyInX1X2(gmOffset, this->rowStep, elementCount, padParams);
            AddX1X2Bias(elementCount, this->rowStep);
            CopyInSmooth12();
            CopyOutAdditionalOutput(gmOffset, this->rowStep, elementCount);
            ComputeLayerNorm(this->rowStep, elementCount, gammaLocal, betaLocal);
            ComputeDynamicQuant(this->rowStep, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowStep);
            gmOffset += this->rowStep * this->numLastDim;
            gmOffsetScale += this->rowStep;
        }
        {
            elementCount = this->numLastDimAligned * this->rowTail_;
            int32_t rowIdx = rowMoveCnt - 1;
            CopyInX1X2(gmOffset, this->rowTail_, elementCount, padParams);
            AddX1X2Bias(elementCount, this->rowTail_);
            CopyInSmooth12();
            CopyOutAdditionalOutput(gmOffset, this->rowTail_, elementCount);
            ComputeLayerNorm(this->rowTail_, elementCount, gammaLocal, betaLocal);
            ComputeDynamicQuant(this->rowTail_, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowTail_);
        }
    }

private:
    __aicore__ inline void CopyInX1X2(
        int32_t gmOffset, int32_t rowCount, int32_t elementCount, DataCopyPadParams &padParams)
    {
        LocalTensor<T> x1x2LocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(x1x2LocalIn[0], this->x1Gm[gmOffset], this->numLastDim, rowCount, padParams);
        DataCopyEx(x1x2LocalIn[elementCount], this->x2Gm[gmOffset], this->numLastDim, rowCount, padParams);
        if constexpr (IS_BIAS_ELEWISE) {
            LocalTensor<T> biasLocal = yBufFp32.template Get<T>();
            if constexpr (is_same<float, T>::value) {
                DataCopyEx(biasLocal, this->biasGm[gmOffset], this->numLastDim, rowCount, padParams);
            } else {
                DataCopyEx(biasLocal[elementCount], this->biasGm[gmOffset], this->numLastDim, rowCount, padParams);
            }
        }
        inRowsQue.EnQue(x1x2LocalIn);
    }

    __aicore__ inline void AddX1X2Bias(int32_t elementCount, int32_t rowCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        LocalTensor<T> x1x2Local = inRowsQue.template DeQue<T>();
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        if constexpr (is_same<float, T>::value) {
            if constexpr (IS_BIAS_BROADCAST) {
                auto biasLocal = biasBuf.template Get<T>();
                Add(xLocalFp32, x1Local, x2Local, elementCount);
                PipeBarrier<PIPE_V>();
                for (int i = 0; i < rowCount; i++) {
                    Add(xLocalFp32[i * this->numLastDimAligned],
                        biasLocal,
                        xLocalFp32[i * this->numLastDimAligned],
                        this->numLastDim);
                }
            } else {
                Add(xLocalFp32, x1Local, x2Local, elementCount);
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32, yLocalFp32, xLocalFp32, elementCount);
            }
        } else {
            if constexpr (IS_BIAS_BROADCAST) {
                auto biasLocal = biasBuf.template Get<T>();
                Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
                Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
                Cast(x1x2Local.template ReinterpretCast<float>(), biasLocal, RoundMode::CAST_NONE, this->numLastDim);
                PipeBarrier<PIPE_V>();
                for (int i = 0; i < rowCount; i++) {
                    Add(xLocalFp32[i * this->numLastDimAligned],
                        x1x2Local.template ReinterpretCast<float>(),
                        xLocalFp32[i * this->numLastDimAligned],
                        this->numLastDim);
                }
            } else {
                auto biasLocal = yLocalFp32.ReinterpretCast<T>()[elementCount];
                Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
                Cast(yLocalFp32, biasLocal, RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
                PipeBarrier<PIPE_V>();
                Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
            }
        }
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(x1x2Local);
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        if (this->isXOut) {
            LocalTensor<float> addBufLocal = xBufFp32.Get<float>();
            auto xLocal = outRowQue.template AllocTensor<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(xLocal, addBufLocal, (float)0.0, elementCount);
            } else if constexpr (is_same<T, half>::value) {
                Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, elementCount);
            }
            PipeBarrier<PIPE_V>();
            outRowQue.template EnQue<T>(xLocal);

            auto x = outRowQue.template DeQue<T>();
            DataCopyEx(this->xGm[gmOffset], x, this->numLastDim, rowCount);
            outRowQue.FreeTensor(x);
        }
    }

    __aicore__ inline void ComputeLayerNorm(
        int32_t nums, int32_t elementCount, LocalTensor<T> &gammaLocal, LocalTensor<T> &betaLocal)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- x1 + x2 + bias
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        Muls(yLocalFp32, xLocalFp32, this->aveNum, elementCount);  // yLocalFp32 <-- x / N
        PipeBarrier<PIPE_V>();

        // reduce#1 for mean
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            auto aveLocalTemp = ReduceSumFP32(yLocalFp32[roundOffset], this->numLastDim);  // aveLocalTemp <-- E(x)
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Adds(xLocalFp32[roundOffset],
                xLocalFp32[roundOffset],
                aveLocalTemp * -1,
                this->numLastDim);  // xLocalFp32 <-- x - E(x)
        }
        PipeBarrier<PIPE_V>();

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, elementCount);  // yLocalFp32 <-- (x - E(x))**2
        PipeBarrier<PIPE_V>();
        Muls(yLocalFp32, yLocalFp32, this->aveNum, elementCount);  // yLocalFp32 <-- (x - E(x))**2 / N
        PipeBarrier<PIPE_V>();

        // reduce#2 for var
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            float varLocalTemp = ReduceSumFP32(yLocalFp32[roundOffset], this->numLastDim);  // varLocalTemp <-- Var(x)
            float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);                       // rstdLocalTemp <-- rstd

            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xLocalFp32[roundOffset],
                xLocalFp32[roundOffset],
                rstdLocalTemp,
                this->numLastDim);  // xLocalFp32 <-- (x - E(x)) * rstd
            PipeBarrier<PIPE_V>();
            if constexpr (!is_same<T, float>::value) {
                Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);
                PipeBarrier<PIPE_V>();
                Mul(xLocalFp32[roundOffset], yLocalFp32, xLocalFp32[roundOffset], this->numLastDim);
                PipeBarrier<PIPE_V>();
                Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim);
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32[roundOffset], yLocalFp32, xLocalFp32[roundOffset], this->numLastDim);
            } else {
                Mul(yLocalFp32, xLocalFp32[roundOffset], gammaLocal, this->numLastDim);
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32[roundOffset], yLocalFp32, betaLocal, this->numLastDim);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t nums, int32_t elementCount)
    {
        if (!this->smooth1Exist) {
            ComputeZeroDynamicQuant(nums, elementCount);
        } else if (!this->smooth2Exist) {
            ComputeSoleDynamicQuant(nums, elementCount);
        } else {
            ComputeDualDynamicQuant(nums, elementCount);
        }
    }

    __aicore__ inline void ComputeZeroDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> scaleLocal = scales1Que.template AllocTensor<float>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- y
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        ScaleTensor(xLocalFp32, yLocalFp32, scaleLocal, elementCount, nums);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> yLocal = outRowQue.template AllocTensor<int8_t>();

        Cast(xLocalFp32.ReinterpretCast<int32_t>(), xLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(xLocalFp32.ReinterpretCast<half>(),
            xLocalFp32.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            elementCount);
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < nums; ++i) {
            Cast(yLocal[i * this->numLastDimRoundUp32],
                xLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
                RoundMode::CAST_TRUNC,
                this->numLastDim);
        }

        PipeBarrier<PIPE_V>();
        outRowQue.EnQue(yLocal);
        scales1Que.EnQue(scaleLocal);
    }

    __aicore__ inline void ComputeSoleDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> scaleLocal = scales1Que.template AllocTensor<float>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- y
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        LocalTensor<T> smoothLocal = inRowsQue.template DeQue<T>();

        // compute smooth1
        auto smoothFp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
        if constexpr (!is_same<T, float>::value) {
            Cast(smoothFp32, smoothLocal, RoundMode::CAST_NONE, this->numLastDim);
        } else {
            Adds(smoothFp32, smoothLocal, (float)0.0, this->numLastDim);
        }
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(smoothLocal);

        for (int32_t rid = 0; rid < nums; ++rid) {
            Mul(yLocalFp32[rid * this->numLastDimAligned],
                xLocalFp32[rid * this->numLastDimAligned],
                smoothFp32,
                this->numLastDim);
        }
        PipeBarrier<PIPE_V>();

        ScaleTensor(yLocalFp32, xLocalFp32, scaleLocal, elementCount, nums);
        PipeBarrier<PIPE_V>();

        LocalTensor<int8_t> yLocal = outRowQue.template AllocTensor<int8_t>();

        Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(yLocalFp32.ReinterpretCast<half>(),
            yLocalFp32.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            elementCount);
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < nums; ++i) {
            Cast(yLocal[i * this->numLastDimRoundUp32],
                yLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
                RoundMode::CAST_TRUNC,
                this->numLastDim);
        }

        outRowQue.EnQue(yLocal);
        scales1Que.EnQue(scaleLocal);
    }

    __aicore__ inline void ComputeDualDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> scale1Local = scales1Que.template AllocTensor<float>();
        LocalTensor<float> scale2Local = scales2Que.template AllocTensor<float>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- y
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        LocalTensor<T> smooth12Local = inRowsQue.template DeQue<T>();
        auto smooth1Local = smooth12Local[0];
        auto smooth2Local = smooth12Local[this->numLastDimAligned];

        // compute smooth1
        LocalTensor<float> tmpLocal = smooth12Local.template ReinterpretCast<float>();
        auto smooth1Fp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
        auto smooth2Fp32 = tmpLocal[0];
        if constexpr (!is_same<T, float>::value) {
            Cast(smooth1Fp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(smooth2Fp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);
        } else {
            Adds(smooth1Fp32, smooth1Local, (float)0.0, this->numLastDim);
            Adds(smooth2Fp32, smooth2Local, (float)0.0, this->numLastDim);
        }
        PipeBarrier<PIPE_V>();

        for (int32_t rid = 0; rid < nums; ++rid) {
            Mul(yLocalFp32[rid * this->numLastDimAligned],
                xLocalFp32[rid * this->numLastDimAligned],
                smooth1Fp32,
                this->numLastDim);  // yLocalFp32 <-- y * smooth1
            Mul(xLocalFp32[rid * this->numLastDimAligned],
                xLocalFp32[rid * this->numLastDimAligned],
                smooth2Fp32,
                this->numLastDim);  // xLocalFp32 <-- y * smooth2
        }
        PipeBarrier<PIPE_V>();

        ScaleTensor(yLocalFp32, tmpLocal, scale1Local, elementCount, nums);
        PipeBarrier<PIPE_V>();
        ScaleTensor(xLocalFp32, tmpLocal, scale2Local, elementCount, nums);
        PipeBarrier<PIPE_V>();

        inRowsQue.FreeTensor(smooth12Local);

        LocalTensor<int8_t> y12Local = outRowQue.template AllocTensor<int8_t>();
        auto y1Local = y12Local[0];

        Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
        Cast(xLocalFp32.ReinterpretCast<int32_t>(), xLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(yLocalFp32.ReinterpretCast<half>(),
            yLocalFp32.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            elementCount);
        Cast(xLocalFp32.ReinterpretCast<half>(),
            xLocalFp32.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            elementCount);
        PipeBarrier<PIPE_V>();

        for (int i = 0; i < nums; ++i) {
            Cast(y1Local[i * this->numLastDimRoundUp32],
                yLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
                RoundMode::CAST_TRUNC,
                this->numLastDim);
            Cast(xLocalFp32.ReinterpretCast<int8_t>()[i * this->numLastDimRoundUp32],
                xLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
                RoundMode::CAST_TRUNC,
                this->numLastDim);
        }
        PipeBarrier<PIPE_V>();

        outRowQue.EnQue(y12Local);
        scales1Que.EnQue(scale1Local);
        scales2Que.EnQue(scale2Local);
    }

    __aicore__ inline void CopyOut(int32_t gmOffset, int32_t gmOffsetScale, int32_t rowCount)
    {
        LocalTensor<int8_t> outY12 = outRowQue.template DeQue<int8_t>();
        LocalTensor<float> scale1Out = scales1Que.template DeQue<float>();
        LocalTensor<float> scale2Out;
        if (this->smooth2Exist) {
            scale2Out = scales2Que.template DeQue<float>();
        }
        auto outY1 = outY12[0];
        DataCopyEx(this->y1Gm[gmOffset], outY1, this->numLastDim, rowCount);
        DataCopyEx(scales1Gm[gmOffsetScale], scale1Out, rowCount);
        if (this->smooth2Exist) {
            LocalTensor<int8_t> xLocalInt8 = xBufFp32.Get<int8_t>();
            DataCopyEx(this->y2Gm[gmOffset], xLocalInt8, this->numLastDim, rowCount);
            DataCopyEx(scales2Gm[gmOffsetScale], scale2Out, rowCount);
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
            scales2Que.FreeTensor(scale2Out);
        }
        outRowQue.FreeTensor(outY12);
        scales1Que.FreeTensor(scale1Out);
    }

    __aicore__ inline void CopyInGammaBeta()
    {
        LocalTensor<T> gammaLocal = gammaBuf.template Get<T>();
        LocalTensor<T> betaLocal = betaBuf.template Get<T>();
        DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
        DataCopyEx(betaLocal, this->betaGm, this->numLastDim);
        if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> biasLocal = biasBuf.template Get<T>();
            DataCopyEx(biasLocal, this->biasGm, this->numLastDim);
        }
    }

    __aicore__ inline void CopyInSmooth12()
    {
        if (this->smooth1Exist) {
            LocalTensor<T> smooth12CopyIn = inRowsQue.template AllocTensor<T>();
            DataCopyEx(smooth12CopyIn[0], smooth1Gm, this->numLastDim);
            if (this->smooth2Exist) {
                DataCopyEx(smooth12CopyIn[this->numLastDimAligned], smooth2Gm, this->numLastDim);
            }
            inRowsQue.EnQue(smooth12CopyIn);
        }
    }

    __aicore__ inline void ScaleTensor(LocalTensor<float> &srcTensor, LocalTensor<float> &tmpTensor,
        LocalTensor<float> &scaleTensor, int32_t size, int32_t nums)
    {
        float maxTemp;
        float scaleTemp;
        event_t eventVS;
        event_t eventSV;
        Abs(tmpTensor, srcTensor, size);  // tmpLocal <-- |y * smooth1|
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            ReduceMaxInplace(tmpTensor[rid * this->numLastDimAligned], this->numLastDim);
            eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            maxTemp = tmpTensor[rid * this->numLastDimAligned].GetValue(0);  // Reduce
            scaleTemp = maxTemp / DYNAMIC_QUANT_DIVIDEND;
            scaleTensor.SetValue(rid, scaleTemp);
            eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            auto srcSlice = srcTensor[rid * this->numLastDimAligned];
            auto dividendTensor = divisorBuf.Get<float>();
            DivScalarFP32(srcSlice, srcSlice, dividendTensor, scaleTemp, this->numLastDim);  // srcTensor <-- y / scale
        }
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outRowQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scales1Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scales2Que;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> divisorBuf;
    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> biasBuf;

    GlobalTensor<T> smooth1Gm, smooth2Gm;
    GlobalTensor<float> scales1Gm, scales2Gm;

    bool smooth1Exist;
    bool smooth2Exist;
};

#endif  // ADD_LAYER_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_

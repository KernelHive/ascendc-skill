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
 * \file add_layer_norm_dynamic_quant_single_row_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
#define ADD_LAYER_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "add_layer_norm_quant_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormDynamicQuantSingleRow : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormDynamicQuantSingleRow(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *smooth1, __gm__ uint8_t *smooth2, __gm__ uint8_t *zeroOffset1,
        __gm__ uint8_t *zeroOffset2, __gm__ uint8_t *y1, __gm__ uint8_t *y2, __gm__ uint8_t *x,
        __gm__ uint8_t *outScale1, __gm__ uint8_t *outScale2, __gm__ uint8_t *workspace,
        const AddLayerNormQuantTilingData *tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y1, y2, x);

        smooth1Exist = tiling->scaleOffsetMode >= 100;
        smooth2Exist = tiling->scaleOffsetMode >= 200;

        smooth1Gm.SetGlobalBuffer((__gm__ T *)smooth1);
        smooth2Gm.SetGlobalBuffer((__gm__ T *)smooth2);
        outScale1Gm.SetGlobalBuffer((__gm__ float *)outScale1 + block_idx * this->firstDimPerCore);
        outScale2Gm.SetGlobalBuffer((__gm__ float *)outScale2 + block_idx * this->firstDimPerCore);

        /*
          UB = 3 * alignedCol * sizeof(T)
              + 2 * alignedCol * sizeof(float)
              + Count(bias) * alignedCol * sizeof(T)
              + 512Btyes(256 + reduceOut)
        */

        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * ROUND_UP32(this->numLastDimAligned * sizeof(T)));  // 2 * D * 2
        Ppipe->InitBuffer(yQue, BUFFER_NUM, ROUND_UP32(this->numLastDimRoundUp32 * sizeof(T)));         // D * 2

        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(this->numLastDimAligned * sizeof(float)));  // D * 4
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(this->numLastDimAligned * sizeof(float)));  // D * 4
        Ppipe->InitBuffer(biasBuf, ROUND_UP32(this->numLastDimAligned * sizeof(T)));       // D * 2

        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, 2 * ROW_FACTOR * sizeof(float));
        Ppipe->InitBuffer(divisorBuf, 8 * sizeof(float));  // D * 2
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> biasLocal = biasBuf.template Get<T>();
        if constexpr (IS_BIAS_BROADCAST) {
            DataCopyEx(biasLocal, this->biasGm, this->numLastDim);
        }

        int32_t outLoopCount = this->rowWork / ROW_FACTOR;
        int32_t outLoopTail = this->rowWork % ROW_FACTOR;

        uint32_t gmOffset = 0;
        uint32_t gmOffsetReduce = 0;

        LocalTensor<float> scalesLocalOut;

        for (int32_t loopIdx = 0; loopIdx < outLoopCount; ++loopIdx) {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < ROW_FACTOR; ++innerIdx) {
                CopyInX1X2(gmOffset, biasLocal);
                AddSingleRow(gmOffset, this->numLastDim, biasLocal);
                ComputeLayerNorm(gmOffset);
                ComputeDynamicQuant(innerIdx, scalesLocalOut, gmOffset);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, ROW_FACTOR);
            gmOffsetReduce += ROW_FACTOR;
        }
        {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < outLoopTail; ++innerIdx) {
                CopyInX1X2(gmOffset, biasLocal);
                AddSingleRow(gmOffset, this->numLastDim, biasLocal);
                ComputeLayerNorm(gmOffset);
                ComputeDynamicQuant(innerIdx, scalesLocalOut, gmOffset);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, outLoopTail);
        }
    }

private:
    __aicore__ inline void CopyInX1X2(int32_t gmOffset, LocalTensor<T> &biasLocal)
    {
        LocalTensor<T> x1x2LocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(x1x2LocalIn[0], this->x1Gm[gmOffset], this->numLastDim);
        DataCopyEx(x1x2LocalIn[this->numLastDimAligned], this->x2Gm[gmOffset], this->numLastDim);
        if constexpr (IS_BIAS_ELEWISE) {
            DataCopyEx(biasLocal, this->biasGm[gmOffset], this->numLastDim);
        }
        inRowsQue.EnQue(x1x2LocalIn);
    }

    __aicore__ inline void AddSingleRow(int32_t gmOffset, int32_t size, LocalTensor<T> &biasLocal)
    {
        auto addBufLocal = xBufFp32.Get<float>();
        auto yBufLocal = yBufFp32.Get<float>();

        auto x1x2Local = inRowsQue.template DeQue<T>();
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[this->numLastDimAligned];
        if constexpr (is_same<float, T>::value) {
            Add(addBufLocal, x1Local, x2Local, size);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(addBufLocal, x1Local, RoundMode::CAST_NONE, size);
            Cast(yBufLocal, x2Local, RoundMode::CAST_NONE, size);
            PipeBarrier<PIPE_V>();
            Add(addBufLocal, yBufLocal, addBufLocal, size);
            PipeBarrier<PIPE_V>();
        }
        inRowsQue.FreeTensor(x1x2Local);

        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            if constexpr (is_same<float, T>::value) {
                Add(addBufLocal, biasLocal, addBufLocal, size);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(yBufLocal, biasLocal, RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                Add(addBufLocal, yBufLocal, addBufLocal, size);
                PipeBarrier<PIPE_V>();
            }
        }

        if (this->isXOut) {
            auto xLocal = yQue.template AllocTensor<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(xLocal, addBufLocal, ZERO, size);
            } else if constexpr (is_same<T, half>::value) {
                Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, size);
            } else {
                Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, size);
            }
            PipeBarrier<PIPE_V>();
            yQue.template EnQue<T>(xLocal);
            auto x = yQue.template DeQue<T>();
            DataCopyEx(this->xGm[gmOffset], x, size);
            yQue.FreeTensor(x);
        }
    }

    __aicore__ inline void ComputeLayerNorm(int32_t gmOffset)
    {
        LocalTensor<T> gammaBetaIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(gammaBetaIn[0], this->gammaGm, this->numLastDim);
        DataCopyEx(gammaBetaIn[this->numLastDimAligned], this->betaGm, this->numLastDim);

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        Muls(yLocalFp32, xLocalFp32, this->aveNum, this->numLastDim);  // yLocalFp32 <- x / N
        PipeBarrier<PIPE_V>();
        float aveLocalTemp = ReduceSumFP32(yLocalFp32, this->numLastDim);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Adds(xLocalFp32, xLocalFp32, aveLocalTemp * -1, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Mul(yLocalFp32, xLocalFp32, xLocalFp32, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Muls(yLocalFp32, yLocalFp32, this->aveNum, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float varLocalTemp = ReduceSumFP32(yLocalFp32, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);
        eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xLocalFp32, xLocalFp32, rstdLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();

        inRowsQue.EnQue(gammaBetaIn);
        auto gammaBetaLocal = inRowsQue.template DeQue<T>();
        auto gammaLocal = gammaBetaLocal[0];
        auto betaLocal = gammaBetaLocal[this->numLastDimAligned];
        if constexpr (!is_same<T, float>::value) {
            Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Mul(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(gammaBetaLocal);
            Add(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
        } else {
            Mul(yLocalFp32, xLocalFp32, gammaLocal, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, yLocalFp32, betaLocal, this->numLastDim);
            inRowsQue.FreeTensor(gammaBetaLocal);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t idx, LocalTensor<float> &scalesLocalOut, int32_t gmOffset)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        if (!smooth1Exist) {
            Abs(yLocalFp32, xLocalFp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
            float maxTemp = ReduceMaxFP32(yLocalFp32, this->numLastDim);
            float scaleTemp = maxTemp / DYNAMIC_QUANT_DIVIDEND;
            scalesLocalOut.SetValue(idx, scaleTemp);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            PipeBarrier<PIPE_V>();
            auto dividendTensor = divisorBuf.Get<float>();
            DivScalarFP32(yLocalFp32, xLocalFp32, dividendTensor, scaleTemp, this->numLastDim);
            LocalTensor<int8_t> yLocal = yQue.template AllocTensor<int8_t>();
            RoundFloat2Int8(yLocal, yLocalFp32, this->numLastDim);
            yQue.EnQue(yLocal);
        } else if (!smooth2Exist) {
            LocalTensor<T> smooth1In = inRowsQue.template AllocTensor<T>();
            DataCopyEx(smooth1In[0], smooth1Gm, this->numLastDim);
            inRowsQue.EnQue(smooth1In);
            auto smooth1Local = inRowsQue.template DeQue<T>();

            if constexpr (!is_same<T, float>::value) {
                Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
                inRowsQue.FreeTensor(smooth1Local);
                PipeBarrier<PIPE_V>();
                Mul(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
            } else {
                Mul(xLocalFp32, xLocalFp32, smooth1Local, this->numLastDim);
                PipeBarrier<PIPE_V>();
                inRowsQue.FreeTensor(smooth1Local);
            }

            PipeBarrier<PIPE_V>();
            ScaleTensor(xLocalFp32, yLocalFp32, scalesLocalOut, idx);
            LocalTensor<int8_t> yLocal = yQue.template AllocTensor<int8_t>();
            RoundFloat2Int8(yLocal, xLocalFp32, this->numLastDim);
            yQue.EnQue(yLocal);
        } else {
            LocalTensor<T> smooth12In = inRowsQue.template AllocTensor<T>();
            DataCopyEx(smooth12In[0], smooth1Gm, this->numLastDim);
            DataCopyEx(smooth12In[this->numLastDimAligned], smooth2Gm, this->numLastDim);
            inRowsQue.EnQue(smooth12In);
            auto smooth12Local = inRowsQue.template DeQue<T>();
            auto smooth1Local = smooth12Local[0];
            auto smooth2Local = smooth12Local[this->numLastDimAligned];
            LocalTensor<float> tmpTensor = smooth12Local.template ReinterpretCast<float>();

            if constexpr (!is_same<T, float>::value) {
                Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
                PipeBarrier<PIPE_V>();
                Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);  // yLocalFp32 <-- y * smooth1
                Cast(tmpTensor, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);
                PipeBarrier<PIPE_V>();
                Mul(xLocalFp32, xLocalFp32, tmpTensor, this->numLastDim);  // xLocalFp32 <-- y * smooth2
                PipeBarrier<PIPE_V>();
            } else {
                Mul(yLocalFp32, xLocalFp32, smooth1Local, this->numLastDim);  // yLocalFp32 <-- y * smooth1
                PipeBarrier<PIPE_V>();
                Mul(xLocalFp32, xLocalFp32, smooth2Local, this->numLastDim);  // xLocalFp32 <-- y * smooth2
                PipeBarrier<PIPE_V>();
            }

            ScaleTensor(yLocalFp32, tmpTensor, scalesLocalOut, idx);
            PipeBarrier<PIPE_V>();
            ScaleTensor(xLocalFp32, tmpTensor, scalesLocalOut, ROW_FACTOR + idx);
            PipeBarrier<PIPE_V>();

            inRowsQue.FreeTensor(smooth12Local);

            LocalTensor<int8_t> y12Local = yQue.template AllocTensor<int8_t>();
            auto y1Local = y12Local[0];
            auto y2Local = y12Local[this->numLastDimRoundUp32];
            Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, this->numLastDim);
            Cast(xLocalFp32.ReinterpretCast<int32_t>(), xLocalFp32, RoundMode::CAST_RINT, this->numLastDim);
            PipeBarrier<PIPE_V>();
            SetDeqScale((half)1.000000e+00f);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32.ReinterpretCast<half>(),
                yLocalFp32.ReinterpretCast<int32_t>(),
                RoundMode::CAST_NONE,
                this->numLastDim);
            Cast(xLocalFp32.ReinterpretCast<half>(),
                xLocalFp32.ReinterpretCast<int32_t>(),
                RoundMode::CAST_NONE,
                this->numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(y1Local, yLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDim);
            Cast(y2Local, xLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDim);
            PipeBarrier<PIPE_V>();
            yQue.EnQue(y12Local);
        }
    }

    __aicore__ inline void ScaleTensor(
        LocalTensor<float> &srcTensor, LocalTensor<float> &tmpTensor, LocalTensor<float> &scaleTensor, int32_t idx)
    {
        Abs(tmpTensor, srcTensor, this->numLastDim);  // tmpLocal <-- |y * smooth|
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor, this->numLastDim);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float maxTemp = tmpTensor.GetValue(0);
        float scaleTemp = maxTemp / DYNAMIC_QUANT_DIVIDEND;
        scaleTensor.SetValue(idx, scaleTemp);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        auto dividendTensor = divisorBuf.Get<float>();
        DivScalarFP32(srcTensor, srcTensor, dividendTensor, scaleTemp, this->numLastDim);  // srcTensor <-- y / scale
    }

    __aicore__ inline void CopyOut(int32_t gmOffset)
    {
        LocalTensor<int8_t> res12 = yQue.template DeQue<int8_t>();
        auto res1 = res12[0];
        auto res2 = res12[this->numLastDimRoundUp32];
        DataCopyEx(this->y1Gm[gmOffset], res1, this->numLastDim);
        if (smooth2Exist) {
            DataCopyEx(this->y2Gm[gmOffset], res2, this->numLastDim);
        }
        yQue.FreeTensor(res12);
    }

    __aicore__ inline void CopyOutScale(int32_t gmOffset, int32_t copyInNums)
    {
        LocalTensor<float> outScalesLocal = scalesQue.template DeQue<float>();
        LocalTensor<float> outScales1Local = outScalesLocal[0];
        LocalTensor<float> outScales2Local = outScalesLocal[ROW_FACTOR];
        DataCopyEx(outScale1Gm[gmOffset], outScales1Local, copyInNums);
        if (smooth2Exist) {
            DataCopyEx(outScale2Gm[gmOffset], outScales2Local, copyInNums);
        }
        scalesQue.FreeTensor(outScalesLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> biasBuf;
    TBuf<TPosition::VECCALC> divisorBuf;

    GlobalTensor<T> smooth1Gm, smooth2Gm;
    GlobalTensor<float> outScale1Gm, outScale2Gm;

    bool smooth1Exist;
    bool smooth2Exist;
};

#endif  // ADD_LAYER_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_

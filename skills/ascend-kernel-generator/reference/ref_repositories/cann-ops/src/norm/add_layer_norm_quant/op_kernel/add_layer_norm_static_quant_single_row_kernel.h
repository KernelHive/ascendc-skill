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
 * \file add_layer_norm_static_quant_single_row_kernel.h
 * \brief
 */
#ifndef ADD_LAYER_NORM_STATIC_QUANT_SINGLE_ROW_KERNEL_H_
#define ADD_LAYER_NORM_STATIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "add_layer_norm_quant_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormDualStaticQuantSingleRow : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormDualStaticQuantSingleRow(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *scales1, __gm__ uint8_t *scales2, __gm__ uint8_t *zeroPoints1,
        __gm__ uint8_t *zeroPoints2, __gm__ uint8_t *y1, __gm__ uint8_t *y2, __gm__ uint8_t *x,
        __gm__ uint8_t *fakeOut1, __gm__ uint8_t *fakeOut2, __gm__ uint8_t *workspace,
        const AddLayerNormQuantTilingData *tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y1, y2, x);

        isZeroPoint1Exist = tiling->scaleOffsetMode % 100 >= 10;
        isZeroPoint2Exist = tiling->scaleOffsetMode % 10 >= 1;

        scales1Gm.SetGlobalBuffer((__gm__ T *)scales1);
        scales2Gm.SetGlobalBuffer((__gm__ T *)scales2);
        if (isZeroPoint1Exist) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ T *)zeroPoints1);
        }
        if (isZeroPoint2Exist) {
            zeroPoints2Gm.SetGlobalBuffer((__gm__ T *)zeroPoints2);
        }
        // single row
        Ppipe->InitBuffer(rowInQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));             // 2
        Ppipe->InitBuffer(rowOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));            // 2
        Ppipe->InitBuffer(quantizeOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(int8_t));  // 1
        Ppipe->InitBuffer(tmpBuf, 3 * this->numLastDimAligned * sizeof(float));                   // 12
    }

    __aicore__ inline void Process()
    {
        uint32_t gm_offset = 0;
        for (uint32_t row_idx = 0; row_idx < this->rowWork; ++row_idx) {
            CopyInAddSingleRow(gm_offset);
            precision_compute_single_row(gm_offset);
            gm_offset += this->numLastDim;
        }
    }

private:
    __aicore__ inline void CopyInAddSingleRow(uint32_t gm_offset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();

        LocalTensor<T> x1x2Fp32 = tmpTensors.ReinterpretCast<T>()[0];
        LocalTensor<T> x1In = x1x2Fp32[0];
        LocalTensor<T> x2In = x1x2Fp32[this->numLastDimAligned];

        LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();

        DataCopyEx(x1In, this->x1Gm[gm_offset], this->numLastDim);
        DataCopyEx(x2In, this->x2Gm[gm_offset], this->numLastDim);
        DataCopyEx(biasIn, this->biasGm, this->numLastDim);

        rowInQue.EnQue(biasIn);
        auto biasTensor = rowInQue.template DeQue<T>();

        if constexpr (is_same<float, T>::value) {
            Add(x1In, x1In, x2In, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(x1In, x1In, biasTensor, this->numLastDim);
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            Adds(xOutTensor, x1In, ZERO, this->numLastDim);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyEx(this->xGm[gm_offset], xOut, this->numLastDim);
            rowOutQue.FreeTensor(xOut);
        } else {
            LocalTensor<float> xTensor = tmpTensors[0];
            LocalTensor<float> yTensor = tmpTensors[this->numLastDimAligned];
            LocalTensor<float> zTensor = tmpTensors[this->numLastDimAligned * 2];

            PipeBarrier<PIPE_V>();
            Cast(yTensor, x1In, RoundMode::CAST_NONE, this->numLastDim);
            Cast(zTensor, x2In, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(yTensor, yTensor, zTensor, this->numLastDim);
            Cast(xTensor, biasTensor, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(xTensor, xTensor, yTensor, this->numLastDim);
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            Cast(xOutTensor, xTensor, RoundMode::CAST_RINT, this->numLastDim);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyEx(this->xGm[gm_offset], xOut, this->numLastDim);
            rowOutQue.FreeTensor(xOut);
        }
        PipeBarrier<PIPE_V>();
        rowInQue.FreeTensor(biasTensor);
    }

    __aicore__ inline void precision_compute_single_row(uint32_t gm_offset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();

        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<float> tmpTensor = tmpTensors[this->numLastDimAligned];
        LocalTensor<float> constsTensor = tmpTensors[this->numLastDimAligned * 2];

        // CopyIn workspace beta gamma
        LocalTensor<T> scales01In = rowInQue.template AllocTensor<T>();
        LocalTensor<T> gammaLocal = constsTensor.ReinterpretCast<T>()[0];
        LocalTensor<T> betaLocal = constsTensor.ReinterpretCast<T>()[this->numLastDimAligned];

        DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
        DataCopyEx(betaLocal, this->betaGm, this->numLastDim);
        DataCopyEx(scales01In, scales1Gm, this->numLastDim);

        Muls(tmpTensor, xTensor, this->aveNum, this->numLastDim);         // x / N
        float aveLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);  // E(X)
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Adds(xTensor, xTensor, -1 * aveLocalTemp, this->numLastDim);  // x - E(X)
        PipeBarrier<PIPE_V>();
        Mul(tmpTensor, xTensor, xTensor, this->numLastDim);  // (x - E(X)) ** 2
        PipeBarrier<PIPE_V>();
        Muls(tmpTensor, tmpTensor, this->aveNum, this->numLastDim);  //  ((x - E(X)) ** 2) / N
        PipeBarrier<PIPE_V>();
        float varLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);  // Var(x)
        PipeBarrier<PIPE_V>();
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);  // rstd = 1 / Sqrt(Var(x) + this->eps)
        event_t eventSV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV1);
        WaitFlag<HardEvent::S_V>(eventSV1);
        Muls(xTensor, xTensor, rstdLocalTemp, this->numLastDim);  // (x - E(X)) * rstd

        rowInQue.EnQue(scales01In);
        auto scales01Tensor = rowInQue.template DeQue<T>();

        Cast(tmpTensor, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);  // tmpTensor <-- gammaFp32
        PipeBarrier<PIPE_V>();
        Mul(xTensor, xTensor, tmpTensor, this->numLastDim);                     // xTensor <-- (x - E(X)) * rstd * gamma
        Cast(constsTensor, betaLocal, RoundMode::CAST_NONE, this->numLastDim);  // constsTensor <-- betaFp32
        PipeBarrier<PIPE_V>();
        Cast(tmpTensor, scales01Tensor, RoundMode::CAST_NONE, this->numLastDim);  // tmpTensor <-- scales01Tensor
        rowInQue.FreeTensor(scales01Tensor);

        LocalTensor<T> offsets01In;
        if (isZeroPoint1Exist) {
            offsets01In = rowInQue.template AllocTensor<T>();
            DataCopyEx(offsets01In, zeroPoints1Gm, this->numLastDim);
        }

        Add(xTensor, xTensor, constsTensor, this->numLastDim);  // xTensor <-- (x - E(X)) * rstd * gamma + beta
        PipeBarrier<PIPE_V>();

        // small operator action
        Cast(constsTensor.ReinterpretCast<T>(), xTensor, RoundMode::CAST_RINT, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Cast(xTensor, constsTensor.ReinterpretCast<T>(), RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();

        Div(constsTensor,
            xTensor,
            tmpTensor,
            this->numLastDim);  // constsTensor <-- ((x - E(X)) * rstd * gamma + beta) / scales
        PipeBarrier<PIPE_V>();

        if (isZeroPoint1Exist) {
            rowInQue.EnQue(offsets01In);
            auto offsets01Local = rowInQue.template DeQue<T>();
            Cast(tmpTensor, offsets01Local, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            rowInQue.FreeTensor(offsets01Local);
            Add(constsTensor, constsTensor, tmpTensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Local = quantizeOutQue.template AllocTensor<int8_t>();
        Cast(constsTensor.ReinterpretCast<int32_t>(), constsTensor, RoundMode::CAST_RINT, this->numLastDim);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(constsTensor.ReinterpretCast<half>(),
            constsTensor.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            this->numLastDim);
        PipeBarrier<PIPE_V>();
        Cast(y1Local, constsTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDim);
        quantizeOutQue.EnQue(y1Local);
        auto y1Out = quantizeOutQue.template DeQue<int8_t>();
        DataCopyEx(this->y1Gm[gm_offset], y1Out, this->numLastDim);
        quantizeOutQue.FreeTensor(y1Out);

        // quantize2
        LocalTensor<T> scales2In = rowInQue.template AllocTensor<T>();
        DataCopyEx(scales2In, scales2Gm, this->numLastDim);
        LocalTensor<T> offsets2Tensor;
        if (isZeroPoint2Exist) {
            offsets2Tensor = rowOutQue.template AllocTensor<T>();
            DataCopyEx(offsets2Tensor, zeroPoints2Gm, this->numLastDim);
        }
        rowInQue.EnQue(scales2In);
        auto scales2Tensor = rowInQue.template DeQue<T>();

        Cast(constsTensor, scales2Tensor, RoundMode::CAST_NONE, this->numLastDim);
        if (isZeroPoint2Exist) {
            Cast(tmpTensor, offsets2Tensor, RoundMode::CAST_NONE, this->numLastDim);
            rowOutQue.FreeTensor(offsets2Tensor);
        }
        PipeBarrier<PIPE_V>();
        rowInQue.FreeTensor(scales2Tensor);

        Div(constsTensor, xTensor, constsTensor, this->numLastDim);  // ((x - E(X)) * rstd * gamma + beta) / scales
        event_t event_v_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte2);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte2);

        PipeBarrier<PIPE_V>();
        if (isZeroPoint2Exist) {
            Add(constsTensor, constsTensor, tmpTensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y2Local = quantizeOutQue.template AllocTensor<int8_t>();
        Cast(constsTensor.ReinterpretCast<int32_t>(), constsTensor, RoundMode::CAST_RINT, this->numLastDim);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(constsTensor.ReinterpretCast<half>(),
            constsTensor.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            this->numLastDim);
        PipeBarrier<PIPE_V>();
        Cast(y2Local, constsTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDim);
        quantizeOutQue.EnQue(y2Local);
        auto y2Out = quantizeOutQue.template DeQue<int8_t>();
        DataCopyEx(this->y2Gm[gm_offset], y2Out, this->numLastDim);
        quantizeOutQue.FreeTensor(y2Out);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> quantizeOutQue;

    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<T> scales1Gm, scales2Gm, zeroPoints1Gm, zeroPoints2Gm;
    GlobalTensor<float> workspaceGm;

    bool isZeroPoint1Exist;
    bool isZeroPoint2Exist;
};

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormSoleStaticQuantSingleRow : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormSoleStaticQuantSingleRow(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *scales, __gm__ uint8_t *fakeScale, __gm__ uint8_t *offsets,
        __gm__ uint8_t *fakeOffsets, __gm__ uint8_t *y, __gm__ uint8_t *fakeY, __gm__ uint8_t *x,
        __gm__ uint8_t *fakeOut1, __gm__ uint8_t *fakeOut2, __gm__ uint8_t *workspace,
        const AddLayerNormQuantTilingData *tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y, fakeY, x);

        isOffsetExist = tiling->scaleOffsetMode % 100 >= 10;

        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        offsetsGm.SetGlobalBuffer((__gm__ T *)offsets);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace + block_idx * 2 * this->numLastDim);

        // single row
        Ppipe->InitBuffer(rowInQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(rowOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(quantizeOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(int8_t));
        Ppipe->InitBuffer(tmpBuf, 2 * this->numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(gammaBetaBuf, 2 * this->numLastDimAligned * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyInBetaGammaBias();
        for (int32_t row_idx = 0; row_idx < this->rowWork; ++row_idx) {
            CopyInAddSingleRow(row_idx);
            precision_compute_single_row(row_idx);
        }
    }

private:
    __aicore__ inline void CopyInAddSingleRow(int32_t row_idx)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();

        LocalTensor<float> x1Fp32 = tmpTensors[0];
        LocalTensor<float> x2Fp32 = tmpTensors[this->numLastDimAligned];

        LocalTensor<T> x1In;
        LocalTensor<T> x2In;

        if constexpr (is_same<float, T>::value) {
            x1In = x1Fp32;
            x2In = x2Fp32;
        } else {
            x1In = x2Fp32.ReinterpretCast<T>()[0];
            x2In = x2Fp32.ReinterpretCast<T>()[this->numLastDimAligned];
        }

        LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();
        uint32_t gm_offset = row_idx * this->numLastDim;
        DataCopyEx(x1In, this->x1Gm[gm_offset], this->numLastDim);
        DataCopyEx(x2In, this->x2Gm[gm_offset], this->numLastDim);

        rowInQue.EnQue(biasIn);
        auto biasTensor = rowInQue.template DeQue<T>();

        if constexpr (is_same<float, T>::value) {
            Add(x1In, x1In, x2In, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(x1In, x1In, biasTensor, this->numLastDim);
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            Adds(xOutTensor, x1In, ZERO, this->numLastDim);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyEx(this->xGm[gm_offset], xOut, this->numLastDim);
            rowOutQue.FreeTensor(xOut);
        } else {
            Cast(x1Fp32, x1In, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(x2Fp32, x2In, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32, x1Fp32, x2Fp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(x2Fp32, biasTensor, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32, x1Fp32, x2Fp32, this->numLastDim);
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            Cast(xOutTensor, x1Fp32, RoundMode::CAST_RINT, this->numLastDim);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyEx(this->xGm[gm_offset], xOut, this->numLastDim);
            rowOutQue.FreeTensor(xOut);
        }
        PipeBarrier<PIPE_V>();
        rowInQue.FreeTensor(biasTensor);
    }

    __aicore__ inline void precision_compute_single_row(int32_t row_idx)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<T> gammaBetaTensor = gammaBetaBuf.Get<T>();

        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<float> tmpTensor = tmpTensors[this->numLastDimAligned];
        LocalTensor<T> gammaTensor = gammaBetaTensor[0];
        LocalTensor<T> betaTensor = gammaBetaTensor[this->numLastDimAligned];

        Muls(tmpTensor, xTensor, this->aveNum, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float aveLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Adds(xTensor, xTensor, -1 * aveLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Mul(tmpTensor, xTensor, xTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Muls(tmpTensor, tmpTensor, this->aveNum, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float varLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        LocalTensor<T> inTensor = rowOutQue.template AllocTensor<T>();
        DataCopyEx(inTensor, scalesGm, this->numLastDim);

        Cast(tmpTensor, gammaTensor, RoundMode::CAST_NONE, this->numLastDim);
        Muls(xTensor, xTensor, rstdLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();

        Mul(xTensor, tmpTensor, xTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Cast(tmpTensor, betaTensor, RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Add(xTensor, tmpTensor, xTensor, this->numLastDim);

        // small operator action
        if constexpr (!is_same<float, T>::value) {
            PipeBarrier<PIPE_V>();
            Cast(tmpTensor.ReinterpretCast<T>(), xTensor, RoundMode::CAST_RINT, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(xTensor, tmpTensor.ReinterpretCast<T>(), RoundMode::CAST_NONE, this->numLastDim);
        }
        PipeBarrier<PIPE_V>();

        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        Cast(tmpTensor, inTensor, RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();

        event_t eventVMTE2;
        if (isOffsetExist) {
            eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            DataCopyEx(inTensor, offsetsGm, this->numLastDim);
        }
        Div(xTensor, xTensor, tmpTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();

        if (isOffsetExist) {
            eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
            Cast(tmpTensor, inTensor, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(xTensor, xTensor, tmpTensor, this->numLastDim);
        }

        rowOutQue.FreeTensor(inTensor);
        eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        LocalTensor<int8_t> yLocal = quantizeOutQue.template AllocTensor<int8_t>();
        Cast(xTensor.ReinterpretCast<int32_t>(), xTensor, RoundMode::CAST_RINT, this->numLastDim);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(xTensor.ReinterpretCast<half>(),
            xTensor.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            this->numLastDim);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, xTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDim);
        quantizeOutQue.EnQue(yLocal);
        auto yOut = quantizeOutQue.template DeQue<int8_t>();
        DataCopyEx(this->y1Gm[row_idx * this->numLastDim], yOut, this->numLastDim);
        quantizeOutQue.FreeTensor(yOut);
    }

    __aicore__ inline void CopyInBetaGammaBias()
    {
        LocalTensor<T> gammaBetaTensor = gammaBetaBuf.Get<T>();

        LocalTensor<T> gammaIn = gammaBetaTensor[0];
        LocalTensor<T> betaIn = gammaBetaTensor[this->numLastDimAligned];
        LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();

        DataCopyEx(gammaIn, this->gammaGm, this->numLastDim);
        DataCopyEx(betaIn, this->betaGm, this->numLastDim);
        DataCopyEx(biasIn, this->biasGm, this->numLastDim);

        rowInQue.EnQue(biasIn);
        auto biasTensor = rowInQue.template DeQue<T>();

        rowInQue.FreeTensor(biasTensor);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> quantizeOutQue;

    TBuf<TPosition::VECCALC> tmpBuf, gammaBetaBuf;

    GlobalTensor<T> scalesGm, offsetsGm;

    GlobalTensor<float> workspaceGm;

    bool isOffsetExist;
};

#endif  // ADD_LAYER_NORM_STATIC_QUANT_SINGLE_ROW_KERNEL_H_
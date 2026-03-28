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
 * \file quantize_add_layer_norm_normal_kernel.h
 * \brief
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_NORMAL_KERNEL_H_
#define QUANTIZE_ADD_LAYER_NORM_NORMAL_KERNEL_H_

#include "quantize_add_layer_norm_base.h"

template <typename T, typename S, int TILING_KEY, int BUFFER_NUM = 1>
class KernelQuantizeAddLayerNormNormalPerTensorKernel {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)

public:
    __aicore__ inline KernelQuantizeAddLayerNormNormalPerTensorKernel(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *scales, __gm__ uint8_t *offsets, __gm__ uint8_t *y, __gm__ uint8_t *x,
        uint32_t numCore_, uint32_t numLastDim_, uint32_t numFirstDim_, uint32_t nlFirstDimPerCore_,
        uint32_t lFirstDimPerCore_, float eps_, float aveNum_)
    {
        numCore = numCore_;
        numLastDim = numLastDim_;
        numFirstDim = numFirstDim_;
        nlFirstDimPerCore = nlFirstDimPerCore_;
        lFirstDimPerCore = lFirstDimPerCore_;
        firstDimPerTime = 1;
        lastDimPerTime = 1;
        aveNum = aveNum_;
        eps = eps_;
        if (block_idx != numCore - 1) {
            rowWork = nlFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = THE_SMALLEST_NUMBER_BETWEEN_TWO_NUMBERS(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);
        gmOffset_ = nlFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T *)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T *)(x2) + block_idx * gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T *)beta);
        biasGm.SetGlobalBuffer((__gm__ T *)bias);

        scalesGm.SetGlobalBuffer((__gm__ S *)scales);
        if (offsets != nullptr) {
            offsetsGm.SetGlobalBuffer((__gm__ S *)offsets);
            hasOffset = true;
        }

        yGm.SetGlobalBuffer((__gm__ int8_t *)(y) + block_idx * gmOffset_);
        xGm.SetGlobalBuffer((__gm__ T *)(x) + block_idx * gmOffset_);

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }

        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * rowStep * numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(xQue, BUFFER_NUM, rowStep * numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(yQue, BUFFER_NUM, rowStep * numLastDimAligned * sizeof(int8_t));

        Ppipe->InitBuffer(xBufFp32, rowStep * numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(yBufFp32, rowStep * numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(zBufFp32, rowStep * numLastDimAligned * sizeof(float));

        Ppipe->InitBuffer(gammaBuf, rowStep * numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(betaBuf, rowStep * numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(biasBuf, rowStep * numLastDimAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t row_move_cnt = CEIL_DIV(rowWork, rowStep);
        CopyInPhase0();
        auto gammaLocal = gammaBuf.Get<float>();
        auto betaLocal = betaBuf.Get<float>();
        auto biasLocal = biasBuf.Get<float>();

        DataCopyPadParams padParams;
        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = numLastDimAligned - numLastDim;
        }

        for (int32_t row_idx = 0; row_idx < row_move_cnt - 1; ++row_idx) {
            CopyInAndAdd(biasLocal, row_idx, rowStep, padParams);
            CopyOutAdditionalOutput(row_idx, rowStep);
            precision_compute(rowStep, gammaLocal, betaLocal);
            CopyOut(row_idx, rowStep);
        }
        {
            int32_t row_idx = row_move_cnt - 1;
            CopyInAndAdd(biasLocal, row_idx, rowTail_, padParams);
            CopyOutAdditionalOutput(row_idx, rowTail_);
            precision_compute(rowTail_, gammaLocal, betaLocal);
            CopyOut(row_idx, rowTail_);
        }
    }

private:
    __aicore__ inline void CopyInAndAdd(
        LocalTensor<float> biasLocal, int32_t proc_id, int32_t row_count, DataCopyPadParams &padParams)
    {
        uint32_t gm_offset = proc_id * rowStep * numLastDim;
        auto elementCount = numLastDimAligned * row_count;

        LocalTensor<T> x1x2CopyIn = inRowsQue.template AllocTensor<T>();
        LocalTensor<T> x1CopyIn = x1x2CopyIn[0];
        LocalTensor<T> x2CopyIn = x1x2CopyIn[elementCount];

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> addBufLocal = zBufFp32.Get<float>();

        DataCopyEx(x1CopyIn, x1Gm[gm_offset], numLastDim, row_count, padParams);
        DataCopyEx(x2CopyIn, x2Gm[gm_offset], numLastDim, row_count, padParams);

        inRowsQue.EnQue(x1x2CopyIn);
        auto x1x2Local = inRowsQue.template DeQue<T>();
        LocalTensor<T> x1Local = x1x2Local[0];
        LocalTensor<T> x2Local = x1x2Local[elementCount];

        if constexpr (IsSame<float, T>::value) {
            Add(addBufLocal, x1Local, x2Local, elementCount);
        } else {
            Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(addBufLocal, xLocalFp32, yLocalFp32, elementCount);
        }
        inRowsQue.FreeTensor(x1x2Local);

        PipeBarrier<PIPE_V>();
        for (int i = 0; i < row_count; i++) {
            Add(addBufLocal[i * numLastDimAligned], biasLocal, addBufLocal[i * numLastDimAligned], numLastDim);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t proc_id, int32_t row_count)
    {
        LocalTensor<float> addBufLocal = zBufFp32.Get<float>();

        uint32_t gm_offset = proc_id * rowStep * numLastDim;
        auto elementCount = numLastDimAligned * row_count;

        auto xLocal = xQue.template AllocTensor<T>();
        if constexpr (IsSame<T, float>::value) {
            Adds(xLocal, addBufLocal, ZERO, elementCount);
        } else if constexpr (IsSame<T, half>::value) {
            Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, elementCount);
        } else {
            Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, elementCount);
        }

        xQue.template EnQue<T>(xLocal);
        auto x = xQue.template DeQue<T>();
        DataCopyEx(xGm[gm_offset], x, numLastDim, row_count);
        xQue.FreeTensor(x);
    }

    __aicore__ inline void precision_compute(
        int32_t nums, LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        LocalTensor<int8_t> yLocal = yQue.template AllocTensor<int8_t>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * numLastDimAligned;
            Muls(yLocalFp32, zLocalFp32[roundOffset], aveNum, numLastDim);
            PipeBarrier<PIPE_V>();
            auto ave_local_temp = ReduceSumFP32(yLocalFp32, numLastDim);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Adds(zLocalFp32[roundOffset], zLocalFp32[roundOffset], ave_local_temp * -1, numLastDim);
            PipeBarrier<PIPE_V>();
            Mul(xLocalFp32, zLocalFp32[roundOffset], zLocalFp32[roundOffset], numLastDim);
            PipeBarrier<PIPE_V>();
            Muls(yLocalFp32, xLocalFp32, aveNum, numLastDim);
            PipeBarrier<PIPE_V>();
            float var_local_temp = ReduceSumFP32(yLocalFp32, numLastDim);
            float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
            eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xLocalFp32, zLocalFp32[roundOffset], rstd_local_temp, numLastDim);
            PipeBarrier<PIPE_V>();

            Mul(yLocalFp32, xLocalFp32, gammaLocal, numLastDim);
            PipeBarrier<PIPE_V>();
            Add(zLocalFp32[roundOffset], yLocalFp32, betaLocal, numLastDim);
            PipeBarrier<PIPE_V>();

            LocalTensor<half> tmpHalfTensor = yLocalFp32.ReinterpretCast<half>();
            Cast(tmpHalfTensor, zLocalFp32[roundOffset], RoundMode::CAST_NONE, numLastDim);
            PipeBarrier<PIPE_V>();
            Muls(tmpHalfTensor, tmpHalfTensor, scale, numLastDim);
            PipeBarrier<PIPE_V>();
            Adds(tmpHalfTensor, tmpHalfTensor, offset, numLastDim);
            PipeBarrier<PIPE_V>();

            Cast(xLocalFp32.ReinterpretCast<int32_t>(), tmpHalfTensor, RoundMode::CAST_RINT, numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(tmpHalfTensor, xLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(yLocal[roundOffset], tmpHalfTensor, RoundMode::CAST_TRUNC, numLastDim);
            PipeBarrier<PIPE_V>();
        }
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t row_count)
    {
        LocalTensor<int8_t> res = yQue.template DeQue<int8_t>();
        uint32_t gm_offset = row_idx * rowStep * numLastDim;
        DataCopyEx(yGm[gm_offset], res, numLastDim, row_count);
        yQue.FreeTensor(res);
    }

    __aicore__ inline void CopyInPhase0()
    {
        // copy in 32 Bytes only
        DataCopyExtParams copyParams{1, 32, 0, 0, 0};
        DataCopyPadExtParams<S> padParams{true, 0, 0, 0};

        LocalTensor<S> constCopyIn = inRowsQue.template AllocTensor<S>();

        LocalTensor<float> gammaFp32 = gammaBuf.Get<float>();
        LocalTensor<float> betaFp32 = betaBuf.Get<float>();
        LocalTensor<float> biasFp32 = biasBuf.Get<float>();

        LocalTensor<T> gammaIn = xBufFp32.Get<T>();
        LocalTensor<T> betaIn = yBufFp32.Get<T>();
        LocalTensor<T> biasIn = zBufFp32.Get<T>();

        DataCopyEx(gammaIn, gammaGm, numLastDim);
        DataCopyEx(betaIn, betaGm, numLastDim);
        DataCopyEx(biasIn, biasGm, numLastDim);
        DataCopyPad(constCopyIn[0], scalesGm, copyParams, padParams);
        // size of 16 elements is 32/64 Bytes
        if (hasOffset) {
            DataCopyPad(constCopyIn[16], offsetsGm, copyParams, padParams);
        }

        event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMTE2S);
        WaitFlag<HardEvent::MTE2_S>(eventMTE2S);
        inRowsQue.EnQue(constCopyIn);
        auto constLocal = inRowsQue.template DeQue<S>();

        LocalTensor<S> scaleConstLocal = constLocal[0];
        // size of 16 elements is 32/64 Bytes
        LocalTensor<S> offsetConstLocal = constLocal[16];

        scale = (half)scaleConstLocal.GetValue(0);
        offset = (hasOffset) ? (half)offsetConstLocal.GetValue(0) : (half)0.0;

        if constexpr (IsSame<float, T>::value) {
            Adds(gammaFp32, gammaIn, (float)0.0, numLastDim);
            Adds(betaFp32, betaIn, (float)0.0, numLastDim);
            Adds(biasFp32, biasIn, (float)0.0, numLastDim);
        } else {
            Cast(gammaFp32, gammaIn, RoundMode::CAST_NONE, numLastDim);
            Cast(betaFp32, betaIn, RoundMode::CAST_NONE, numLastDim);
            Cast(biasFp32, biasIn, RoundMode::CAST_NONE, numLastDim);
        }
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(constLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;

    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> biasBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> xGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<S> scalesGm;
    GlobalTensor<S> offsetsGm;

    uint32_t numCore;
    uint32_t numFirstDim;
    uint32_t numLastDim;
    uint32_t rowStep;
    uint32_t rowWork;
    uint32_t gmOffset_;
    uint32_t rowTail_;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    uint32_t nlFirstDimPerCore;
    uint32_t lFirstDimPerCore;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t numLastDimAligned;
    half offset;
    half scale;
    bool hasOffset = false;
};

#endif  // QUANTIZE_ADD_LAYER_NORM_NORMAL_KERNEL_H_
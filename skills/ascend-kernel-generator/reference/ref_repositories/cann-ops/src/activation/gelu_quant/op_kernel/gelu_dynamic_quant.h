/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_dynamic_quant.h
 * \brief
 */

#ifndef GELU_DYNAMIC_QUANT_H
#define GELU_DYNAMIC_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base.h"

namespace GeluQuantALL {
using namespace AscendC;
// dynamic quant 基础模板  ub内单个尾轴或者多个尾轴
template <typename T1, typename T2> class GeluDynamicQuant : public GeluQuantBase {
public:
    __aicore__ inline GeluDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale,
        GM_ADDR workspace, const GeluQuantTilingData &tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMulRows(LocalTensor<float> &scaleLocalFp32, int64_t offset, int32_t rowCount);
    __aicore__ inline void ProcessOptionalScale(LocalTensor<float> &scaleLocalFp32, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t offset, int32_t rowCount);
    __aicore__ inline void Compute(LocalTensor<float> &scaleLocalFp32, int32_t rowCount);
    __aicore__ inline void CopyOut(int64_t offset, int32_t rowCount);
    __aicore__ inline void ComputeGelu(LocalTensor<float> &geluRes, int32_t rowCount);
    __aicore__ inline void ComputeDynamicQuant(LocalTensor<float> &geluRes, LocalTensor<float> &scaleLocalFp32,
        int32_t rowCount);

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleOutQueue_;

    TBuf<TPosition::VECCALC> castQueue_;
    TBuf<TPosition::VECCALC> geluQueue_;
    TBuf<TPosition::VECCALC> tempQueue_;
    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> outScaleGm_;

    /* variable */
    uint32_t mulRows_;
    uint32_t endAxisLenAlignTo32_;
    uint32_t endAxisLenAlignTo16_;
    uint32_t endAxisLenAlignTo8_;
};

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y,
    GM_ADDR outScale, GM_ADDR workspace, const GeluQuantTilingData &tilingData)
{
    // Init tiling data
    GeluQuantBase::ParseTilingData(tilingData);
    endAxisLenAlignTo32_ = (endAxisLen_ + FP32_BLOCK_NUM - 1) / FP32_BLOCK_NUM * FP32_BLOCK_NUM;
    endAxisLenAlignTo16_ = (endAxisLen_ + FP16_BLOCK_NUM - 1) / FP16_BLOCK_NUM * FP16_BLOCK_NUM;
    endAxisLenAlignTo8_ = (endAxisLen_ + INT8_BLOCK_NUM - 1) / INT8_BLOCK_NUM * INT8_BLOCK_NUM;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset1 = blockIdx_ * normalCoreProcessNum_ * endAxisLen_;
    uint64_t intraCoreOffset2 = blockIdx_ * normalCoreProcessNum_;

    // shield normal core and tail core
    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ T1 *)x + intraCoreOffset1);
    inputScaleGm_.SetGlobalBuffer((__gm__ T2 *)inputScale);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y + intraCoreOffset1);
    outScaleGm_.SetGlobalBuffer((__gm__ float *)outScale + intraCoreOffset2);

    pipe.InitBuffer(inQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(geluQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(castQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(tempQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleOutQueue_, BUFFER_NUM, MAX_ROWS_NUM * sizeof(float));
    pipe.InitBuffer(scaleQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ProcessMulRows(LocalTensor<float> &scaleLocalFp32, int64_t offset,
    int32_t rowCount)
{
    CopyIn(offset, rowCount);
    Compute(scaleLocalFp32, rowCount);
    CopyOut(offset, rowCount);
}

template <typename T1, typename T2> __aicore__ inline void GeluDynamicQuant<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    LocalTensor<float> scaleLocalFp32 = scaleQueue_.AllocTensor<float>();
    if (inputScaleType_ == NORMAL_TENSOR) {
        ProcessOptionalScale(scaleLocalFp32, endAxisLen_);
    } else if (inputScaleType_ == SCALAR_TENSOR) {
        ProcessOptionalScale(scaleLocalFp32, 1);
        PipeBarrier<PIPE_ALL>();
        inputScaleScalar_ = scaleLocalFp32.GetValue(0);
    }

    mulRows_ = coexistentNodeElementNum_ / endAxisLenAlignTo32_;
    int64_t ubLoopNum = curCoreProcessNum_ / mulRows_;
    int64_t ubLoopTail = curCoreProcessNum_ % mulRows_;
    for (int64_t loopIndex = 0; loopIndex < ubLoopNum; loopIndex++) {
        ProcessMulRows(scaleLocalFp32, mulRows_ * endAxisLen_ * loopIndex, mulRows_);
    }
    if (ubLoopTail != 0) {
        ProcessMulRows(scaleLocalFp32, mulRows_ * endAxisLen_ * ubLoopNum, ubLoopTail);
    }
    scaleQueue_.FreeTensor(scaleLocalFp32);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ProcessOptionalScale(LocalTensor<float> &scaleLocalFp32,
    int32_t calCount)
{
    LocalTensor<T2> optionalScale = inQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T2> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };

    DataCopyPad(optionalScale, inputScaleGm_[0], copyParams, padParams);

    inQueue_.EnQue(optionalScale);
    LocalTensor<T2> scaleInput = inQueue_.DeQue<T2>();

    if constexpr (IsSameType<T2, float>::value) {
        Muls(scaleLocalFp32, scaleInput, 1.0f, calCount);
    } else {
        Cast(scaleLocalFp32, scaleInput, RoundMode::CAST_NONE, calCount);
    }

    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(optionalScale);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::CopyIn(int64_t offset, int32_t rowCount)
{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(rowCount), static_cast<uint32_t>(endAxisLen_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T1> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };
    DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::Compute(LocalTensor<float> &scaleLocalFp32, int32_t rowCount)
{
    LocalTensor<float> geluRes = geluQueue_.Get<float>();
    ComputeGelu(geluRes, rowCount);
    ComputeDynamicQuant(geluRes, scaleLocalFp32, rowCount);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::CopyOut(int64_t offset, int32_t rowCount)
{
    LocalTensor<float> scaleOutLocal = scaleOutQueue_.DeQue<float>();
    DataCopyExtParams copyParamsFloat{ static_cast<uint16_t>(1), static_cast<uint32_t>(rowCount * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPad(outScaleGm_[offset / endAxisLen_], scaleOutLocal, copyParamsFloat);
    scaleOutQueue_.FreeTensor(scaleOutLocal);

    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{ static_cast<uint16_t>(rowCount),
        static_cast<uint32_t>(endAxisLen_ * sizeof(int8_t)), static_cast<uint32_t>(0), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0) };

    DataCopyPad(yGm_[offset], outLocal, copyParamsInt8);
    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ComputeGelu(LocalTensor<float> &geluRes, int32_t rowCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();
    LocalTensor<float> castFp32 = castQueue_.Get<float>();
    for (int32_t i = 0; i < rowCount; i++) {
        if constexpr (IsSameType<T1, float>::value) {
            Muls(castFp32[i * endAxisLenAlignTo32_], xLocal[i * endAxisLenAlignTo32_], 1.0f, endAxisLen_);
        } else {
            Cast(castFp32[i * endAxisLenAlignTo32_], xLocal[i * endAxisLenAlignTo16_], RoundMode::CAST_NONE,
                endAxisLen_);
        }

        PipeBarrier<PIPE_V>();
        if (approximate_ == APPROXIMATE_NONE) {
            LocalTensor<float> xSquared = tempQueue_.Get<float>();
            ComputeGeluErf(castFp32[i * endAxisLenAlignTo32_], geluRes[i * endAxisLenAlignTo32_], xSquared,
                endAxisLen_);
        } else {
            ComputeGeluTanh(castFp32[i * endAxisLenAlignTo32_], geluRes[i * endAxisLenAlignTo32_], endAxisLen_);
        }
    }

    inQueue_.FreeTensor(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ComputeDynamicQuant(LocalTensor<float> &geluRes,
    LocalTensor<float> &scaleLocalFp32, int32_t rowCount)
{
    LocalTensor<float> tempRes = tempQueue_.Get<float>();
    LocalTensor<float> scaleOutLocal = scaleOutQueue_.AllocTensor<float>();
    LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
    LocalTensor<half> castFp16 = castQueue_.Get<half>();

    for (int32_t i = 0; i < rowCount; i++) {
        if (inputScaleType_ == SCALAR_TENSOR) {
            Muls(geluRes[i * endAxisLenAlignTo32_], geluRes[i * endAxisLenAlignTo32_], inputScaleScalar_, endAxisLen_);
        } else if (inputScaleType_ == NORMAL_TENSOR) {
            Mul(geluRes[i * endAxisLenAlignTo32_], geluRes[i * endAxisLenAlignTo32_], scaleLocalFp32, endAxisLen_);
        }

        PipeBarrier<PIPE_V>(); // geluRes more consumption

        Abs(tempRes[i * endAxisLenAlignTo32_], geluRes[i * endAxisLenAlignTo32_], endAxisLen_);
        PipeBarrier<PIPE_V>();

        float maxValue = 0;
        ComputeReduceMax(tempRes[i * endAxisLenAlignTo32_], endAxisLen_, maxValue);
        float scale = MAX_INT8 / maxValue;
        scaleOutLocal.SetValue(i, 1 / scale);
        Muls(tempRes[i * endAxisLenAlignTo32_], geluRes[i * endAxisLenAlignTo32_], scale, endAxisLen_);
        PipeBarrier<PIPE_V>();

        Cast(castFp16[i * endAxisLenAlignTo16_], tempRes[i * endAxisLenAlignTo32_], RoundMode::CAST_ODD, endAxisLen_);
        PipeBarrier<PIPE_V>();

        Cast(outLocal[i * endAxisLenAlignTo8_], castFp16[i * endAxisLenAlignTo16_], RoundMode::CAST_RINT, endAxisLen_);
        PipeBarrier<PIPE_V>();
    }

    scaleOutQueue_.EnQue<float>(scaleOutLocal);
    outQueue_.EnQue<int8_t>(outLocal);
}
} // namespace GeluQuantALL
#endif
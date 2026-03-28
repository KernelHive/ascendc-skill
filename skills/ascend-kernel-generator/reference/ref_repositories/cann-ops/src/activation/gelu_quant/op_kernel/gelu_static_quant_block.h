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
 * \file gelu_static_quant_block.h
 * \brief
 */

#ifndef GELU_STATIC_QUANT_BLOCK_H
#define GELU_STATIC_QUANT_BLOCK_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base.h"

namespace GeluQuantALL {
using namespace AscendC;
// static quant performance template   单个矩形块处理，满核情况下单次处理多尾轴或者非满核时切分尾轴分核

template <typename T1, typename T2> class StaticQuantBlock : public GeluQuantBase {
public:
    __aicore__ inline StaticQuantBlock(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale,
        GM_ADDR workspace, const GeluQuantTilingData &tilingData);
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessOptionalInput(LocalTensor<float> &optionalLocalFp32, GlobalTensor<T2> &optionalGlobal,
        int64_t endAxisOffset_, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t blockOffset, int32_t rowCount, int32_t colCount);
    __aicore__ inline void Compute(LocalTensor<float> &scaleLocalFp32, LocalTensor<float> &offsetLocalFp32);
    __aicore__ inline void CopyOut(int64_t blockOffset, int32_t rowCount, int32_t colCount);
    __aicore__ inline void ComputeGelu(LocalTensor<float> &geluRes);
    __aicore__ inline void ComputeQuant(LocalTensor<float> &geluRes, LocalTensor<float> &scaleLocalFp32,
        LocalTensor<float> &offsetLocalFp32);

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> castQueue_;
    TBuf<TPosition::VECCALC> tempQueue_;

    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<T2> inputOffsetGm_;
    GlobalTensor<int8_t> yGm_;

    /* variable */

    uint32_t rowInner_;
    uint32_t rowTail_;
    uint32_t rowOuter_;

    uint32_t colInner_;
    uint32_t colTail_;
    uint32_t colOuter_;

    int64_t inputOffset_;
    int64_t endAxisOffset_;
    uint32_t rowActual_;
    uint32_t colActual_;
    uint32_t colActualAlignTo32_;
    uint32_t colActualAlignTo16_;
    uint32_t colActualAlignTo8_;
};

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y,
    GM_ADDR outScale, GM_ADDR workspace, const GeluQuantTilingData &tilingData)
{
    // Init tiling data
    GeluQuantBase::ParseTilingData(tilingData);
    rowInner_ = tilingData.rowInner;
    rowTail_ = tilingData.rowTail;
    colInner_ = tilingData.colInner;
    colTail_ = tilingData.colTail;
    rowOuter_ = tilingData.rowOuter;
    colOuter_ = tilingData.colOuter;

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield normal core and tail core
    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ T1 *)x);
    inputScaleGm_.SetGlobalBuffer((__gm__ T2 *)inputScale);
    inputOffsetGm_.SetGlobalBuffer((__gm__ T2 *)inputOffset);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y);

    pipe.InitBuffer(inQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(castQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(tempQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(offsetQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
}

template <typename T1, typename T2> __aicore__ inline void StaticQuantBlock<T1, T2>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIndex = blockIdx_ * normalCoreProcessNum_ + loopNum;
    int64_t rowIndex = baseBlockIndex / colOuter_;
    int64_t colIndex = baseBlockIndex % colOuter_;
    inputOffset_ =endAxisLen_ * rowIndex * rowInner_ + colIndex * colInner_;
    endAxisOffset_ = colIndex * colInner_;

    rowActual_ = rowIndex == rowOuter_ - 1 ? rowTail_ : rowInner_;
    colActual_ = colIndex == colOuter_ - 1 ? colTail_ : colInner_;
    colActualAlignTo32_ = (colActual_ + FP32_BLOCK_NUM - 1) / FP32_BLOCK_NUM * FP32_BLOCK_NUM;
    colActualAlignTo16_ = (colActual_ + FP16_BLOCK_NUM - 1) / FP16_BLOCK_NUM * FP16_BLOCK_NUM;
    colActualAlignTo8_ = (colActual_ + INT8_BLOCK_NUM - 1) / INT8_BLOCK_NUM * INT8_BLOCK_NUM;
}

template <typename T1, typename T2> __aicore__ inline void StaticQuantBlock<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        LocalTensor<float> scaleLocalFp32 = scaleQueue_.AllocTensor<float>();
        LocalTensor<float> offsetLocalFp32 = offsetQueue_.AllocTensor<float>();
        if (inputScaleType_ == NORMAL_TENSOR) {
            ProcessOptionalInput(scaleLocalFp32, inputScaleGm_, endAxisOffset_, colActual_);
        }

        if (inputOffsetType_ == NORMAL_TENSOR) {
            ProcessOptionalInput(offsetLocalFp32, inputOffsetGm_, endAxisOffset_, colActual_);
        }
        CopyIn(inputOffset_, rowActual_, colActual_);
        Compute(scaleLocalFp32, offsetLocalFp32);
        CopyOut(inputOffset_, rowActual_, colActual_);

        scaleQueue_.FreeTensor(scaleLocalFp32);
        offsetQueue_.FreeTensor(offsetLocalFp32);
    }
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::CopyIn(int64_t blockOffset, int32_t rowCount, int32_t colCount)
{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(rowCount), static_cast<uint32_t>(colCount * sizeof(T1)),
        static_cast<uint32_t>((endAxisLen_ - colCount) * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T1> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };
    DataCopyPad(xLocal, xGm_[blockOffset], copyParams, padParams);

    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::ProcessOptionalInput(LocalTensor<float> &optionalLocalFp32,
    GlobalTensor<T2> &optionalGlobal, int64_t endAxisOffset_, int32_t calCount)
{
    LocalTensor<T2> optionalInput = inQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T2> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };

    DataCopyPad(optionalInput, optionalGlobal[endAxisOffset_], copyParams, padParams);

    inQueue_.EnQue(optionalInput);
    LocalTensor<T2> tempLocal = inQueue_.DeQue<T2>();

    if constexpr (IsSameType<T2, float>::value) {
        Muls(optionalLocalFp32, tempLocal, 1.0f, calCount);
    } else {
        Cast(optionalLocalFp32, tempLocal, RoundMode::CAST_NONE, calCount);
    }
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(optionalInput);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::Compute(LocalTensor<float> &scaleLocalFp32,
    LocalTensor<float> &offsetLocalFp32)
{
    LocalTensor<float> geluRes = tempQueue_.Get<float>();
    ComputeGelu(geluRes);
    ComputeQuant(geluRes, scaleLocalFp32, offsetLocalFp32);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::CopyOut(int64_t blockOffset, int32_t rowCount, int32_t colCount)
{
    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{ static_cast<uint16_t>(rowCount), static_cast<uint32_t>(colCount * sizeof(int8_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>((endAxisLen_ - colCount) * sizeof(int8_t)),
        static_cast<uint32_t>(0) };

    DataCopyPad(yGm_[blockOffset], outLocal, copyParamsInt8);

    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::ComputeGelu(LocalTensor<float> &geluRes)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();
    LocalTensor<float> castFp32 = castQueue_.Get<float>();

    for (int32_t i = 0; i < rowActual_; i++) {
        if constexpr (IsSameType<T1, float>::value) {
            Muls(castFp32[i * colActualAlignTo32_], xLocal[i * colActualAlignTo32_], 1.0f, colActual_);
        } else {
            Cast(castFp32[i * colActualAlignTo32_], xLocal[i * colActualAlignTo16_], RoundMode::CAST_NONE, colActual_);
        }
        PipeBarrier<PIPE_V>();
    }

    inQueue_.FreeTensor(xLocal);

    int32_t calCount = rowActual_ * colActualAlignTo32_;
    if (approximate_ == APPROXIMATE_NONE) {
        LocalTensor<float> xSquared = inQueue_.AllocTensor<float>();
        ComputeGeluErf(castFp32, geluRes, xSquared, calCount);
        inQueue_.FreeTensor(xSquared);
    } else {
        ComputeGeluTanh(castFp32, geluRes, calCount);
    }
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantBlock<T1, T2>::ComputeQuant(LocalTensor<float> &geluRes,
    LocalTensor<float> &scaleLocalFp32, LocalTensor<float> &offsetLocalFp32)
{
    LocalTensor<half> castFp16 = castQueue_.Get<half>();
    LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
    for (int32_t i = 0; i < rowActual_; i++) {
        if (inputScaleType_ == NORMAL_TENSOR) {
            Mul(geluRes[i * colActualAlignTo32_], geluRes[i * colActualAlignTo32_], scaleLocalFp32, colActual_);
            PipeBarrier<PIPE_V>();
        }

        if (inputOffsetType_ == NORMAL_TENSOR) {
            Add(geluRes[i * colActualAlignTo32_], geluRes[i * colActualAlignTo32_], offsetLocalFp32, colActual_);
            PipeBarrier<PIPE_V>();
        }


        Cast(castFp16[i * colActualAlignTo16_], geluRes[i * colActualAlignTo32_], RoundMode::CAST_ODD, colActual_);
        PipeBarrier<PIPE_V>();
        Cast(outLocal[i * colActualAlignTo8_], castFp16[i * colActualAlignTo16_], RoundMode::CAST_RINT, colActual_);
        PipeBarrier<PIPE_V>();
    }

    outQueue_.EnQue<int8_t>(outLocal);
}
} // namespace GeluQuantALL
#endif
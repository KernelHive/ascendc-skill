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
 * \file gelu_static_quant.h
 * \brief
 */

#ifndef GELU_STATIC_QUANT_H
#define GELU_STATIC_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base.h"

namespace GeluQuantALL {
using namespace AscendC;
// static quant function template   单行处理或者大尾轴UB内for循环

template <typename T1, typename T2> class GeluQuant : public GeluQuantBase {
public:
    __aicore__ inline GeluQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale,
        GM_ADDR workspace, const GeluQuantTilingData &tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerLoop(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void ProcessOptionalInput(LocalTensor<float> &optionalLocalFp32, GlobalTensor<T2> &optionalGlobal,
        int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t endAxisOffset, int32_t calCount, int64_t loopNum);
    __aicore__ inline void Compute(LocalTensor<float> &scaleLocalFp32, LocalTensor<float> &offsetLocalFp32,
        int32_t calCount);
    __aicore__ inline void CopyOut(int64_t endAxisOffset, int32_t calCount, int64_t loopNum);
    __aicore__ inline void ComputeGelu(LocalTensor<float> &geluRes, int32_t calCount);
    __aicore__ inline void ComputeQuant(LocalTensor<float> &geluRes, LocalTensor<float> &scaleLocalFp32,
        LocalTensor<float> &offsetLocalFp32, int32_t calCount);

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> castQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> tempQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<T2> inputOffsetGm_;
    GlobalTensor<int8_t> yGm_;
};

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y,
    GM_ADDR outScale, GM_ADDR workspace, const GeluQuantTilingData &tilingData)
{
    // Init tiling data
    GeluQuantBase::ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_ * endAxisLen_;

    // shield normal core and tail core
    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ T1 *)x + intraCoreOffset);
    inputScaleGm_.SetGlobalBuffer((__gm__ T2 *)inputScale);
    inputOffsetGm_.SetGlobalBuffer((__gm__ T2 *)inputOffset);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y + intraCoreOffset);

    pipe.InitBuffer(inQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(castQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(tempQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(offsetQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
}

template <typename T1, typename T2> __aicore__ inline void GeluQuant<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t endAxisLoopNum = endAxisLen_ / coexistentNodeElementNum_;
    int64_t endAxisLoopTail = endAxisLen_ % coexistentNodeElementNum_;
    for (int64_t endAxisLoopIndex = 0; endAxisLoopIndex < endAxisLoopNum; endAxisLoopIndex++) {
        ProcessPerLoop(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
    }
    if (endAxisLoopTail != 0) {
        ProcessPerLoop(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ProcessPerLoop(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<float> scaleLocalFp32 = scaleQueue_.AllocTensor<float>();
    LocalTensor<float> offsetLocalFp32 = offsetQueue_.AllocTensor<float>();
    if (inputScaleType_ == NORMAL_TENSOR) {
        ProcessOptionalInput(scaleLocalFp32, inputScaleGm_, endAxisOffset, calCount);
    }

    if (inputOffsetType_ == NORMAL_TENSOR) {
        ProcessOptionalInput(offsetLocalFp32, inputOffsetGm_, endAxisOffset, calCount);
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        CopyIn(endAxisOffset, calCount, loopNum);
        Compute(scaleLocalFp32, offsetLocalFp32, calCount);
        CopyOut(endAxisOffset, calCount, loopNum);
    }
    scaleQueue_.FreeTensor(scaleLocalFp32);
    offsetQueue_.FreeTensor(offsetLocalFp32);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::CopyIn(int64_t endAxisOffset, int32_t calCount, int64_t loopNum)

{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T1> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };
    DataCopyPad(xLocal, xGm_[endAxisLen_ * loopNum + endAxisOffset], copyParams, padParams);

    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ProcessOptionalInput(LocalTensor<float> &optionalLocalFp32,
    GlobalTensor<T2> &optionalGlobal, int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<T2> optionalInput = inQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T2> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };

    DataCopyPad(optionalInput, optionalGlobal[endAxisOffset], copyParams, padParams);

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
__aicore__ inline void GeluQuant<T1, T2>::Compute(LocalTensor<float> &scaleLocalFp32,
    LocalTensor<float> &offsetLocalFp32, int32_t calCount)
{
    LocalTensor<float> geluRes = tempQueue_.AllocTensor<float>();
    ComputeGelu(geluRes, calCount);
    ComputeQuant(geluRes, scaleLocalFp32, offsetLocalFp32, calCount);
    tempQueue_.FreeTensor(geluRes);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::CopyOut(int64_t endAxisOffset, int32_t calCount, int64_t loopNum)
{
    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int8_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPad(yGm_[endAxisLen_ * loopNum + endAxisOffset], outLocal, copyParamsInt8);

    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ComputeGelu(LocalTensor<float> &geluRes, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();
    LocalTensor<float> castFp32 = castQueue_.AllocTensor<float>();

    if constexpr (IsSameType<T1, float>::value) {
        Muls(castFp32, xLocal, 1.0f, calCount);
    } else {
        Cast(castFp32, xLocal, RoundMode::CAST_NONE, calCount);
    }

    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(xLocal);

    if (approximate_ == APPROXIMATE_NONE) {
        LocalTensor<float> xSquared = inQueue_.AllocTensor<float>();
        ComputeGeluErf(castFp32, geluRes, xSquared, calCount);
        inQueue_.FreeTensor(xSquared);
    } else {
        ComputeGeluTanh(castFp32, geluRes, calCount);
    }

    castQueue_.FreeTensor(castFp32);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ComputeQuant(LocalTensor<float> &geluRes, LocalTensor<float> &scaleLocalFp32,
    LocalTensor<float> &offsetLocalFp32, int32_t calCount)
{
    if (inputScaleType_ == NORMAL_TENSOR) {
        Mul(geluRes, geluRes, scaleLocalFp32, calCount);
        PipeBarrier<PIPE_V>();
    }


    if (inputOffsetType_ == NORMAL_TENSOR) {
        Add(geluRes, geluRes, offsetLocalFp32, calCount);
        PipeBarrier<PIPE_V>();
    }


    LocalTensor<half> castFp16 = castQueue_.AllocTensor<half>();
    LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();

    Cast(castFp16, geluRes, RoundMode::CAST_ODD, calCount);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, castFp16, RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_V>();

    castQueue_.FreeTensor(castFp16);
    outQueue_.EnQue<int8_t>(outLocal);
}
} // namespace GeluQuantALL
#endif
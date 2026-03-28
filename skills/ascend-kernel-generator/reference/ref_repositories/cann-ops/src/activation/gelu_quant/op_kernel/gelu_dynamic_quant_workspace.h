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
 * \file gelu_dynamic_quant_workspace.h
 * \brief
 */

#ifndef GELU_DYNAMIC_QUANT_WORKSPACE_H
#define GELU_DYNAMIC_QUANT_WORKSPACE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base.h"

namespace GeluQuantALL {
using namespace AscendC;
// dynamic quant 基础泛化模板  ub内单个大尾轴
template <typename T1, typename T2> class GeluDynamicQuantWorkspace : public GeluQuantBase {
public:
    __aicore__ inline GeluDynamicQuantWorkspace(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale,
        GM_ADDR workspace, const GeluQuantTilingData &tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessScalarInput();
    __aicore__ inline void ProcessPerBlock(int64_t endAxisOffset, int32_t calCount, float &maxUpdateValue);
    __aicore__ inline void ProcessPerEndAxis();
    __aicore__ inline void ProcessOptionalScale(LocalTensor<float> &scaleLocalFp32, int64_t endAxisOffset,
        int32_t calCount);
    __aicore__ inline void ScaleResToWorkspace(LocalTensor<float> &geluRes, LocalTensor<float> &scaleLocalFp32,
        int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void WorkspaceToScaleRes(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void Compute(LocalTensor<float> &scaleLocalFp32, int64_t endAxisOffset, int32_t calCount,
        float &maxUpdateValue);
    __aicore__ inline void ComputeMaxValue(LocalTensor<float> &scaleRes, int32_t calCount, float &maxUpdateValue);
    __aicore__ inline void CopyOut(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void CopyOutScaleOut(float value);
    __aicore__ inline void ComputeGelu(LocalTensor<float> &geluRes, int32_t calCount);
    __aicore__ inline void ComputeDynamicQuant(float scale, int32_t calCount);

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workspaceQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleOutQueue_;

    TBuf<TPosition::VECCALC> castQueue_;
    TBuf<TPosition::VECCALC> geluQueue_;
    TBuf<TPosition::VECCALC> tempQueue_;

    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> outScaleGm_;
    GlobalTensor<float> workspaceGm_;

    /* variable */
    int64_t loopNum_;
};

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::Init(GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset,
    GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace, const GeluQuantTilingData &tilingData)
{
    // Init tiling data
    GeluQuantBase::ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset1 = blockIdx_ * normalCoreProcessNum_ * endAxisLen_;
    uint64_t intraCoreOffset2 = blockIdx_ * normalCoreProcessNum_;
    uint64_t intraCoreOffset3 = blockIdx_ * endAxisLen_;

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
    workspaceGm_.SetGlobalBuffer((__gm__ float *)workspace + intraCoreOffset3);

    pipe.InitBuffer(inQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(geluQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(castQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(tempQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(int8_t));
    pipe.InitBuffer(workspaceQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleOutQueue_, BUFFER_NUM, FP32_BLOCK_NUM * sizeof(float));
}

template <typename T1, typename T2> __aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessScalarInput()
{
    if (inputScaleType_ == SCALAR_TENSOR) {
        LocalTensor<float> scaleLocalFp32 = scaleQueue_.AllocTensor<float>();
        ProcessOptionalScale(scaleLocalFp32, 0, 1);
        PipeBarrier<PIPE_ALL>();
        inputScaleScalar_ = scaleLocalFp32.GetValue(0);
        scaleQueue_.FreeTensor(scaleLocalFp32);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessPerBlock(int64_t endAxisOffset, int32_t calCount,
    float &maxUpdateValue)
{
    LocalTensor<float> scaleLocalFp32 = scaleQueue_.AllocTensor<float>();
    if (inputScaleType_ == NORMAL_TENSOR) {
        ProcessOptionalScale(scaleLocalFp32, endAxisOffset, calCount);
    }
    CopyIn(endAxisOffset, calCount);
    Compute(scaleLocalFp32, endAxisOffset, calCount, maxUpdateValue);
    scaleQueue_.FreeTensor(scaleLocalFp32);
}

template <typename T1, typename T2> __aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessPerEndAxis()
{
    int64_t endAxisLoopNum = endAxisLen_ / coexistentNodeElementNum_;
    int64_t endAxisLoopTail = endAxisLen_ % coexistentNodeElementNum_;
    float maxUpdateValue = 0;
    for (int64_t endAxisLoopIndex = 0; endAxisLoopIndex < endAxisLoopNum; endAxisLoopIndex++) {
        ProcessPerBlock(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_, maxUpdateValue);
    }
    if (endAxisLoopTail != 0) {
        ProcessPerBlock(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail, maxUpdateValue);
    }
    
    float scale = MAX_INT8 / maxUpdateValue;
    CopyOutScaleOut(1 / scale);

    for (int64_t endAxisLoopIndex = 0; endAxisLoopIndex < endAxisLoopNum; endAxisLoopIndex++) {
        WorkspaceToScaleRes(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
        ComputeDynamicQuant(scale, coexistentNodeElementNum_);
        CopyOut(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
    }
    if (endAxisLoopTail != 0) {
        WorkspaceToScaleRes(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail);
        ComputeDynamicQuant(scale, endAxisLoopTail);
        CopyOut(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail);
    }
}

template <typename T1, typename T2> __aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    ProcessScalarInput();
    for (loopNum_ = 0; loopNum_ < curCoreProcessNum_; loopNum_++) {
        ProcessPerEndAxis();
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessOptionalScale(LocalTensor<float> &scaleLocalFp32,
    int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<T2> optionalScale = inQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T2> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };

    DataCopyPad(optionalScale, inputScaleGm_[endAxisOffset], copyParams, padParams);

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
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyIn(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<T1> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };
    DataCopyPad(xLocal, xGm_[endAxisLen_ * loopNum_ + endAxisOffset], copyParams, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::Compute(LocalTensor<float> &scaleLocalFp32,
    int64_t endAxisOffset, int32_t calCount, float &maxUpdateValue)
{
    LocalTensor<float> geluRes = geluQueue_.Get<float>();
    ComputeGelu(geluRes, calCount);
    ScaleResToWorkspace(geluRes, scaleLocalFp32, endAxisOffset, calCount);
    ComputeMaxValue(geluRes, calCount, maxUpdateValue);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ScaleResToWorkspace(LocalTensor<float> &geluRes,
    LocalTensor<float> &scaleLocalFp32, int64_t endAxisOffset, int32_t calCount)
{
    if (inputScaleType_ == SCALAR_TENSOR) {
        Muls(geluRes, geluRes, inputScaleScalar_, calCount);
    } else if (inputScaleType_ == NORMAL_TENSOR) {
        Mul(geluRes, geluRes, scaleLocalFp32, calCount);
    }

    PipeBarrier<PIPE_V>();// geluRes more consumption
    LocalTensor<float> workspaceLocal = workspaceQueue_.AllocTensor<float>();
    Muls(workspaceLocal, geluRes, 1.0f, calCount);
    PipeBarrier<PIPE_V>();
    workspaceQueue_.EnQue<float>(workspaceLocal);

    LocalTensor<float> workspaceLocalOut = workspaceQueue_.DeQue<float>();
    DataCopyExtParams copyParamsFloat{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPad(workspaceGm_[endAxisOffset], workspaceLocalOut, copyParamsFloat);
    workspaceQueue_.FreeTensor(workspaceLocalOut);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::WorkspaceToScaleRes(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<float> scaleResLocal = inQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<float> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<float>(0) };
    DataCopyPad(scaleResLocal, workspaceGm_[endAxisOffset], copyParams, padParams);
    inQueue_.EnQue(scaleResLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyOut(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int8_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPad(yGm_[endAxisLen_ * loopNum_ + endAxisOffset], outLocal, copyParamsInt8);
    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyOutScaleOut(float value)
{
    LocalTensor<float> scaleOutLocal = scaleOutQueue_.AllocTensor<float>();
    scaleOutLocal.SetValue(0, value);
    scaleOutQueue_.EnQue<float>(scaleOutLocal);
    LocalTensor<float> scaleOut = scaleOutQueue_.DeQue<float>();
    DataCopyExtParams copyParamsFloat{ static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPad(outScaleGm_[loopNum_], scaleOut, copyParamsFloat);
    scaleOutQueue_.FreeTensor(scaleOut);
}
template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ComputeGelu(LocalTensor<float> &geluRes, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();
    LocalTensor<float> castFp32 = castQueue_.Get<float>();

    if constexpr (IsSameType<T1, float>::value) {
        Muls(castFp32, xLocal, 1.0f, calCount);
    } else {
        Cast(castFp32, xLocal, RoundMode::CAST_NONE, calCount);
    }

    PipeBarrier<PIPE_V>();

    if (approximate_ == APPROXIMATE_NONE) {
        LocalTensor<float> xSquared = tempQueue_.Get<float>();
        ComputeGeluErf(castFp32, geluRes, xSquared, calCount);
    } else {
        ComputeGeluTanh(castFp32, geluRes, calCount);
    }
    inQueue_.FreeTensor(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ComputeDynamicQuant(float scale, int32_t calCount)
{
    LocalTensor<float> geluRes = inQueue_.DeQue<float>();
    Muls(geluRes, geluRes, scale, calCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> castFp16 = castQueue_.Get<half>();
    Cast(castFp16, geluRes, RoundMode::CAST_ODD, calCount);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(geluRes);

    LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
    Cast(outLocal, castFp16, RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_V>();

    outQueue_.EnQue<int8_t>(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ComputeMaxValue(LocalTensor<float> &scaleRes,
    int32_t calCount, float &maxUpdateValue)
{
    LocalTensor<float> tempRes = tempQueue_.Get<float>();
    Abs(tempRes, scaleRes, calCount);
    PipeBarrier<PIPE_V>();

    float maxValue = 0;
    ComputeReduceMax(tempRes, calCount, maxValue);
    maxUpdateValue = maxUpdateValue > maxValue ? maxUpdateValue : maxValue;
}
} // namespace GeluQuantALL
#endif
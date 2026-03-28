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
 * \file foreach_norm.cpp
 * \brief
 */

#include "foreach_reduce_unary.h"

using namespace Common::OpKernel;
using namespace AscendC;

/**
 * modelCode:
 * 0 p=else... the default operator(not used now)
 * 1 p=0 Calculate the count of not-zero element in each tensor. (not used now)
 * 2 p=1 AbsAndNotNeedPower NotNeedSqrt(now is default as we now only consider p=1 || p=2)
 * 3 p=2 MulSelf(we don't need abs this time) Sqrt(not Power(self,1/p))
 * 4 p=+inf Calculate the max Abs(element) in each tensor. (not used now)
 * 5 p=-inf Calculate the min Abs(element) in each tensor. (not used now)
 */

constexpr uint8_t ZERO_SCALAR_NORM_MODEL_CODE = 0;
constexpr uint8_t ONE_SCALAR_NORM_MODEL_CODE = 1;
constexpr uint8_t TWO_SCALAR_NORM_MODEL_CODE = 2;
constexpr uint8_t POSITIVE_INF_SCALAR_NORM_MODEL_CODE = 3;
constexpr uint8_t NEGATIVE_INF_SCALAR_NORM_MODEL_CODE = 4;
constexpr uint8_t DEFAULT_SCALAR_NORM_MODEL_CODE = 5;
constexpr uint8_t NORM_MODEL_CODE = 2;

// this is actually ord=1
template <typename P, uint8_t modelCode>
class NormAdapter: public ReduceAdapter<P, NormAdapter<P, modelCode>> {
public:
    using Reduce =  ReduceAdapter<P, NormAdapter<P, modelCode>>;

    __aicore__ inline NormAdapter() : ReduceAdapter<P, NormAdapter<P, modelCode>>(*this) {};

    __aicore__ inline void BeforeReduceOp(const LocalTensor<P> &dstLocal, const LocalTensor<P> &srcLocal, int64_t dataCount) {
        pipe_barrier(PIPE_V);
        Abs(dstLocal, srcLocal, dataCount);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void AfterReduceOp(const LocalTensor<P> &dstLocal, const LocalTensor<P> &srcLocal, int64_t dataCount) {
        uint64_t mask = 64;
        uint32_t repeatTimes = 4;
        CopyRepeatParams copyRepeatParams{1, 1, 8, 8};
        Copy(dstLocal, srcLocal, mask, repeatTimes, copyRepeatParams);
    }

    __aicore__ inline void ReduceOp(const LocalTensor<P>& dstLocal, const LocalTensor<P>& srcLocal, const LocalTensor<P>& workLocal, int32_t count) {
        ReduceSum<P>(dstLocal, srcLocal, workLocal, count);
    }
};

// this is actually ord=2
template <typename P>
class NormAdapter<P, NORM_MODEL_CODE> : public ReduceAdapter<P, NormAdapter<P, NORM_MODEL_CODE>> {
public:
    using Reduce =  ReduceAdapter<P, NormAdapter<P, NORM_MODEL_CODE>>;

    __aicore__ inline NormAdapter() : ReduceAdapter<P, NormAdapter<P, NORM_MODEL_CODE>>(*this) {};

    __aicore__ inline void BeforeReduceOp(const LocalTensor<P> &dstLocal, const LocalTensor<P> &srcLocal, int64_t dataCount) {
        pipe_barrier(PIPE_V);
        Mul(dstLocal, srcLocal, srcLocal, dataCount);
        pipe_barrier(PIPE_V); 
    }

    __aicore__ inline void AfterReduceOp(const LocalTensor<P> &dstLocal, const LocalTensor<P> &srcLocal, int64_t dataCount) {
        pipe_barrier(PIPE_V);
        Sqrt(dstLocal, srcLocal, dataCount);
        pipe_barrier(PIPE_V); 
    }

    __aicore__ inline void ReduceOp(const LocalTensor<P>& dstLocal, const LocalTensor<P>& srcLocal, const LocalTensor<P>& workLocal, int32_t count) {
        ReduceSum<P>(dstLocal, srcLocal, workLocal, count);
    }
};

extern "C" __global__ __aicore__ void foreach_norm(GM_ADDR inputs, GM_ADDR scalar, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);

    uint8_t modelCode = TWO_SCALAR_NORM_MODEL_CODE;
    GlobalTensor<DTYPE_SCALAR> inScalarGM;
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
    float scalarVal = inScalarGM.GetValue(0);

    if (static_cast<int>(scalarVal) == 1)  {
        modelCode = ONE_SCALAR_NORM_MODEL_CODE;
    }

    if (TILING_KEY_IS(1)) {
        if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
            NormAdapter<float, ONE_SCALAR_NORM_MODEL_CODE> normAdapter;
            ForeachReduceUnary<half, float, NormAdapter<float, ONE_SCALAR_NORM_MODEL_CODE>> op(normAdapter);
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        } else {
            NormAdapter<float, TWO_SCALAR_NORM_MODEL_CODE> normAdapter;
            ForeachReduceUnary<half, float, NormAdapter<float, TWO_SCALAR_NORM_MODEL_CODE>> op(normAdapter);
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
            NormAdapter<float, ONE_SCALAR_NORM_MODEL_CODE> normAdapter;
            ForeachReduceUnary<float, float, NormAdapter<float, ONE_SCALAR_NORM_MODEL_CODE>> op(normAdapter);
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        } else {
            NormAdapter<float, TWO_SCALAR_NORM_MODEL_CODE> normAdapter;
            ForeachReduceUnary<float, float, NormAdapter<float, TWO_SCALAR_NORM_MODEL_CODE>> op(normAdapter);
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(4)) {
        if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
            NormAdapter<float, ONE_SCALAR_NORM_MODEL_CODE> normAdapter;
            ForeachReduceUnary<bfloat16_t, float, NormAdapter<float, ONE_SCALAR_NORM_MODEL_CODE>> op(normAdapter);
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        } else {
            NormAdapter<float, TWO_SCALAR_NORM_MODEL_CODE> normAdapter;
            ForeachReduceUnary<bfloat16_t, float, NormAdapter<float, TWO_SCALAR_NORM_MODEL_CODE>> op(normAdapter);
            op.Init(inputs, output, userWS, &tilingData);
            op.Process();
        }
    }
}

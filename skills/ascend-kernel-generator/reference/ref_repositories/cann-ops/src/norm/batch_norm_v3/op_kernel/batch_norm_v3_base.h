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
 * \file batch_norm_v3_base.h
 * \brief
 */

#ifndef BATCH_NORM_V3_BASE_H
#define BATCH_NORM_V3_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace BatchNormV3Ops {
using namespace AscendC;

template <typename T1, typename T2>
class BatchNormV3Base {
public:
    __aicore__ inline BatchNormV3Base()
    {}

protected:
    /* global memory address */
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> weightGm;
    GlobalTensor<T2> biasGm;
    GlobalTensor<float> runningMeanGm;
    GlobalTensor<float> runningVarGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<float> saveMeanGm;
    GlobalTensor<float> saveVarGm;
    GlobalTensor<float> runningMeanOutGm;
    GlobalTensor<float> runningVarOutGm;

    /* variable */
    float epsilon = 1e-5;
    float momentum = 0.1;
    float momentumReverse;
    float batchVarScale;

    /* ascendc variable */
    TPipe *pipe_ = nullptr;
    uint32_t blockIdx = GetBlockIdx();
    uint32_t useCoreNum = GetBlockNum();
    // 公共函数声明
};
// 公共函数实现

}  // namespace BatchNormV3Ops
#endif  // BATCH_NORM_V3_BASE_H
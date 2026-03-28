/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_LAYER_NORM_V3_H_
#define OP_API_INC_LEVEL0_LAYER_NORM_V3_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t LAYER_NORM_V3_OUT_NUM = 3;

const std::array<aclTensor *, LAYER_NORM_V3_OUT_NUM> LayerNormV3(const aclTensor *input, const aclTensor *weight,
    const aclTensor *bias, int64_t beginAxis, double eps, aclOpExecutor *executor);

const std::array<aclTensor *, LAYER_NORM_V3_OUT_NUM> LayerNormV3WithImplMode(const aclTensor *input,
    const aclTensor *weight, const aclTensor *bias, int64_t beginAxis, double eps, int32_t implMode,
    aclOpExecutor *executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_LAYER_NORM_V3_H_

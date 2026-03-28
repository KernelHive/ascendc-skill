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
 * \file grid_sampler2d_grad.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_GRID_SAMPLER2D_GRAD_H_
#define OP_API_INC_LEVEL0_GRID_SAMPLER2D_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor *, aclTensor *> GridSampler2DGrad(const aclTensor *gradOutput, const aclTensor *input,
    const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclOpExecutor *executor);
const std::tuple<aclTensor *, aclTensor *> GridSamplerGrad(const aclTensor *gradOutput, const aclTensor *input,
    const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclOpExecutor *executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_GRID_SAMPLER2D_GRAD_H_
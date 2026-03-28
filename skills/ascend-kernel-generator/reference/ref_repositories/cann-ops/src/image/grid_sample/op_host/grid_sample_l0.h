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
 * \file grid_sample_l0.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_GRID_SAMPLE_H_
#define OP_API_INC_LEVEL0_GRID_SAMPLE_H_

#include "opdev/op_executor.h"
#define OP_ATTR_NAMES ::op::FVector<std::string>

namespace l0op {
const aclTensor *GridSample(const aclTensor *input, const aclTensor *grid, int64_t interpolationMode,
    int64_t paddingMode, bool alignCorners, bool channelLast, int64_t schedulerMode, aclOpExecutor *executor);
const aclTensor *GridSample3D(const aclTensor *input, const aclTensor *grid, int64_t interpolationMode,
    int64_t paddingMode, bool alignCorners, bool channelLast, aclOpExecutor *executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_GRID_SAMPLE_H_
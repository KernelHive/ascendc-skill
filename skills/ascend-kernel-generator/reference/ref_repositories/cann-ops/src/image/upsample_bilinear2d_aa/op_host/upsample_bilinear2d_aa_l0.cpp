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
 * \file upsample_bilinear2d_aa_l0.cpp
 * \brief
 */
#include "upsample_bilinear2d_aa_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBilinear2dAA);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

const aclTensor *UpsampleBilinear2dAA(const aclTensor *input, const aclIntArray *outputSize, const aclTensor *output,
    bool alignCorners, float scales_h, float scales_w, aclOpExecutor *executor)
{
    L0_DFX(UpsampleBilinear2dAA, input, outputSize, output, alignCorners, scales_h, scales_w);

    auto out = executor->AllocTensor(output->GetViewShape(), op::DataType::DT_FLOAT, output->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        UpsampleBilinear2dAA, OP_INPUT(input), OP_OUTPUT(out), OP_ATTR(outputSize, alignCorners, scales_h, scales_w));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UpsampleBilinear2dAAAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);

    return out;
}
}  // namespace l0op
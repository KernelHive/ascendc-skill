/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_grad_v3_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormGradV3);

const std::array<aclTensor *, GRAD_V3_OUT_NUM> LayerNormGradV3(const aclTensor *gradOut, const aclTensor *input,
    const aclTensor *rstd, const aclTensor *mean, const aclTensor *weight, aclOpExecutor *executor)
{
    L0_DFX(LayerNormGradV3, gradOut, input, rstd, mean, weight);
    auto gradInputOut = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), Format::FORMAT_ND);
    auto gradWeight = executor->AllocTensor(weight->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
    auto gradBias = executor->AllocTensor(weight->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);

    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormGradV3, OP_INPUT(gradOut, input, rstd, mean, weight), OP_OUTPUT(gradInputOut, gradWeight, gradBias));

    return {gradInputOut, gradWeight, gradBias};
}

}  // namespace l0op

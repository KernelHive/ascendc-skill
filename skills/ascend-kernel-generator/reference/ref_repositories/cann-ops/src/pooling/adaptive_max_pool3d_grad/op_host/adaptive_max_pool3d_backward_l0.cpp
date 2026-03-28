/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptive_max_pool3d_backward_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AdaptiveMaxPool3DGrad);

const inline aclTensor* AdaptiveMaxPool3DGradAiCore(const aclTensor* gradOutput, const aclTensor* self,
                                                    const aclTensor* indices, aclTensor* gradInput,
                                                    aclOpExecutor* executor) {
    L0_DFX(AdaptiveMaxPool3DGradAiCore, self, gradOutput, indices, gradInput);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveMaxPool3DGrad, OP_INPUT(self, gradOutput, indices), OP_OUTPUT(gradInput));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveMaxPool3DGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return gradInput;
}

const aclTensor* AdaptiveMaxPool3DGrad(const aclTensor* gradOutput, const aclTensor* self,
                                       const aclTensor* indices, aclOpExecutor* executor) {
    auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_NCDHW);
    if (gradInput == nullptr){
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradInput is nullptr.");
        return nullptr;
    }
    return AdaptiveMaxPool3DGradAiCore(gradOutput, self, indices, gradInput, executor);
}
} // l0op
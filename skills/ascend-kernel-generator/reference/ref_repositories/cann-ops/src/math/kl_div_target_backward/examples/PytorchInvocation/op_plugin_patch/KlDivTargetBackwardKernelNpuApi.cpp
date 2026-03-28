/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file KlDivTargetBackwardKernelNpuApi.cpp
 */
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_kl_div_target_backward(const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &target, int64_t reduction, bool log_target)
{
    at::Tensor result = npu_preparation::apply_tensor_without_format(target); // Create output memory

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnKlDivTargetBackward, grad_output, self, target, reduction, log_target, result);
    return result;
}
} // namespace op_api

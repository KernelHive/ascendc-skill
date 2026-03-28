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
 * \file embedding_Dense_grad.cpp
 * \brief
 */

#include "group_norm_grad_l0.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(GroupNormGrad);

const std::tuple<aclTensor *, aclTensor *, aclTensor *> GroupNormGrad(const aclTensor *dy, const aclTensor *mean,
    const aclTensor *rstd, const aclTensor *x, const aclTensor *gamma, uint64_t numGroups,
    const std::string &dataFormat, bool dxIsRequire, bool dgammaIsRequire, bool dbetaIsRequire, aclTensor *dxOutput,
    aclTensor *dgammaOutput, aclTensor *dbetaOutput, aclOpExecutor *executor)
{
    L0_DFX(GroupNormGrad,
        dy,
        mean,
        rstd,
        x,
        gamma,
        numGroups,
        dataFormat,
        dxIsRequire,
        dgammaIsRequire,
        dbetaIsRequire,
        dxOutput,
        dgammaOutput,
        dbetaOutput);
    // 创建输出Tensor
    // 调用device的GroupNormGrad算子
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupNormGrad,
        OP_INPUT(dy, mean, rstd, x, gamma),
        OP_OUTPUT(dxOutput, dgammaOutput, dbetaOutput),
        OP_ATTR(numGroups, dataFormat, dxIsRequire, dgammaIsRequire, dbetaIsRequire));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GroupNormGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor *, aclTensor *, aclTensor *>(nullptr, nullptr, nullptr);
    }
    return std::tie(dxOutput, dgammaOutput, dbetaOutput);
}
}  // namespace l0op
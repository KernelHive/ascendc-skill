/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "instance_norm_v3_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

OP_TYPE_REGISTER(InstanceNormV3);

std::tuple<aclTensor *, aclTensor *, aclTensor *> InstanceNormV3(const aclTensor *x, const aclTensor *gamma,
    const aclTensor *beta, const char *dataFormat, double eps, aclOpExecutor *executor)
{
    OP_LOGD("Enter InstanceNorm Level 0");
    L0_DFX(InstanceNormV3, x, gamma, beta, dataFormat, eps);

    op::Shape reduceShape = x->GetStorageShape();
    reduceShape.SetDim(DIM_2, 1);
    reduceShape.SetDim(DIM_3, 1);

    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), Format::FORMAT_ND);
    auto mean = executor->AllocTensor(reduceShape, x->GetDataType(), Format::FORMAT_ND);
    auto variance = executor->AllocTensor(reduceShape, x->GetDataType(), Format::FORMAT_ND);

    ADD_TO_LAUNCHER_LIST_AICORE(InstanceNormV3,
        OP_INPUT(x, gamma, beta),
        OP_OUTPUT(y, mean, variance),
        OP_ATTR(dataFormat, static_cast<float>(eps)));

    OP_LOGD("Finish InstanceNorm level 0");
    return std::tuple<aclTensor *, aclTensor *, aclTensor *>(y, mean, variance);
}
}  // namespace l0op

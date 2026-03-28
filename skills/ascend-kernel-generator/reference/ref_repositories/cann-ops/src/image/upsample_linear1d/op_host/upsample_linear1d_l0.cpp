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
 * \file upsample_linear1d_l0.cpp
 * \brief
 */
#include "upsample_linear1d_l0.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/common_types.h"

#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleLinear1d);

static const string LINEAR_MODE = "linear";
static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DIM_THREE = 3;

const aclTensor *UpsampleLinear1dNcdhw(const aclTensor *x, const aclIntArray *outputSize, const bool alignCorners,
    const aclTensor *y, const double scale, aclOpExecutor *executor)
{
    L0_DFX(UpsampleLinear1dNcdhw, x, outputSize, alignCorners, scale);

    auto dataType = x->GetDataType();
    auto outputShape = y->GetViewShape();

    float realScale = 0.0f;
    if (scale > 0) {
        realScale = static_cast<float>(1.0 / scale);
    }

    auto size1 = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT64));
    auto castSize = l0op::Cast(size1, op::DataType::DT_INT32, executor);

    // fp16和bf16转fp32计算
    if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
        x = l0op::Cast(x, op::DataType::DT_FLOAT, executor);
    }
    const aclTensor *out = executor->AllocTensor(y->GetViewShape(), op::DataType::DT_FLOAT, x->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);
    // AICORE
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        UpsampleLinear1d, OP_INPUT(x, castSize), OP_OUTPUT(out), OP_ATTR(alignCorners, realScale));
    if (op::DataType::DT_FLOAT16 == dataType) {
        out = l0op::Cast(out, op::DataType::DT_FLOAT16, executor);
    } else if (op::DataType::DT_BF16 == dataType) {
        out = l0op::Cast(out, op::DataType::DT_BF16, executor);
    }
    return out;
}
}  // namespace l0op

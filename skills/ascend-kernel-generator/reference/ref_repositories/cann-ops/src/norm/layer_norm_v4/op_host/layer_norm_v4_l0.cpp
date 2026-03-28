/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_v4_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormV4);

const std::array<aclTensor *, LAYER_NORM_V4_OUT_NUM> LayerNormV4(const aclTensor *input,
    const aclIntArray *normalizedShape, const aclTensor *weight, const aclTensor *bias, double eps,
    aclOpExecutor *executor)
{
    L0_DFX(LayerNormV4, input, normalizedShape, weight, bias, eps);

    // 根据array构造算子所需tensor输入
    auto normTensor = executor->ConvertToTensor(normalizedShape, DataType::DT_INT32);

    // 根据输入与normalizedShape关系构造输出shape
    Shape meanOutShape = input->GetViewShape();
    for (size_t index = meanOutShape.GetDimNum() - normalizedShape->Size(); index < meanOutShape.GetDimNum(); index++) {
        meanOutShape[index] = 1;
    }
    auto output = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), Format::FORMAT_ND);
    auto meanOut = executor->AllocTensor(meanOutShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    auto rstdOut = executor->AllocTensor(meanOutShape, DataType::DT_FLOAT, Format::FORMAT_ND);

    ADD_TO_LAUNCHER_LIST_AICORE(LayerNormV4,
        OP_INPUT(input, normTensor, weight, bias),
        OP_OUTPUT(output, meanOut, rstdOut),
        OP_ATTR(static_cast<float>(eps)));

    return {output, meanOut, rstdOut};
}

}  // namespace l0op

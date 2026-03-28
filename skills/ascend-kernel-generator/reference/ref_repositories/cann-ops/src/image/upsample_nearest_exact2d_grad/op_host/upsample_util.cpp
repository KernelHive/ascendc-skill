/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "upsample_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

bool CheckInputElement(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize)
{
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();
    FVector<int64_t> fullOutputSize = {batch, channels, outH, outW};

    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputH = (*inputSize)[DIM_ONE];
        inputW = (*inputSize)[DIM_TWO];
        channels = (*inputSize)[DIM_THREE];
        fullOutputSize[DIM_ONE] = outH;
        fullOutputSize[DIM_TWO] = outW;
        fullOutputSize[DIM_THREE] = channels;
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (H: %ld,"
            " W: %ld) output (H: %ld, W: %ld)",
            inputH,
            inputW,
            outH,
            outW),
        return false);

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected grad_output to have the same shape as output;"
                " output.size(%zu) = %ld but got grad_output.size(%zu) = %ld",
                i,
                fullOutputSize[i],
                i,
                gradOutShape.GetDim(i));
            return false;
        }
    }
    return true;
}

#ifdef __cplusplus
}
#endif

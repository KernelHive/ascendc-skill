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
 * \file resize_bilinear_v2_grad_l0.cpp
 * \brief
 */
#include "resize_bilinear_v2_grad_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeBilinearV2Grad);

static aclTensor *ResizeBilinearV2Grad5HdAICORE(const aclTensor *gradOut, const aclTensor *image, bool alignCorners,
    bool halfPixelCenters, aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(ResizeBilinearV2Grad5HdAICORE, gradOut, image, alignCorners, halfPixelCenters, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeBilinearV2Grad, OP_INPUT(gradOut, image), OP_OUTPUT(out), OP_ATTR(alignCorners, halfPixelCenters));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeBilinearV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
const aclTensor *ResizeBilinearV2Grad5Hd(
    const aclTensor *gradOut, const aclTensor *image, bool alignCorners, bool halfPixelCenters, aclOpExecutor *executor)
{
    auto out = executor->AllocTensor(image->GetStorageShape(),
        image->GetOriginalShape(),
        image->GetDataType(),
        image->GetStorageFormat(),
        image->GetOriginalFormat());
    CHECK_RET(out != nullptr, nullptr);

    return ResizeBilinearV2Grad5HdAICORE(gradOut, image, alignCorners, halfPixelCenters, out, executor);
}
}  // namespace l0op

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_backward.cpp
 * \brief
 */

#include "adaptive_avg_pool3d_backward.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;

namespace l0op{

OP_TYPE_REGISTER(AdaptiveAvgPool3dGrad);
static constexpr size_t DIM_D = 3;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 1;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;

static const aclTensor* AdaptiveAvgPool3dGradAiCore(const aclTensor* gradOutput, const aclTensor* self,
                                                    aclTensor* out, aclOpExecutor* executor) {
    L0_DFX(AdaptiveAvgPool3dGradAiCore, gradOutput, self, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveAvgPool3dGrad, 
                                           OP_INPUT(gradOutput, self), 
                                           OP_OUTPUT(out));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool3dGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;                      
}

const aclTensor* AdaptiveAvgPool3dGrad(const aclTensor* gradOutput, const aclTensor* self,
                                       aclOpExecutor* executor) {
    L0_DFX(AdaptiveAvgPool3dGrad, gradOutput, self);
    op::Shape outShape = gradOutput->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    size_t dimNum = selfShape.GetDimNum();
    outShape.SetDim( DIM_0, selfShape.GetDim(dimNum-DIM_D));
    outShape.SetDim( DIM_1, selfShape.GetDim(dimNum-DIM_H));
    outShape.SetDim( DIM_2, selfShape.GetDim(dimNum-DIM_W));
    auto out = executor->AllocTensor(outShape, gradOutput->GetDataType(), gradOutput->GetStorageFormat());
    if (out == nullptr){
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr.");
        return nullptr;
    }
    return AdaptiveAvgPool3dGradAiCore(gradOutput, self, out, executor);
}
}
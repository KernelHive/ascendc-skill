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
 * \file avgpool3d_backward.cpp
 * \brief
 */

#include "avgpool3d_backward.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

namespace l0op {
constexpr int64_t D_DIM = 1;
constexpr int64_t H_DIM = 2;
constexpr int64_t W_DIM = 3;
constexpr int64_t D_DIM_OFFSET = 3;
constexpr int64_t H_DIM_OFFSET = 2;
constexpr int64_t W_DIM_OFFSET = 1;
constexpr int64_t OUTD_DIM = 2;
constexpr int64_t OUTH_DIM = 3;
constexpr int64_t OUTW_DIM = 4;
constexpr int64_t EXPECT_GRAD_SHAPE = 5;

OP_TYPE_REGISTER(AvgPool3DGrad);

const aclTensor *AvgPool3DGrad(const aclTensor* self, const aclTensor* shapeOrigInput,
                               const aclTensor* gradOutput, const aclIntArray* ksize,
                               const aclIntArray* strides, const aclIntArray* pads,
                               bool ceilMode, bool countIncludePad,
                               int divisorOverride, const std::string& dataFormat,
                               aclOpExecutor *executor) {
  L0_DFX(AvgPool3DGrad, gradOutput, shapeOrigInput, ksize, strides, pads, ceilMode,
         countIncludePad, divisorOverride, dataFormat);

  auto gradShape = gradOutput->GetViewShape();
  auto gradSize = gradShape.GetDimNum();
  auto selfShape = self->GetViewShape();
  auto selfSize = selfShape.GetDimNum();

  if (gradSize != EXPECT_GRAD_SHAPE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad size [%li] is not equal to 5", gradSize);
    return nullptr;
  }

  op::Shape outShape;
  outShape.SetDimNum(EXPECT_GRAD_SHAPE);
  if (dataFormat == "NDHWC") {
    auto ncSize = gradShape.GetDim(gradSize - 1);
    auto depth = selfShape.GetDim(selfSize - D_DIM_OFFSET);
    auto height = selfShape.GetDim(selfSize - H_DIM_OFFSET);
    auto weight = selfShape.GetDim(selfSize - W_DIM_OFFSET);
    outShape.SetDim(0, 1);
    outShape.SetDim(1, depth);
    outShape.SetDim(OUTD_DIM, height);
    outShape.SetDim(OUTH_DIM, weight);
    outShape.SetDim(OUTW_DIM, ncSize);
  } else if (dataFormat == "NCDHW") {
    outShape = selfShape;
  }

  auto out = executor->AllocTensor(outShape, gradOutput->GetDataType(), gradOutput->GetStorageFormat());

//   调用device的AvgPool3DGrad算子
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPool3DGrad,
                              OP_INPUT(shapeOrigInput, gradOutput),
                              OP_OUTPUT(out),
                              OP_ATTR(ksize, strides, pads, ceilMode,
                                      countIncludePad, divisorOverride, dataFormat));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvgPool3DGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
} // l0op

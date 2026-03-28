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
 * \file upsample_trilinear_3d_l0.cpp
 * \brief
 */
#include "upsample_trilinear_3d_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleTrilinear3d);
OP_TYPE_REGISTER(ResizeUpsampleTrilinear);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr float MAX_SUPPORT_SCALE = 50.0;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_310P_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

bool CheckScales(float scaleW, float scaleH, float scaleD) {
  return (scaleW <= MAX_SUPPORT_SCALE && scaleH <= MAX_SUPPORT_SCALE && scaleD <= MAX_SUPPORT_SCALE);
}

const aclTensor* UpsampleTrilinear3dNcdhw(const aclTensor* self, const aclIntArray* outputSize, bool alignCorners,
                                          const aclFloatArray* scales, const aclFloatArray* castScales, float checkScaleW, float checkScaleH, float checkScaleD, 
                                          aclOpExecutor* executor) {
  L0_DFX(UpsampleTrilinear3dNcdhw, self, outputSize, alignCorners, scales);

  auto selfShape = op::ToShapeVector(self->GetViewShape());
  float scales_d = 0.0;
  float scales_h = 0.0;
  float scales_w = 0.0;

  auto socVer = GetCurrentPlatformInfo().GetSocVersion();
  auto dimZero = DIM_ZERO;
  auto dimOne = DIM_ONE;
  auto dimTwo = DIM_TWO;
  auto dimThree = DIM_THREE;
  auto dimFour = DIM_FOUR;
 
  if ((socVer == SocVersion::ASCEND310P) && CheckType(self->GetDataType(), AICORE_310P_SUPPORT_LIST) && CheckScales(checkScaleW, checkScaleH, checkScaleD)) {
    dimTwo = DIM_ZERO;
    dimThree = DIM_ONE;
    dimFour = DIM_TWO;
  }
  
  if (outputSize->Size() == DIM_THREE) {
    selfShape[dimTwo] = (*outputSize)[DIM_ZERO];
    selfShape[dimThree] = (*outputSize)[DIM_ONE];
    selfShape[dimFour] = (*outputSize)[DIM_TWO];
  } else {
    selfShape[dimTwo] *= (*scales)[DIM_ZERO];
    selfShape[dimThree] *= (*scales)[DIM_ONE];
    selfShape[dimFour] *= (*scales)[DIM_TWO];
    scales_d = (*castScales)[DIM_ZERO];
    scales_h = (*castScales)[DIM_ONE];
    scales_w = (*castScales)[DIM_TWO];
  }
  op::Shape outShape;
  op::ToShape(selfShape.data(), selfShape.size(), outShape);
  auto dataType = self->GetDataType();
  if ((socVer == SocVersion::ASCEND910B) &&
      CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) && CheckScales(checkScaleW, checkScaleH, checkScaleD)) {
    if (op::DataType::DT_FLOAT16 == dataType || op::DataType::DT_BF16 == dataType) {
      self = l0op::Cast(self, op::DataType::DT_FLOAT, executor);
    }
    const aclTensor* out = executor->AllocTensor(outShape, op::DataType::DT_FLOAT, self->GetStorageFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeUpsampleTrilinear, OP_INPUT(self), OP_OUTPUT(out),
                                           OP_ATTR(outputSize, alignCorners, scales_d, scales_h, scales_w));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeUpsampleTrilinearAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
      return nullptr);
    // cast back to fp16
    if (op::DataType::DT_FLOAT16 == dataType) {
      out = l0op::Cast(out, op::DataType::DT_FLOAT16, executor);
    } else if (op::DataType::DT_BF16 == dataType) {
      // cast back to bf16
      out = l0op::Cast(out, op::DataType::DT_BF16, executor);
    }
    return out;
  } else if ((socVer == SocVersion::ASCEND310P) && CheckType(self->GetDataType(), AICORE_310P_SUPPORT_LIST) && CheckScales(checkScaleW, checkScaleH, checkScaleD)) {
    if (op::DataType::DT_FLOAT16 == dataType) {
      self = l0op::Cast(self, op::DataType::DT_FLOAT, executor);
    }
    const aclTensor* out = executor->AllocTensor(outShape, op::DataType::DT_FLOAT, self->GetStorageFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeUpsampleTrilinear, OP_INPUT(self), OP_OUTPUT(out),
                                           OP_ATTR(outputSize, alignCorners, scales_d, scales_h, scales_w));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeUpsampleTrilinear310PAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
      return nullptr);
    if (op::DataType::DT_FLOAT16 == dataType) {
      out = l0op::Cast(out, op::DataType::DT_FLOAT16, executor);
    }
    return out;
  }
}
}  // namespace l0op
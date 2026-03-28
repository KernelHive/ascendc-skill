/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_upsample_trilinear_3d.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/reshape.h"
#include "upsample_trilinear_3d_l0.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_DOUBLE, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static constexpr size_t EXPECT_SIZE = 3;
static constexpr float MAX_SUPPORT_SCALE = 50.0;

static const int64_t UPSAMPLE_DIM_LIMIT = 5;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(outputSize, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
  if (curSoc == op::SocVersion::ASCEND310P || curSoc == op::SocVersion::ASCEND310B) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
  } else {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  }
  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
  return true;
}

static bool CheckIsPlatform310p(const aclTensor *self, const aclTensor *out) {
  if (GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P ||
      GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310B) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
  } else {
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* outputSize) {
  size_t outputSizeNum = outputSize->Size();
  OP_CHECK_WRONG_DIMENSION(self, UPSAMPLE_DIM_LIMIT, return false);
  OP_CHECK(outputSizeNum == EXPECT_SIZE,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 3, but got size %zu", outputSizeNum),
           return false);
  const op::Format inputFormat = self->GetStorageFormat();
  OP_CHECK(inputFormat == op::Format::FORMAT_ND || inputFormat == op::Format::FORMAT_NCDHW || 
           inputFormat == op::Format::FORMAT_NDHWC,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input storage format support NCDHW or NDHWC, but got %s.",
                  op::ToString(inputFormat).GetString()),
           return false);
  return true;
}

static bool CheckInputElement(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out) {
  auto selfShape = self->GetViewShape();
  int64_t inputN = selfShape.GetDim(DIM_ZERO);
  int64_t outC = 0;
  int64_t inputD = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  int64_t outD = (*outputSize)[DIM_ZERO];
  int64_t outH = (*outputSize)[DIM_ONE];
  int64_t outW = (*outputSize)[DIM_TWO];
  if (self->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
    outC = selfShape.GetDim(DIM_FOUR);
    inputD = selfShape.GetDim(DIM_ONE);
    inputH = selfShape.GetDim(DIM_TWO);
    inputW = selfShape.GetDim(DIM_THREE);
  } else {
    outC = selfShape.GetDim(DIM_ONE);
    inputD = selfShape.GetDim(DIM_TWO);
    inputH = selfShape.GetDim(DIM_THREE);
    inputW = selfShape.GetDim(DIM_FOUR);
  }

  OP_CHECK(inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Input and output sizes should greater than 0, bug got input (D: %ld,"
                   " H: %ld, W: %ld) output (D: %ld, H: %ld, W: %ld)",
                   inputD, inputH, inputW, outD, outH, outW),
           return false);

  OP_CHECK(outC > 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Non-empty 5D data tensor expected but got a tensor with sizes %s.",
                   op::ToString(self->GetViewShape()).GetString()),
           return false);
  op::Shape expectShape = op::Shape{inputN, outC, outD, outH, outW};
  if (out->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
    expectShape = op::Shape{inputN, outD, outH, outW, outC};
  }
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
  return true;
}

static float ComputeScales(float scale, uint32_t input_size, uint32_t output_size) {
  auto zero = static_cast<float>(0.);
  if (scale > zero) {
    return scale;
  } else {
    if (output_size != 0) {
      return (static_cast<float>(input_size) / output_size);
    } else {
      static_cast<float>(0);
    }
  }
}

static float AsComputeScale(bool align_corners, int64_t input_size, int64_t output_size, float scale) {
  if (output_size == input_size) {
    return static_cast<float>(1);
  }
  if (align_corners) {
    if ((output_size > 1) && (output_size != 1)) {
      return static_cast<float>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<float>(0);
    }
  } else {
    return ComputeScales(scale, input_size, output_size);
  }
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否支持
  CHECK_RET(CheckShape(self, outputSize), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入元素是否合法
  CHECK_RET(CheckInputElement(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

const aclTensor* upsampleTrilinear3dCompute(const aclTensor* selfContiguous, const aclIntArray* outputSize,
                                            bool alignCorners, const aclFloatArray* scales, 
                                            const aclFloatArray* castScales, float scaleW, float scaleH, float scaleD, aclOpExecutor* executor) {                                           
  if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
    const int64_t permuteNCDHWList[] = {0, 4, 1, 2, 3};
    auto permuteNCDHWArray = executor->AllocIntArray(permuteNCDHWList, UPSAMPLE_DIM_LIMIT);
    CHECK_RET(permuteNCDHWArray != nullptr, nullptr);

    auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCDHWArray, executor);
    CHECK_RET(selfTranspose != nullptr, nullptr);

    auto selfUpsampleTrilinear =
        l0op::UpsampleTrilinear3dNcdhw(selfTranspose, outputSize, alignCorners, scales, castScales, scaleW, scaleH, scaleD, executor);
    CHECK_RET(selfUpsampleTrilinear != nullptr, nullptr);
    const int64_t permuteNDHWCList[] = {0, 2, 3, 4, 1};
    auto permuteNDHWCArray = executor->AllocIntArray(permuteNDHWCList, UPSAMPLE_DIM_LIMIT);
    CHECK_RET(permuteNDHWCArray != nullptr, nullptr);

    return l0op::Transpose(selfUpsampleTrilinear, permuteNDHWCArray, executor);
  } else {
    return l0op::UpsampleTrilinear3dNcdhw(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW, scaleH, scaleD, executor);
  }
}

static bool CheckScales(float scaleW, float scaleH, float scaleD) {
  return (scaleW <= MAX_SUPPORT_SCALE && scaleH <= MAX_SUPPORT_SCALE && scaleD <= MAX_SUPPORT_SCALE);
}

aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                     bool alignCorners, double scalesD, double scalesH, double scalesW,
                                                     aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnUpsampleTrilinear3d, DFX_IN(self, outputSize, alignCorners, scalesD, scalesH, scalesW),
                 DFX_OUT(out));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(self, outputSize, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  vector<float> scalesList{};
  if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
    scalesList.push_back(scalesD);
    scalesList.push_back(scalesH);
    scalesList.push_back(scalesW);
    outputSize = uniqueExecutor.get()->AllocIntArray({}, 0);
  }
  const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
  CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

  vector<float> scalesCastList{};
  if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
    scalesCastList.push_back(static_cast<float>(1.0 / scalesD));
    scalesCastList.push_back(static_cast<float>(1.0 / scalesH));
    scalesCastList.push_back(static_cast<float>(1.0 / scalesW));
  }
  const aclFloatArray* castScales = uniqueExecutor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
  CHECK_RET(castScales != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfShape = op::ToShapeVector(self->GetViewShape());
  float scales_d = 0.0;
  float scales_h = 0.0;
  float scales_w = 0.0;

  if (outputSize->Size() == DIM_THREE) {
    selfShape[DIM_TWO] = (*outputSize)[DIM_ZERO];
    selfShape[DIM_THREE] = (*outputSize)[DIM_ONE];
    selfShape[DIM_FOUR] = (*outputSize)[DIM_TWO];
  } else {
    selfShape[DIM_TWO] *= (*scales)[DIM_ZERO];
    selfShape[DIM_THREE] *= (*scales)[DIM_ONE];
    selfShape[DIM_FOUR] *= (*scales)[DIM_TWO];
    scales_d = (*castScales)[DIM_ZERO];
    scales_h = (*castScales)[DIM_ONE];
    scales_w = (*castScales)[DIM_TWO];
  }

  op::Shape outShape;
  op::ToShape(selfShape.data(), selfShape.size(), outShape);

  auto inputShape = self->GetViewShape();
  // if scale is bigger than 50,back to AICPU
  float scaleW = AsComputeScale(alignCorners, inputShape.GetDim(DIM_FOUR), outShape.GetDim(DIM_FOUR), scales_w);
  float scaleH = AsComputeScale(alignCorners, inputShape.GetDim(DIM_THREE), outShape.GetDim(DIM_THREE), scales_h);
  float scaleD = AsComputeScale(alignCorners, inputShape.GetDim(DIM_TWO), outShape.GetDim(DIM_TWO), scales_d);
  if (CheckIsPlatform310p(self, out) && CheckScales(scaleW, scaleH, scaleD)) {
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
      //将输入self进行transpose，shape：NDHWC-->DHWNC
      const int64_t permuteDHWNCList[] = {1, 2, 3, 0, 4};
      auto permuteDHWNCArray = uniqueExecutor.get()->AllocIntArray(permuteDHWNCList, UPSAMPLE_DIM_LIMIT);
      CHECK_RET(permuteDHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto selfTranspose = l0op::Transpose(selfContiguous, permuteDHWNCArray, uniqueExecutor.get());
      CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto selfUpsampleTrilinear =
          l0op::UpsampleTrilinear3dNcdhw(selfTranspose, outputSize, alignCorners, scales, castScales, scaleW, scaleH, scaleD, uniqueExecutor.get());
      CHECK_RET(selfUpsampleTrilinear != nullptr, ACLNN_ERR_INNER_NULLPTR);

      const int64_t permuteNDHWList[] = {3, 0, 1, 2, 4};
      auto permuteNDHWCArray = uniqueExecutor.get()->AllocIntArray(permuteNDHWList, UPSAMPLE_DIM_LIMIT);
      CHECK_RET(permuteNDHWCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto result = l0op::Transpose(selfUpsampleTrilinear, permuteNDHWCArray, uniqueExecutor.get());
      auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
      CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
      //将输入self进行transpose，shape：NCDHW-->DHWNC
      const int64_t permuteHWNCList[] = {2, 3, 4, 0, 1};
      auto permuteHWNCArray = uniqueExecutor.get()->AllocIntArray(permuteHWNCList, UPSAMPLE_DIM_LIMIT);
      CHECK_RET(permuteHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto selfTranspose = l0op::Transpose(selfContiguous, permuteHWNCArray, uniqueExecutor.get());
      CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto selfUpsampleTrilinear =
          l0op::UpsampleTrilinear3dNcdhw(selfTranspose, outputSize, alignCorners, scales, castScales, scaleW, scaleH, scaleD, uniqueExecutor.get());
      CHECK_RET(selfUpsampleTrilinear != nullptr, ACLNN_ERR_INNER_NULLPTR);

      const int64_t permuteNDHWList[] = {3, 4, 0, 1, 2};
      auto permuteNCDHWArray = uniqueExecutor.get()->AllocIntArray(permuteNDHWList, UPSAMPLE_DIM_LIMIT);
      CHECK_RET(permuteNCDHWArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto result = l0op::Transpose(selfUpsampleTrilinear, permuteNCDHWArray, uniqueExecutor.get());
      auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
      CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  } else {
    auto result =
        upsampleTrilinear3dCompute(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW, scaleH, scaleD, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleTrilinear3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnUpsampleTrilinear3d);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
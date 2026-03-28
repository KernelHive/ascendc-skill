/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "aclnn_pow.h"
#include "aclnn_kernels/contiguous.h"
#include "pow.h"
#include "pows_l0.h"
#include "aclnn_kernels/cast.h"
#include "square.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

const float SQRT_EXP = 0.5;
const float SQUARE_EXP = 2.0;
const float CUBE_EXP = 3.0;
const float NEGTIVE_SQRT_EXP = -0.5;
const float NEGTIVE_ONE_EXP = -1.0;
const float NEGTIVE_SQUARE_EXP = -2.0;

static const uint64_t MAX_SUPPORT_DIMS_NUMS = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_INT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SQUARE_NEED_CAST_DTYPE_LIST = {
  op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_INT16
};

static const std::initializer_list<op::DataType> POWS_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT, op::DataType::DT_BF16
};

static bool CheckPowTensorScalarNotNull(const aclTensor *self, const aclScalar *exponent, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(exponent, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckPowScalarTensorNotNull(const aclScalar *self, const aclTensor *exponent, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(exponent, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const op::DataType selfDtype, const op::DataType expDtype, const op::DataType outDtype) {
  if (!CheckSocVersionIsSupportBf16() &&
     (selfDtype == op::DataType::DT_BF16 || expDtype == op::DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of pow is not support bfloat16 in current socversion.");
    return false;
  }
  if (!CheckType(selfDtype, DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s should be in dtype support list %s.",
            op::ToString(selfDtype).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  if (!CheckType(expDtype, DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "exp dtype %s should be in dtype support list %s.",
            op::ToString(expDtype).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  // 检查self和exponent能否做数据类型推导
  op::DataType promoteType = op::PromoteType(selfDtype, expDtype);
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and exponent dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(expDtype).GetString());
    return false;
  }
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
  return true;
}

static inline bool isFloatType(const DataType type) {
  return type == op::DataType::DT_DOUBLE || type == op::DataType::DT_FLOAT ||
         type == op::DataType::DT_FLOAT16 || type == op::DataType::DT_BF16;
}

static inline op::DataType InferTensorScalarDtype(const aclTensor *self, const aclScalar* exponent,
                                                  const aclTensor *out) {
  if (exponent->GetDataType() == op::DataType::DT_DOUBLE && out->GetDataType() == op::DataType::DT_FLOAT) {
    return op::DataType::DT_FLOAT;
  }

  if (IsComplexType(exponent->GetDataType())) {
    return PromoteType(self->GetDataType(), exponent->GetDataType());
  }
  return isFloatType(self->GetDataType()) ? self->GetDataType() :
         ((isFloatType(exponent->GetDataType()) || self->GetDataType() == op::DataType::DT_BOOL) ?
          PromoteType(self->GetDataType(), exponent->GetDataType()) : self->GetDataType());
}

static inline op::DataType InferScalarTensorDtype(const aclScalar *self, const aclTensor* exponent,
                                                  const aclTensor *out) {
  if (exponent->GetDataType() == op::DataType::DT_DOUBLE && out->GetDataType() == op::DataType::DT_FLOAT) {
    return op::DataType::DT_FLOAT;
  }

  if (IsComplexType(self->GetDataType())) {
    return PromoteType(self->GetDataType(), exponent->GetDataType());
  }
  return isFloatType(exponent->GetDataType()) ? exponent->GetDataType() :
         ((isFloatType(self->GetDataType()) || exponent->GetDataType() == op::DataType::DT_BOOL) ?
          PromoteType(exponent->GetDataType(), self->GetDataType()) : exponent->GetDataType());
}

static bool CheckPromoteType(const op::DataType selfDtype, const op::DataType exponentDtype,
                             const op::DataType outDtype, op::DataType promoteType) {
  if (promoteType == op::DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and exponent dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(exponentDtype).GetString());
    return false;
  }
  if ((selfDtype == op::DataType::DT_BOOL) && (exponentDtype == op::DataType::DT_BOOL)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self and exponent dtype are bool is not supported.");
    return false;
  }
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus CheckPowTensorScalarParams(const aclTensor *self, const aclScalar* exponent,
                                              const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckPowTensorScalarNotNull(self, exponent, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self->GetDataType(), exponent->GetDataType(), out->GetDataType()),
            ACLNN_ERR_PARAM_INVALID);

  op::DataType promoteType = InferTensorScalarDtype(self, exponent, out);
  CHECK_RET(CheckPromoteType(self->GetDataType(), exponent->GetDataType(), out->GetDataType(), promoteType),
            ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入shape
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckPowScalarTensorParams(const aclScalar *self, const aclTensor* exponent,
                                              const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckPowScalarTensorNotNull(self, exponent, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self->GetDataType(), exponent->GetDataType(), out->GetDataType()),
            ACLNN_ERR_PARAM_INVALID);

  op::DataType promoteType = InferScalarTensorDtype(self, exponent, out);
  CHECK_RET(CheckPromoteType(self->GetDataType(), exponent->GetDataType(), out->GetDataType(), promoteType),
            ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入shape
  CHECK_RET(CheckShape(exponent, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckSupportPows(const aclTensor *selfCast, const aclScalar *exponent) {
  if (exponent->ToFloat() != SQRT_EXP && exponent->ToFloat() != SQUARE_EXP &&
      exponent->ToFloat() != CUBE_EXP && exponent->ToFloat() != NEGTIVE_SQRT_EXP &&
      exponent->ToFloat() != NEGTIVE_ONE_EXP && exponent->ToFloat() != NEGTIVE_SQUARE_EXP) {
    return false; 
  }

  if (!CheckType(selfCast->GetDataType(), POWS_DTYPE_SUPPORT_LIST)) {
    return false; 
  }

  if(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P && 
     GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B && 
     GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
    return false;
  }

  return true;
}

aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                 const aclScalar *exponent,
                                                 const aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPowTensorScalar, DFX_IN(self, exponent), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckPowTensorScalarParams(self, exponent, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // pow算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto promoteType = InferTensorScalarDtype(self, exponent, out);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclTensor* powOut = nullptr;
  if (CheckSupportPows(selfCast, exponent)) {
    auto expTensor = uniqueExecutor.get()->ConvertToTensor(exponent, promoteType);
    CHECK_RET(expTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用pows进行计算
    powOut = const_cast<aclTensor *>(l0op::Pows(selfCast, expTensor, uniqueExecutor.get()));
  } else if (exponent->ToFloat() == SQUARE_EXP) {
    const aclTensor *squareInput = selfCast;
    if (CheckType(selfCast->GetDataType(), SQUARE_NEED_CAST_DTYPE_LIST)) {
      squareInput = l0op::Cast(selfCast, op::DataType::DT_INT32, uniqueExecutor.get());
      CHECK_RET(squareInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 当exponent为2.0时，使用square算子计算
    powOut = const_cast<aclTensor *>(l0op::Square(squareInput, uniqueExecutor.get()));
  } else {
    auto expTensor = uniqueExecutor.get()->ConvertToTensor(exponent, promoteType);
    CHECK_RET(expTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用pow进行计算
    powOut = const_cast<aclTensor *>(l0op::Pow(selfCast, expTensor, uniqueExecutor.get()));
  }
  CHECK_RET(powOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(powOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPowTensorScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                        const aclScalar *exponent,
                                                        uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {
  auto out = const_cast<aclTensor*>(self);
  return aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, workspaceSize, executor);
}

aclnnStatus aclnnInplacePowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplacePowTensorScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar *self,
                                                 const aclTensor *exponent,
                                                 const aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPowScalarTensor, DFX_IN(self, exponent), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

   // 固定写法，参数检查
  auto ret = CheckPowScalarTensorParams(self, exponent, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // pow算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (exponent->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto promoteType = InferScalarTensorDtype(self, exponent, out);

  // 固定写法，将输入self转换成连续的tensor
  auto expContiguous = l0op::Contiguous(exponent, uniqueExecutor.get());
  CHECK_RET(expContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto expCast = l0op::Cast(expContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(expCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfTensor = uniqueExecutor.get()->ConvertToTensor(self, promoteType);
  CHECK_RET(selfTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用pow进行计算
  auto powOut = l0op::Pow(selfTensor, expCast, uniqueExecutor.get());
  CHECK_RET(powOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(powOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPowScalarTensor(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPowScalarTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif

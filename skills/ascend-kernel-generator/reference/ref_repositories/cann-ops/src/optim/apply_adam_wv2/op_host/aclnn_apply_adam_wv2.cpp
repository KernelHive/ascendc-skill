/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_apply_adam_wv2.cpp
 */

#include "aclnn_apply_adam_wv2.h"
#include "apply_adam_wv2_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
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

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> STEP_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensor* varRef, const aclTensor* mRef, const aclTensor* vRef,
                         const aclTensor* maxGradNormOptionalRef, const aclTensor* grad,
                         const aclTensor* step, bool amsgrad) {
  OP_CHECK_NULL(varRef, return false);
  OP_CHECK_NULL(mRef, return false);
  OP_CHECK_NULL(vRef, return false);
  if (amsgrad) {
    OP_CHECK_NULL(maxGradNormOptionalRef, return false);
  }
  OP_CHECK_NULL(grad, return false);
  OP_CHECK_NULL(step, return false);
  return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910B: {
      return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    case SocVersion::ASCEND910: {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
    default: {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
  }
}

static bool CheckShape(const aclTensor* varRef, const aclTensor* mRef, const aclTensor* vRef,
                       const aclTensor* maxGradNormOptionalRef, const aclTensor* grad) {
  OP_CHECK_SHAPE_NOT_EQUAL(mRef, varRef, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(vRef, varRef, return false);
  if (maxGradNormOptionalRef != nullptr) {
    OP_CHECK_SHAPE_NOT_EQUAL(maxGradNormOptionalRef, varRef, return false);
  }
  OP_CHECK_SHAPE_NOT_EQUAL(grad, varRef, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* varRef, const aclTensor* mRef, const aclTensor* vRef,
                               const aclTensor* maxGradNormOptionalRef, const aclTensor* grad,
                               const aclTensor* step, bool amsgrad) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(varRef, mRef, vRef, maxGradNormOptionalRef, grad, step, amsgrad), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
  // 检查varRef的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(varRef, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
  // 检查mRef的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(mRef, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
  // 检查vRef的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(vRef, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
  // 检查maxGradNormOptionalRef的数据类型是否在支持列表内
  if (maxGradNormOptionalRef != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(maxGradNormOptionalRef, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
  }
  // 检查grad的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(grad, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
  // 检查step的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(step, STEP_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入的shape是否满足约束
  CHECK_RET(CheckShape(varRef, mRef, vRef, maxGradNormOptionalRef, grad), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static inline bool IsNeedDtypeCast(const aclTensor* var, const aclTensor* m, const aclTensor* v,
                                   const aclTensor* maxGradNorm, const aclTensor* grad) {
  bool isSameDtype = (var->GetDataType() == m->GetDataType()) && (var->GetDataType() == v->GetDataType()) &&
                     (var->GetDataType() == grad->GetDataType()) &&
                     (maxGradNorm == nullptr || var->GetDataType() == maxGradNorm->GetDataType());
  if (isSameDtype) {
    return false;
  }
  bool isOpsSupport = (var->GetDataType() == DataType::DT_FLOAT) && (m->GetDataType() == DataType::DT_FLOAT) &&
                      (v->GetDataType() == DataType::DT_FLOAT) &&
                      ((grad->GetDataType() == DataType::DT_FLOAT16 || grad->GetDataType() == DataType::DT_BF16)) &&
                      (maxGradNorm == nullptr || grad->GetDataType() == maxGradNorm->GetDataType());
  return !isOpsSupport;
}

static inline aclTensor* CheckAndExecuteTypeCast(bool isNeedCast, const aclTensor* input, aclOpExecutor *executor) {
  if (input == nullptr) {
    return nullptr;
  }
  if (isNeedCast) {
    return const_cast<aclTensor*>(l0op::Cast(input, DataType::DT_FLOAT, executor));
  }
  return const_cast<aclTensor*>(input);
}

aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                                              aclTensor* maxGradNormOptionalRef, const aclTensor* grad,
                                              const aclTensor* step, float lr, float beta1, float beta2,
                                              float weightDecay, float eps, bool amsgrad, bool maximize,
                                              uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnApplyAdamWV2, DFX_IN(varRef, mRef, vRef, maxGradNormOptionalRef, grad, step, lr, beta1, beta2,
                                           weightDecay, eps, amsgrad, maximize), DFX_OUT());
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(varRef, mRef, vRef, maxGradNormOptionalRef, grad, step, amsgrad);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor场景处理
  if (varRef->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入var转换成连续的tensor
  auto varContiguous = l0op::Contiguous(varRef, uniqueExecutor.get());
  CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入m转换成连续的tensor
  auto mContiguous = l0op::Contiguous(mRef, uniqueExecutor.get());
  CHECK_RET(mContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入v转换成连续的tensor
  auto vContiguous = l0op::Contiguous(vRef, uniqueExecutor.get());
  CHECK_RET(vContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入maxGradNorm转换成连续的tensor
  auto maxGradNormContiguous = maxGradNormOptionalRef == nullptr ? nullptr :
    l0op::Contiguous(maxGradNormOptionalRef, uniqueExecutor.get());
  CHECK_RET(maxGradNormOptionalRef == nullptr || maxGradNormContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入grad转换成连续的tensor
  auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
  CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  bool isNeedCast = IsNeedDtypeCast(varContiguous, mContiguous, vContiguous, maxGradNormContiguous, gradContiguous);

  auto varCast = CheckAndExecuteTypeCast(isNeedCast, varContiguous, uniqueExecutor.get());
  CHECK_RET(varCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto mCast = CheckAndExecuteTypeCast(isNeedCast, mContiguous, uniqueExecutor.get());
  CHECK_RET(mCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto vCast = CheckAndExecuteTypeCast(isNeedCast, vContiguous, uniqueExecutor.get());
  CHECK_RET(vCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto maxGradNormCast = CheckAndExecuteTypeCast(isNeedCast, maxGradNormContiguous, uniqueExecutor.get());
  CHECK_RET(maxGradNormContiguous == nullptr || maxGradNormCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gradCast = isNeedCast ? l0op::Cast(gradContiguous, DataType::DT_FLOAT, uniqueExecutor.get()) : gradContiguous;
  CHECK_RET(gradCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，调用ApplyAdamWV2算子完成计算
  l0op::ApplyAdamWV2(varCast, mCast, vCast, maxGradNormCast, gradCast, step, lr, beta1, beta2, weightDecay, eps,
                     amsgrad, maximize, uniqueExecutor.get());

  // 固定写法，将计算结果转换成输出的数据类型
  auto varOut = isNeedCast ? l0op::Cast(varCast, varRef->GetDataType(), uniqueExecutor.get()) : varCast;
  CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto mOut = isNeedCast ? l0op::Cast(mCast, mRef->GetDataType(), uniqueExecutor.get()) : mCast;
  CHECK_RET(mOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto vOut = isNeedCast ? l0op::Cast(vCast, vRef->GetDataType(), uniqueExecutor.get()) : vCast;
  CHECK_RET(vOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出上，输出可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(varOut, varRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  viewCopyResult = l0op::ViewCopy(mOut, mRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  viewCopyResult = l0op::ViewCopy(vOut, vRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (maxGradNormOptionalRef != nullptr) {
    auto maxGradNormOut = (isNeedCast ?
      l0op::Cast(maxGradNormCast, maxGradNormOptionalRef->GetDataType(), uniqueExecutor.get()) : maxGradNormCast);
    CHECK_RET(maxGradNormOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(maxGradNormOut, maxGradNormOptionalRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnApplyAdamWV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnApplyAdamWV2);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
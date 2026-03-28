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

/*!
 * \file aclnn_mse_loss_backward.cpp
 * \brief
 */

#include "level2/aclnn_mse_loss_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/mse_loss_grad.h"
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
#include "common/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::string REDUCTION_STR[] = {"none", "mean", "sum"};
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                        op::DataType::DT_FLOAT16,
                                                                                        op::DataType::DT_BF16};

static inline bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                                   const aclTensor* out) {
  const auto& supportList = GetDtypeSupportListV1(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SAME(gradOutput, self, return false);
  OP_CHECK_DTYPE_NOT_SAME(target, self, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  return true;
}

static inline bool CheckReduction(int64_t reduction) {
  if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction to be between 0 and 2, but got %ld.", reduction);
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                       const aclTensor* out) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
  op::Shape broadcastShape1;
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, broadcastShape1, return false);
  OP_CHECK_BROADCAST_WITH_SHAPE(target, broadcastShape1, return false);
  BroadcastInferShape(target->GetViewShape(), broadcastShape1, broadcastShape);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                                      int64_t reduction, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull4Tensor(gradOutput, self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, target, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查reduction是否符合规则
  CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输出shape
  CHECK_RET(CheckShape(gradOutput, self, target, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMseLossBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                 const aclTensor* target, int64_t reduction, aclTensor* out,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnMseLossBackward, DFX_IN(gradOutput, self, target, reduction), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, target, reduction, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入gradOutput转换成连续的tensor
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入target转换成连续的tensor
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行计算
  auto grad = l0op::MseLossGradV2(gradOutputContiguous, selfContiguous, targetContiguous,
                                REDUCTION_STR[reduction], uniqueExecutor.get());
  CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(grad, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMseLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMseLossBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

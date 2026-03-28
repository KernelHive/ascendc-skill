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
 * \file aclnn_index_select.cpp
 * \brief
 */
#include "aclnn_index_select.h"
#include "gather_v2.h"
#include "gather_v3_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "unsqueeze.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

/* IndexSelect 算子的完整计算流程如下:
 * self             index         dim
 *   |                |        (workspace_3)
 *   \                |            |
 * Contiguous      Contiguous      |
 * (workspace_0)   (workspace_2)   |
 *      \             |            /
 *       \            |           /
 *         \          |         /
 *          GatherV2(workspace_4)
 *                    |
 *                    |
 *                ViewCopy
 *                    |
 *                   out
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL,
    op::DataType::DT_UINT64, op::DataType::DT_UINT32, op::DataType::DT_UINT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static constexpr uint64_t MAX_INPUT_DIM_NUM = 8;
static const std::initializer_list<op::DataType> INDEX_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static inline bool HasEmptyTensor(const aclTensor *self, const aclTensor *index) {
  return self->IsEmpty() || index->IsEmpty();
}

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  auto ver = GetCurrentPlatformInfo().GetSocVersion();
  if (self->GetDataType() == op::DataType::DT_BF16 && ver != SocVersion::ASCEND910B && ver != SocVersion::ASCEND910_93) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "socVersion %s does not support BF16.",
            op::ToString(ver).GetString());
    return false;
  }

  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

  // 检查index的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_SUPPORT_LIST, return false);

  return true;
}

static inline bool CheckFormat(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(index->GetStorageFormat())
          || op::IsPrivateFormat(out->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *out) {
  int64_t selfDimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (self->GetViewShape().IsScalar()) {
    selfDimSize = 1;
    if (!out->GetViewShape().IsScalar()) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self is scalar, and out should also be scalar");
      return false;
    }
  }

  // self维度有上限
  OP_CHECK_MAX_DIM(self,  MAX_INPUT_DIM_NUM, return false);

  // return false if dim is not in valid range
  if (!(dim >= -selfDimSize && dim < selfDimSize)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "IndexError: Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
            -selfDimSize, selfDimSize-1, dim);
    return false;
  } else if (dim < 0) {
    dim += selfDimSize;
  }

  int64_t indexDimSize = static_cast<int64_t>(index->GetViewShape().GetDimNum());
  if (indexDimSize > 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index dimNum is %ld, should not greater than 1", indexDimSize);
    return false;
  }

  op::Shape outShape = self->GetViewShape();
  outShape.SetDim(dim, index->GetViewShape().IsScalar() ? 1 : index->GetViewShape().GetDim(0));
  if (self->Size() && index->Size() && out->GetViewShape() != outShape) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(outShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, index, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, index, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, dim, index, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnIndexSelect, DFX_IN(self, dim, index), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dim, index, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 算子的空tensor的支持情况
  if (HasEmptyTensor(self, index)) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // index如果非连续，需要转连续
  auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
  CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto indexParam = indexContiguous;

  // 判断index是零维的情况：将它转成一维
  if (indexContiguous->GetViewShape().GetDimNum()==0) {
    const int64_t zero = 0;
    indexParam = l0op::UnsqueezeNd(indexContiguous, zero, uniqueExecutor.get());
  }

  // 调用l0算子IndexSelect进行计算
  auto selectResult = l0op::GatherV3(selfContiguous, dim, indexParam, uniqueExecutor.get());
  CHECK_RET(selectResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(selectResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                             const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnIndexSelect);

  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

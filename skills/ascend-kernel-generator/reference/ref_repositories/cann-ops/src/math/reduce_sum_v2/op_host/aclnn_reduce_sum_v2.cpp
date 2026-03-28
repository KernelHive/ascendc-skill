/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <bitset>
#include "aclnn_kernels/cast.h"
#include "reduce_sum_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_reduce_sum_v2.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_LEN = 64;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> EMPTY_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclIntArray *dims, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(dims, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclDataType dtype, const aclTensor *out) {
  // 检查self和out的数据类型是否在支持列表内
  bool isAscend910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                                 GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);

  OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
  // 检查dtype指定的数据类型是否支持
  if (!CheckType(op::ToOpDataType(dtype), ASCEND910B_DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "type %s should be in dtype support list [%s].",
            op::ToString(op::ToOpDataType(dtype)).GetString(), op::ToString(ASCEND910B_DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  // 检查dtype数据类型与out是否一致
  OP_CHECK_DTYPE_NOT_MATCH(out, op::ToOpDataType(dtype), return false);
  return true;
}

static bool CheckMaxDimension(const aclTensor *self) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static inline uint64_t GetPosDim(int64_t dim, int64_t dimNum) {
  if (dimNum <= 0) {
    dimNum = 1;
  }
  return dim >= 0 ? dim : dim + dimNum;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dims) {
  auto selfViewShape = self->GetViewShape();
  auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
  // self为标量时，dim range [-1, 0]
  if (selfDimNum <= 0) {
    selfDimNum = 1;
  }
  // dim为负时需要转正校验
  std::bitset<MAX_MASK_LEN> dimMask = std::bitset<MAX_MASK_LEN>();

  for (size_t i = 0; i < dims->Size(); i++) {
    int64_t curDim = (*dims)[i];
    if (curDim >= selfDimNum || curDim < (-selfDimNum)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Provided dim %ld must be in the range of [%ld, %ld].",
              curDim, -selfDimNum, selfDimNum - 1);
      return false;
    }
    uint64_t index = GetPosDim(curDim, selfDimNum);
    // dim重复
    if (dimMask[index]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %lu appears multiple times in the list of dims.", index);
      return false;
    }

    dimMask.set(index);
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *dims, const aclDataType dtype, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, dims, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查self、out的数据类型是否合法
  CHECK_RET(CheckDtypeValid(self, dtype, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查最大维度是否超过8
  CHECK_RET(CheckMaxDimension(self), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查reduce的轴是否超出self维度范围
  CHECK_RET(CheckDimValid(self, dims), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor *out, float val, aclOpExecutor *executor)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(out, EMPTY_INPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
  FVector<int64_t> shape;
  size_t dimNum = out->GetViewShape().GetDimNum();

  if (out->IsEmpty()) {
    return ACLNN_SUCCESS;
  }

  for (size_t idx = 0; idx < dimNum; idx++) {
    int64_t tmpVal = out->GetViewShape().GetDim(idx);
    shape.push_back(tmpVal);
  }

  auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
  auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

  FVector<float> valVector = {val};
  auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
  auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
  CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnReduceSumV2GetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, bool keepDims,
                                           aclDataType dtype, aclTensor *out, uint64_t *workspaceSize,
                                           aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnReduceSumV2, DFX_IN(self, dims, keepDims, dtype), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dims, dtype, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 输入self为空tensor时，直接返回dtype类型的空tensor
  if (self->IsEmpty()) {
    ret = FillScalar(out, 0.0f, uniqueExecutor.get());
    if (ret == ACLNN_SUCCESS) {
      *workspaceSize = uniqueExecutor->GetWorkspaceSize();
      uniqueExecutor.ReleaseTo(executor);
    }
    return ret;
  }

  // 数据类型转换处理
  op::DataType selfType = self->GetDataType();
  op::DataType dataType = op::ToOpDataType(dtype);
  op::DataType promoteType = dataType;
  promoteType = (selfType == promoteType) ? selfType : promoteType;
  
  // 空dims处理
  op::Shape shape = self->GetViewShape();
  if (dims->Size() == 0) {
    size_t dimDum = shape.GetDimNum();
    int64_t appendDim[dimDum];
    for (uint64_t i = 0; i < dimDum; i++) {
      appendDim[i] = i;
    }
    dims = uniqueExecutor.get()->AllocIntArray(appendDim, dimDum);
  }
 
  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型进行转换
  auto selfContiguousCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用ReduceSum算子kernel,将输入self的数据类型转换成指定的数据类型
  const aclTensor *reduceSumOut = l0op::ReduceSumV2(selfContiguousCasted, dims, keepDims, uniqueExecutor.get());
  CHECK_RET(reduceSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckReduceOutShape(reduceSumOut, out), ACLNN_ERR_PARAM_INVALID);

  // 将计算结果转换成输出out的数据类型
  reduceSumOut = l0op::Cast(reduceSumOut, out->GetDataType(), uniqueExecutor.get());

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(reduceSumOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnReduceSumV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnReduceSumV2);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file aclnn_histc.cpp
 * \brief
 */
#include "aclnn_histc.h"
#include <cmath>
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "zero_op.h"
#include "histogram.h"
#include "reduce_min.h"
#include "reduce_max.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_INT_LIST = {
  op::DataType::DT_INT64, op::DataType::DT_INT32, op::DataType::DT_INT16,
  op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_FLOAT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor *self, const aclScalar *min, const aclScalar *max, aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  OP_CHECK_NULL(max, return false);
  OP_CHECK_NULL(min, return false);
  return true;
}

static bool SelfDTypeInt(op::DataType selfDType) {
  auto it = std::find(DTYPE_INT_LIST.begin(), DTYPE_INT_LIST.end(), selfDType);
  if (it != DTYPE_INT_LIST.end()) {
    return true;
  }
  return false;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在Histogram算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_910B, return false);

  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_910B, return false);

  return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *out, op::DataType promoteType) {
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s can not cast to promote dtype %s.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(DataType::DT_UNDEFINED).GetString());
    return false;
  }

  // 检查self的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, self->GetDataType(), return false);

  // 检查out的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  return true;
}

static bool CheckValueRange(int64_t bins, const aclScalar *min, const aclScalar *max, op::DataType selfDType) {
  if (bins <= 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of bins can not less or equal 0.");
    return false;
  }

  if (SelfDTypeInt(selfDType)) {
    int32_t minValue = min->ToInt32();
    int32_t maxValue = max->ToInt32();
    if (minValue > maxValue) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of max should greater than min.");
      return false;
    }
  } else {
    float minValue = min->ToFloat();
    float maxValue = max->ToFloat();
    if (minValue > maxValue) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of max should greater than min.");
      return false;
    }
  }
  return true;
}

static bool CheckShape(const aclTensor *self, int64_t bins, const aclTensor *out) {
  OP_CHECK_WRONG_DIMENSION(out, 1, return false);

  OP_CHECK_MAX_DIM(self, MAX_DIM, return false);

  if (out->Size() != bins) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of out tensor should be same as bins.");
    return false;
  }

  return true;
}

static bool NeedComputeMinMax(const aclScalar *min, const aclScalar *max, op::DataType selfDType) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (!(socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E)) {
    return false;
  }

  if (SelfDTypeInt(selfDType)) {
    int64_t minValue = min->ToInt64();
    int64_t maxValue = max->ToInt64();
    return minValue == maxValue;
  }
  float minValue = min->ToFloat();
  float maxValue = max->ToFloat();
  return maxValue - minValue < static_cast<float>(1e-6) && maxValue - minValue > static_cast<float>(-1e-6);
}

static std::tuple<const aclTensor*, const aclTensor*> AllMinMax(const aclTensor *self, aclOpExecutor* executor) {
  if (self->GetViewShape().GetDimNum() == 0) {
    return std::tuple<const aclTensor*, const aclTensor*>(self, self);
  }
  size_t dimDum = self->GetViewShape().GetDimNum();
  int64_t appendDim[dimDum];
  for (int64_t i = 0; i < dimDum; i++) {
      appendDim[i] = i;
  }
  auto dim = executor->AllocIntArray(appendDim, dimDum);

  // 进行reducemin计算
  const aclTensor *min = l0op::ReduceMin(self, dim, false, executor);
  // 进行reducemax计算
  const aclTensor *max = l0op::ReduceMax(self, dim, false, executor);
  return std::tuple<const aclTensor*, const aclTensor*>(min, max);
}

static aclnnStatus CheckHistcParams(const aclTensor *self, int64_t bins, const aclScalar *min,
                                         const aclScalar *max, aclTensor *out) {
  // 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, min, max, out), ACLNN_ERR_PARAM_NULLPTR);

  // 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 检查bins, min, max的取值范围
  CHECK_RET(CheckValueRange(bins, min, max, self->GetDataType()), ACLNN_ERR_PARAM_INVALID);

  // 检查self和out能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, out, out->GetDataType()), ACLNN_ERR_PARAM_INVALID);

  // 检查shape是否一致
  CHECK_RET(CheckShape(self, bins, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus EmptyTensor(aclTensor *out, aclOpExecutor* executor) {
  auto outContiguous = l0op::Contiguous(out, executor);
  CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 调用ZerosLike算子kernel
  auto zeroOut = l0op::ZerosLike(outContiguous, executor);
  CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyOut = l0op::ViewCopy(zeroOut, out, executor);
  CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static const aclTensor* CastSelf(const aclTensor *selfContiguous, aclOpExecutor* executor) {
  bool selfIsInt = SelfDTypeInt(selfContiguous->GetDataType());
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto castType = selfIsInt ? op::DataType::DT_INT32 : op::DataType::DT_FLOAT;
  if (!(socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E)) {
    return selfContiguous;
  }
  auto selfContiguousCasted = l0op::Cast(selfContiguous, castType, executor);
  return selfContiguousCasted;
}

aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor *self, int64_t bins, const aclScalar *min, const aclScalar *max,
                                       aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnHistc, DFX_IN(self, bins, min, max), DFX_OUT(out));

  // 参数检查
  auto ret = CheckHistcParams(self, bins, min, max, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty()) {
    auto status = EmptyTensor(out, uniqueExecutor.get());
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return status;
  }

  // 将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // min, max转tensor
  aclOpExecutor *executorP = uniqueExecutor.get();
  auto minTensor = executorP->ConvertToTensor(min, selfContiguous->GetDataType());
  CHECK_RET(minTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  auto maxTensor = executorP->ConvertToTensor(max, selfContiguous->GetDataType());
  CHECK_RET(maxTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 重新求self中的min max值
  if (NeedComputeMinMax(min, max, selfContiguous->GetDataType())) {
    auto minMaxResult = AllMinMax(selfContiguous, uniqueExecutor.get());
    minTensor = std::get<0>(minMaxResult);
    maxTensor = std::get<1>(minMaxResult);
    CHECK_RET(minTensor != nullptr && maxTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  }

  float minValue = min->ToFloat();
  float maxValue = max->ToFloat();

  // 调用Histogram算子kernel
  auto HistogramCal = l0op::Histogram(selfContiguous, minTensor, maxTensor, out,
                                      bins, minValue, maxValue, uniqueExecutor.get());
  CHECK_RET(HistogramCal != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(HistogramCal, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHistc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHistc);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

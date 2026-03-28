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
 * \file aclnn_inplace_index_add_with_sorted.cpp
 * \brief
 */
#include "aclnn_inplace_index_add_with_sorted.h"
#include "inplace_index_add_with_sorted_l0.h"
#include "sort.h"
#include "mul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
// FLOAT类型在2139095040时为inf，不能sort
static constexpr int64_t MAX_SORT_SHAPE_DIM = 2139095040;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT64,
    op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
    op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_BF16};

static bool IsAICoreSupport(const aclTensor *self) {
  // 根据芯片类型和输入self类型判断是否走aicore
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    if (CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
      return true;
    }
  } else {
    if (CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST)) {
      return true;
    }
  }
  return false;
}

static bool CheckNotNull(const aclTensor * self, const aclTensor * index, const aclTensor * source,
                         const aclScalar * alpha, const aclTensor * out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(source, return false);
  OP_CHECK_NULL(alpha, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const op::DataType selfDtype, const op::DataType indexDtype,
                            const op::DataType sourceDtype, const op::DataType outDtype) {
  // 获取芯片支持的数据类型                      
  bool isAscend910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
                                   GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
  const std::initializer_list<op::DataType> dtypeSupportList =
      isAscend910BSocVersion ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;

  if (!CheckType(selfDtype, dtypeSupportList)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s should be in dtype support list [%s].",
            op::ToString(selfDtype).GetString(), op::ToString(dtypeSupportList).GetString());
    return false;
  }
  if (!CheckType(indexDtype, INDEX_DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index dtype %s should be in dtype support list [%s].",
            op::ToString(indexDtype).GetString(), op::ToString(INDEX_DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  if (!CheckType(sourceDtype, dtypeSupportList)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "source dtype %s should be in dtype support list [%s].",
            op::ToString(sourceDtype).GetString(), op::ToString(dtypeSupportList).GetString());
    return false;
  }
  if (!CheckType(outDtype, dtypeSupportList)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outDtype dtype %s should be in dtype support list [%s].",
            op::ToString(sourceDtype).GetString(), op::ToString(dtypeSupportList).GetString());
    return false;
  }
  if (selfDtype != sourceDtype) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and source dtype %s must same.",
            op::ToString(selfDtype).GetString(), op::ToString(sourceDtype).GetString());
    return false;
  }
  if (selfDtype != outDtype) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and source dtype %s must same.",
            op::ToString(selfDtype).GetString(), op::ToString(outDtype).GetString());
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *index, const aclTensor *source,
                       const aclTensor *out, int64_t dim) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(source, MAX_DIM_LEN, return false);
  int64_t selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());

  if ((dim < -selfDimNum || dim >= selfDimNum) && (
      !(dim == 0 && selfDimNum == 0)
      ))  {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be in [-self.dims(), self.dims), but dim is %ld.", dim);
    return false;
  }
  
  if (self->GetViewShape().GetDimNum() != 0 && self->GetViewShape().GetDimNum() != source->GetViewShape().GetDimNum()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of self must be equal to dimNum of source.");
    return false;
  }
  int64_t dimStd = dim >= 0 ? dim : self->GetViewShape().GetDimNum() + dim;
  for (size_t idx = 0; idx < self->GetViewShape().GetDimNum(); idx++) {
    if (dimStd != static_cast<int64_t>(idx)) {
      if (self->GetViewShape().GetDim(idx) != source->GetViewShape().GetDim(idx)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of self must match the size of source at dimension %ld.", dim);
        return false;
      }
    }
  }
  if (index->GetViewShape().GetDimNum() > 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index dimension %ld is supposed to be 1.", index->GetViewShape().GetDimNum());
    return false;
  }
  if (index->GetViewShape().GetDim(0) != source->GetViewShape().GetDim(dimStd)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Number of index should be equal to number of source in dim  %ld.", dim);
    return false;
  }

  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const int64_t dim, const aclTensor *index,
                               const aclTensor *source, const aclScalar *alpha, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, source, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self->GetDataType(), index->GetDataType(), source->GetDataType(),
                            out->GetDataType()), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(self, index, source, out, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static void ViewDataType(const aclTensor *input, const op::DataType dtype) {
  auto tmpTensor = const_cast<aclTensor *>(input);
  tmpTensor->SetDataType(dtype);
  input = tmpTensor;
}

static const aclTensor* DoIndexAddWithSorted(const aclTensor *self, const int64_t dim, const aclTensor *index,
                            const aclTensor *source, const aclTensor *alpha, aclOpExecutor* executor) {
  auto indexSize = index->GetViewShape().GetDim(0);
  // sort index
  const aclTensor* indexInt32 = nullptr;
  const aclTensor* indexAddRes = nullptr;
  indexInt32 = l0op::Cast(index, op::DataType::DT_INT32, executor);
  CHECK_RET(indexInt32 != nullptr, nullptr);
  // 将INT32的索引看作float数据类型以便传给sort算子进行排序
  // 注：由于sort在元素个数为1时会走到aicpu算子，耗时为几百us，性能大幅劣化，在sort修复前先用此逻辑规避
  if (indexSize > 1) {
    const aclTensor *indicesViewFloat = executor->CreateView(indexInt32, {indexSize}, indexInt32->GetViewOffset());
    ViewDataType(indicesViewFloat, op::DataType::DT_FLOAT);
    auto sortResult = l0op::Sort(indicesViewFloat, -1, false, true, executor);
    auto sortIdxOut = std::get<0>(sortResult);
    auto posIdx = std::get<1>(sortResult);
    CHECK_RET(sortIdxOut != nullptr && posIdx != nullptr, nullptr);
    // 将sort的结果已INT32的数据类型来进行读取
    auto sortIndices = executor->CreateView(sortIdxOut, {indexSize}, sortIdxOut->GetViewOffset());
    sortIndices->SetDataType(op::DataType::DT_INT32);
    // inplace index add
    indexAddRes = l0op::InplaceIndexAddWithSorted(self, dim, sortIndices, posIdx, source, alpha, executor);
  } else {
    const aclTensor* posTensor = executor->ConvertToTensor(executor->AllocScalar(0), op::DataType::DT_INT32);
    CHECK_RET(posTensor != nullptr, nullptr);
    indexAddRes = l0op::InplaceIndexAddWithSorted(self, dim, indexInt32, posTensor, source, alpha, executor);
  }
  return indexAddRes;
}

aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor *self, const int64_t dim, const aclTensor *index,
    const aclTensor *source, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  // 固定写法，参数检查
  L2_DFX_PHASE_1(aclnnIndexAdd, DFX_IN(self, dim, index, source, alpha), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  if (index != nullptr && index->GetViewShape().GetDimNum() == 0) {
    int64_t zero = 0;
    index = l0op::UnsqueezeNd(index, zero, uniqueExecutor.get());
    CHECK_RET(index != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (source != nullptr && source->GetViewShape().GetDimNum() == 0) {
    int64_t zero = 0;
    source = l0op::UnsqueezeNd(source, zero, uniqueExecutor.get());
    CHECK_RET(index != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  auto ret = CheckParams(self, dim, index, source, alpha, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if (self->IsEmpty() || source->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto sourceContiguous = l0op::Contiguous(source, uniqueExecutor.get());
  CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* alphaTensor = nullptr;
  bool useNewOp = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
    GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) && dim == 0 &&
    self->GetViewShape().GetDim(0) < MAX_SORT_SHAPE_DIM && self->GetDataType() == op::DataType::DT_BF16;
  if (self->GetDataType() == op::DataType::DT_BF16 && !useNewOp) {
    selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    sourceContiguous = l0op::Cast(sourceContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, op::DataType::DT_FLOAT);
  } else if ((self->GetDataType() == op::DataType::DT_BF16 || self->GetDataType() == op::DataType::DT_FLOAT16) && useNewOp) {
    // 由于算子内部升精度计算，因此不能将高精度传入的aclScalar先降低精度，再算子内部升精度，会造成精度损失，因此在aclnn处将alpha作为float的tensor
    alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, op::DataType::DT_FLOAT);
  } else {
    alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, self->GetDataType());
  }
  CHECK_RET(alphaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
  CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* indexAddOut = nullptr;
  // 当设备类型为A2或A3且index为int32类型时，切为InplaceIndexAddWithSorted算子
  if (useNewOp) {
    indexAddOut = DoIndexAddWithSorted(selfContiguous, dim, indexContiguous, sourceContiguous, alphaTensor,
                                       uniqueExecutor.get());
  } else if (IsAICoreSupport(self)) {
    indexAddOut = l0op::InplaceIndexAddAiCore(selfContiguous, dim, indexContiguous, sourceContiguous, alphaTensor,
                                              uniqueExecutor.get());
  } else {
    indexAddOut = l0op::InplaceIndexAddAiCpu(selfContiguous, dim, indexContiguous, sourceContiguous, alphaTensor,
                                             uniqueExecutor.get());
  }
  CHECK_RET(indexAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (self->GetDataType() == op::DataType::DT_BF16) {
    indexAddOut = l0op::Cast(indexAddOut, op::DataType::DT_BF16, uniqueExecutor.get());
  }
  CHECK_RET(indexAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyResult = l0op::ViewCopy(indexAddOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                          aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnIndexAdd);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

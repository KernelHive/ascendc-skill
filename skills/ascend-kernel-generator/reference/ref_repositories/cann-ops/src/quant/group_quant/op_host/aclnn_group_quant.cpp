/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_group_quant.h"
#include "group_quant_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

static constexpr size_t MAX_DIM_LEN = 8;
static constexpr uint64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr uint64_t INT4_NUMS_IN_INT8_SPACE = 2;
static constexpr int32_t X_DIM_NUM = 2;

static const std::initializer_list<op::DataType> EMPTY_LIST = {};

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> GROUP_INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<DataType> Y_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT4};

static const std::map<op::SocVersion, const std::initializer_list<op::DataType>*> SOC_X_SUPPORT_DTYPES = {
    {SocVersion::ASCEND910B, &X_DTYPE_SUPPORT_LIST}
};

static const std::map<op::SocVersion, const std::initializer_list<op::DataType>*> SOC_GROUP_INDEX_SUPPORT_DTYPES = {
    {SocVersion::ASCEND910B, &GROUP_INDEX_DTYPE_SUPPORT_LIST}
};

static const std::map<op::SocVersion, const std::initializer_list<op::DataType>*> SOC_Y_SUPPORT_DTYPES = {
    {SocVersion::ASCEND910B, &Y_DTYPE_SUPPORT_LIST}
};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList(
    const std::map<op::SocVersion, const std::initializer_list<op::DataType>*> &socSupportDtypes) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto found = socSupportDtypes.find(socVersion);
  if (found != socSupportDtypes.end()) {
    return *(found->second);
  }
  OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
  return EMPTY_LIST;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* y, const aclTensor*& outTensor,
                                          const int32_t& dstType, aclOpExecutor *executor) {
  OP_CHECK(dstType == op::DataType::DT_INT32,
           OP_LOGD("aclnnGroupQuant output do not need to pack."), return ACLNN_SUCCESS);

  // if dstType is int32, pack output
  auto viewShape = y->GetViewShape();
  auto viewShapeDim = viewShape.GetDimNum();
  viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
  auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
  CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

  outTemp->SetDataType(DataType::DT_INT32);
  outTensor = outTemp;
  OP_LOGD("aclnnGroupQuant output real dtype is int4, pack to int32 to y.");

  return ACLNN_SUCCESS;
}

static bool CheckSpecialIOShape(const aclTensor *x, const aclTensor *y, const op::DataType &yDtype) {
  // output with data type int4/int32
  auto dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
  OP_CHECK(dimNum != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x not support scalar."), return false);

  OP_CHECK(y->GetViewShape().GetDimNum() == dimNum,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x dim num should be same with output y dim num."), return false);

  // other dims should be same
  for (int64_t i = 0; i < dimNum - 1; ++i) {
    OP_CHECK(x->GetViewShape().GetDim(i) == y->GetViewShape().GetDim(i),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "the dim(%ld) of x must be same with y, x is (%ld), y is (%ld).", i,
                     x->GetViewShape().GetDim(i), y->GetViewShape().GetDim(i)),
             return false);
  }

  // check last dim
  auto xLastDim = x->GetViewShape().GetDim(dimNum - 1);
  auto yLastDim = y->GetViewShape().GetDim(dimNum - 1);
  if (yDtype == op::DataType::DT_INT32) {
    OP_CHECK(xLastDim == yLastDim * INT4_NUMS_IN_INT32_SPACE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "if dstType is int32, the y last dim must be 1/8 of x last dim,"
                     " x last dim is (%ld), y last dim is (%ld)", xLastDim, yLastDim),
             return false);
  } else {
    if (yDtype == op::DataType::DT_INT4) {
      OP_CHECK(xLastDim % INT4_NUMS_IN_INT8_SPACE == 0,
               OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                       "if dstType is int4, x last dim must be divisible by 2, last dim is (%ld).", xLastDim),
               return false);
    }
    OP_CHECK(xLastDim == yLastDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "x last dim must be equal with y last dim. "
                     "x last dim is (%ld), y last dim is (%ld)",
                     xLastDim, yLastDim),
             return false);
  }
  return true;
}

static bool CheckDim(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                     const aclTensor *offsetOptional, const aclTensor *y) {
  // shape:
  //  x(S, H)
  //  scale(E，H)
  //  groupIndex(E)
  //  offset(1)
  if (offsetOptional != nullptr) {
    OP_CHECK(offsetOptional->GetViewShape().GetShapeSize() <= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "offsetOptional tensor should be scalar or tensor with shape is [1, ]."), return false);
  }

  auto xShape = x->GetViewShape();
  auto scaleShape = scale->GetViewShape();
  auto groupShape = groupIndex->GetViewShape();
  OP_CHECK(xShape.GetDimNum() == X_DIM_NUM,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor should be 2D."), return false);
  OP_CHECK(scaleShape.GetDimNum() == X_DIM_NUM,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale tensor should be 2D."), return false);
  OP_CHECK(groupShape.GetDimNum() == 1,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupIndex tensor should be 1D."), return false);

  int64_t xDim1 = xShape.GetDim(1);
  int64_t scaleDim0 = scaleShape.GetDim(0);
  int64_t scaleDim1 = scaleShape.GetDim(1);
  int64_t groupDim = groupShape.GetDim(0);
  OP_CHECK(xDim1 == scaleDim1,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim1 should be same with scale dim1."), return false);
  OP_CHECK(scaleDim0 == groupDim,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dim0 should be same with groupIndex dim0."), return false);

  return true;
}

static bool CheckShape(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                       const aclTensor *offsetOptional, const aclTensor *y) {
  // 维度限制校验
  OP_CHECK(CheckDim(x, scale, groupIndex, offsetOptional, y),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check output shape failed."), return false);

  auto outputDtype = y->GetDataType();
  if (outputDtype == op::DataType::DT_INT32 || outputDtype == op::DataType::DT_INT4) {
    OP_CHECK(CheckSpecialIOShape(x, y, outputDtype),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check output shape failed."), return false);
  } else {
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
  }
  return true;
}

static inline bool CheckNotNull(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                                const aclTensor *y) {
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(scale, return false);
  OP_CHECK_NULL(groupIndex, return false);
  OP_CHECK_NULL(y, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                            const aclTensor *offsetOptional, int32_t dstType, const aclTensor *y) {
  OP_CHECK_DTYPE_NOT_SUPPORT(x, GetDtypeSupportList(SOC_X_SUPPORT_DTYPES), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(scale, GetDtypeSupportList(SOC_X_SUPPORT_DTYPES), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(groupIndex, GetDtypeSupportList(SOC_GROUP_INDEX_SUPPORT_DTYPES), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(y, GetDtypeSupportList(SOC_Y_SUPPORT_DTYPES), return false);
  if (offsetOptional != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(offsetOptional, GetDtypeSupportList(SOC_X_SUPPORT_DTYPES), return false);
    OP_CHECK_DTYPE_NOT_SAME(scale, offsetOptional, return false);
  }

  OP_CHECK(static_cast<int32_t>(y->GetDataType()) == dstType,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dtype must be the same as dstType."),
           return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                               const aclTensor *offsetOptional, int32_t dstType, const aclTensor *y) {
  CHECK_RET(CheckNotNull(x, scale, groupIndex, y), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(x, scale, groupIndex, offsetOptional, dstType, y), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(x, scale, groupIndex, offsetOptional, y), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* scale,
                                            const aclTensor* groupIndex, const aclTensor* offsetOptional,
                                            int32_t dstType, aclTensor* y,
                                            uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnGroupQuant, DFX_IN(x, scale, groupIndex, offsetOptional, dstType), DFX_OUT(y));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(x, scale, groupIndex, offsetOptional, dstType, y);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (x->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // x如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
  CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto groupIndexContiguous = l0op::Contiguous(groupIndex, uniqueExecutor.get());
  CHECK_RET(groupIndexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* offsetContiguous = nullptr;
  if (offsetOptional != nullptr) {
    offsetContiguous = l0op::Contiguous(offsetOptional, uniqueExecutor.get());
    CHECK_RET(offsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    offsetContiguous = offsetOptional;
  }

  // 输出dtype支持int8/int4, int32是int4伪装，实际需要按照int4计算
  auto yDtype = y->GetDataType();
  if (yDtype == op::DataType::DT_INT32) {
    yDtype = op::DataType::DT_INT4;
    OP_LOGD("aclnnGroupQuant real output is int4.");
  }
  auto result = l0op::GroupQuant(selfContiguous, scaleContiguous, groupIndexContiguous, offsetContiguous, yDtype,
                                 uniqueExecutor.get());
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果输出是Int4，需要转成Int32，8个Int4输出拼成1个Int32
  const aclTensor *outTensor = result;
  ret = Int42Int32PackedTensor(result, outTensor, dstType, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto finalResult = l0op::ViewCopy(outTensor, y, uniqueExecutor.get());
  CHECK_RET(finalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupQuant);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
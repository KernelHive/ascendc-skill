/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ascend_quant.h"
#include "ascend_quant_v2_l0.h"
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
static constexpr int32_t DEFAULT_AXIS = -1;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT4};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType> SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType> SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType>& GetInDtypeSupportList() {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910B: {
      return X_DTYPE_SUPPORT_LIST_ASCEND910B;
    }
    case SocVersion::ASCEND310P:
      return X_DTYPE_SUPPORT_LIST_ASCEND310P;
    default: {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
      return EMPTY_LIST;
    }
  }
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList() {
  return OUT_DTYPE_SUPPORT_LIST;
}

static const std::initializer_list<DataType>& GetScaleOffsetDtypeSupportList() {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910B: {
      return SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND910B;
    }
    case SocVersion::ASCEND310P:
      return SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND310P;
    default: {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
      return EMPTY_LIST;
    }
  }
}

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *scale, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(scale, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeCompatible(const aclTensor *self, const aclTensor *scale) {
  if (self->GetDataType() == op::DataType::DT_FLOAT) {
    return true;
  } else if (self->GetDataType() == op::DataType::DT_FLOAT16) {
    if (scale->GetDataType() != op::DataType::DT_FLOAT16) {
      return false;
    }
  } else if (self->GetDataType() == op::DataType::DT_BF16) {
    if (scale->GetDataType() != op::DataType::DT_BF16) {
      return false;
    }
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *scale,
                            const aclTensor *offset, const aclTensor *out,
                            int32_t dstType) {
  OP_CHECK_DTYPE_NOT_SUPPORT(self, GetInDtypeSupportList(), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, GetOutDtypeSupportList(), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(scale, GetScaleOffsetDtypeSupportList(), return false);
  if (offset != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(offset, GetScaleOffsetDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SAME(scale, offset, return false);
  }
  if (!CheckDtypeCompatible(self, scale)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x:%s not compatible with scale:%s.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(scale->GetDataType()).GetString());
    return false;
  }

  if (static_cast<int32_t>(out->GetDataType()) != dstType) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dtype must be the same as dstType.");
    return false;
  }

  return true;
}

static bool CheckDim(const aclTensor *self, const aclTensor *scale, const aclTensor *offset) {
  int64_t scaleDimNum = static_cast<int64_t>(scale->GetViewShape().GetDimNum());
  int64_t scaleDim = scale->GetViewShape().GetDim(scaleDimNum - 1);
  if (offset != nullptr) {
    int64_t offsetDimNum = static_cast<int64_t>(offset->GetViewShape().GetDimNum());
    int64_t offsetDim = offset->GetViewShape().GetDim(offsetDimNum - 1);
    if (scaleDim != offsetDim) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim of scale must be the same as dim of offset");
      return false;
    }
  }

  int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dimNum == 0) {
    dimNum = 1;
  }
  int64_t lastDim = self->GetViewShape().GetDim(dimNum - 1);
  if (scaleDim != lastDim && scaleDim != 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "the last dim size(%ld) of x must be same as scale and offset(%ld).", lastDim, scaleDim);
    return false;
  }
  return true;
}

static bool CheckDimOne(const aclTensor *tensor) {
  int64_t dimNum = static_cast<int64_t>(tensor->GetViewShape().GetDimNum());
  for (int64_t i = 0; i < dimNum - 1; ++i) {
    int64_t dim = tensor->GetViewShape().GetDim(i);
    if (dim != 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "the dim(%ld) of scale or offset must be 1, but it is(%ld).", i, dim);
      return false;
    }
  }
  return true;
}

static bool CheckInt32OutputShape(const aclTensor *self, const aclTensor *out) {
  int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dimNum == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dtype int32 not support scalar.");
    return false;
  }
  int64_t dimInput = self->GetViewShape().GetDim(dimNum - 1);
  int64_t dimOutput = out->GetViewShape().GetDim(dimNum - 1);
  // check last dim
  bool inputIsNot8xOutput = 
    ((static_cast<uint64_t>(dimOutput) * INT4_NUMS_IN_INT32_SPACE) != static_cast<uint64_t>(dimInput)) ? true : false;
  if (inputIsNot8xOutput) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "the y last dim must be 1/8 of input last dim, input last dim is (%ld), output last dim is (%ld).",
            dimInput, dimOutput);
    return false;
  }
  // check others dim
  for (int64_t i = 0; i < dimNum - 1; ++i) {
    dimInput = self->GetViewShape().GetDim(i);
    dimOutput = out->GetViewShape().GetDim(i);
    if (dimInput != dimOutput) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "the dim(%ld) of input must be same with output, input is (%ld), output is (%ld).", i, dimInput, dimOutput);
      return false;
    }
  }

  return true;
}

static bool CheckInt4LastDim(const aclTensor *self) {
  int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dimNum == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dtype int4 not support scalar.");
    return false;
  }
  int64_t lastDimInput = self->GetViewShape().GetDim(dimNum - 1);
  bool lastDimOdd = ((static_cast<uint64_t>(lastDimInput) % INT4_NUMS_IN_INT8_SPACE) > 0) ? true : false;
  if (lastDimOdd) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "x last dim must be divisible by 2, last dim is (%ld).", lastDimInput);
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *x, const aclTensor *y,
                       const aclTensor *scale, const aclTensor *offset) {
  // self和out的shape必须一致
  if (y->GetDataType() != op::DataType::DT_INT32) {
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
  }

  // check input int32
  if (y->GetDataType() == op::DataType::DT_INT32 && !CheckInt32OutputShape(x, y)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check output shape failed.");
    return false;
  }

  // check input int4
  if (y->GetDataType() == op::DataType::DT_INT4 && !CheckInt4LastDim(x)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check input shape failed.");
    return false;
  }

  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);

  // check x scale offset last dim size
  CHECK_RET(CheckDim(x, scale, offset), false);

  if (!CheckDimOne(scale)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dim num must be 1. scale:%zu",
              scale->GetViewShape().GetDimNum());
      return false;
  }
  OP_CHECK_BROADCAST_WITH_SHAPE(scale, x->GetViewShape(), return false);

  if (offset != nullptr) {
    if (!CheckDimOne(offset)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offset dim num must be 1. offset:%zu",
              offset->GetViewShape().GetDimNum());
      return false;
    }
    OP_CHECK_BROADCAST_WITH_SHAPE(offset, x->GetViewShape(), return false);
  }

  return true;
}

static bool CheckRoundMode(const char *roundMode) {
  if (roundMode == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode cannot be empty.");
    return false;
  }
  const std::string mode = std::string(roundMode);
  if (mode != "round" && mode != "floor" && mode != "ceil" && mode != "trunc") {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check roundMode failed,roundMode not is 'round','floor','ceil','trunc'.");
    return false;
  }
  return true;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* out, const aclTensor*& outTensor,
                                          const int32_t& dstType, aclOpExecutor *executor) {
  if (dstType != op::DataType::DT_INT32) {
    OP_LOGD("AclnnAscendQuant output do not need to pack.");
    return ACLNN_SUCCESS;
  }
  // if dstType is int32, pack output
  auto viewShape = out->GetViewShape();
  auto viewShapeDim = viewShape.GetDimNum();
  viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
  auto outTemp = executor->CreateView(out, viewShape, out->GetViewOffset());
  CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

  outTemp->SetDataType(DataType::DT_INT32);
  outTensor = outTemp;
  OP_LOGD("AclnnAscendQuant output real dtype is int4, pack to int32 to out.");

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *scale,
                               const aclTensor *offset, const char *roundMode, int32_t dstType, const aclTensor *out) {
  CHECK_RET(CheckNotNull(self, scale, out), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(self, scale, offset, out, dstType), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(self, out, scale, offset), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckRoundMode(roundMode), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset,
                                             bool sqrtMode, const char *roundMode, int32_t dstType, const aclTensor *y,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnAscendQuant, DFX_IN(x, scale, offset, roundMode, sqrtMode, dstType), DFX_OUT(y));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(x, scale, offset, roundMode, dstType, y);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (x->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto scaleCast = l0op::Cast(scale, x->GetDataType(), uniqueExecutor.get());
  CHECK_RET(scaleCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* offsetCast = nullptr;
  if (offset != nullptr) {
    offsetCast = l0op::Cast(offset, x->GetDataType(), uniqueExecutor.get());
    CHECK_RET(offsetCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    offsetCast = offset;
  }

  // 输出dtype支持int8/int4, int32是int4伪装，实际需要按照int4计算
  auto yDtype = y->GetDataType();
  if (yDtype == op::DataType::DT_INT32) {
    yDtype = op::DataType::DT_INT4;
    OP_LOGD("AclnnAscendQuant real output is int4.");
  }
  auto ascendQuantV2Result = l0op::AscendQuantV2(selfContiguous, scaleCast, offsetCast,
                                                 sqrtMode, roundMode, yDtype, DEFAULT_AXIS, uniqueExecutor.get());
  CHECK_RET(ascendQuantV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果输出是Int4，需要转成Int32，8个Int4输出拼成1个Int32
  const aclTensor *outTensor = ascendQuantV2Result;
  ret = Int42Int32PackedTensor(ascendQuantV2Result, outTensor, dstType, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(outTensor, y, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAscendQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnAscendQuant);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
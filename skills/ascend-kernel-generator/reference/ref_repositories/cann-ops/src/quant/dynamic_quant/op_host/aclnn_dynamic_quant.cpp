/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_dynamic_quant.h"
#include "aclnn_dynamic_quant_v2.h"
#include "dynamic_quant_l0.h"
#include "dynamic_quant_v2_l0.h"
#include "fault_injection.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

#include <dlfcn.h>

#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr int64_t INT4_NUMS_IN_INT8_SPACE = 2;
using DtypeCheck = std::initializer_list<op::DataType>;

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                             op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> GROUP_INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                                   op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT4, op::DataType::DT_INT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> INPUT_DTYPE_EMPTY_LIST = {};

struct DynamicQuantParams {
  const aclTensor* x = nullptr;
  const aclTensor* smoothScales = nullptr;
  const aclTensor* groupIndex = nullptr;
  int64_t dstType;
  const aclTensor* y = nullptr;
  const aclTensor* scale = nullptr;
  const aclTensor* offset = nullptr;
};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910B: {
      return INPUT_DTYPE_SUPPORT_LIST;
    }
    default: {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
      return EMPTY_LIST;
    }
  }
}

static aclnnStatus CheckNotNull(const DynamicQuantParams& dynamicQuantParams) {
  CHECK_COND(dynamicQuantParams.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
  CHECK_COND(dynamicQuantParams.y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
  CHECK_COND(dynamicQuantParams.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale must not be nullptr.");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckOpDim(const op::Shape& shape1, const op::Shape& shape2, uint32_t shape1Dim,
                              uint32_t shape2Dim) {
  if (shape1Dim != shape2Dim) {
    return ACLNN_ERR_PARAM_INVALID;
  }

  for (uint32_t i = 0; i < shape1Dim; i++) {
    if (shape1.GetDim(i) != shape2.GetDim(i)) {
      return ACLNN_ERR_PARAM_INVALID;
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckInt32OutputShape(const op::Shape& shape1, const op::Shape& shape2, uint32_t shape1Dim,
                                         uint32_t shape2Dim) {
  if (shape1Dim != shape2Dim) {
    return ACLNN_ERR_PARAM_INVALID;
  }

  int64_t dimInput = shape1.GetDim(shape1Dim - 1);
  int64_t dimOutput = shape2.GetDim(shape2Dim - 1);
  // check last dim
  if (dimInput != (dimOutput * INT4_NUMS_IN_INT32_SPACE)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "the y last dim must be 1/8 of input last dim, input last dim is (%ld), output last dim is (%ld).",
            dimInput, dimOutput);
    return ACLNN_ERR_PARAM_INVALID;
  }

  for (int64_t i = 0; i < shape1Dim - 1; i++) {
    if (shape1.GetDim(i) != shape2.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "the dim(%ld) of input must be same with output, input is (%ld), output is (%ld).", i, dimInput,
              dimOutput);
      return ACLNN_ERR_PARAM_INVALID;
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const DynamicQuantParams& dynamicQuantParams) {
  auto xDimNum = dynamicQuantParams.x->GetViewShape().GetDimNum();
  int64_t xLastDimInput = dynamicQuantParams.x->GetViewShape().GetDim(xDimNum - 1);
  CHECK_COND(xDimNum > 1, ACLNN_ERR_PARAM_INVALID, "X DimNum should is less than 2");

  if (dynamicQuantParams.smoothScales) {
    if (dynamicQuantParams.groupIndex) {
      auto groupDimNum = dynamicQuantParams.groupIndex->GetViewShape().GetDimNum();
      CHECK_COND(groupDimNum == 1, ACLNN_ERR_PARAM_INVALID, "group_indexs DimNum not one.");
    } else {
      auto smoothDimNum = dynamicQuantParams.smoothScales->GetViewShape().GetDimNum();
      CHECK_COND(smoothDimNum == 1, ACLNN_ERR_PARAM_INVALID, "smooth_scales DimNum not one.");
      CHECK_COND(dynamicQuantParams.x->GetViewShape().GetDim(xDimNum - 1) == xLastDimInput, ACLNN_ERR_PARAM_INVALID,
                 "x and smooth_scales last dim is not equal.");
    }
  }

  auto yDtype = dynamicQuantParams.y->GetDataType();
  auto yDimNum = dynamicQuantParams.y->GetViewShape().GetDimNum();
  if (yDtype != op::DataType::DT_INT32) {
    CHECK_COND(CheckOpDim(dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.y->GetViewShape(), xDimNum,
                          yDimNum) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "x and y shape is inconsistency.");
  }

  // check y dtype int32
  if (yDtype == op::DataType::DT_INT32) {
    CHECK_RET(CheckInt32OutputShape(dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.y->GetViewShape(), xDimNum,
                                    yDimNum) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
  }

  // check y dtype int4
  if (yDtype == op::DataType::DT_INT4) {
    if (xLastDimInput % INT4_NUMS_IN_INT8_SPACE) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "if y dtype is int4, x last dim must be divisible by 2, last dim is (%ld).",
              xLastDimInput);
      return ACLNN_ERR_PARAM_INVALID;
    }
  }

  auto scaleNum = dynamicQuantParams.scale->GetViewShape().GetDimNum();
  CHECK_COND(CheckOpDim(dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.scale->GetViewShape(), xDimNum - 1,
                        scaleNum) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "scale shape is wrong, must be same as before x exclude the last dim.");

  if (dynamicQuantParams.offset) {
    auto offsetDimNum = dynamicQuantParams.offset->GetViewShape().GetDimNum();
    CHECK_COND(CheckOpDim(dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.offset->GetViewShape(), xDimNum - 1,
                          scaleNum) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "offset shape is wrong, must be same as before x exclude the last dim.");
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const DynamicQuantParams& dynamicQuantParams) {
  // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
  OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.x, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
  if (dynamicQuantParams.smoothScales) {
    // 检查smooth的数据类型是否在add算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.smoothScales, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(dynamicQuantParams.x->GetDataType() == dynamicQuantParams.smoothScales->GetDataType(),
               ACLNN_ERR_PARAM_INVALID, "x and smooth_scales dtype not equal.");
    if (dynamicQuantParams.groupIndex) {
      OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.groupIndex, GROUP_INDEX_DTYPE_SUPPORT_LIST,
                                 return ACLNN_ERR_PARAM_INVALID);
    }
  }
  OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.y, OUTPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
  OP_CHECK_DTYPE_NOT_MATCH(dynamicQuantParams.scale, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
  if (dynamicQuantParams.offset) {
    OP_CHECK_DTYPE_NOT_MATCH(dynamicQuantParams.offset, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
  }
  return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParams(DynamicQuantParams& dynamicQuantParams) {
  CHECK_COND(CheckNotNull(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  CHECK_COND(CheckDtype(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
  CHECK_RET(CheckShape(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* out, const aclTensor*& outTensor, aclOpExecutor* executor) {
  auto viewShape = out->GetViewShape();
  auto viewShapeDim = viewShape.GetDimNum();
  viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
  auto outTemp = executor->CreateView(out, viewShape, out->GetViewOffset());
  CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
  outTemp->SetDataType(DataType::DT_INT32);
  outTensor = outTemp;
  return ACLNN_SUCCESS;
}

aclnnStatus InputsContiguous(const aclTensor* tensor, const aclTensor*& reformatedTensor, const std::string& tensorName,
                             aclOpExecutor* executor) {
  if (tensor == nullptr) {
    return ACLNN_SUCCESS;
  }
  reformatedTensor = l0op::Contiguous(tensor, executor);
  CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s Contiguous failed.", tensorName.c_str());
  return ACLNN_SUCCESS;
}

aclnnStatus GetDynamicQuantResultByL0Api(DynamicQuantParams& dynamicQuantParams, uint64_t* workspaceSize,
                                         aclOpExecutor** executor) {
#ifdef CFG_BUILD_DEBUG
  // 获取环境变量
  auto faultInjectionEnv = getenv("FOR_FAULT_INJECTION");
  int32_t faultInjectionFlag = 0;
  if (faultInjectionEnv != nullptr) {
    faultInjectionFlag = std::atoi(faultInjectionEnv);
  }
  if (faultInjectionFlag) {
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(dynamicQuantParams.x, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(dynamicQuantParams.smoothScales, return ACLNN_ERR_INNER_NULLPTR);

    auto xContiguous = l0op::Contiguous(dynamicQuantParams.x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto smoothScalesOptionalContiguous = l0op::Contiguous(dynamicQuantParams.smoothScales, uniqueExecutor.get());
    CHECK_RET(smoothScalesOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用FaultInjection算子kernel
    auto out = uniqueExecutor->AllocTensor(dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.x->GetDataType());
    auto opOut = l0op::FaultInjection(xContiguous, smoothScalesOptionalContiguous, out, uniqueExecutor.get());
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
#endif
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  aclnnStatus ret = CheckParams(dynamicQuantParams);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (dynamicQuantParams.x->IsEmpty() || dynamicQuantParams.y->IsEmpty() || dynamicQuantParams.scale->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  const aclTensor* x = nullptr;
  const aclTensor* smoothScales = nullptr;
  const aclTensor* groupIndex = nullptr;
  const aclTensor* outputTensor = nullptr;
  const aclTensor* y = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  ret = InputsContiguous(dynamicQuantParams.x, x, "x", uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  ret = InputsContiguous(dynamicQuantParams.smoothScales, smoothScales, "smooth_scales", uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  ret = InputsContiguous(dynamicQuantParams.groupIndex, groupIndex, "group_index", uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 输出dtype支持int8/int4, int32是int4伪装，实际需要按照int4计算
  auto yDtype = dynamicQuantParams.y->GetDataType();
  if (yDtype == op::DataType::DT_INT32) {
    yDtype = op::DataType::DT_INT4;
    OP_LOGD("AclnnDynamicQuant real output is int4.");
  }

  bool isSymmetrical = dynamicQuantParams.offset == nullptr ? true : false;
  if (isSymmetrical) {
    auto dynamicQuantResult = l0op::DynamicQuant(x, smoothScales, groupIndex, yDtype, uniqueExecutor.get());
    y = std::get<0>(dynamicQuantResult);
    outputTensor = std::get<0>(dynamicQuantResult);
    scale = std::get<1>(dynamicQuantResult);
    CHECK_RET(y != nullptr && scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    auto dynamicQuantV2Result = l0op::DynamicQuantV2(x, smoothScales, groupIndex, yDtype, uniqueExecutor.get());
    y = std::get<0>(dynamicQuantV2Result);
    outputTensor = std::get<0>(dynamicQuantV2Result);
    scale = std::get<1>(dynamicQuantV2Result);
    offset = std::get<2>(dynamicQuantV2Result);  // 2 is offset index
    CHECK_RET(y != nullptr && scale != nullptr && offset != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyOffsetResult = l0op::ViewCopy(offset, dynamicQuantParams.offset, uniqueExecutor.get());
    CHECK_RET(viewCopyOffsetResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 针对pta调用方式，由于pta不支持int4，故当输出是int4时，pta那边会将dstType转换成DT_INT32
  if (dynamicQuantParams.dstType == op::DataType::DT_INT32) {
    OP_LOGD("dynamicQuantParams.dstType is DT_INT32");
    // 如果输出是Int4，需要转成Int32，8个Int4输出拼成1个Int32
    ret = Int42Int32PackedTensor(outputTensor, y, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
  }

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyYResult = l0op::ViewCopy(y, dynamicQuantParams.y, uniqueExecutor.get());
  CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyScaleResult = l0op::ViewCopy(scale, dynamicQuantParams.scale, uniqueExecutor.get());
  CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDynamicQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* smoothScalesOptional,
                                              const aclTensor* yOut, const aclTensor* scaleOut, uint64_t* workspaceSize,
                                              aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnDynamicQuant, DFX_IN(x, smoothScalesOptional), DFX_OUT(yOut, scaleOut));
  DynamicQuantParams dynamicQuantParams{x, smoothScalesOptional, nullptr, 2, yOut, scaleOut, nullptr};
  return GetDynamicQuantResultByL0Api(dynamicQuantParams, workspaceSize, executor);
}

aclnnStatus aclnnDynamicQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDynamicQuant);
  auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER, "This is an error in DynamicQuant launch aicore");
    return ACLNN_ERR_INNER;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor* x, const aclTensor* smoothScalesOptional,
                                                const aclTensor* groupIndexOptional, int64_t dstType,
                                                const aclTensor* yOut, const aclTensor* scaleOut,
                                                const aclTensor* offsetOut, uint64_t* workspaceSize,
                                                aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnDynamicQuantV2, DFX_IN(x, smoothScalesOptional, groupIndexOptional, dstType),
                 DFX_OUT(yOut, scaleOut, offsetOut));
  DynamicQuantParams dynamicQuantParams{x, smoothScalesOptional, groupIndexOptional, dstType, yOut, scaleOut,
                                        offsetOut};
  return GetDynamicQuantResultByL0Api(dynamicQuantParams, workspaceSize, executor);
}

aclnnStatus aclnnDynamicQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDynamicQuantV2);
  auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER, "This is an error in DynamicQuant launch aicore");
    return ACLNN_ERR_INNER;
  }
  return ACLNN_SUCCESS;
}
#ifdef __cplusplus
}
#endif
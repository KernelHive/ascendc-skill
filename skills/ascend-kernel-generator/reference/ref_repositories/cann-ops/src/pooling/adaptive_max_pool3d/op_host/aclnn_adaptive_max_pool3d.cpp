/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "adaptive_max_pool3d_l0.h"
#include "max_pool3d_with_argmax_v2_l0.h"
#include "unsqueeze_l0.h"
#include "squeeze_l0.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "reduce_mean_l0.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_adaptive_max_pool3d.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t INDEX_DIM0 = 0;
static const int64_t INDEX_DIM1 = 1;
static const int64_t INDEX_DIM2 = 2;
static const int64_t INDEX_DIM3 = 3;
static const int64_t INDEX_DIM4 = 4;
static const int64_t CDHW_SHAPE_SIZE = 4;
static const int64_t NCDHW_SHAPE_SIZE = 5;
static const int64_t OUTPUT_SIZE_LIMIT = 3;
static bool isSelf4D = false;
static const std::initializer_list<op::DataType> dtypeSupportList = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> indicesSupportList = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> nullSupportList = {};
static inline bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out, const aclTensor* indices) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

//校验输入Tensor维度，AdaptiveMaxPool3d支持4维或5维
static bool CheckInputOutputDims(const aclTensor* self, const aclTensor* out, const aclTensor* indices) {
    auto inputShape = self->GetViewShape();
    auto outputShape = out->GetViewShape();
    auto indicesShape = indices->GetViewShape();
    size_t inputDimNum = inputShape.GetDimNum();
    size_t outputDimNum = outputShape.GetDimNum();
    size_t indicesDimNum = indicesShape.GetDimNum();
    if (inputDimNum != NCDHW_SHAPE_SIZE && 
        inputDimNum != CDHW_SHAPE_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dims %zu should be 5 or 4.", inputDimNum);
        return false;
    }
    if (outputDimNum != NCDHW_SHAPE_SIZE && 
        outputDimNum != CDHW_SHAPE_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dims %zu should be 5 or 4.", outputDimNum);
        return false;
    }
    if (indicesDimNum != NCDHW_SHAPE_SIZE && 
        indicesDimNum != CDHW_SHAPE_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Indices dims %zu should be 5 or 4.", indicesDimNum);
        return false;
    }
    return true;
}

static inline const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return dtypeSupportList;
  } 
  return nullSupportList;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out, const aclTensor* indices) {
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, indicesSupportList, return false);
  return true;
}

//校验输入输出格式
static inline bool CheckFormat(const aclTensor* self, const aclTensor* out) {
  if (self->GetStorageFormat() != ge::FORMAT_NCDHW && self->GetStorageFormat() != ge::FORMAT_NCHW) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input should be NCDHW or NCHW.");
    return false;
  }
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and out should be equal, input [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }

  return true;
}

static bool CheckOutShape(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out, const aclTensor* indices) {
  uint64_t size = outputSize->Size();
  if (size != OUTPUT_SIZE_LIMIT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputSize length should be 3.");
    return false;
  }
  auto selfShape = self->GetViewShape();
  auto outputShape = out->GetViewShape();
  auto indicesShape = indices->GetViewShape();
  size_t outputDimNum = outputShape.GetDimNum();
  size_t offset = outputDimNum == CDHW_SHAPE_SIZE ? 1 : 2;
  // 检查输入self shape与output shape的前2维是否一致
  for (size_t i = 0; i < offset; ++i) {
    if (selfShape[i] != outputShape[i]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The [%lu] dim of outShape value should match the selfShape's.", i);
      return false;
    }
  }
  // 检查输出的output size与output tensor shape是否一致
  for (size_t i = 0; i < size; ++i) {
    if ((*outputSize)[i] != outputShape[i + offset]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outShape value should match the outputSize value.");
      return false;
    }
  }
  // 检查output shape 与indices shape是否一致
  for (size_t i = 0; i < size; ++i) {
    if (indicesShape[i] != outputShape[i]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The [%lu] dim of outShape value should match the indicesShape value.", i);
      return false;
    }
  }
  return true;
}

static inline bool checkDtypeSame(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

static inline bool checkPlatformValid() {
  return (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
}

static bool checkExceptiveValue(const aclTensor* self, const aclIntArray* outputSize) {
  // self 在非第一维度上的size小于1
  auto selfShape = self->GetViewShape();
  size_t selfDimNum = selfShape.GetDimNum();
  for (size_t i = INDEX_DIM1; i < selfDimNum; i++) {
    if (selfShape[i] < 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The [%lu] dim of selfShape is with size smaller than 1.", i);
      return false;
    }
  }
  
  // self d,h,w维度展开超过int32表达范围
  size_t offset =  selfDimNum == CDHW_SHAPE_SIZE ? INDEX_DIM1 : INDEX_DIM2;
  uint64_t dhwFlattenEleNum = ((uint64_t) selfShape[offset]) * ((uint64_t)selfShape[offset + INDEX_DIM1]) * ((uint64_t)selfShape[offset + INDEX_DIM2]);
  if (dhwFlattenEleNum > 0Xffffffff) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DHW flatten ele num should not exceed upper bound of uint32_t.");
    return false;
  }

  // outputSize中元素值小于0
  for (size_t i = 0; i < outputSize->Size(); i++) {
    if ((*outputSize)[i] < 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The [%lu] dim of outputSize is less than 1.", i);
      return false;
    }
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out, const aclTensor* indices) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, outputSize, out, indices), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 校验输入tensor维度/校验是否为空tensor
  CHECK_RET(CheckInputOutputDims(self, out, indices), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入输出数据类型
  CHECK_RET(CheckDtypeValid(self, out, indices), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);
  // 5. 校验outputSize是否合法
  CHECK_RET(CheckOutShape(self, outputSize, out, indices), ACLNN_ERR_PARAM_INVALID);
  // 6. 检查输入输出数据类型是否一致
  CHECK_RET(checkDtypeSame(self, out), ACLNN_ERR_PARAM_INVALID);
  // 7. 检查平台
  CHECK_RET(checkPlatformValid(), ACLNN_ERR_PARAM_INVALID);
  // 8. 检查输入异常值
  CHECK_RET(checkExceptiveValue(self, outputSize), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static inline const aclTensor *View4Das5D(const aclTensor *input, aclOpExecutor *executor)
{
    // CDHW -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 5D
    const int64_t dim = 0;  // Unsqueeze at dimension 0
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor *View5Das4D(const aclTensor *input, const op::Format &format, aclOpExecutor *executor)
{
    // NCDHW -> squeeze -> reformat -> CDHW
    // squeeze out into 4D
    const int64_t dim = 0;  // Squeeze out dimension 0
    auto squeezedInput = l0op::SqueezeNd(input, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to CDHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

//execute maxpool
static aclnnStatus DoMaxPool3D(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out,
                                           aclTensor* indices, aclOpExecutor* executor) {
    size_t kernelDSize = self->GetViewShape()[INDEX_DIM2] / (*outputSize)[INDEX_DIM0];
    size_t kernelHSize = self->GetViewShape()[INDEX_DIM3] / (*outputSize)[INDEX_DIM1];
    size_t kernelWSize = self->GetViewShape()[INDEX_DIM4] / (*outputSize)[INDEX_DIM2];
    std::vector<int64_t> kernelSizeArr = {};
    kernelSizeArr.push_back(kernelDSize);
    kernelSizeArr.push_back(kernelHSize);
    kernelSizeArr.push_back(kernelWSize);
    const aclIntArray *kernelSize = aclCreateIntArray(kernelSizeArr.data(), kernelSizeArr.size());
    const aclIntArray *stride = aclCreateIntArray(kernelSizeArr.data(), kernelSizeArr.size());
    std::vector<int64_t> paddingArr = {0, 0, 0};
    const aclIntArray *padding = aclCreateIntArray(paddingArr.data(), paddingArr.size());
    std::vector<int64_t> dilationArr = {1, 1, 1};
    const aclIntArray *dilation = aclCreateIntArray(dilationArr.data(), dilationArr.size());
    bool ceilMode = false;
    std::string dataFormat = "NCDHW";

    auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(self, kernelSize, stride, padding, dilation, ceilMode, dataFormat, executor);
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format &outFormat = out->GetStorageFormat();
    auto outResultSqueezed = isSelf4D ? View5Das4D(outResult, outFormat, executor) :
                                         l0op::ReFormat(outResult, outFormat, executor);
    CHECK_RET(outResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(outResultSqueezed, out), ACLNN_ERR_PARAM_INVALID);

    const op::Format &indicesFormat = indices->GetStorageFormat();
    auto indicesResultSqueezed = isSelf4D ? View5Das4D(indicesResult, indicesFormat, executor) :
                                         l0op::ReFormat(indicesResult, indicesFormat, executor);
    CHECK_RET(indicesResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(indicesResultSqueezed, indices), ACLNN_ERR_PARAM_INVALID);

    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewOutCopyResult = l0op::ViewCopy(outResultSqueezed, out, executor);
    CHECK_RET(viewOutCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the indices is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewIndicesCopyResult = l0op::ViewCopy(indicesResultSqueezed, indices, executor);
    CHECK_RET(viewIndicesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    // 释放资源
    aclDestroyIntArray(kernelSize);
    aclDestroyIntArray(stride);
    aclDestroyIntArray(padding);
    aclDestroyIntArray(dilation);
    return ACLNN_SUCCESS;
}

// input shape不能与output shape整除
static aclnnStatus DoAdaptiveMaxPool3D(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out,
                                       aclTensor* indices, aclOpExecutor* executor) {
    auto [outResult, indicesResult] = l0op::AdaptiveMaxPool3d(self, outputSize, executor);
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format &outFormat = out->GetStorageFormat();
    auto outResultSqueezed = isSelf4D ? View5Das4D(outResult, outFormat, executor) : outResult;
    CHECK_RET(outResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(outResultSqueezed, out), ACLNN_ERR_PARAM_INVALID);

    const op::Format &indicesFormat = indices->GetStorageFormat();
    auto indicesResultSqueezed = isSelf4D ? View5Das4D(indicesResult, indicesFormat, executor) : indicesResult;
    CHECK_RET(indicesResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(indicesResultSqueezed, indices), ACLNN_ERR_PARAM_INVALID);
    
    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewOutCopyResult = l0op::ViewCopy(outResultSqueezed, out, executor);
    CHECK_RET(viewOutCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the indices is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewIndicesCopyResult = l0op::ViewCopy(indicesResultSqueezed, indices, executor);
    CHECK_RET(viewIndicesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus SelectAdaptiveMaxPool3D(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out,
                                           aclTensor* indices, aclOpExecutor* executor) {
    int64_t dValueRemainder = self->GetViewShape()[INDEX_DIM2] % (*outputSize)[INDEX_DIM0];
    int64_t hValueRemainder = self->GetViewShape()[INDEX_DIM3] % (*outputSize)[INDEX_DIM1];
    int64_t wValueRemainder = self->GetViewShape()[INDEX_DIM4] % (*outputSize)[INDEX_DIM2];
    if(dValueRemainder == 0 && hValueRemainder == 0 && wValueRemainder == 0) {
        return DoMaxPool3D(self, outputSize, out, indices, executor);
    }                                        
    return DoAdaptiveMaxPool3D(self, outputSize, out, indices, executor);
}

aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out,
                                                   aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAdaptiveMaxPool3d, DFX_IN(self, outputSize), DFX_OUT(out, indices));
    //固定写法，参数检查
    auto ret = CheckParams(self, outputSize, out, indices);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    //固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty() || (*outputSize)[INDEX_DIM0] == 0 || (*outputSize)[INDEX_DIM1] == 0 || (*outputSize)[INDEX_DIM2] == 0){
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    isSelf4D = self->GetViewShape().GetDimNum() == CDHW_SHAPE_SIZE;

    auto selfUnsqueezed = isSelf4D ? View4Das5D(selfContiguous, uniqueExecutor.get()) : selfContiguous;
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret= SelectAdaptiveMaxPool3D(selfUnsqueezed, outputSize, out, indices, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    //固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveMaxPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnAdaptiveMaxPool3d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
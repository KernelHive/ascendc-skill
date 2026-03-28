/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aclnn_max_pool3d_with_argmax_backward.h"
#include "max_pool3d_grad_with_argmax_l0.h"
#include "unsqueeze.h"
#include "squeeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> NULL_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static const size_t CDHW_DIMS = 4;
static const size_t NCDHW_DIMS = 5;

const int W_DIM = -1;
const int H_DIM = -2;
const int D_DIM = -3;

const int64_t MAX_INT32 = 2147483647;

static bool CheckNotNullPtr(const aclTensor* gradOutput, const aclTensor* self,
                            const aclTensor* indices, const aclIntArray* kernelSize,
                            const aclIntArray* stride, const aclIntArray* padding,
                            const aclIntArray* dilation, aclTensor* gradInput) {
    // gradOutput, self, indices, kernelSize, stride, padding, dilation, gradInput cannot be null pointers
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static const std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910_93:
    case SocVersion::ASCEND910B: {
      return GRAD_DTYPE_SUPPORT_LIST;
    }
    case SocVersion::ASCEND910: {
      return NULL_DTYPE_SUPPORT_LIST;
    }
    default: {
      return NULL_DTYPE_SUPPORT_LIST;
    }
  }
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self,
                            const aclTensor* indices, const aclTensor* gradInput) {
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradOutput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);
    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self,
                        const aclTensor* indices, const aclTensor* gradInput) {
    if ((self->GetStorageFormat() != gradOutput->GetStorageFormat())
        || (self->GetStorageFormat() != gradInput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format of self and gradOutput and gradInput should be same, self[%s], gradOutput[%s], gradInput[%s].",
                op::ToString(self->GetStorageFormat()).GetString(),
                op::ToString(gradOutput->GetStorageFormat()).GetString(),
                op::ToString(gradInput->GetStorageFormat()).GetString());
        return false;
    }

    if (op::IsPrivateFormat(self->GetStorageFormat())
        || op::IsPrivateFormat(gradInput->GetStorageFormat())
        || op::IsPrivateFormat(gradOutput->GetStorageFormat())
        || op::IsPrivateFormat(indices->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCDHW or CDHW");
        return false;
    }

    if (self->GetViewShape().GetDimNum() != CDHW_DIMS 
        && self->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "4D or 5D tensor expected for self but got dim num:%zu", self->GetViewShape().GetDimNum());
        return false;
    }

    if (gradInput->GetViewShape().GetDimNum() != CDHW_DIMS
        && gradInput->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "4D or 5D tensor expected for gradInput but got dim num:%zu", gradInput->GetViewShape().GetDimNum());
        return false;
    }

    if (gradOutput->GetViewShape().GetDimNum() != CDHW_DIMS
        && gradOutput->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "4D or 5D tensor expected for gradOutput but got dim num:%zu", gradOutput->GetViewShape().GetDimNum());
        return false;
    }

    if (indices->GetViewShape().GetDimNum() != CDHW_DIMS
        && indices->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "4D or 5D tensor expected for indices but got dim num:%zu", indices->GetViewShape().GetDimNum());
        return false;
    }

    return true;
}

static bool CheckParamsValid(const aclIntArray *kernelSize, const aclIntArray *stride, 
                             const aclIntArray *padding, const aclIntArray *dilation) {
	// Kernel check
    const aclIntArray& kernelRef = *kernelSize;
    OP_CHECK(((kernelRef.Size() == 1) || (kernelRef.Size() == 3)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The length of kernelSize can be only 1 or 3"),
             return false);
    const int64_t kernelD = kernelRef[0];
    const int64_t kernelH = (kernelRef.Size() == 1) ? kernelD : kernelRef[1];
    const int64_t kernelW = (kernelRef.Size() == 1) ? kernelD : kernelRef[2];
    OP_CHECK(((kernelD > 0) && (kernelH > 0) && (kernelW > 0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The size of kernel should be greater than 0, but got kernelD:%ld, kernelH:%ld, kernelW:%ld",
                    kernelD, kernelH, kernelW),
             return false);

    // Stride check
    const aclIntArray& strideRef = *stride;
    OP_CHECK(((strideRef.Size() == 0) || (strideRef.Size() == 1) || (strideRef.Size() == 3)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The length of stride can be only 0, 1 or 3"),
             return false);
    const int64_t strideD = (strideRef.Size() == 0) ? kernelD : strideRef[0];
    const int64_t strideH = (strideRef.Size() == 0) ? kernelH : (strideRef.Size() == 1) ? strideD : strideRef[1];
    const int64_t strideW = (strideRef.Size() == 0) ? kernelW : (strideRef.Size() == 1) ? strideD : strideRef[2];
    OP_CHECK(((strideD > 0) && (strideH > 0) && (strideW > 0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The size of stride should be greater than 0, but got strideD:%ld, strideH:%ld, strideW:%ld",
                    strideD, strideH, strideW),
             return false);

    // Padding check
    const aclIntArray& paddingRef = *padding;
    OP_CHECK(((paddingRef.Size() == 1) || (paddingRef.Size() == 3)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The length of padding can be only 1 or 3"),
             return false);
    const int64_t paddingD = paddingRef[0];
    const int64_t paddingH = (paddingRef.Size() == 1) ? paddingD : paddingRef[1];
    const int64_t paddingW = (paddingRef.Size() == 1) ? paddingD : paddingRef[2];
    const int ratioKernelPad = 2;
    OP_CHECK(((paddingD >= 0) && (paddingH >= 0) && (paddingW >= 0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The size of padding should be greater than or equal to 0, but got paddingD:%ld, paddingH:%ld, paddingW:%ld",
                    paddingD, paddingH, paddingW),
             return false);
    OP_CHECK(((paddingD <= (kernelD / ratioKernelPad)) && (paddingH <= (kernelH / ratioKernelPad)) &&
              (paddingW <= (kernelW / ratioKernelPad))),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "padding should be smaller than or equal to half of kernel size,\
                    but got kernelD:%ld, kernelH:%ld, kernelW:%ld, paddingD:%ld, paddingH:%ld, paddingW:%ld",
                    kernelD, kernelH, kernelW, paddingD, paddingH, paddingW),
             return false);

    // Dilation check
    const aclIntArray& dilationRef = *dilation;
    OP_CHECK(((dilationRef.Size() == 1) || (dilationRef.Size() == 3)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The length of dilation can be only 1 or 3"),
             return false);
    const int64_t dilationD = dilationRef[0];
    const int64_t dilationH = (dilationRef.Size() == 1) ? dilationD : dilationRef[1];
    const int64_t dilationW = (dilationRef.Size() == 1) ? dilationD : dilationRef[2];
    OP_CHECK(((dilationD > 0) && (dilationH > 0) && (dilationW > 0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The size of dilation should be greater than 0, but got dilationD:%ld, dilationH:%ld, dilationW:%ld",
                    dilationD, dilationH, dilationW),
             return false);

    return true;
}

static bool CheckSelfShapeSupport(const aclTensor* self) {
    const auto& selfShape = self->GetViewShape();
    const auto& selfDimNum = selfShape.GetDimNum();
    const auto& selfDimW = selfShape.GetDim(selfDimNum + W_DIM);
    const auto& selfDimH = selfShape.GetDim(selfDimNum + H_DIM);
    const auto& selfDimD = selfShape.GetDim(selfDimNum + D_DIM);

    const int64_t selfSize = selfDimW * selfDimH * selfDimD;
    OP_CHECK((selfSize <= MAX_INT32),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The size of self should be less than or equal to 2^32 - 1, but got selfSize:%ld", selfSize),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self,
                               const aclTensor* indices, const aclIntArray* kernelSize,
                               const aclIntArray* stride, const aclIntArray* padding,
                               const aclIntArray* dilation, aclTensor* gradInput) {
    CHECK_RET(CheckNotNullPtr(gradOutput, self, indices, kernelSize, stride, padding, dilation, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParamsValid(kernelSize, stride, padding, dilation), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSelfShapeSupport(self), ACLNN_ERR_PARAM_INVALID);
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

aclnnStatus aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                             const aclTensor* indices, const aclIntArray* kernelSize,
                                                             const aclIntArray* stride, const aclIntArray* padding,
                                                             const aclIntArray* dilation, bool ceilMode,
                                                             aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnMaxPool3dWithArgmaxBackward, DFX_IN(gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode),
                   DFX_OUT(gradInput));
    OP_LOGD("MaxPool3DGradWithArgmax: getWorkspaceSize");
    
    // Create an OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Сhecking parameters
    auto ret = CheckParams(gradOutput, self, indices, kernelSize, stride, padding, dilation, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Check whether the tensor is empty (the operator does not support empty tensors)
    if (gradOutput->IsEmpty() || self->IsEmpty() || indices->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Convert the self, gradOutput, indices into consecutive tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If it's 4D, it needs to be expanded to 5D before calling the MaxPool3dGradWithArgMax interface.
    const bool isSelf4D = self->GetViewShape().GetDimNum() == CDHW_DIMS;

    auto selfUnsqueezed = isSelf4D ? View4Das5D(selfContiguous, uniqueExecutor.get()) :
                                     l0op::ReFormat(selfContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputUnsqueezed = isSelf4D ?
                                View4Das5D(gradOutputContiguous, uniqueExecutor.get()) :
                                l0op::ReFormat(gradOutputContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(gradOutputUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesUnsqueezed = isSelf4D ?
                             View4Das5D(indicesContiguous, uniqueExecutor.get()) :
                             l0op::ReFormat(indicesContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(indicesUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputResult = l0op::MaxPool3DGradWithArgmax(gradOutputUnsqueezed, selfUnsqueezed, indicesUnsqueezed, kernelSize, 
                                                         stride, padding, dilation, ceilMode, uniqueExecutor.get());
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    const op::Format &dstFormat = gradInput->GetStorageFormat();
    auto gradResultSqueezed = isSelf4D ? View5Das4D(gradInputResult, dstFormat, uniqueExecutor.get()) :
                                         l0op::ReFormat(gradInputResult, dstFormat, uniqueExecutor.get());
    CHECK_RET(gradResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(gradResultSqueezed, gradInput), ACLNN_ERR_PARAM_INVALID); 

    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewGradInputCopyResult = l0op::ViewCopy(gradResultSqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(viewGradInputCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPool3dWithArgmaxBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
	L2_DFX_PHASE_2(aclnnMaxPool3dWithArgmaxBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
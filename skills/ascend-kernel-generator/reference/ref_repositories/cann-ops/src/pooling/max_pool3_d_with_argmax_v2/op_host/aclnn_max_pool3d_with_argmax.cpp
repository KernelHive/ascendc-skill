/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_pool3d_with_argmax.h"
#include "max_pool3d_with_argmax_v2_l0.h"
#include "unsqueeze.h"
#include "squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
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

static const std::initializer_list<DataType> NULL_SUPPORT_LIST = {};
static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16,
                                                                       DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static const size_t CDHW_DIMS = 4;
static const size_t NCDHW_DIMS = 5;

static bool CheckNotNullPtr(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
                            const aclIntArray* padding, const aclIntArray* dilation, aclTensor* out,
                            aclTensor* indices) {
    // self, kernelSize, stride, padding, dilation, out, indices cannot be null pointers
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        return OUT_DTYPE_SUPPORT_LIST;
    } else {
        return NULL_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out, const aclTensor* indices) {
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out, const aclTensor* indices) {
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same, self [%s], out [%s].",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (self->GetStorageFormat() != indices->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and indices should be same, self [%s], indices [%s].",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(indices->GetStorageFormat()).GetString());
        return false;
    }
    
    // Only 4D or 5D input tensors are supported
    if (self->GetViewShape().GetDimNum() != CDHW_DIMS && self->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for inputs");
        return false;
    }

    return true;
}

static bool CheckParamsValid(const aclIntArray* kernelSize, const aclIntArray* stride, 
                             const aclIntArray* padding, const aclIntArray* dilation) {
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
                    "The value of kernel should be greater than 0, but got kernelD:%ld, kernelH:%ld, kernelW:%ld",
                    kernelD, kernelH, kernelW),
             return false);

    // Stride check
    const aclIntArray& strideRef = *stride;
    OP_CHECK(((strideRef.Size() == 0) || (strideRef.Size() == 1) || (strideRef.Size() == 3)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "The length of stride can be only 1 or 3"),
             return false);
    const int64_t strideD = (strideRef.Size() == 0) ? kernelD : strideRef[0];
    const int64_t strideH = (strideRef.Size() == 0) ? kernelH : ((strideRef.Size() == 1) ? strideD : strideRef[1]);
    const int64_t strideW = (strideRef.Size() == 0) ? kernelW : ((strideRef.Size() == 1) ? strideD : strideRef[2]);
    OP_CHECK(((strideD > 0) && (strideH > 0) && (strideW > 0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The value of stride should be greater than 0, but got strideD:%ld, strideH:%ld, strideW:%ld",
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
                    "The value of padding should be greater than or equal to 0, but got paddingD:%ld, paddingH:%ld, paddingW:%ld",
                    paddingD, paddingH, paddingW),
             return false);
    OP_CHECK(((paddingD <= (kernelD / ratioKernelPad)) && (paddingH <= (kernelH / ratioKernelPad)) &&
              (paddingW <= (kernelW / ratioKernelPad))),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Padding should be smaller than or equal to half of kernel size, "
                    "but got kernelD:%ld, kernelH:%ld, kernelW:%ld, paddingD:%ld, paddingH:%ld, paddingW:%ld",
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
    OP_CHECK(((dilationD == 1) && (dilationH == 1) && (dilationW == 1)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The value of dilation should be equal 1, but got dilationD:%ld, dilationH:%ld, dilationW:%ld",
                    dilationD, dilationH, dilationW),
             return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* kernelSize, 
                               const aclIntArray* stride, const aclIntArray* padding, 
                               const aclIntArray* dilation, aclTensor* out, aclTensor* indices) {
    CHECK_RET(CheckNotNullPtr(self, kernelSize, stride, padding, dilation, out, indices), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(self, out, indices), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(self, out, indices), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParamsValid(kernelSize, stride, padding, dilation), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckPlatform() {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        return true;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnMaxPool3dWithArgmax is not supported on this platform");
        return false;
    }
}

static inline const aclTensor *View4Das5D(const aclTensor *input, aclOpExecutor *executor)
{
    // NCHW -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 5D
    const int64_t dim = 0;  // Unsqueeze at dimension 0
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor *View5Das4D(const aclTensor *input, const op::Format &format, aclOpExecutor *executor)
{
    // NCDHW -> squeeze -> reformat -> NCHW
    // squeeze out into 4D
    const int64_t dim = 0;  // Squeeze out dimension 0
    auto squeezedInput = l0op::SqueezeNd(input, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize,
                                                     const aclIntArray* stride, const aclIntArray* padding,
                                                     const aclIntArray* dilation, bool ceilMode, aclTensor* out,
                                                     aclTensor* indices, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnMaxPool3dWithArgmax, DFX_IN(self, kernelSize, stride, padding, dilation, ceilMode), 
                   DFX_OUT(out, indices));
    OP_LOGD("aclnnMaxPool3dWithArgmax: getWorkspaceSize");
    
    // Create an OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    // Сhecking platform
    bool result = CheckPlatform();
    CHECK_RET(result == true, ACLNN_ERR_PARAM_INVALID);

    // Сhecking parameters
    auto ret = CheckParams(self, kernelSize, stride, padding, dilation, out, indices);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    std::string dataFormat = "NCDHW";

    // Check whether the tensor is empty (the operator does not support empty tensors)
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Convert the self into consecutive tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    // If it's 4D, it needs to be expanded to 5D before calling the MaxPool3DWithArgmaxV2Ncdhw interface.
    const bool isSelf4D = self->GetViewShape().GetDimNum() == CDHW_DIMS;

    auto selfUnsqueezed = isSelf4D ? View4Das5D(selfContiguous, uniqueExecutor.get()) :
                                     l0op::ReFormat(selfContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Returns a tuple containing out and indices
    auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(selfUnsqueezed, kernelSize, stride, padding, dilation,
                                                                       ceilMode, dataFormat, uniqueExecutor.get());
    
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    const op::Format &outFormat = out->GetStorageFormat();
    auto outResultSqueezed = isSelf4D ? View5Das4D(outResult, outFormat, uniqueExecutor.get()) : 
                                        l0op::ReFormat(outResult, outFormat, uniqueExecutor.get());
    CHECK_RET(outResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format &indicesFormat = indices->GetStorageFormat();
    auto indicesResultSqueezed = isSelf4D ? View5Das4D(indicesResult, indicesFormat, uniqueExecutor.get()) : 
                                            l0op::ReFormat(indicesResult, indicesFormat, uniqueExecutor.get());
    CHECK_RET(indicesResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CheckReduceOutShape(outResultSqueezed, out), ACLNN_ERR_PARAM_INVALID); 
    CHECK_RET(CheckReduceOutShape(indicesResultSqueezed, indices), ACLNN_ERR_PARAM_INVALID); 
    
    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewOutCopyResult = l0op::ViewCopy(outResultSqueezed, out, uniqueExecutor.get());
    CHECK_RET(viewOutCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the indices is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewIndicesCopyResult = l0op::ViewCopy(indicesResultSqueezed, indices, uniqueExecutor.get());
    CHECK_RET(viewIndicesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPool3dWithArgmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnMaxPool3dWithArgmax);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
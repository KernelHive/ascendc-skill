/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_avgpool3d_backward.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/framework_op.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "avgpool3d_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
static int64_t NCDHW_SHAPE = 5;
static int64_t CDHW_SHAPE = 4;
static size_t MIN_ARRAY_DIM_SIZE = 1;
static size_t MAX_ARRAY_DIM_SIZE = 3;
static int64_t DEPTH_IDX = 2;
static int64_t HEIGHT_IDX = 3;
static int64_t WEIGHT_IDX = 4;
static const std::string NDHWC_FORMAT = "NDHWC";
static const std::string NCDHW_FORMAT = "NCDHW";

static const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<DataType> INT_ARRAY_SUPPORT_LIST = {
    op::DataType::DT_INT32};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST= {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self,
                         const aclIntArray* kernelSize, const aclIntArray* stride,
                         const aclIntArray* padding, const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor*self, const aclTensor* out)
{
    const std::initializer_list<DataType> gradDtypeSupportList = GRAD_DTYPE_SUPPORT_LIST;
    const std::initializer_list<DataType> outDtypeSupportList = OUT_DTYPE_SUPPORT_LIST;

    // 检查grad数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, gradDtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, gradDtypeSupportList, return false);
    // 检查out的数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(out, outDtypeSupportList, return false);
    return true;
}

static bool CheckDimVlid(const aclTensor* gradOutput, const aclTensor* self,
                        const aclIntArray* kernelSize, const aclIntArray* stride,
                        const aclIntArray* padding, const aclTensor* out)
{
    auto gradShape = gradOutput->GetViewShape();
    auto outShape = out->GetViewShape();
    auto inputShape = self->GetViewShape();
    auto kSize = kernelSize->Size();
    auto stridesSize = stride->Size();
    auto padsSize = padding->Size();
    auto gradShapeDimNum = static_cast<int64_t>(gradShape.GetDimNum());
    if (gradShapeDimNum != CDHW_SHAPE && gradShapeDimNum != NCDHW_SHAPE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad dim [%zu] should be 4 or 5", gradShape.GetDimNum());
        return false;
    }
    if (inputShape.GetDimNum() != gradShape.GetDimNum() || inputShape.GetDimNum() != outShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dim num [%zu] should be equal to grad dim num [%zu] and out dim num [%zu]",
            inputShape.GetDimNum(), gradShape.GetDimNum(), outShape.GetDimNum());
        return false;
    }
    for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
        if (inputShape.GetDim(i) != outShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dim [%zu] is [%zu] should be equal to out dim [%zu]",
                i, inputShape.GetDim(i), outShape.GetDim(i));
        return false;
        }
    }

    if (kSize != MIN_ARRAY_DIM_SIZE && kSize != MAX_ARRAY_DIM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernel size [%lu] must be 1 or 3", kSize);
        return false;
    }
    if (padsSize != MIN_ARRAY_DIM_SIZE && padsSize != MAX_ARRAY_DIM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding size [%lu] must be 1 or 3", padsSize);
        return false;
    }
    if (stridesSize != 0 && stridesSize != MIN_ARRAY_DIM_SIZE && stridesSize != MAX_ARRAY_DIM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride size [%lu] must be 0 or 1 or 3", stridesSize);
        return false;
    }

    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* out)
{
    if (gradOutput->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and out should be equal, input [%s], out [%s].",
            op::ToString(gradOutput->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOutput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only don't support private format.");
        return false;
    }
    return true;
}

static aclnnStatus CheckArrayData(const aclIntArray* kernelSize, const aclIntArray* stride,
                                  const aclIntArray* padding)
{
    for (uint64_t i = 0; i < kernelSize->Size(); i++) {
        auto size = (*kernelSize)[i];
        if (size <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize [%lu] data [%li] is less than or equal to 0",
                    i, size);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    for (int i = 0; i < stride->Size(); i++) {
        auto size = (*stride)[i];
        if (size <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride [%i] data [%li] is less than or equal to 0",
                    i, size);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (kernelSize->Size() != padding->Size()) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (int i = 0; i < kernelSize->Size(); i++) {
        auto halfKsize = (*kernelSize)[i] / 2;
        auto padSize = (*padding)[i];
        if (halfKsize < padSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding [%i] data [%li] should less than kernelSize / 2 [%lu]",
                    i, padSize, halfKsize);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (padSize < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding [%i] data [%li] is less than 0",
                    i, padSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self,
                               const aclIntArray* kernelSize, const aclIntArray* stride,
                               const aclIntArray* padding, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, kernelSize, stride, padding, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查数据类型是否满足要求
    CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查维度
    CHECK_RET(CheckDimVlid(gradOutput, self, kernelSize, stride, padding, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查数据格式
    CHECK_RET(CheckFormat(gradOutput, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclIntArray* GetPerm(const aclTensor* gradOutput, const int startIdx, aclOpExecutor* executor) {
    auto gradShape = gradOutput->GetViewShape();
    auto dimSize = gradShape.GetDimNum();
    std::vector<int64_t> valuePerm(dimSize);
    // NCDHW -> DHWNC
    for (size_t i = startIdx; i < dimSize; i++) {
        valuePerm[i - startIdx] = i;
    }

    auto idx = dimSize - startIdx + 1;
    for (uint64_t i = 0; i < startIdx; i++) {
        valuePerm[idx + i] = i;
    }
    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

static std::tuple<const aclIntArray*, const aclIntArray*, const aclIntArray*> FillIntArray(
    const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, aclOpExecutor* executor)
{
    std::vector<int64_t> kvec(MAX_ARRAY_DIM_SIZE, 0);
    std::vector<int64_t> svec(MAX_ARRAY_DIM_SIZE, 0);
    std::vector<int64_t> pvec(MAX_ARRAY_DIM_SIZE, 0);
    auto kSize = kernelSize->Size();
    auto stridesSize = stride->Size();
    auto padsSize = padding->Size();
    for (size_t i = 0; i < MAX_ARRAY_DIM_SIZE; i++) {
        kvec[i] = kSize == 1 ? (*kernelSize)[0] : (*kernelSize)[i];
        pvec[i] = padsSize == 1 ? (*padding)[0] : (*padding)[i];
        svec[i] = stridesSize == 0 ? kvec[i] : stridesSize == 1 ? (*stride)[0] : (*stride)[i];
    }
    const aclIntArray* kernelFill = executor->AllocIntArray(kvec.data(), MAX_ARRAY_DIM_SIZE);
    const aclIntArray* stridesFill = executor->AllocIntArray(svec.data(), MAX_ARRAY_DIM_SIZE);
    const aclIntArray* padsFill = executor->AllocIntArray(pvec.data(), MAX_ARRAY_DIM_SIZE);
    return std::tuple<const aclIntArray*, const aclIntArray*, const aclIntArray*>(kernelFill, stridesFill, padsFill);
}

const aclTensor* TransGrad2CDHW(const aclTensor* gradOutput, aclOpExecutor* executor)
{
    auto gradShape = gradOutput->GetViewShape();
    auto gradDimNum = gradShape.GetDimNum();
    int64_t num = 1;
    int64_t mergeNC = gradDimNum == CDHW_SHAPE ?
                      gradShape.GetDim(0) : gradShape.GetDim(0) * gradShape.GetDim(1);
    int64_t depth = gradShape.GetDim(gradDimNum - 3);
    int64_t height = gradShape.GetDim(gradDimNum - 2);
    int64_t weight = gradShape.GetDim(gradDimNum - 1);
    FVector<int64_t> reshapeVec = {num, mergeNC, depth, height, weight};
    aclIntArray* reshapeArray = executor->AllocIntArray(reshapeVec.data(), reshapeVec.size());
    CHECK_RET(reshapeArray != nullptr, nullptr);
    auto grad5hd = l0op::Reshape(gradOutput, reshapeArray, executor);
    return grad5hd;
}

const aclTensor* TransOut2OrigShape(const aclTensor* result, const aclTensor* gradOutput, aclOpExecutor* executor)
{
    auto gradShape = gradOutput->GetViewShape();
    auto resShape = result->GetViewShape();
    auto gradDimNum = gradShape.GetDimNum();
    auto resDimNum = resShape.GetDimNum();

    int64_t num = gradDimNum == CDHW_SHAPE ? 1 : gradShape.GetDim(0);
    int64_t channel = gradDimNum == CDHW_SHAPE ? gradShape.GetDim(0) : gradShape.GetDim(1);
    int64_t depth = resShape.GetDim(resDimNum - 3);
    int64_t height = resShape.GetDim(resDimNum - 2);
    int64_t weight = resShape.GetDim(resDimNum - 1);
    FVector<int64_t> cdhwVec = {channel, depth, height, weight};
    FVector<int64_t> ncdhwVec = {num, channel, depth, height, weight};
    FVector<int64_t> reshapeVec = gradDimNum == CDHW_SHAPE ? cdhwVec : ncdhwVec;
    aclIntArray* reshapeArray = executor->AllocIntArray(reshapeVec.data(), reshapeVec.size());
    CHECK_RET(reshapeArray != nullptr, nullptr);
    auto res = l0op::Reshape(result, reshapeArray, executor);
    return res;
}

const aclTensor* CopyShape2OneDim(const aclTensor* self, aclOpExecutor* executor)
{
    FVector<int> shapeVec;
    for (int i = 0; i < self->GetViewShape().GetDimNum(); i++) {
        shapeVec.push_back(self->GetViewShape().GetDim(i));
    }
    auto shapeCopy = executor->ConvertToTensor(shapeVec.data(), shapeVec.size(), DataType::DT_INT32);
    return shapeCopy;
}

aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                   const aclIntArray* kernelSize, const aclIntArray* stride,
                                                   const aclIntArray* padding, bool ceilMode, bool countIncludePad,
                                                   int64_t divisorOverride, aclTensor* output,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAvgPool3dBackward,
                   DFX_IN(gradOutput, self, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride),
                   DFX_OUT(output));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, kernelSize, stride, padding, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 先将tensor转为连续性
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR); 

    // intarray
    // 维度补齐
    auto intArrayFill = FillIntArray(kernelSize, stride, padding, uniqueExecutor.get());
    const aclIntArray* ksizeFill = std::get<0>(intArrayFill);
    const aclIntArray* stridesFill = std::get<1>(intArrayFill);
    const aclIntArray* padsFill = std::get<2>(intArrayFill);
    CHECK_RET(ksizeFill != nullptr && stridesFill != nullptr && padsFill != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    // 校验数值
    auto arrayRet = CheckArrayData(ksizeFill, stridesFill, padsFill);
    CHECK_RET(arrayRet == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // gradoutput转成5维
    auto gradCDHW = TransGrad2CDHW(gradOutputContiguous, uniqueExecutor.get());
    CHECK_RET(gradCDHW != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool isOnlyT = ((*ksizeFill)[1] == (*stridesFill)[1] && (*ksizeFill)[1] == 1) &&
                    ((*ksizeFill)[2] == (*stridesFill)[2] && (*ksizeFill)[2] == 1);           
    
    const aclTensor* gradTrans = gradCDHW;
    if (!isOnlyT) {
        // transdata to dhwnc
        std::vector<int64_t> transPreDim = {0, 2, 3, 4, 1};
        auto permPre = uniqueExecutor.get()->AllocIntArray(transPreDim.data(), transPreDim.size());
        CHECK_RET(permPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradTrans = l0op::Transpose(gradCDHW, permPre, uniqueExecutor.get());
        CHECK_RET(gradTrans != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* shapeOrigInput = CopyShape2OneDim(selfContiguous, uniqueExecutor.get());
    CHECK_RET(shapeOrigInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用l0op
    std::string dataFormat = isOnlyT ? NCDHW_FORMAT : NDHWC_FORMAT;
    auto avgPool3dBackwardResult = l0op::AvgPool3DGrad(selfContiguous, shapeOrigInput, gradTrans, ksizeFill,
                                                  stridesFill, padsFill, ceilMode, countIncludePad,
                                                  divisorOverride, dataFormat, uniqueExecutor.get());
    CHECK_RET(avgPool3dBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (!isOnlyT) {
        // transdata to ncdhw
        std::vector<int64_t> transAfterDim = {0, 4, 1, 2, 3};
        auto permAfter = uniqueExecutor.get()->AllocIntArray(transAfterDim.data(), transAfterDim.size());
        CHECK_RET(permAfter != nullptr, ACLNN_ERR_INNER_NULLPTR);
        avgPool3dBackwardResult = l0op::Transpose(avgPool3dBackwardResult, permAfter, uniqueExecutor.get());
        CHECK_RET(avgPool3dBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    avgPool3dBackwardResult = TransOut2OrigShape(avgPool3dBackwardResult, gradOutput, uniqueExecutor.get());
    CHECK_RET(avgPool3dBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // viewcopy
    auto viewCopyResult = l0op::ViewCopy(avgPool3dBackwardResult, output, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvgPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAvgPool3dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

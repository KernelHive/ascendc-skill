/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_avgpool3d.h"

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

#include "avgpool3d.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

static const int64_t DIM_NUM_5D = 5;
static const int64_t DIM_NUM_4D = 4;

static const int64_t DIM0 = 0;
static const int64_t DIM1 = 1;
static const int64_t DIM2 = 2;
static const int64_t DIM3 = 3;
static const int64_t DIM4 = 4;

static const int64_t KERNEL_SIZE = 3;
static const int64_t PAD_SIZE = 3;
static const int64_t STRIDE_SIZE = 3;

#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNullPtr(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
    const aclIntArray *padding, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernel, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16 };

static const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    return DTYPE_SUPPORT_LIST;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    // 根据API定义，需要列出所能支持的所有dtype
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查self的数据类型是否和out一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out)
{
    // 检查self和out的format是否一致
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and out format should be same. Actual: self is [%s], out is [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support format [%s].",
            op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

// 计算经过avgpool3d后的shape的d和h和w（n,c与input一致，不用计算）
static inline int64_t PoolingOutShape(const int64_t inputEdgeLen, const int64_t kernEdgeLen, const int64_t padLeftLen,
    const int64_t strideEdgeLen, const bool ceilMode)
{
    float outputTempSize = 1.0f;
    if (strideEdgeLen > 0) {
        // 2: double
        outputTempSize = 1.0f * (inputEdgeLen + padLeftLen * 2 - kernEdgeLen + strideEdgeLen) / strideEdgeLen;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of stride should be greater than 0, but got strideEdgeLen:%ld", strideEdgeLen);
        return -1;
    }
    if (ceilMode) {
        // The sliding window starting from the right filled area will be ignored
        int64_t outHeight = static_cast<int64_t>(std::ceil(outputTempSize));
        if ((outHeight - 1) * strideEdgeLen >= inputEdgeLen + padLeftLen) {
            --outHeight;
        }
        return outHeight;
    } else {
        return static_cast<int64_t>(std::floor(outputTempSize));
    }
}

static bool CheckAttrValue(const int64_t kernel, const int64_t stride, const int64_t pad, const int64_t input)
{
    if (kernel <= 0 || kernel > input) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernel value(%ld) is invaild, must be (0, %ld].", kernel, input);
        return false;
    }

    if (stride <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride value(%ld) is less than or equal to 0.", stride);
        return false;
    }

    if (!(pad >= 0 && pad <= kernel / 2)) { // 2: double
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding value(%ld) is invaild, must be [0, kernel/2].", pad);
        return false;
    }

    return true;
}

static bool CheckOutputShape(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
    const aclIntArray *padding, const bool ceilMode, const aclTensor *out)
{
    const bool is4d = self->GetViewShape().GetDimNum() == DIM_NUM_4D;
    const int64_t nBatch = is4d ? 1 : self->GetViewShape().GetDim(0);
    const int64_t nInputPlane = is4d ? self->GetViewShape().GetDim(0) : self->GetViewShape().GetDim(DIM1);
    const int64_t depth = is4d ? self->GetViewShape().GetDim(DIM1) : self->GetViewShape().GetDim(DIM2);
    const int64_t height = is4d ? self->GetViewShape().GetDim(DIM2) : self->GetViewShape().GetDim(DIM3);
    const int64_t width = is4d ? self->GetViewShape().GetDim(DIM3) : self->GetViewShape().GetDim(DIM4);

    const int64_t kD = (*kernel)[0];
    const int64_t kH = kernel->Size() == 1 ? kD : (*kernel)[DIM1];
    const int64_t kW = kernel->Size() == 1 ? kD : (*kernel)[DIM2];

    const int64_t sD = stride->Size() == 0 ? kD : (*stride)[0];
    const int64_t sH = stride->Size() == 0 ? kH : stride->Size() == 1 ? sD : (*stride)[DIM1];
    const int64_t sW = stride->Size() == 0 ? kW : stride->Size() == 1 ? sD : (*stride)[DIM2];

    const int64_t padD = (*padding)[0];
    const int64_t padH = padding->Size() == 1 ? padD : (*padding)[DIM1];
    const int64_t padW = padding->Size() == 1 ? padD : (*padding)[DIM2];

    CHECK_RET(CheckAttrValue(kD, sD, padD, depth), false);
    CHECK_RET(CheckAttrValue(kH, sH, padH, height), false);
    CHECK_RET(CheckAttrValue(kW, sW, padW, width), false);

    const int64_t outDepth = PoolingOutShape(depth, kD, padD, sD, ceilMode);
    const int64_t outHeight = PoolingOutShape(height, kH, padH, sH, ceilMode);
    const int64_t outWidth = PoolingOutShape(width, kW, padW, sW, ceilMode);

    OP_CHECK((outDepth >= 0 && outHeight >= 0 && outWidth >= 0),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "Given input size [%ld,%ld,%ld,%ld], calc out size [%ld,%ld,%ld,%ld] dim value should be non negative",
        nInputPlane, depth, height, width, nInputPlane, outDepth, outHeight, outWidth), return false);

    const op::Shape calcOutShape = is4d ? op::Shape({nInputPlane, outDepth, outHeight, outWidth})
                                        : op::Shape({nBatch, nInputPlane, outDepth, outHeight, outWidth});
    // 判断out的shape与推导出的输出shape是否相等
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, calcOutShape, return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
    const aclIntArray *padding, const bool ceilmode, const aclTensor *out)
{
    // 检查self/out是否是4D或5D
    if (!((self->GetViewShape().GetDimNum() == DIM_NUM_5D || self->GetViewShape().GetDimNum() == DIM_NUM_4D) &&
        (self->GetViewShape().GetDimNum() == out->GetViewShape().GetDimNum()))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and out must maintain same format, support 4D or 5D.");
        return false;
    }

    if (kernel->Size() != 1 && kernel->Size() != KERNEL_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected length of kernel is 1 or 3, but got %ld.", kernel->Size());
        return false;
    }

    if (padding->Size() != 1 && padding->Size() != PAD_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected length of padding is 1 or 3, but got %ld.", padding->Size());
        return false;
    }

    if (stride->Size() != 0 && stride->Size() != 1 && stride->Size() != STRIDE_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected length of stride is 0 or 1 or 3, but got: %ld.", stride->Size());
        return false;
    }
    // 检查out的shape是否符合预期输出
    const bool ret = CheckOutputShape(self, kernel, stride, padding, ceilmode, out);
    CHECK_RET(ret, false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *stride,
    const aclIntArray *padding, const bool ceilMode, const aclTensor *out)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNullPtr(self, kernelSize, stride, padding, out), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查参数&数据维度是否合法
    CHECK_RET(CheckShape(self, kernelSize, stride, padding, ceilMode, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsDimDDownsamping(const aclIntArray *kernelSize, const aclIntArray *stride)
{
    const int64_t kD = (*kernelSize)[0];
    const int64_t kH = kernelSize->Size() == 1 ? kD : (*kernelSize)[DIM1];
    const int64_t kW = kernelSize->Size() == 1 ? kD : (*kernelSize)[DIM2];

    const int64_t sD = stride->Size() == 0 ? kD : (*stride)[0];
    const int64_t sH = stride->Size() == 0 ? kH : stride->Size() == 1 ? sD : (*stride)[DIM1];
    const int64_t sW = stride->Size() == 0 ? kW : stride->Size() == 1 ? sD : (*stride)[DIM2];

    return kH == 1 && kW == 1 && sH == 1 && sW == 1;
}

// 构建averagepool3d计算图, 通过Vector实现
static aclnnStatus BuildAvgPool3dGraph(UniqueExecutor &uniqueExecutor, const aclTensor *self,
    const aclIntArray *kernelSize, const aclIntArray *stride, const aclIntArray *pad, const bool ceilMode,
    const bool countIncludePad, const int64_t divisorOverride, const aclTensor *out)
{
    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool isDimDDownsamping = IsDimDDownsamping(kernelSize, stride);

    std::string dataFormat = "NCDHW";
    const aclTensor* avgpoolIn = selfContiguous;

    auto selfDimNum = self->GetViewShape().GetDimNum();
    // Reshape 4D to 5D   C,D,H,W -> 1,C,D,H,W
    if (selfDimNum == DIM_NUM_4D) {
        auto shape = op::ToShapeVector(self->GetViewShape());
        FVector<int64_t> newShape = {1, shape[DIM0], shape[DIM1], shape[DIM2], shape[DIM3]};
        aclIntArray* shapeArray = uniqueExecutor->AllocIntArray(newShape.data(), newShape.size());
        CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        avgpoolIn = l0op::Reshape(avgpoolIn, shapeArray, uniqueExecutor.get());
        CHECK_RET(avgpoolIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (!isDimDDownsamping) {
        dataFormat = "NDHWC";
        // Transpose N,C,D,H,W -> N,D,H,W,C
        FVector<int64_t> inputDims = { 0, 2, 3, 4, 1 };
        auto permPre = uniqueExecutor->AllocIntArray(inputDims.data(), inputDims.size());
        CHECK_RET(permPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        avgpoolIn = l0op::Transpose(avgpoolIn, permPre, uniqueExecutor.get());
        CHECK_RET(avgpoolIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    // 执行L0算子
    auto avgpoolOut = l0op::AvgPool3D(avgpoolIn, kernelSize, stride, pad, ceilMode,
        countIncludePad, divisorOverride, dataFormat, uniqueExecutor.get());
    CHECK_RET(avgpoolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!isDimDDownsamping) {
        // Transpose N,D,H,W,C -> N,C,D,H,W
        FVector<int64_t> outputDims = { 0, 4, 1, 2, 3 };
        auto permPost = uniqueExecutor->AllocIntArray(outputDims.data(), outputDims.size());
        CHECK_RET(permPost!= nullptr, ACLNN_ERR_INNER_NULLPTR);

        avgpoolOut = l0op::Transpose(avgpoolOut, permPost, uniqueExecutor.get());
        CHECK_RET(avgpoolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Reshape 5D to 4D   1,C,D,H,W -> C,D,H,W
    if (selfDimNum == DIM_NUM_4D) {
        auto shape = op::ToShapeVector(out->GetViewShape());
        aclIntArray* shapeArray = uniqueExecutor->AllocIntArray(shape.data(), shape.size());
        CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        avgpoolOut = l0op::Reshape(avgpoolOut, shapeArray, uniqueExecutor.get());
        CHECK_RET(avgpoolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::ViewCopy(avgpoolOut, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvgPool3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelSize,
                                           const aclIntArray *stride, const aclIntArray *padding,
                                           bool ceilMode, bool countIncludePad,
                                           int64_t divisorOverride, aclTensor *out,
                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnAvgPool3d, DFX_IN(self, kernelSize, stride, padding,\
        ceilMode, countIncludePad, divisorOverride), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, stride, padding, ceilMode, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是空tensor，直接返回
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 解析kernel,stride,pad等参数
    const int64_t kD = (*kernelSize)[0];
    const int64_t kH = kernelSize->Size() == 1 ? kD : (*kernelSize)[DIM1];
    const int64_t kW = kernelSize->Size() == 1 ? kD : (*kernelSize)[DIM2];
    FVector<int64_t> newKernel { kD, kH, kW };
    const aclIntArray *kernelList = uniqueExecutor.get()->AllocIntArray(newKernel.data(), KERNEL_SIZE);

    const int64_t sD = stride->Size() == 0 ? kD : (*stride)[0];
    const int64_t sH = stride->Size() == 0 ? kH : stride->Size() == 1 ? sD : (*stride)[DIM1];
    const int64_t sW = stride->Size() == 0 ? kW : stride->Size() == 1 ? sD : (*stride)[DIM2];
    FVector<int64_t> newStride { sD, sH, sW };
    const aclIntArray *strideList = uniqueExecutor.get()->AllocIntArray(newStride.data(), STRIDE_SIZE);

    const int64_t padD = (*padding)[0];
    const int64_t padH = padding->Size() == 1 ? padD : (*padding)[DIM1];
    const int64_t padW = padding->Size() == 1 ? padD : (*padding)[DIM2];
    FVector<int64_t> newPad { padD, padH, padW };
    const aclIntArray *padList = uniqueExecutor.get()->AllocIntArray(newPad.data(), PAD_SIZE);

    // 构建常规averagepool3d计算图, 通过Vector实现
    CHECK_RET(ACLNN_SUCCESS == BuildAvgPool3dGraph(uniqueExecutor, self, kernelList, strideList, padList,
        ceilMode, countIncludePad, divisorOverride, out), ACLNN_ERR_INNER);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvgPool3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAvgPool3d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

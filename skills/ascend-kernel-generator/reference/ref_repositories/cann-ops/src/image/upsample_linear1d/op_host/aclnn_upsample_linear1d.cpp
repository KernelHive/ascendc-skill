/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_upsample_linear1d.h"
#include "upsample_linear1d_l0.h"
#include "resize_d_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "squeeze_l0.h"
#include "unsqueeze_l0.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static constexpr size_t DIM_LIMIT = 3;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

static const float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
static const float MAX_SUPPORT_ZOOM_SCALE_REV = 0.00125f;

// outputSizeNum 的维度限制为1
static constexpr int64_t EXPECT_SIZE = 1;

static const std::string LINEAR_MODE = "linear";

static bool CheckNotNull(const aclTensor *self, const aclTensor *out, const aclIntArray *outputSize)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(outputSize, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out, const aclIntArray *outputSize, const double scale)
{
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    size_t outDimNum = out->GetViewShape().GetDimNum();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);

    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected outputSize equals to 1, but got size %zu", outputSizeNum),
        return false);
    int64_t outL = (*outputSize)[DIM_ZERO];
    int64_t batch = self->GetViewShape().GetDim(DIM_ZERO);
    int64_t channels = self->GetViewShape().GetDim(DIM_ONE);
    auto outShape = out->GetViewShape();
    FVector<int64_t> fullOutputSize = {batch, channels, outL};
    if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(DIM_TWO) == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Non-empty 3D data tensor expected but got a tensor with sizes %s.",
            op::ToString(self->GetViewShape()).GetString());
        return false;
    }
    OP_CHECK(outL > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0,"
            "bug got output (L: %ld)",
            outL),
        return false);

    for (size_t i = 0; i < outDimNum; ++i) {
        if (outShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected out to have shape"
                "size(%zu) = %ld but got out.size(%zu) = %ld",
                i,
                fullOutputSize[i],
                i,
                outShape.GetDim(i));
            return false;
        }
    }
    // 在scale合法的情况下，判断scale和outputSize是否冲突
    if (scale >= 0) {
        const double odim = static_cast<double>(self->GetViewShape().GetDim(DIM_TWO)) * scale;
        if (static_cast<int64_t>(odim) != outL) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Scale conflicts with outputSize."
                "scale * input[2] should be equal to outputSize[0]");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *selfRef, const aclTensor *out, const aclIntArray *outputSize, const double scale)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, out, outputSize), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckShape(selfRef, out, outputSize, scale), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *View3dAs4d(const aclTensor *input, aclOpExecutor *executor)
{
    // NCL -> contigious -> unsqueeze(2)
    // contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    const int64_t appendDim[] = {2};
    aclIntArray *dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);

    return unsqueezedInput;
}

static bool CheckLinear1dScales(
    const aclTensor *x, const aclTensor *y, const aclIntArray *size, const double scale, const bool alignCorners)
{
    float scalesW = 0.0;
    auto inputShape = x->GetViewShape();
    auto outputShape = y->GetViewShape();
    int64_t input_size = inputShape.GetDim(DIM_TWO);
    int64_t output_size = (*size)[DIM_ZERO];

    if (scale > 0) {
        output_size = outputShape.GetDim(DIM_TWO);
    }

    if (alignCorners) {
        if (output_size > 1) {
            scalesW = static_cast<float>(input_size - 1) / (output_size - 1);
        } else {
            scalesW = static_cast<float>(0);
        }
    } else {
        scalesW = (scale > 0) ? static_cast<float>(1.0 / scale) : (static_cast<float>(input_size) / output_size);
    }
    return (scalesW <= MAX_SUPPORT_SHRINK_SCALE && scalesW >= MAX_SUPPORT_ZOOM_SCALE_REV);
}

aclnnStatus aclnnUpsampleLinear1dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    const bool alignCorners, const double scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleLinear1d, DFX_IN(self, outputSize, alignCorners, scale), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out, outputSize, scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    int64_t outL = (*outputSize)[DIM_ZERO];
    if (scale == 1.0 || outL == self->GetViewShape().GetDim(DIM_TWO)) {
        // 直接返回输入tensor
        auto viewCopyResult = l0op::ViewCopy(self, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 固定写法，将输入self升维
        auto selfRefContiguous = View3dAs4d(self, uniqueExecutor.get());
        CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // scale转为aclFloatArray
        const float scalesList[] = {(scale < 0) ? 0 : scale};
        const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList, 1);
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto socVer = GetCurrentPlatformInfo().GetSocVersion();
        auto dataType = selfRefContiguous->GetDataType();
        auto inputShape = selfRefContiguous->GetViewShape();
        const aclTensor *ResizeDOut;

        if ((socVer == SocVersion::ASCEND910B || socVer == SocVersion::ASCEND910_93) &&
            CheckLinear1dScales(self, out, outputSize, scale, alignCorners)) {
            // 调用UpsampleLinear1d算子kernel
            ResizeDOut = l0op::UpsampleLinear1dNcdhw(
                selfRefContiguous, outputSize, alignCorners, out, scale, uniqueExecutor.get());
        } else {
            // 调用ResizeD算子kernel
            ResizeDOut = l0op::ResizeD(
                selfRefContiguous, outputSize, alignCorners, out, scales, LINEAR_MODE, uniqueExecutor.get());
        }
        CHECK_RET(ResizeDOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(ResizeDOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleLinear1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleLinear1d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

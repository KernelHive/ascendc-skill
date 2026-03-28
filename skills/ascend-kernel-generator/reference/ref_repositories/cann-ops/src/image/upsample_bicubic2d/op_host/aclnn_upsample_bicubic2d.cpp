/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_upsample_bicubic2d.h"
#include "resize_d.h"
#include "upsample_bicubic2d_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static constexpr size_t EXPECT_SIZE = 2;
static constexpr float MAX_SUPPORT_SCALE = 40.0;
static constexpr float MAX_SUPPORT_SCALE_DOUBLE = 30.0;
static const int64_t DIM_LIMIT = 4;
static const int64_t FOURDIMS = 4;
static const std::string MODE = "cubic";

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

static bool CheckNotNull(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == op::SocVersion::ASCEND910B) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected outputSize's size is 2, but got size %zu", outputSizeNum),
        return false);
    auto selfShape = self->GetViewShape();
    int64_t inputC = selfShape.GetDim(DIM_ONE);
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];
    OP_CHECK(inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (H: %ld,"
            " W: %ld) output (H: %ld, W: %ld)",
            inputH,
            inputW,
            outH,
            outW),
        return false);
    OP_CHECK(inputC > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Non-empty 4D data tensor expected but got a tensor with sizes %s.",
            op::ToString(selfShape).GetString()),
        return false);
    OP_CHECK(
        (self->GetStorageFormat() == op::Format::FORMAT_ND || self->GetStorageFormat() == op::Format::FORMAT_NCHW) &&
            (out->GetStorageFormat() == op::Format::FORMAT_ND || out->GetStorageFormat() == op::Format::FORMAT_NCHW),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output storage format only support NCHW, but got Input %s and output %s.",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(out->GetStorageFormat()).GetString()),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckMaxScaleSupport(
    const aclTensor *self, const aclIntArray *outputSize, const double scalesH, const double scalesW)
{
    auto selfShape = self->GetViewShape();
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    const float scale_h = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : static_cast<float>(inputH / outputH);
    const float scale_w = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : static_cast<float>(inputW / outputW);
    if (scale_h > MAX_SUPPORT_SCALE || scale_w > MAX_SUPPORT_SCALE) {
        return false;
    }
    if (scale_h > MAX_SUPPORT_SCALE_DOUBLE && scale_w > MAX_SUPPORT_SCALE_DOUBLE) {
        return false;
    }
    return true;
}

static bool CheckIsBicubic2dPlatform(const aclTensor *self, const aclTensor *out)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    } else {
        return false;
    }
    return true;
}

static bool CheckIsBicubic2dPlatform310p(const aclTensor *self, const aclTensor *out)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P ||
        GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310B) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    } else {
        return false;
    }
    return true;
}

aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleBicubic2d, DFX_IN(self, outputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (CheckIsBicubic2dPlatform(self, out) && CheckMaxScaleSupport(self, outputSize, scalesH, scalesW)) {
        auto dtype = self->GetDataType();
        // 将fp16/bf16类型cast成fp32处理，保证精度
        if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
            selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 先用double算好1/scale再转float，减少精度损失
        const float realScales_h = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        const float realScales_w = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

        // 调用Bicubic2d算子kernel
        const aclTensor *Bicubic2dOut = l0op::UpsampleBicubic2d(
            selfContiguous, outputSize, alignCorners, realScales_h, realScales_w, out, uniqueExecutor.get());
        CHECK_RET(Bicubic2dOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (dtype == op::DataType::DT_BF16) {
            // CAST回bf16
            Bicubic2dOut = l0op::Cast(Bicubic2dOut, op::DataType::DT_BF16, uniqueExecutor.get());
        } else if (dtype == op::DataType::DT_FLOAT16) {
            // CAST回fp16
            Bicubic2dOut = l0op::Cast(Bicubic2dOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        }
        CHECK_RET(Bicubic2dOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(Bicubic2dOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto selfshape = self->GetViewShape();
        int64_t batch = selfshape.GetDim(DIM_ZERO);
        int64_t channels = selfshape.GetDim(DIM_ONE);
        int64_t outH = (*outputSize)[DIM_ZERO];
        int64_t outW = (*outputSize)[DIM_ONE];

        const float scalesList[] = {scalesH, scalesW};
        const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList, DIM_TWO);
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self进行transpose，shape：NCHW-->HWNC
        const int64_t permuteHWNCList[] = {2, 3, 0, 1};
        auto permuteHWNCArray = uniqueExecutor.get()->AllocIntArray(permuteHWNCList, FOURDIMS);
        CHECK_RET(permuteHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfTranspose = l0op::Transpose(selfContiguous, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将cast移至transpose后，转换成连续的tensor
        auto dtype = selfTranspose->GetDataType();
        if (dtype == op::DataType::DT_BF16) {
            selfTranspose = l0op::Cast(selfTranspose, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // out reshape
        const int64_t new_reshape[4] = {batch, channels, outH, outW};
        aclIntArray *shapeArray = uniqueExecutor.get()->AllocIntArray(new_reshape, FOURDIMS);
        auto OutReshape = l0op::Reshape(out, shapeArray, uniqueExecutor.get());
        CHECK_RET(OutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // out Transpose
        auto outTranspose = l0op::Transpose(OutReshape, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(outTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // CAST
        if (dtype == op::DataType::DT_BF16) {
            outTranspose = l0op::Cast(outTranspose, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(outTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor *resizeOut = nullptr;
        if (CheckIsBicubic2dPlatform310p(self, out)) {
            // 调用Bicubic2d算子kernel
            // 先用double算好1/scale再转float，可以减少精度损失
            const float realScales_h = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
            const float realScales_w = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;
            resizeOut = l0op::UpsampleBicubic2d(selfTranspose,
                outputSize,
                alignCorners,
                realScales_h,
                realScales_w,
                outTranspose,
                uniqueExecutor.get());
        } else {
            // 调用ResizeD算子kernel
            resizeOut = l0op::ResizeD(
                selfTranspose, outputSize, alignCorners, outTranspose, scales, MODE, uniqueExecutor.get());
        }
        CHECK_RET(resizeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // resizeDOut reshape
        const int64_t new_reshape_reverse[4] = {outH, outW, batch, channels};
        aclIntArray *shapeArrayReverse = uniqueExecutor.get()->AllocIntArray(new_reshape_reverse, FOURDIMS);
        auto resizeDOutReshape = l0op::Reshape(resizeOut, shapeArrayReverse, uniqueExecutor.get());
        CHECK_RET(resizeDOutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // CAST回bf16
        if (dtype == op::DataType::DT_BF16) {
            resizeDOutReshape = l0op::Cast(resizeDOutReshape, op::DataType::DT_BF16, uniqueExecutor.get());
        }
        CHECK_RET(resizeDOutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // resizeDOut Transpose, shape：HWNC-->NCHW
        auto resizeDOutTranspose = l0op::Transpose(resizeDOutReshape, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(resizeDOutTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(resizeDOutTranspose, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2d(
    void *workspace, uint64_t workspace_size, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBicubic2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

#ifdef __cplusplus
}
#endif

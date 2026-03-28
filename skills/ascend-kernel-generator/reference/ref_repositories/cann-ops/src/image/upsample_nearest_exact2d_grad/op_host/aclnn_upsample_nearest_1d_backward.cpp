/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/transdata.h"
#include "squeeze_l0.h"
#include "unsqueeze_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "resize_grad_l0.h"
#include "resize_nearest_neighbor_v2_grad_l0.h"
#include "upsample_nearest_exact2d_grad_l0.h"
#include "aclnn_upsample_nearest_1d_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/cast.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 输入维度限制为3
static constexpr size_t DIM_LIMIT = 3;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
// outputSizeNum 的维度限制为1
static constexpr int64_t EXPECT_SIZE = 1;
// 浮点数-1和0
static constexpr float FLOAT_NEGONE = -1.0;
static constexpr float FLOAT_ZERO = 0.0;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(
    const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *out, const aclIntArray *inputSize)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize)
{
    size_t gradOutDimNum = gradOut->GetViewShape().GetDimNum();
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK(gradOutDimNum == DIM_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected gradOut to be 3d Tensor, instead got: %zu", gradOutDimNum),
        return false);

    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 1, but got size %zu", outputSizeNum),
        return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input_size equals to 3, but got size %zu", inputSizeNum),
        return false);
    return true;
}

static bool CheckInputElement(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize)
{
    int64_t outL = (*outputSize)[DIM_ZERO];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputL = (*inputSize)[DIM_TWO];
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();
    FVector<int64_t> fullOutputSize = {batch, channels, outL};

    OP_CHECK(inputL > 0 && outL > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0,"
            "bug got input (L: %ld) output (L: %ld)",
            inputL,
            outL),
        return false);

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected grad_output to have the same shape as output;"
                " output.size(%zu) = %ld but got grad_output.size(%zu) = %ld",
                i,
                fullOutputSize[i],
                i,
                gradOutShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckNCDimEqual(const aclTensor *self, const aclTensor *out)
{
    int64_t selfDimN = self->GetViewShape().GetDim(DIM_ZERO);
    int64_t selfDimC = self->GetViewShape().GetDim(DIM_ONE);
    int64_t outDimN = out->GetViewShape().GetDim(DIM_ZERO);
    int64_t outDimC = out->GetViewShape().GetDim(DIM_ONE);
    if ((outDimC != selfDimC) || (outDimN != selfDimN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "selfDimC[%ld]/outDimC[%ld] or selfDimN[%ld]/outDimN[%ld] not equal .",
            selfDimC,
            outDimC,
            selfDimN,
            outDimN);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查shape
    CHECK_RET(CheckShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOut, out, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 校验gradOut的shape是否与输出的output的shape一致
    CHECK_RET(CheckInputElement(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 5.检查gradOut和gradIn N/C轴的大小是否一致
    CHECK_RET(CheckNCDimEqual(gradOut, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *View4dAs3d(const aclTensor *input, const aclTensor *out, bool ifAiCpu, aclOpExecutor *executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t removeDim[] = {2};
    aclIntArray *dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    auto reformatInput = ifAiCpu ? squeezedInput : l0op::ReFormat(squeezedInput, out->GetStorageFormat());
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor *View3dAs4d(const aclTensor *input, bool ifAiCpu, aclOpExecutor *executor)
{
    // NCL -> contigious -> unsqueeze(2) -> reformat -> NCHW
    // contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    const int64_t appendDim[] = {2};
    aclIntArray *dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);

    // reformat
    auto reformatInput = ifAiCpu ? unsqueezedInput : l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor *upsampleNearest1dBackwardAiCpuCompute(const aclTensor *gradOutContiguous,
    const aclIntArray *outputSize, double scales, const aclTensor *size, aclOpExecutor *executor)
{
    const float scalesList[] = {FLOAT_NEGONE, FLOAT_NEGONE, FLOAT_NEGONE, static_cast<float>(scales)};
    auto scalesArray = executor->AllocFloatArray(scalesList, 4);
    CHECK_RET(scalesArray != nullptr, nullptr);

    auto scalesTensor = executor->ConvertToTensor(scalesArray, op::ToOpDataType(ACL_FLOAT));
    CHECK_RET(scalesTensor != nullptr, nullptr);

    const aclTensor *resizeNearestOutAiCpu =
        l0op::ResizeGrad(gradOutContiguous, outputSize, scalesTensor, size, executor);
    CHECK_RET(resizeNearestOutAiCpu != nullptr, nullptr);

    return resizeNearestOutAiCpu;
}

aclnnStatus aclnnUpsampleNearest1dBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, double scales, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleNearest1dBackward, DFX_IN(gradOut, outputSize, inputSize, scales), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradOut, outputSize, inputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOut->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const aclTensor *out3d = nullptr;
    bool isExactSupport = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                          GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93;
    const int64_t inputL = (*inputSize)[DIM_TWO];
    const int64_t gradOutL = gradOut->GetViewShape().GetDim(DIM_TWO);
    bool check_scales = static_cast<int64_t>(inputL * scales) == gradOutL;
    if (CheckType(gradOut->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) && scales > FLOAT_ZERO && isExactSupport &&
        check_scales) {
        auto gradOutContiguous = View3dAs4d(gradOut, false, uniqueExecutor.get());
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto outContiguous = View3dAs4d(out, false, uniqueExecutor.get());
        CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const int64_t inputSizeList[] = {(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], 1, (*inputSize)[DIM_TWO]};
        auto inputSizeArray = uniqueExecutor.get()->AllocIntArray(inputSizeList, 4);
        CHECK_RET(inputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t outputSizeList[] = {1, (*outputSize)[DIM_ZERO]};
        auto outputSizeArray = uniqueExecutor.get()->AllocIntArray(outputSizeList, 2);
        CHECK_RET(outputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
        const float realScales_w = scales > 0 ? static_cast<float>(scales) : 0;
        const float realScales_h = static_cast<float>(1.0);
        // 调用算子计算
        const aclTensor *upsampleOut = l0op::UpsampleNearestExact2dGrad(gradOutContiguous,
            outputSizeArray,
            inputSizeArray,
            const_cast<aclTensor *>(outContiguous),
            realScales_h,
            realScales_w,
            false,
            uniqueExecutor.get());
        CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor *resizeNearestOut = nullptr;
        resizeNearestOut = l0op::TransData(upsampleOut, outContiguous->GetStorageFormat(), 0, uniqueExecutor.get());
        CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        out3d = View4dAs3d(resizeNearestOut, out, false, uniqueExecutor.get());
    } else {
        // 当支持类型不在aicore范围内或者传入的scales大于0时使用aicpu算子
        bool ifAiCpu = ((gradOut->GetDataType() != op::DataType::DT_FLOAT || scales > FLOAT_ZERO) &&
                        gradOut->GetDataType() != op::DataType::DT_BF16 && check_scales);
        // l0算子要求传入4维tensor
        auto gradOutContiguous = View3dAs4d(gradOut, ifAiCpu, uniqueExecutor.get());
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const int64_t originSizeList[] = {(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], 1, (*inputSize)[DIM_TWO]};
        auto originSizeArray = uniqueExecutor.get()->AllocIntArray(originSizeList, 4);
        CHECK_RET(originSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor *resizeNearestOut = nullptr;
        if (ifAiCpu) {
            auto dataType = gradOutContiguous->GetDataType();
            if (op::DataType::DT_FLOAT16 == dataType) {
                gradOutContiguous = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            }
            auto originSizeTensor = uniqueExecutor.get()->ConvertToTensor(originSizeArray, op::DataType::DT_INT64);
            CHECK_RET(originSizeTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
            resizeNearestOut = upsampleNearest1dBackwardAiCpuCompute(
                gradOutContiguous, originSizeArray, scales, originSizeTensor, uniqueExecutor.get());
            CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            if (op::DataType::DT_FLOAT16 == dataType) {
                resizeNearestOut = l0op::Cast(resizeNearestOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
            }
        } else {
            // aicore算子ResizeNearestNeighborV2Grad要求格式为NC1HWC0
            auto gradOutTransdata =
                l0op::TransDataSpecial(gradOutContiguous, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
            CHECK_RET(gradOutTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // ResizeNearestNeighborV2Grad算子要求传入的默认参数
            bool alignCorners = false;
            bool halfPixelCenters = false;
            const aclTensor *resizeNearestOutAiCore = l0op::ResizeNearestNeighborV2Grad5Hd(
                gradOutTransdata, originSizeArray, alignCorners, halfPixelCenters, uniqueExecutor.get());
            CHECK_RET(resizeNearestOutAiCore != nullptr, ACLNN_ERR_INNER_NULLPTR);

            resizeNearestOut =
                l0op::TransData(resizeNearestOutAiCore, gradOutContiguous->GetStorageFormat(), 0, uniqueExecutor.get());
            CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // 转回3维tensor
        out3d = View4dAs3d(resizeNearestOut, out, ifAiCpu, uniqueExecutor.get());
        CHECK_RET(CheckReduceOutShape(out3d, out), ACLNN_ERR_PARAM_INVALID);
    }

    auto viewCopyResult = l0op::ViewCopy(out3d, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest1dBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest1dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

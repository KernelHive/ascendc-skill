/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_upsample_bilinear2d.h"
#include "upsample_bilinear2d_l0.h"
#include "resize_bilinear_v2_l0.h"
#include "broadcast_to_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
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

static const int64_t AICPU_SHAPE = 2L;
static const int64_t AICPU_OFFSET_NHWC = 1L;

static const double MAX_SUPPORT_SCALE = 50;

static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DIM_THREE = 3;
static const float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
static const float MAX_SUPPORT_ZOOM_SCALE_REV = 0.02f;
static const float UNSUPPORT_SCALES_TWO = 2.0f;
static const float UNSUPPORT_SCALES_ZERO = 0.0f;
static const int64_t FOURDIMS = 4;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_ALL = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_FOR_AICORE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const int64_t DIMLIMIT = 4;

static bool CheckNotNull(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_ALL, return false);
    return true;
}

static bool CheckDtypeEqual(const aclTensor *selfRef, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_MATCH(selfRef, out->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out)
{
    // 需要根据算子实际情况添加校验
    const op::Format selfFormat = self->GetStorageFormat();
    if (selfFormat != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Format of input and output should be equal, self [%s], out [%s].",
            op::ToString(selfFormat).GetString(),
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    // 如果输入格式是不为4D，记录日志，直接报错
    OP_CHECK_WRONG_DIMENSION(self, DIMLIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIMLIMIT, return false);

    const op::DataType selfType = self->GetDataType();
    if ((selfFormat == op::Format::FORMAT_NCHW) && (selfType == op::DataType::DT_DOUBLE)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "When dtype is %s, only support NHWC format", op::ToString(selfType).GetString());
        return false;
    }
    return true;
}

static bool CheckScalesAndShapeValid(
    const aclTensor *self, const aclTensor *out, const double scaleH, const double scaleW)
{
    auto inputShape = self->GetViewShape();
    auto outputShape = out->GetViewShape();
    int64_t input_h = inputShape.GetDim(DIM_TWO);
    int64_t input_w = inputShape.GetDim(DIM_THREE);
    int64_t output_h = outputShape.GetDim(DIM_TWO);
    int64_t output_w = outputShape.GetDim(DIM_THREE);
    int64_t scaleSizeH = static_cast<int64_t>(input_h * scaleH);
    int64_t scaleSizeW = static_cast<int64_t>(input_w * scaleW);
    return scaleSizeH == output_h && scaleSizeW == output_w;
}

static bool CheckScalesValid(const double weight, const double high)
{
    if ((weight < 0) || (high < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "w scales and h scales cannot be negative , w_scales [%f], h_scales [%f].",
            weight,
            high);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *selfRef, const aclTensor *out, const double scalesW, const double scalesH)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, out), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查selfRef和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckDtypeEqual(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 5.检查self和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 6.检验scalesW/scalesH，与资料保持一致
    CHECK_RET(CheckScalesValid(scalesW, scalesH), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static double GetBilinearScales(int64_t input_size, int64_t output_size, double scale, const bool alignCorners)
{
    double realScale = 0.0;

    if (output_size == input_size) {
        return static_cast<double>(1);
    }

    if (alignCorners) {
        if (output_size > 1) {
            realScale = static_cast<double>(input_size - 1) / (output_size - 1);
        } else {
            realScale = static_cast<double>(0);
        }
    } else {
        realScale = (scale > 0) ? static_cast<double>(1.0 / scale) : (static_cast<double>(input_size) / output_size);
    }

    return realScale;
}

static bool CheckBilinear2dScales(const aclTensor *x, const aclTensor *y, const aclIntArray *size, const double scaleH,
    const double scaleW, const bool alignCorners)
{
    auto dataType = x->GetDataType();
    auto inputShape = x->GetViewShape();
    auto outputShape = y->GetViewShape();
    int64_t input_h = inputShape.GetDim(DIM_TWO);
    int64_t input_w = inputShape.GetDim(DIM_THREE);
    int64_t output_h = outputShape.GetDim(DIM_TWO);
    int64_t output_w = outputShape.GetDim(DIM_THREE);
    double scaleH1 = scaleH;
    double scaleW1 = scaleW;

    if (scaleW == UNSUPPORT_SCALES_TWO && scaleH == UNSUPPORT_SCALES_TWO && dataType != op::DataType::DT_BF16) {
        return false;
    }

    if (scaleH == UNSUPPORT_SCALES_ZERO || scaleW == UNSUPPORT_SCALES_ZERO) {
        return false;
    }

    // size(fp16&fp32) go resizeBilinearV2, size(bf16)go upsampleBilinear2d
    if (scaleH == 1.0 && scaleW == 1.0 && (output_h != input_h || output_w != input_w)) {
        if (dataType != op::DataType::DT_BF16) {
            return false;
        } else {
            scaleH1 = 0.0;
            scaleW1 = 0.0;
        }
    }

    float scalesH = GetBilinearScales(input_h, output_h, scaleH1, alignCorners);
    float scalesW = GetBilinearScales(input_w, output_w, scaleW1, alignCorners);

    return scalesH <= MAX_SUPPORT_SHRINK_SCALE && scalesW <= MAX_SUPPORT_SHRINK_SCALE;
}

static const aclTensor *GoResizeBilinearV2AICORE(const aclTensor *selfRefContiguous, const aclIntArray *outputSize,
    const bool alignCorners, const aclTensor *outContiguous, aclTensor *out, aclOpExecutor *executor)
{
    auto dstFormat = out->GetStorageFormat();
    auto size = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT64));
    auto castSize = l0op::Cast(size, op::DataType::DT_INT32, executor);

    auto dataType = selfRefContiguous->GetDataType();
    if (op::DataType::DT_BF16 == dataType) {
        selfRefContiguous = l0op::Cast(selfRefContiguous, op::DataType::DT_FLOAT, executor);
        outContiguous = l0op::Cast(outContiguous, op::DataType::DT_FLOAT, executor);
    }

    auto selfTransdata = l0op::TransDataSpecial(selfRefContiguous, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(selfTransdata != nullptr, nullptr);

    auto outTransdata = l0op::TransDataSpecial(outContiguous, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(outTransdata != nullptr, nullptr);

    // 调用UpsampleBilinear算子kernel
    const aclTensor *upsampleBilinearout =
        l0op::ResizeBilinearV2(selfTransdata, castSize, alignCorners, outTransdata, executor);
    CHECK_RET(upsampleBilinearout != nullptr, nullptr);

    auto upsampleBilinearoutTransdata = l0op::TransData(upsampleBilinearout, dstFormat, 0, executor);
    CHECK_RET(upsampleBilinearoutTransdata != nullptr, nullptr);

    if (op::DataType::DT_BF16 == dataType) {
        upsampleBilinearoutTransdata = l0op::Cast(upsampleBilinearoutTransdata, op::DataType::DT_BF16, executor);
        return upsampleBilinearoutTransdata;
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    const aclTensor *castOut = l0op::Cast(upsampleBilinearoutTransdata, selfRefContiguous->GetDataType(), executor);

    return castOut;
}

static const aclTensor *GoUpsampleBilinear2DAICORE(const aclTensor *selfRefContiguous, const aclIntArray *outputSize,
    const bool alignCorners, const double scalesH, const double scalesW, const aclTensor *outContiguous,
    aclOpExecutor *executor)
{
    auto dataType = selfRefContiguous->GetDataType();
    auto size = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT64));
    auto castSize = l0op::Cast(size, op::DataType::DT_INT32, executor);
    auto inputShape = selfRefContiguous->GetViewShape();
    auto outputShape = outContiguous->GetViewShape();

    int64_t batch = inputShape.GetDim(DIM_ZERO);
    int64_t channels = inputShape.GetDim(DIM_ONE);
    int64_t input_h = inputShape.GetDim(DIM_TWO);
    int64_t input_w = inputShape.GetDim(DIM_THREE);
    int64_t output_h = outputShape.GetDim(DIM_TWO);
    int64_t output_w = outputShape.GetDim(DIM_THREE);

    const int64_t permuteHWNCList[] = {0, 1, 3, 2};
    auto permuteHWNCArray = executor->AllocIntArray(permuteHWNCList, FOURDIMS);

    const aclTensor *upsampleBilinearout;
    if (input_w == 1 && input_h != 1) {
        auto selfTranspose = l0op::Transpose(selfRefContiguous, permuteHWNCArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        const int64_t new_out_reshape[4] = {batch, channels, output_h, output_w};
        aclIntArray *out_shape_array = executor->AllocIntArray(new_out_reshape, FOURDIMS);
        auto outReshape = l0op::Reshape(outContiguous, out_shape_array, executor);
        CHECK_RET(outReshape != nullptr, nullptr);

        auto outTranspose = l0op::Transpose(outReshape, permuteHWNCArray, executor);
        CHECK_RET(outTranspose != nullptr, nullptr);

        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            selfTranspose = l0op::Cast(selfTranspose, op::DataType::DT_FLOAT, executor);
            outTranspose = l0op::Cast(outTranspose, op::DataType::DT_FLOAT, executor);
        }
        auto outRes = l0op::UpsampleBilinear2dNcdhw(
            selfTranspose, castSize, alignCorners, scalesW, scalesH, outTranspose, executor);

        const int64_t out_reshape2[4] = {batch, channels, output_w, output_h};
        aclIntArray *out_shape2 = executor->AllocIntArray(out_reshape2, FOURDIMS);
        upsampleBilinearout = l0op::Reshape(outRes, out_shape2, executor);
    } else {
        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            selfRefContiguous = l0op::Cast(selfRefContiguous, op::DataType::DT_FLOAT, executor);
            outContiguous = l0op::Cast(outContiguous, op::DataType::DT_FLOAT, executor);
        }
        upsampleBilinearout = l0op::UpsampleBilinear2dNcdhw(
            selfRefContiguous, castSize, alignCorners, scalesH, scalesW, outContiguous, executor);
    }
    CHECK_RET(upsampleBilinearout != nullptr, nullptr);

    const aclTensor *castOut;
    if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
        castOut = l0op::Cast(upsampleBilinearout, dataType, executor);
    } else {
        castOut = l0op::Cast(upsampleBilinearout, selfRefContiguous->GetDataType(), executor);
    }

    if (input_w == 1 && input_h != 1) {
        castOut = l0op::Transpose(castOut, permuteHWNCArray, executor);
    }
    return castOut;
}

aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleBilinear2d, DFX_IN(self, outputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out, scalesW, scalesH);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // IndexPut算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfRefContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入out转换成连续的tensor
    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *castOut;

    auto socVer = GetCurrentPlatformInfo().GetSocVersion();
    if ((socVer == SocVersion::ASCEND910B || socVer == SocVersion::ASCEND910_93) &&
        CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_FOR_AICORE) &&
        CheckBilinear2dScales(self, out, outputSize, scalesH, scalesW, alignCorners) &&
        CheckScalesAndShapeValid(self, out, scalesH, scalesW)) {
        castOut = GoUpsampleBilinear2DAICORE(
            selfRefContiguous, outputSize, alignCorners, scalesH, scalesW, outContiguous, uniqueExecutor.get());
    } else if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_FOR_AICORE)) {
        castOut = GoResizeBilinearV2AICORE(
            selfRefContiguous, outputSize, alignCorners, outContiguous, out, uniqueExecutor.get());
    } else {
        auto outShape = op::ToShapeVector(outContiguous->GetViewShape());
        aclIntArray *newOutputSize = uniqueExecutor.get()->AllocIntArray(outShape.data(), outShape.size());
        auto size = uniqueExecutor.get()->ConvertToTensor(newOutputSize, op::ToOpDataType(ACL_INT32));
        const aclTensor *newSize = uniqueExecutor.get()->CreateView(size, op::Shape({AICPU_SHAPE}), AICPU_OFFSET_NHWC);

        const aclTensor *upsampleBilinearout =
            l0op::ResizeBilinearV2(selfRefContiguous, newSize, alignCorners, outContiguous, uniqueExecutor.get());
        CHECK_RET(upsampleBilinearout != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果转换成输出out的数据类型
        castOut = l0op::Cast(upsampleBilinearout, self->GetDataType(), uniqueExecutor.get());
    }
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2d(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBilinear2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

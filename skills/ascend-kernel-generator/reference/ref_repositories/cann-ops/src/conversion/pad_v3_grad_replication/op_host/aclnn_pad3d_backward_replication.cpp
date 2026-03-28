/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_pad3d_backward.cpp
 */
#include "aclnn_replication_pad3d_backward.h"
#include "../../circular_pad_grad/op_host/padv3grad.h"
#include "aclnn_kernels/contiguous.h"
#include "squeeze_pad_v3_grad_replication.h"
#include "unsqueeze_pad_v3_grad_replication.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const string REFLECTION_PAD_MODE = "reflect";
static const string REPLICATION_PAD_MODE = "edge";
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

inline static bool isPadSupport(const aclIntArray * padding ) {
    int MAX_PADDING_VALUE = 7;
    for (size_t i = 0; i < padding->Size(); i++){
        if ((*padding)[i] > MAX_PADDING_VALUE){
            return false;
        }
    }
    return true;
}

inline static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding,
    const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput)
{
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查gradInput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, dtypeSupportList, return false);

    // gradOutput, self和gradInput数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradInput, self->GetDataType(), return false);
    return true;
}

inline static bool CheckFormat(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(self->GetStorageFormat()) ||
        op::IsPrivateFormat(gradInput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND.");
        return false;
    }

    OP_CHECK(gradOutput->GetViewFormat() == self->GetViewFormat() &&
        gradOutput->GetViewFormat() == gradInput->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format of input and output should be equal, gradOutput [%s], self [%s], gradInput [%s].",
                op::ToString(gradOutput->GetViewFormat()).GetString(), op::ToString(self->GetViewFormat()).GetString(),
                op::ToString(gradInput->GetViewFormat()).GetString()),
        return false);
    return true;
}

inline static bool CheckShape(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding,
                              const string& mode, const aclTensor *gradInput)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self和gradInput的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradInput, return false);

    // 4 and 5 are dims, self只支持4维和5维
    OP_CHECK_MIN_DIM(self, 4, return false);
    OP_CHECK_MAX_DIM(self, 5, return false);

    // gradOutput, self, gradInput维度需要一致
    OP_CHECK(gradOutput->GetViewShape().GetDimNum() == selfDimnum &&
             gradInput->GetViewShape().GetDimNum() == selfDimnum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput, self, gradInput dim should be same."),
             return false);

    // padding长度为6
    OP_CHECK(padding->Size() == 6,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 6, but got %zu.",
                     padding->Size()),
             return false);

    // reflection时padding的每一维度的数值要小于对应self的dim
    if (mode == "reflect") {
        // 0, 1, 2, 3, 4, 5 are indexes
        OP_CHECK((*padding)[0] < self->GetViewShape().GetDim(selfDimnum - 1) &&
                 (*padding)[1] < self->GetViewShape().GetDim(selfDimnum - 1) &&
                 (*padding)[2] < self->GetViewShape().GetDim(selfDimnum - 2) &&
                 (*padding)[3] < self->GetViewShape().GetDim(selfDimnum - 2) &&
                 (*padding)[4] < self->GetViewShape().GetDim(selfDimnum - 3) &&
                 (*padding)[5] < self->GetViewShape().GetDim(selfDimnum - 3),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding size should be less than the corresponding self dimention."),
                 return false);
    }

    // check the last 2 dim value of out. 0, 1, 2, 3 are indexes
    OP_CHECK(gradOutput->GetViewShape().GetDim(selfDimnum - 2) ==
             self->GetViewShape().GetDim(selfDimnum - 2) + (*padding)[2] + (*padding)[3] &&
             gradOutput->GetViewShape().GetDim(selfDimnum - 1) ==
             self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1] &&
             gradOutput->GetViewShape().GetDim(selfDimnum - 3) ==
             self->GetViewShape().GetDim(selfDimnum - 3) + (*padding)[4] + (*padding)[5],
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "wrong gradOutput shape."),
             return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *self,
                                      const aclIntArray *padding, const string& mode, const aclTensor *gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, padding, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(gradOutput, self, padding, mode, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *GetPaddingTensor(int64_t dim, const aclIntArray *padding, aclOpExecutor *executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> paddingsVector;
    // 2 is the magnification
    for (size_t i = 2 * dim; i > 0; i -= 2) {
        if (i <= (size_t)padding->Size()) {
        // 2 and 1 indicate the element of padding is put into paddingsVector from the back to the front
            paddingsVector.emplace_back((*padding)[i - 2]);
            paddingsVector.emplace_back((*padding)[i - 1]);
        } else {
            paddingsVector.emplace_back(0);
            paddingsVector.emplace_back(0);
        }
    }
    // 2 is the magnification
    auto newpadding = executor->AllocIntArray(paddingsVector.data(), 2 * dim);
    auto paddingsTensor = executor->ConvertToTensor(newpadding, static_cast<op::DataType>(ACL_INT64));
    return paddingsTensor;
}

static aclnnStatus InputPreprocess(const aclTensor *&gradOutput, const aclTensor *&self,
                                   int64_t dimCp, aclOpExecutor *executor)
{
    // 如果非连续，需要转连续
    gradOutput = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果tensor为4维, 需要扩展成5维
    if (dimCp == 4) {
        // 0 is index
        const int64_t appendDim[] = {0};
        // 1 is the dim num to be unsqueezed
        aclIntArray *dimArray = executor->AllocIntArray(appendDim, 1);
        self = l0op::UnsqueezeNd(self, dimArray, executor);
        gradOutput = l0op::UnsqueezeNd(gradOutput, dimArray, executor);
        CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CommonPad3dBackward(const aclTensor *gradOutput, const aclTensor *self,
                                       const aclIntArray *padding, const string& mode, aclTensor *gradInput,
                                       uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, padding, mode, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (gradOutput->IsEmpty() || self->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0;
        // 4 is dim number
        if (self->GetViewShape().GetDimNum() == 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected 4D or 5D tensor with possibly 0 batch size and other non-zero dimentions for input.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 5 is dim number
        if (self->GetViewShape().GetDimNum() == 5) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0 ||
                // 3, 4 are indexes
                self->GetViewShape().GetDim(3) == 0 || self->GetViewShape().GetDim(4) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "Expected 4D or 5D tensor with possibly 0 batch size and other non-zero dimentions for input.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子进行计算
    auto dim = self->GetViewShape().GetDimNum();
    auto dimCp = dim;
    ret = InputPreprocess(gradOutput, self, dimCp, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = GetPaddingTensor(dim, padding, uniqueExecutor.get());
    const aclTensor *pad3dbackwardResult = nullptr;
    bool padSupport = isPadSupport(padding);
    auto originOutDataType = gradOutput->GetDataType();
    // cast to f32
    if (mode == "reflect" && originOutDataType == op::DataType::DT_FLOAT16 && padSupport==false) {
        gradOutput = l0op::Cast(gradOutput, op::DataType::DT_FLOAT, uniqueExecutor.get());
        OP_LOGD("[PadV3GradReflect] F16 Cast to F32: true");
    }
    pad3dbackwardResult = l0op::PadV3Grad(gradOutput, paddingsTensor, mode, true, padSupport, uniqueExecutor.get());
    CHECK_RET(pad3dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果tensor为4维, 需要扩展成5维
    if (dimCp == 4) {
        // 0 is index
        const int64_t appendDim[] = {0};
        // 1 is the dim num to be squeezed
        aclIntArray *dimArray = (uniqueExecutor.get())->AllocIntArray(appendDim, 1);
        pad3dbackwardResult = l0op::SqueezeNd(pad3dbackwardResult, dimArray, uniqueExecutor.get());
        CHECK_RET(pad3dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // cast to f16
    if (mode == "reflect" && originOutDataType == op::DataType::DT_FLOAT16 && padSupport==false) {
        pad3dbackwardResult = l0op::Cast(pad3dbackwardResult, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        OP_LOGD("[PadV3GradReflect] F32 Cast to F16: true");
    }

    // 如果出参gradInput是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(pad3dbackwardResult, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReplicationPad3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
    const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnReplicationPad3dBackward, DFX_IN(gradOutput, self, padding), DFX_OUT(gradInput));
    return CommonPad3dBackward(gradOutput, self, padding, REPLICATION_PAD_MODE, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnReplicationPad3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReplicationPad3dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
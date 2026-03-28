/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_circular_pad2d.h"
#include "padv3.h"
#include "unsqueeze.h"
#include "squeeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const string CIRCULAR_MODE = "circular";
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT32, op::DataType::DT_INT8};

inline static bool CheckNotNull(const aclTensor *self, const aclIntArray *padding, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);

    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

inline static bool CheckFormat(const aclTensor *self, const aclTensor *out)
{
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW、NCL");
        return false;
    }
    OP_CHECK(self->GetViewFormat() == out->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format of input and output should be equal, self [%s], gradInoutput [%s].",
                op::ToString(self->GetViewFormat()).GetString(),
                op::ToString(out->GetViewFormat()).GetString()),
        return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *padding, const aclTensor *out)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self只支持3维和4维
    OP_CHECK_MIN_DIM(self, 3, return false);
    OP_CHECK_MAX_DIM(self, 4, return false);

    // self, out维度需要一致
    OP_CHECK(selfDimnum == out->GetViewShape().GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self, out dim should be same."),
             return false);

    // padding长度为4
    OP_CHECK(padding->Size() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 4, but got %zu.",
                     padding->Size()),
             return false);

    op::Shape expectShape;
    expectShape.SetDimNum(selfDimnum);
    size_t paddingDim = 2;
    if (selfDimnum > paddingDim){
        size_t dimToCompare = selfDimnum - paddingDim;
        for (size_t i = 0; i < dimToCompare; i++){
            expectShape.SetDim(i, self->GetViewShape().GetDim(i));
        }
    }
    // 0, 1, 2, 3 are indexes
    expectShape.SetDim(selfDimnum - 1, self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1]);
    // 0, 1, 2, 3 are indexes
    expectShape.SetDim(selfDimnum - 2, self->GetViewShape().GetDim(selfDimnum - 2) + (*padding)[2] + (*padding)[3]);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *padding, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, padding, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, padding, out), ACLNN_ERR_PARAM_INVALID);

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

inline static aclnnStatus InputPreprocess(const aclTensor *&self, int64_t dimCp, aclOpExecutor *executor)
{
    // 如果非连续，需要转连续
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCircularPad2dGetWorkspaceSize(const aclTensor *self,
    const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnCircularPad2d, DFX_IN(self, padding), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, padding, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        // 3 is dim num
        if (self->GetViewShape().GetDimNum() == 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected 3D or 4D tensor with possibly 0 batch size and other non-zero dimentions for inpit.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 4 is dim num
        if (self->GetViewShape().GetDimNum() == 4) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0 ||
                // 3 is index
                self->GetViewShape().GetDim(3) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "Expected 3D or 4D tensor with possibly 0 batch size and other non-zero dimentions for inpit.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子进行计算
    auto dim = self->GetViewShape().GetDimNum();
    auto dimCp = dim;
    ret = InputPreprocess(self, dimCp, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = GetPaddingTensor(dim, padding, uniqueExecutor.get());
    aclScalar* constantValueScalar = (uniqueExecutor.get())->AllocScalar(0);
    CHECK_RET(constantValueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto constantValueTensor = (uniqueExecutor.get())->ConvertToTensor(constantValueScalar, self->GetDataType());
    const aclTensor *pad2dResult = nullptr;
    pad2dResult = l0op::PadV3(self, paddingsTensor, constantValueTensor, CIRCULAR_MODE, true, uniqueExecutor.get());
    CHECK_RET(pad2dResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(pad2dResult, out, uniqueExecutor.get());
    CHECK_RET((viewCopyResult != nullptr), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
    }

aclnnStatus aclnnCircularPad2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCircularPad2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "aclnn_kl_div_target_backward.h"
#include "kl_div_target_backward_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "broadcast_to.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* MaskedSelect 算子的完整计算流程如下:
 * self                               mask
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_1)
 *      \                             /
 *          \                 Cast(workspace_2)
 *             \                 /
 *             MaskedSelect(workspace_3)
 *                    |
 *               Cast(workspace_4)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> TARGET_DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> TARGET_DTYPE_SUPPORT_LIST_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                                const aclTensor* gradTarget) {
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(target, return false);
    OP_CHECK_NULL(gradTarget, return false);
    return true;
}

static const std::initializer_list<op::DataType> CheckSocVersionIsSupportBf16(void) {
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E)
    {
        return TARGET_DTYPE_SUPPORT_LIST_SUPPORT_BF16;
    }
    return TARGET_DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                            const aclTensor* gradTarget) {
    auto DTYPE_SUPPORT_LIST = CheckSocVersionIsSupportBf16();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(target, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradTarget, DTYPE_SUPPORT_LIST, return false);

    return true;
}

inline static bool isOutSizeSameWithBroadcastShapeSize(const aclTensor* y, op::Shape broadcastShape) {
    int64_t broadcastShapeSize = broadcastShape.GetShapeSize();
    if (y->GetViewShape().GetShapeSize() == broadcastShapeSize) {
        return true;
    }
    return false;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                       const aclTensor* gradTarget) {
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
    const aclTensor* gradTarget) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *target) {
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return CheckType(target->GetDataType(), TARGET_DTYPE_SUPPORT_LIST_SUPPORT_BF16);
    }
    return false;
}

aclnnStatus aclnnKlDivTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    int64_t reduction,
    bool logTarget,
    const aclTensor *gradTarget,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnKlDivTargetBackward, DFX_IN(gradOutput, self, target, reduction, logTarget), DFX_OUT(gradTarget));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, target, gradTarget);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty() || self->IsEmpty() || target->IsEmpty() || gradTarget->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入target转换成连续的tensor
    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* gradOutputBroadcast;
    const aclTensor* selfBroadcast;
    const aclTensor* targetBroadcast;
    gradOutputBroadcast = gradOutputContiguous;
    selfBroadcast = selfContiguous;
    targetBroadcast = targetContiguous;

    if(IsAiCoreSupport(target)){
        // 判断输入shape不相等需要调用BroadcastTo
        if (gradOutput->GetViewShape() != self->GetViewShape() ||
            gradOutput->GetViewShape() != target->GetViewShape() ||
            self->GetViewShape() != target->GetViewShape()) {
            op::Shape broadcastShape;
            if (BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape) &&
                BroadcastInferShape(broadcastShape, target->GetViewShape(), broadcastShape)) {
                op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
                auto broadcastShapeArray =
                    uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
                CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE);
                gradOutputBroadcast =
                    l0op::BroadcastTo(gradOutputContiguous, broadcastShapeArray, uniqueExecutor.get());
                CHECK_RET(gradOutputBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                selfBroadcast = l0op::BroadcastTo(selfContiguous, broadcastShapeArray, uniqueExecutor.get());
                CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                targetBroadcast = l0op::BroadcastTo(targetContiguous, broadcastShapeArray, uniqueExecutor.get());
                CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
        // 调用MaskedSelect算子
        auto out = l0op::KlDivTargetBackward(gradOutputBroadcast, selfBroadcast, targetBroadcast,
            reduction, logTarget, uniqueExecutor.get());
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(out, gradTarget, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKlDivTargetBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnKlDivTargetBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

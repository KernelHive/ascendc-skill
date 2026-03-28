/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "zero_op.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ZerosLike);

static const std::initializer_list<op::DataType> AICORE910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BOOL};


static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST  = {
    op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static constexpr size_t MAX_DIM_LEN = 8;

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self)
{
    // 获取芯片类型,判断是1971还是1980
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
    }

    // 1980 & other
    return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *ZerosLikeAiCore(const aclTensor *self, aclTensor *zeroslike_out, aclOpExecutor *executor)
{
    L0_DFX(ZerosLikeAiCore, self);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ZerosLike, OP_INPUT(self), OP_OUTPUT(zeroslike_out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "ZerosLike ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return zeroslike_out;
}

static bool CheckDim(const aclTensor *self)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  
    return true;
}
  
// AICPU算子kernel
static const aclTensor *ZerosLikeAiCpu(const aclTensor *self, aclTensor *zeroslike_out, aclOpExecutor *executor)
{
    L0_DFX(ZerosLikeAiCpu, self);
    if (!CheckDim(self)) {
        return nullptr;
    }

    static internal::AicpuTaskSpace space("ZerosLike", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ZerosLike, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(zeroslike_out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return zeroslike_out;
}

const aclTensor *ZerosLike(const aclTensor *self, aclOpExecutor *executor)
{
    auto zeroslike_out = executor->AllocTensor(self->GetStorageShape(), self->GetDataType(), self->GetStorageFormat());

    if (IsAiCoreSupport(self)) {
        return ZerosLikeAiCore(self, zeroslike_out, executor);
    } else {
        return ZerosLikeAiCpu(self, zeroslike_out, executor);
    }
}

}
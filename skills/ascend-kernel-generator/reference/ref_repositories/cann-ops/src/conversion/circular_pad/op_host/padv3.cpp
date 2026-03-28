/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "padv3.h"
#include "opdev/op_log.h"
#include "shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PadV3);
OP_TYPE_REGISTER(CircularPad);

static const std::initializer_list<op::DataType> CONSTANT_PAD_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> CONSTANT_PAD_ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> REPLICATION_PAD_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> REPLICATION_PAD_ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> CIRCULAR_PAD_ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_INT32};

static const string CONSTANT_MODE = "constant";
static const string REFLECTION_MODE = "reflect";
static const string REPLICATION_MODE = "edge";
static const string CIRCULAR_MODE = "circular";
static const std::unordered_set<std::string> VALID_MODES = {CONSTANT_MODE, REFLECTION_MODE, REPLICATION_MODE};
static const size_t AI_CORE_CONSTANT_PAD_DIM_BOUND = 8;
static const size_t AI_CORE_REPLICATION_PAD_DIM_BOUND = 5;

inline static bool IsAiCoreSupport(const aclTensor *self, const std::string& mode)
{
    if (VALID_MODES.find(mode) == VALID_MODES.end()) {
        return false;
    }
    if (mode == CONSTANT_MODE){
        if (self->GetViewShape().GetDimNum() > AI_CORE_CONSTANT_PAD_DIM_BOUND) {
            return false;
        }
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            return CheckType(self->GetDataType(), CONSTANT_PAD_ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
        }
        return CheckType(self->GetDataType(), CONSTANT_PAD_AICORE_DTYPE_SUPPORT_LIST);
    } else if (mode == REPLICATION_MODE){
        if (self->GetViewShape().GetDimNum() > AI_CORE_REPLICATION_PAD_DIM_BOUND) {
            return false;
        }
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            return CheckType(self->GetDataType(), REPLICATION_PAD_ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
        }
        return CheckType(self->GetDataType(), REPLICATION_PAD_AICORE_DTYPE_SUPPORT_LIST);
    }
}

inline const aclTensor *CircularPadAiCore(const aclTensor *self, const aclTensor *paddings,
    aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(CircularPadAiCore, self, paddings, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CircularPad, OP_INPUT(self, paddings), OP_OUTPUT(out));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CircularPadAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

inline const aclTensor *PadV3AiCore(const aclTensor *self, const aclTensor *paddings,
    const aclTensor *constant_values, const std::string& mode, const bool paddingsContiguous,
    aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(PadV3AiCore, self, paddings, constant_values, mode, paddingsContiguous, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PadV3, OP_INPUT(self, paddings, constant_values), OP_OUTPUT(out),
                                           OP_ATTR(mode, paddingsContiguous));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadV3AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor *PadV3(const aclTensor *self,
                       const aclTensor *paddings,
                       const aclTensor *constant_values,
                       const std::string& mode,
                       const bool paddingsContiguous,
                       aclOpExecutor *executor)
{
    auto out = executor->AllocTensor(self->GetDataType(), self->GetViewFormat(), self->GetViewFormat());
    auto ret = INFER_SHAPE(PadV3, OP_INPUT(self, paddings), OP_OUTPUT(out), OP_ATTR(mode, paddingsContiguous));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PadV3 InferShape failed.");
        return nullptr;
    }
    if (mode == CIRCULAR_MODE) {
        return CircularPadAiCore(self, paddings, out, executor);
    }
    
    if (IsAiCoreSupport(self, mode)) {
        return PadV3AiCore(self, paddings, constant_values, mode, paddingsContiguous, out, executor);
    }
}
}

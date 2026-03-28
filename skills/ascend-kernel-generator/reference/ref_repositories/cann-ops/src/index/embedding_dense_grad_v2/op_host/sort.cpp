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

/*!
 * \file sort.cpp
 * \brief
 */
#include "sort.h"
#include "opdev/platform.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sort);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DATA_LIMIT = 100000;
static const int64_t AXIS_LIMIT = 8;

// 根据排序轴的数据量大小判断是否支持aicore
static bool SocSupportDimSize(const aclTensor *self)
{
    // 该维度数据量为1 或数据量>100000 走AICPU
    auto shapeSize = (int64_t)(self->GetViewShape().GetDimNum());
    auto lastDimSize = (self->GetViewShape())[shapeSize-1];
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND310 || socVersion == SocVersion::ASCEND310B) {
        // sort轴数据量大于100k
        if (lastDimSize > DATA_LIMIT) {
            return false;
        }
    } else {
        // sort轴数据量为1
        if (lastDimSize == 1) {
            return false;
        }
    }
    return true;
}

// 根据dtype判断是否支持aicore
static bool SocSupportDtype(const aclTensor *self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    // AiCore只支持FLOAT16 + FLOAT32
    if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        // 910和310芯片 + tensor为FLOAT32 则不支持AiCore
        if (((socVersion == SocVersion::ASCEND910) || (socVersion == SocVersion::ASCEND310)) &&
            (self->GetDataType()==op::DataType::DT_FLOAT || self->GetDataType()==op::DataType::DT_BF16)) {
            return false;
        }
        return true;
    }
    return false;
}

static bool IsAiCoreSupport(const aclTensor *self)
{
    return (SocSupportDimSize(self) && SocSupportDtype(self));
}


void SortAiCore(const aclTensor *self, bool stable, int64_t dim, bool descending, aclTensor *values, aclTensor *indices,
    aclOpExecutor* executor)
{
    L0_DFX(SortAiCore, self, stable, dim, descending, values, indices);
    auto dimSize = (int64_t)(self->GetViewShape().GetDimNum());
    if ((dimSize!= dim + 1) && (dim != -1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must equal to the (number of dimensions - 1 ) or -1.");
    }

    ADD_TO_LAUNCHER_LIST_AICORE(Sort, OP_INPUT(self), OP_OUTPUT(values, indices), OP_ATTR(dim, descending, stable));
}


std::tuple<aclTensor*, aclTensor*> SortAiCpu(const aclTensor *self, bool stable, int64_t dim, bool descending,
                                             aclTensor *values, aclTensor *indices, aclOpExecutor* executor)
{
    L0_DFX(SortAiCpu, self, stable, dim, descending, values, indices);

    auto dimSize = (int64_t)(self->GetViewShape().GetDimNum());
    if ((dim > (dimSize-1)) || (dim + dimSize < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be in range [-N, N-1]. Current dim is %ld.", dim);
    }

    static internal::AicpuTaskSpace space("Sort");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Sort, OP_ATTR_NAMES({"axis", "descending", "stable"}), OP_INPUT(self),
        OP_OUTPUT(values, indices), OP_ATTR(dim, descending, stable));
    if (ret != ACLNN_SUCCESS) {
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tie(values, indices);
}

const std::tuple<aclTensor*, aclTensor*> Sort(const aclTensor *self, int64_t dim, bool descending, bool stable,
    aclOpExecutor* executor)
{
    auto selfShape = self->GetViewShape();
    auto selfFormat = self->GetViewFormat();

    auto dimSize = (int64_t)(selfShape.GetDimNum());
    if (dimSize < 1 || dimSize > AXIS_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor self dimension size must be in range [1, 8]. Current size is [%ld].",
            dimSize);
    }
    auto lastDimSize = selfShape[dimSize - 1];
    // The Sort Op not support sort axis is 1 when input type is BF16..
    bool  isNotSupport = (1 == lastDimSize && op::DataType::DT_BF16 == self->GetDataType());
    if (isNotSupport) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The sort axis value is not support 1 when input type is BF16.");
      return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    auto values = executor->AllocTensor(selfShape, self->GetDataType(), selfFormat);
    auto indices = executor->AllocTensor(selfShape, DataType::DT_INT32, selfFormat);

    if (IsAiCoreSupport(self)) {
        SortAiCore(self, stable, dim, descending, values, indices, executor);
    } else {
        return SortAiCpu(self, stable, dim, descending, values, indices, executor);
    }
    return std::tie(values, indices);
}

}

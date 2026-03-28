/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest_l0.cpp
 * \brief
 */
#include "upsample_nearest_l0.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearest);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;

const aclTensor *UpsampleNearestExact2d(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *output,
    float scalesH, float scalesW, bool exactMode, aclOpExecutor *executor)
{
    L0_DFX(UpsampleNearestExact2d, self, outputSize, scalesH, scalesW);

    op::Shape selfStorageShape = self->GetStorageShape();
    op::Shape selfOriginalShape = self->GetOriginalShape();
    op::Format selfFormat = self->GetStorageFormat();
    if (self->GetViewShape().GetDimNum() == DIM_THREE) {
        // NLC
        const int64_t sizeL = (*outputSize)[DIM_ZERO];
        selfStorageShape.SetDim(DIM_ONE, sizeL);
        selfOriginalShape.SetDim(DIM_ONE, sizeL);
    } else {
        const int64_t sizeH = (*outputSize)[DIM_ZERO];
        const int64_t sizeW = (*outputSize)[DIM_ONE];
        if (selfFormat == op::Format::FORMAT_NCHW) {
            selfStorageShape.SetDim(DIM_TWO, sizeH);
            selfStorageShape.SetDim(DIM_THREE, sizeW);
            selfOriginalShape.SetDim(DIM_TWO, sizeH);
            selfOriginalShape.SetDim(DIM_THREE, sizeW);
        } else {
            selfStorageShape.SetDim(DIM_ONE, sizeH);
            selfStorageShape.SetDim(DIM_TWO, sizeW);
            selfOriginalShape.SetDim(DIM_ONE, sizeH);
            selfOriginalShape.SetDim(DIM_TWO, sizeW);
        }
    }

    const aclTensor *out = executor->AllocTensor(
        selfStorageShape, selfOriginalShape, self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    CHECK_RET(out != nullptr, nullptr);

    ADD_TO_LAUNCHER_LIST_AICORE(
        UpsampleNearest, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize, scalesH, scalesW, exactMode));
    return out;
}
}  // namespace l0op

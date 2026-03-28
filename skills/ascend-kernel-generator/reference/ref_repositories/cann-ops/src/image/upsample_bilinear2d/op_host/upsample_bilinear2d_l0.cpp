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
 * \file upsample_bilinear2d_l0.cpp
 * \brief
 */
#include "upsample_bilinear2d_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_def.h"

#include "opdev/op_executor.h"

#include "opdev/common_types.h"
#include "opdev/shape_utils.h"

#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBilinear2d);

static const string LINEAR_MODE = "linear";
static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DIM_THREE = 3;

const aclTensor *UpsampleBilinear2dNcdhw(const aclTensor *x, const aclTensor *size, const bool alignCorners,
    const double scalesH, const double scalesW, const aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(UpsampleBilinear2dNcdhw, x, size, alignCorners, scalesH, scalesW, y);

    auto dataType = x->GetDataType();
    auto inputShape = x->GetViewShape();
    auto outputShape = y->GetViewShape();
    auto input_h = inputShape.GetDim(DIM_TWO);
    auto input_w = inputShape.GetDim(DIM_THREE);
    auto output_h = outputShape.GetDim(DIM_TWO);
    auto output_w = outputShape.GetDim(DIM_THREE);

    vector<float> scalesList{};

    if (scalesH > 0 && scalesW > 0 && (scalesH != 1.0 || scalesW != 1.0)) {
        scalesList.push_back(static_cast<float>(1 / scalesH));
        scalesList.push_back(static_cast<float>(1 / scalesW));
    } else if (scalesH == 1.0 && scalesW == 1.0 && (output_h != input_h || output_w != input_w)) {
        scalesList.push_back(0.0);
        scalesList.push_back(0.0);
    } else {
        scalesList.push_back(1.0);
        scalesList.push_back(1.0);
    }

    const aclFloatArray *realScales = executor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(realScales != nullptr, nullptr);
    // AICORE
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        UpsampleBilinear2d, OP_INPUT(x, size), OP_OUTPUT(y), OP_ATTR(alignCorners, realScales));
    return y;
}
}  // namespace l0op

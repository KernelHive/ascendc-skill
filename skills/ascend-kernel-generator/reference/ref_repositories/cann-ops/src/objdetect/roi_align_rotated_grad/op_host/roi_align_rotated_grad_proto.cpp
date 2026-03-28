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
 * @file roi_align_rotated_grad_proto.cpp
 */
#include "register/op_def_registry.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_X_GRAD = 0;
const uint32_t INPUT_ROIS = 1;

const uint32_t INPUT_Y_GRAD_SHAPE = 0;
const uint32_t INPUT_POOLED_H = 1;
const uint32_t INPUT_POOLED_W = 2;
const uint32_t INPUT_SPATIAL_SCALE = 3;
const uint32_t INPUT_SAMPLING_RATIO = 4;
const uint32_t INPUT_ALIGNED = 5;
const uint32_t INPUT_CLOCKWISE = 6;

const uint32_t BOX_SIZE_DIM = 1;
const uint32_t BATCH_SIZE_DIM = 0;
const uint32_t HEIGHT_DIM = 1;
const uint32_t WIDTH_DIM = 2;
const uint32_t CHANNEL_DIM = 3;

const uint32_t OUTPUT_Y_GRAD = 0;
} // namespace

namespace ge {
static ge::graphStatus InferShapeForRoiAlignRotatedGrad(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto inputShape = attrs->GetListInt(INPUT_Y_GRAD_SHAPE)->GetData();
    uint32_t batchSize = inputShape[BATCH_SIZE_DIM];
    uint32_t channels = inputShape[CHANNEL_DIM];
    uint32_t height = inputShape[HEIGHT_DIM];
    uint32_t width = inputShape[WIDTH_DIM];

    gert::Shape* gradInputShape = context->GetOutputShape(OUTPUT_Y_GRAD);
    if (gradInputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gradInputShape->AppendDim(batchSize);
    gradInputShape->AppendDim(height);
    gradInputShape->AppendDim(width);
    gradInputShape->AppendDim(channels);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForRoiAlignRotatedGrad(gert::InferDataTypeContext* context)
{
    auto gradOutputDtype = context->GetInputDataType(INPUT_X_GRAD);
    context->SetOutputDataType(OUTPUT_Y_GRAD, gradOutputDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiAlignRotatedGrad).InferShape(InferShapeForRoiAlignRotatedGrad).InferDataType(InferDataTypeForRoiAlignRotatedGrad);
}
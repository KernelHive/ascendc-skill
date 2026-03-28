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
 * @file roi_align_rotated_grad.cpp
 */
#include <cmath>
#include "roi_align_rotated_grad_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_type.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_X_GRAD = 0;
const uint32_t INPUT_ROIS = 1;

const uint32_t INPUT_Y_GRAD_LIST = 0;
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

const uint32_t WORKSAPCE_16MBYTE_SIZE = 16 * 1024 * 1024;
template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
    return context->GetCompiledInfo<T>();
}

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
    if ((ptr) == nullptr) {                                                                        \
        std::printf("nullptr error!");                                                               \
        return ge::GRAPH_FAILED;                                                                     \
    }
} // namespace

namespace optiling {
static ge::graphStatus TilingPrepareForRoiAlignRotatedGrad(gert::TilingParseContext* context) {
    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform_info = platform_ascendc::PlatformAscendC(platform);
    uint64_t ub_total_size;
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
    uint32_t coreNum = platform_info.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = GetCompileInfoPtr<RoiAlignRotatedGradCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->totalCoreNum = coreNum;
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ub_total_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncForRoiAlignRotatedGrad(gert::TilingContext* context)
{
    RoiAlignRotatedGradTilingData tiling;
    auto roiTensorPtr = context->GetInputTensor(INPUT_ROIS);
    auto gradOutputTensorPtr = context->GetInputTensor(INPUT_X_GRAD);
    if (roiTensorPtr == nullptr || gradOutputTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto roiShape = roiTensorPtr->GetStorageShape();

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t boxSize = roiShape.GetDim(BOX_SIZE_DIM);
    int32_t pooledHeight = *(attrs->GetAttrPointer<uint32_t>(INPUT_POOLED_H));
    int32_t pooledWidth = *(attrs->GetAttrPointer<uint32_t>(INPUT_POOLED_W));

    auto inputShape = attrs->GetListInt(INPUT_Y_GRAD_LIST)->GetData();
    uint32_t batchSize = inputShape[BATCH_SIZE_DIM];
    uint32_t channelNum = inputShape[CHANNEL_DIM];
    uint32_t height = inputShape[HEIGHT_DIM];
    uint32_t width = inputShape[WIDTH_DIM];

    bool aligned = *(attrs->GetAttrPointer<bool>(INPUT_ALIGNED));
    bool clockwise = *(attrs->GetAttrPointer<bool>(INPUT_CLOCKWISE));
    int32_t samplingRatio = *(attrs->GetAttrPointer<uint32_t>(INPUT_SAMPLING_RATIO));
    float spatialScale = *(attrs->GetAttrPointer<float>(INPUT_SPATIAL_SCALE));

    uint32_t coreRoisNums = boxSize / coreNum;
    uint32_t coreRoisTail = boxSize % coreNum;

    tiling.set_coreRoisNums(coreRoisNums);
    tiling.set_coreRoisTail(coreRoisTail);
    tiling.set_boxSize(boxSize);
    tiling.set_pooledHeight(pooledHeight);
    tiling.set_pooledWidth(pooledWidth);
    tiling.set_batchSize(batchSize);
    tiling.set_channelNum(channelNum);
    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_aligned(aligned);
    tiling.set_clockwise(clockwise);
    tiling.set_samplingRatio(samplingRatio);
    tiling.set_spatialScale(spatialScale);

    tiling.set_coreNum(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSAPCE_16MBYTE_SIZE;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RoiAlignRotatedGrad)
    .Tiling(TilingFuncForRoiAlignRotatedGrad)
    .TilingParse<RoiAlignRotatedGradCompileInfo>(TilingPrepareForRoiAlignRotatedGrad);
}
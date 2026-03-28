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
 * @file roi_align_rotated_grad_tiling.h
 */
#ifndef ROI_ALIGN_ROTATED_GRAD_TILING_H
#define ROI_ALIGN_ROTATED_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
struct RoiAlignRotatedGradCompileInfo {
  uint32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
};
BEGIN_TILING_DATA_DEF(RoiAlignRotatedGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreRoisNums);
    TILING_DATA_FIELD_DEF(uint32_t, coreRoisTail);
    TILING_DATA_FIELD_DEF(uint32_t, boxSize);
    TILING_DATA_FIELD_DEF(int32_t, pooledHeight);
    TILING_DATA_FIELD_DEF(int32_t, pooledWidth);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelNum);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(bool, aligned);
    TILING_DATA_FIELD_DEF(bool, clockwise);
    TILING_DATA_FIELD_DEF(int32_t, samplingRatio);
    TILING_DATA_FIELD_DEF(float, spatialScale);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoiAlignRotatedGrad, RoiAlignRotatedGradTilingData)
} // namespace optiling
#endif // ROI_ALIGN_ROTATED_GRAD_TILING_H
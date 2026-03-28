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
 * @file roi_align_rotated_tiling.h
 */
#ifndef ROI_ALIGN_ROTATED_TILING_H
#define ROI_ALIGN_ROTATED_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
struct RoiAlignRotatedCompileInfo {
  uint32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
};

BEGIN_TILING_DATA_DEF(RoiAlignRotatedTilingData)
    TILING_DATA_FIELD_DEF(uint8_t, aligned);
    TILING_DATA_FIELD_DEF(uint8_t, clockwise);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, rois_num_per_Lcore);
    TILING_DATA_FIELD_DEF(uint32_t, rois_num_per_Score);
    TILING_DATA_FIELD_DEF(uint32_t, Lcore_num);
    TILING_DATA_FIELD_DEF(uint32_t, Score_num);
    TILING_DATA_FIELD_DEF(uint32_t, input_buffer_size);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, batch_size);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, channels_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, input_h);
    TILING_DATA_FIELD_DEF(uint32_t, input_w);
    TILING_DATA_FIELD_DEF(uint32_t, rois_num_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, tail_num);
    TILING_DATA_FIELD_DEF(float, spatial_scale);
    TILING_DATA_FIELD_DEF(int32_t, sampling_ratio);
    TILING_DATA_FIELD_DEF(int32_t, pooled_height);
    TILING_DATA_FIELD_DEF(int32_t, pooled_width);
    TILING_DATA_FIELD_DEF(uint64_t, ub_total_size);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoiAlignRotated, RoiAlignRotatedTilingData)
} // namespace optiling
#endif // ROI_ALIGN_ROTATED_TILING_H
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POINTS_TO_VOXEL_TILING_H
#define POINTS_TO_VOXEL_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PointsToVoxelTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(int32_t, ndimlength);
  TILING_DATA_FIELD_DEF(int32_t, Nlength);
  TILING_DATA_FIELD_DEF(float, voxel_size_x);
  TILING_DATA_FIELD_DEF(float, voxel_size_y);
  TILING_DATA_FIELD_DEF(float, voxel_size_z);
  TILING_DATA_FIELD_DEF(float, coors_range_xL);
  TILING_DATA_FIELD_DEF(float, coors_range_yL);
  TILING_DATA_FIELD_DEF(float, coors_range_zL);
  TILING_DATA_FIELD_DEF(int32_t, voxelmap_shape_xR);
  TILING_DATA_FIELD_DEF(int32_t, voxelmap_shape_yR);
  TILING_DATA_FIELD_DEF(int32_t, voxelmap_shape_zR);
  TILING_DATA_FIELD_DEF(int32_t, max_points);
  TILING_DATA_FIELD_DEF(bool, reverse_index);
  TILING_DATA_FIELD_DEF(int32_t, max_voxels);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PointsToVoxel, PointsToVoxelTilingData)
}

#endif  // POINTS_TO_VOXEL_TILING_H
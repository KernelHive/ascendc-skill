/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOTION_COMPENSATION_TILING_H
#define MOTION_COMPENSATION_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MotionCompensationTilingData)
  TILING_DATA_FIELD_DEF(int64_t, N);
  TILING_DATA_FIELD_DEF(int64_t, ndim);
  TILING_DATA_FIELD_DEF(float, f);
  TILING_DATA_FIELD_DEF(float, theta);
  TILING_DATA_FIELD_DEF(float, r_sin_theta);
  TILING_DATA_FIELD_DEF(float, d_sign);
  TILING_DATA_FIELD_DEF(int32_t, doRotation);
  TILING_DATA_FIELD_DEF(float, tMax);
  TILING_DATA_FIELD_DEF(int32_t, tMaxLow32);

  TILING_DATA_FIELD_DEF_ARR(float, 3, trans);
  TILING_DATA_FIELD_DEF_ARR(float, 4, qRel);
  
  TILING_DATA_FIELD_DEF(int64_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(int64_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(int64_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(int64_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(int64_t, tileDataNum);
  TILING_DATA_FIELD_DEF(int64_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(int64_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(int64_t, tailBlockNum);
  
  TILING_DATA_FIELD_DEF(int64_t, smallCoreDataNum2);
  TILING_DATA_FIELD_DEF(int64_t, bigCoreDataNum2);
  TILING_DATA_FIELD_DEF(int64_t, finalBigTileNum2);
  TILING_DATA_FIELD_DEF(int64_t, finalSmallTileNum2);
  TILING_DATA_FIELD_DEF(int64_t, tileDataNum2);
  TILING_DATA_FIELD_DEF(int64_t, smallTailDataNum2);
  TILING_DATA_FIELD_DEF(int64_t, bigTailDataNum2);
  TILING_DATA_FIELD_DEF(int64_t, tailBlockNum2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MotionCompensation, MotionCompensationTilingData)
}

#endif  // MOTION_COMPENSATION_TILING_H
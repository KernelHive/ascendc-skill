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
 * \file adaptive_avg_pool3d_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdaptiveAvgPool3dTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, dimC);
  TILING_DATA_FIELD_DEF(uint64_t, cTileLength);
  TILING_DATA_FIELD_DEF(uint64_t, inD);
  TILING_DATA_FIELD_DEF(uint64_t, inH);
  TILING_DATA_FIELD_DEF(uint64_t, inW);
  TILING_DATA_FIELD_DEF(uint64_t, outD);
  TILING_DATA_FIELD_DEF(uint64_t, outH);
  TILING_DATA_FIELD_DEF(uint64_t, outW);
  TILING_DATA_FIELD_DEF(uint64_t, formerLength);
  TILING_DATA_FIELD_DEF(uint64_t, formerNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailNum);
  TILING_DATA_FIELD_DEF(uint64_t, indexBufLen);
  TILING_DATA_FIELD_DEF(uint64_t, windowWNum);
  TILING_DATA_FIELD_DEF(uint64_t, maxWindowWLength);
  TILING_DATA_FIELD_DEF(uint64_t, inputTileNum);
  TILING_DATA_FIELD_DEF(uint64_t, atomicAddNum);
END_TILING_DATA_DEF;

struct AdaptiveAvgPool3dCompileInfo {
  int32_t totalCoreNum = 0;
  uint32_t sysWorkspaceSize = 0;
  uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(AdaptiveAvgPool3d, AdaptiveAvgPool3dTilingData)
} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_H_

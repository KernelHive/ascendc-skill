/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdaptiveAvgPool3dGradTilingData)
  TILING_DATA_FIELD_DEF(int64_t, ncNum);

  TILING_DATA_FIELD_DEF(int64_t, dIn);
  TILING_DATA_FIELD_DEF(int64_t, hIn);
  TILING_DATA_FIELD_DEF(int64_t, wIn);
  
  TILING_DATA_FIELD_DEF(int64_t, dOut);
  TILING_DATA_FIELD_DEF(int64_t, hOut);
  TILING_DATA_FIELD_DEF(int64_t, wOut);

  TILING_DATA_FIELD_DEF(int64_t, taskCoreUsed);
  TILING_DATA_FIELD_DEF(int64_t, taskNumPerCore);
  TILING_DATA_FIELD_DEF(int64_t, taskNumLastCore);
  TILING_DATA_FIELD_DEF(int64_t, yNumPerCalc);

  TILING_DATA_FIELD_DEF(int64_t, ncSliceNum);
  TILING_DATA_FIELD_DEF(int64_t, ncAlignSliceLength);
  TILING_DATA_FIELD_DEF(int64_t, ncAlignSliceTail);

  TILING_DATA_FIELD_DEF(int64_t, isAtomicAdd);
  TILING_DATA_FIELD_DEF(int64_t, deterministicFlag);
END_TILING_DATA_DEF;

struct AdaptiveAvgPool3dGradCompileInfo {
  uint32_t coreNum = 0;
  uint64_t sysWorkspaceSize = 0;
  uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingData)
}

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
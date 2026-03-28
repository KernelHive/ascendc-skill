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
 * \file avg_pool3d_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_AVG_POOL3D_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_AVG_POOL3D_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AvgPool3DTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, inN);
  TILING_DATA_FIELD_DEF(uint64_t, inC);
  TILING_DATA_FIELD_DEF(uint64_t, tileC);
  TILING_DATA_FIELD_DEF(uint64_t, inD);
  TILING_DATA_FIELD_DEF(uint64_t, inH);
  TILING_DATA_FIELD_DEF(uint64_t, inW);
  TILING_DATA_FIELD_DEF(uint64_t, outD);
  TILING_DATA_FIELD_DEF(uint64_t, outH);
  TILING_DATA_FIELD_DEF(uint64_t, outW);
  TILING_DATA_FIELD_DEF(uint64_t, kD);
  TILING_DATA_FIELD_DEF(uint64_t, kH);
  TILING_DATA_FIELD_DEF(uint64_t, kW);
  TILING_DATA_FIELD_DEF(uint64_t, dD);
  TILING_DATA_FIELD_DEF(uint64_t, dH);
  TILING_DATA_FIELD_DEF(uint64_t, dW);
  TILING_DATA_FIELD_DEF(uint64_t, pD);
  TILING_DATA_FIELD_DEF(uint64_t, pH);
  TILING_DATA_FIELD_DEF(uint64_t, pW);
  TILING_DATA_FIELD_DEF(int64_t, divisorOverride);
  TILING_DATA_FIELD_DEF(uint64_t, countIncludePad);
  TILING_DATA_FIELD_DEF(uint64_t, ceilMode);
  TILING_DATA_FIELD_DEF(uint64_t, formerLength);
  TILING_DATA_FIELD_DEF(uint64_t, formerNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailNum);
  TILING_DATA_FIELD_DEF(uint64_t, indexBufLen);
  TILING_DATA_FIELD_DEF(uint64_t, windowWNum);
  TILING_DATA_FIELD_DEF(uint64_t, tileInput);
  TILING_DATA_FIELD_DEF(uint64_t, tileHW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AvgPool3D, AvgPool3DTilingData)
}  // namespace optiling

#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_AVG_POOL3D_H_
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
 * \file ge_glu_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GEGLUV2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GEGLUV2_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeGluV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, group);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, tailLoopNum);
TILING_DATA_FIELD_DEF(int64_t, nLastTailGroup);
TILING_DATA_FIELD_DEF(int64_t, lastTailGroup);
TILING_DATA_FIELD_DEF(int64_t, splitSize);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, activateLeft);
TILING_DATA_FIELD_DEF(int64_t, ny);
TILING_DATA_FIELD_DEF(int64_t, approximate);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeGluV2, GeGluV2TilingData)

struct TilingParam {
  int64_t x = 1;
  int64_t ny = 1;
  int64_t coreNum = 0;
  int64_t blockSize = 0;
  int64_t bufSize = 1;
  int64_t activateLeft = 0;
  int64_t approximate = 0;
  bool isAscend310P = false;
};

enum class GeGluV2TilingKey : int64_t {
    TILINGKEY_101 = 101,
    TILINGKEY_102 = 102,
    TILINGKEY_103 = 103,
    TILINGKEY_201 = 201,
    TILINGKEY_202 = 202,
    TILINGKEY_203 = 203,
    TILINGKEY_301 = 301,
    TILINGKEY_302 = 302,
    TILINGKEY_303 = 303,
    TILINGKEY_111 = 111,
    TILINGKEY_112 = 112,
    TILINGKEY_113 = 113,
    TILINGKEY_211 = 211,
    TILINGKEY_212 = 212,
    TILINGKEY_213 = 213,
    TILINGKEY_311 = 311,
    TILINGKEY_312 = 312,
    TILINGKEY_313 = 313
};
} // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GEGLUV2_H_

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
 * @file heaviside_tiling.h
 */
#ifndef HEAVISIDE_TILING_H
#define HEAVISIDE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HeavisideTilingData)
TILING_DATA_FIELD_DEF(uint32_t, smallSize);
TILING_DATA_FIELD_DEF(uint16_t, incSize);
TILING_DATA_FIELD_DEF(uint16_t, formerNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Heaviside, HeavisideTilingData)
BEGIN_TILING_DATA_DEF(HeavisideTilingData_BroadCast)
TILING_DATA_FIELD_DEF(uint32_t, size);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmInputDims);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmValuesDims);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmOutputDims);
TILING_DATA_FIELD_DEF(uint8_t, nOutputDims);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Heaviside_5, HeavisideTilingData_BroadCast)
} // namespace optiling
#endif // HEAVISIDE_TILING_H
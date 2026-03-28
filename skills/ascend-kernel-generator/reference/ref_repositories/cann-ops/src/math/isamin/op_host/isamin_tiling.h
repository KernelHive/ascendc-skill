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
 * @file isamin_tiling.h
 */
#ifndef ISAMIN_TILING_H
#define ISAMIN_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {

constexpr static int MAX_ARRAY_NUM = 40;

BEGIN_TILING_DATA_DEF(IsaminTilingData)
TILING_DATA_FIELD_DEF(uint32_t, incx);
TILING_DATA_FIELD_DEF(uint32_t, needVecCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dytpeFlag);
TILING_DATA_FIELD_DEF(uint32_t, rstLenAllCoreBytes);
TILING_DATA_FIELD_DEF(uint32_t, tailCount);
TILING_DATA_FIELD_DEF(uint32_t, maxRepeatLen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, startOffset);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, eleTotalEachCore);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, dealTimesEachCore);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, dealLenEachTime);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, reduceMaxRstsLenEachCore);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, dealLenUpBlockEachTime);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, totalRptCntNor);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, totalRptCntNorRemainder);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, rptBatchCntNor);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, rptBatchCntNorRemainder);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, rmdRptLenNor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Isamin, IsaminTilingData)
} // namespace optiling
#endif // ISAMIN_TILING_H

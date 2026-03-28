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
 * @file sasum_tiling.h
 */
#ifndef SASUM_TILING_H
#define SASUM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {

constexpr static int MAX_ARRAY_NUM = 40;

BEGIN_TILING_DATA_DEF(SasumTilingData)
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, startOffset);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, calNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Sasum, SasumTilingData)
} // namespace optiling
#endif // SASUM_TILING_H

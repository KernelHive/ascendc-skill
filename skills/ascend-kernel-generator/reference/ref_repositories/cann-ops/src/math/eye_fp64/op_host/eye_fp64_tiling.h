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
 * @file eye_fp64_tiling.h
 */
#ifndef EYE_FP64_TILING_H
#define EYE_FP64_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(EyeFp64TilingData)
TILING_DATA_FIELD_DEF(uint16_t, totalMatrixNum);
TILING_DATA_FIELD_DEF(uint16_t, numRows);
TILING_DATA_FIELD_DEF(uint16_t, numColumns);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EyeFp64, EyeFp64TilingData)
BEGIN_TILING_DATA_DEF(EyeFp64TilingData_slice)
TILING_DATA_FIELD_DEF(uint64_t, mask0);
TILING_DATA_FIELD_DEF(uint64_t, mask1);
TILING_DATA_FIELD_DEF(uint64_t, mask_remain0);
TILING_DATA_FIELD_DEF(uint64_t, mask_remain1);
TILING_DATA_FIELD_DEF(int, totalMatrixNum);
TILING_DATA_FIELD_DEF(int, numRows);
TILING_DATA_FIELD_DEF(int, numColumns);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EyeFp64_2, EyeFp64TilingData_slice)
} // namespace optiling
#endif // EYE_FP64_TILING_H

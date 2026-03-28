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
 * @file mul_mul_reduce_mean_d_twice_tiling.h
 */
#ifndef MUL_MUL_REDUCE_MEAN_D_TWICE_TILING_H
#define MUL_MUL_REDUCE_MEAN_D_TWICE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MulMulReduceMeanDTwiceTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint64_t, formerNum); 
  TILING_DATA_FIELD_DEF(uint64_t, formerLength); 
  TILING_DATA_FIELD_DEF(uint64_t, tailNum); 
  TILING_DATA_FIELD_DEF(uint64_t, tailLength); 
  TILING_DATA_FIELD_DEF(uint64_t, tileLength); 
  TILING_DATA_FIELD_DEF(uint64_t, shareSize); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulMulReduceMeanDTwice, MulMulReduceMeanDTwiceTilingData)
}
#endif // MUL_MUL_REDUCE_MEAN_D_TWICE_TILING_H
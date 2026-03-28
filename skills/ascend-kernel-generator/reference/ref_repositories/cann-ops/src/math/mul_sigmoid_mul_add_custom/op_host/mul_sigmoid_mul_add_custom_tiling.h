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
 * @file mul_sigmoid_mul_add_custom_tiling.h
 */

#ifndef MUL_SIGMOID_MUL_ADD_CUSTOM_TILING_H
#define MUL_SIGMOID_MUL_ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MulSigmoidMulAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLen);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, completeTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, partTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, completeTileLen);
  TILING_DATA_FIELD_DEF(uint32_t, partTileLen);
  TILING_DATA_FIELD_DEF(uint32_t, totalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, frontBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, latterBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumInFrontBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumInLatterBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulSigmoidMulAddCustom, MulSigmoidMulAddCustomTilingData)
}
#endif // MUL_SIGMOID_MUL_ADD_CUSTOM_TILING_H

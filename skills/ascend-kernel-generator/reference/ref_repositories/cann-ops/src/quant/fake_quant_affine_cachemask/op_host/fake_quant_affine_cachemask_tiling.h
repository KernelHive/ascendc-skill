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
 * \file fake_quant_affine_cachemask_tiling.h
 * \brief
 */

#ifndef FAKE_QUANT_AFFINE_CACHEMASK_TILING_H
#define FAKE_QUANT_AFFINE_CACHEMASK_TILING_H
#include "register/tilingdata_base.h"

// 只copy中间和头文件，宏定义和copyright要保持规范
namespace optiling {
BEGIN_TILING_DATA_DEF(FakeQuantAffineCachemaskTilingData)
  TILING_DATA_FIELD_DEF(int64_t, quantMin);
  TILING_DATA_FIELD_DEF(int64_t, quantMax);
  TILING_DATA_FIELD_DEF(uint32_t, loopNum);
  TILING_DATA_FIELD_DEF(uint32_t, remainNum);
  TILING_DATA_FIELD_DEF(uint32_t, calcLength);
  TILING_DATA_FIELD_DEF(uint32_t, headNum);
  TILING_DATA_FIELD_DEF(uint32_t, dataPerRepeat);
  TILING_DATA_FIELD_DEF(uint32_t, totalLengthAligned);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FakeQuantAffineCachemask, FakeQuantAffineCachemaskTilingData)
} // namespace optiling
#endif // FAKE_QUANT_AFFINE_CACHEMASK_TILING_H

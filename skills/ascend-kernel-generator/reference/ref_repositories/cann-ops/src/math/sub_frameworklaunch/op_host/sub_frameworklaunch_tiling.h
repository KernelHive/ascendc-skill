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
 * @file sub_frameworklaunch_tiling.h
 */
#ifndef SUB_FRAMEWORKLAUNCH_TILING_H
#define SUB_FRAMEWORKLAUNCH_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SubFrameworklaunchTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);     // number of cores allocated to a larger amount of data, i.e., large blocks
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);       // number of cores allocated to a smaller amount of data, i.e., small blocks
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);  // length of the large block
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);    // length of the small block
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);      // minimum data amount that needs to be aligned
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubFrameworklaunch, SubFrameworklaunchTilingData)
} // namespace optiling
#endif // SUB_FRAMEWORKLAUNCH_TILING_H
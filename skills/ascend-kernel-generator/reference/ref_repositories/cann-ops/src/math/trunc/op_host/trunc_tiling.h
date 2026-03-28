/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef TRUNC_TILING_H
#define TRUNC_TILING_H
#include "register/tilingdata_base.h"

#define MAX_INPUT_DIM 8
#define BROADCAST_TILING_KEY 10

namespace optiling {
BEGIN_TILING_DATA_DEF(TruncTilingData)
     // 核内切分参数
    TILING_DATA_FIELD_DEF(uint32_t, Len);
    TILING_DATA_FIELD_DEF(uint32_t, fNum);
    TILING_DATA_FIELD_DEF(uint32_t, fLen);
    TILING_DATA_FIELD_DEF(uint32_t, tLen);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Trunc, TruncTilingData)

}
#endif
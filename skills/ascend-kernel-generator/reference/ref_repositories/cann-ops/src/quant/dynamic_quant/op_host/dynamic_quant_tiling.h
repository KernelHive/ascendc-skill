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
 * \file dynamic_quant_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowPerHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, rowPerTailCore);
TILING_DATA_FIELD_DEF(uint32_t, multiRowNumHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, multiRowNumTailCore);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopEle);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopTail);
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, alignGroupNum);
TILING_DATA_FIELD_DEF(uint32_t, hasSmooth);
TILING_DATA_FIELD_DEF(uint32_t, unused);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicQuant, DynamicQuantTilingData)
REGISTER_TILING_DATA_CLASS(DynamicQuantV2, DynamicQuantTilingData)
struct DynamicQuantCompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H

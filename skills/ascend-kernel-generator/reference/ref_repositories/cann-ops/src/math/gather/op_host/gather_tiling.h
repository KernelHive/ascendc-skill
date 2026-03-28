#ifndef GATHER_TILING_H 
#define GATHER_TILING_H
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(GatherTilingData)
        TILING_DATA_FIELD_DEF(uint32_t, placeHolder);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(GatherTilingDataWithDataCopy)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, sliceLength);
        TILING_DATA_FIELD_DEF(uint32_t, maxLength);
        TILING_DATA_FIELD_DEF(uint32_t, tileLength);
        TILING_DATA_FIELD_DEF(uint32_t, reminder);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(GatherTilingDataScalarCopy)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, sliceLength);
    END_TILING_DATA_DEF;
    
    REGISTER_TILING_DATA_CLASS(Gather, GatherTilingData)
    REGISTER_TILING_DATA_CLASS(Gather_0, GatherTilingDataWithDataCopy)
    REGISTER_TILING_DATA_CLASS(Gather_1, GatherTilingDataScalarCopy)
}
#endif
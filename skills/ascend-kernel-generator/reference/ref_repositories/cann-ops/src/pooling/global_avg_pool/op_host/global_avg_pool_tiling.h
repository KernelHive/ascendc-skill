
/*
* @file global_avg_pool_tiling.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This program is free software; you can redistribute it and/or modify
* it under the terms of the Apache License Version 2.0.
* You may not use this file except in compliance with the License.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef GLOBAL_AVG_POOL_TILING_H
#define GLOBAL_AVG_POOL_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(GlobalAvgPoolTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outDim);
    TILING_DATA_FIELD_DEF(int32_t, dimLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lasttileLength);
    TILING_DATA_FIELD_DEF(uint32_t, workLength);
    TILING_DATA_FIELD_DEF(uint32_t, actLastLen);
    TILING_DATA_FIELD_DEF(uint32_t, typeKey);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(GlobalAvgPool, GlobalAvgPoolTilingData)
}

#endif


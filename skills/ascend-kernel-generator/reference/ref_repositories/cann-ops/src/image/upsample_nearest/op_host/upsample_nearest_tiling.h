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
 * \file upsample_nearest_tiling.h
 * \brief
 */
#ifndef OPS_UPSAMPLE_NEAREST_TILING_H
#define OPS_UPSAMPLE_NEAREST_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint16_t MAX_CORE_CONT = 50;

struct UpsampleNearestCompileInfo {
    int64_t totalCoreNum = 0;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(UpsampleNearestTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dataType);
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(bool, exactMode);

TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);

TILING_DATA_FIELD_DEF_ARR(int64_t, 4, inputShapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, outputShapes);

TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_CONT, tailColStartList);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_CONT, tailColEndList);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_CONT, tailRowStartList);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_CONT, tailRowEndList);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleNearest, UpsampleNearestTilingData)
}  // namespace optiling

#endif

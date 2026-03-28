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
 * \file grid_sampler_2d_grad_tiling_data.h
 * \brief
 */
#ifndef __GRID_SAMPLER_2D_GRAD_TILINGDATA_H__
#define __GRID_SAMPLER_2D_GRAD_TILINGDATA_H__

#include "register/tilingdata_base.h"

namespace optiling {

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(GridSampler2DGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, pNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailPNum);
TILING_DATA_FIELD_DEF(uint32_t, channel);
TILING_DATA_FIELD_DEF(uint32_t, height);
TILING_DATA_FIELD_DEF(uint32_t, width);
TILING_DATA_FIELD_DEF(uint32_t, gridH);
TILING_DATA_FIELD_DEF(uint32_t, gridW);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
TILING_DATA_FIELD_DEF(uint32_t, interpolation);
TILING_DATA_FIELD_DEF(uint32_t, padding);
TILING_DATA_FIELD_DEF(uint32_t, alignCorners);
TILING_DATA_FIELD_DEF(uint32_t, group);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumCast);
TILING_DATA_FIELD_DEF(uint32_t, pNumPerCoreCast);
TILING_DATA_FIELD_DEF(uint32_t, tailPNumCast);
TILING_DATA_FIELD_DEF(uint32_t, castElement);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GridSampler2DGrad, GridSampler2DGradTilingData)

struct Tiling4GridSampler2DGradCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
};

struct InputParamsInfo {
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t tilingKey = 0;
    int interpolation = 0;
    int padding = 0;
    int alignCorners = 0;
};
}  // namespace optiling
#endif  // __GRID_SAMPLER_2D_GRAD_TILINGDATA_H__
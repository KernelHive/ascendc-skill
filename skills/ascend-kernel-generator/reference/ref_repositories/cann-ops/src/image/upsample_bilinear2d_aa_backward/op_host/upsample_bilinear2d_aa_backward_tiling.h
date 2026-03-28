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
 * \file upsample_bilinear2d_aa_backward_tiling.h
 * \brief
 */
#ifndef OPS_UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_H
#define OPS_UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint16_t MAX_CORE_CONT = 50;

struct UpsampleBilinear2dAABackwardCompileInfo {
    uint32_t coreNum;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(UpsampleBilinear2dAABackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, slideSize);
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, invscaleW);
TILING_DATA_FIELD_DEF(float, invscaleH);

TILING_DATA_FIELD_DEF(float, supportW);
TILING_DATA_FIELD_DEF(float, supportH);
TILING_DATA_FIELD_DEF(int16_t, maxInterpSizeW);
TILING_DATA_FIELD_DEF(int16_t, maxInterpSizeH);
TILING_DATA_FIELD_DEF(uint64_t, intermediateMatrixSize);
TILING_DATA_FIELD_DEF(uint32_t, radioMatrixSizeW);
TILING_DATA_FIELD_DEF(uint32_t, radioMatrixSizeH);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNumW);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNumH);
TILING_DATA_FIELD_DEF(bool, needResizeW);
TILING_DATA_FIELD_DEF(bool, needResizeH);

TILING_DATA_FIELD_DEF_ARR(int64_t, 4, inputShapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, outputShapes);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideStartListW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideEndListW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideStartListW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideEndListW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowStartListW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowEndListW);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideStartListH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideEndListH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideStartListH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideEndListH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowStartListH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowEndListH);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingW);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingH);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleBilinear2dAABackward, UpsampleBilinear2dAABackwardTilingData)
}  // namespace optiling

#endif  // OPS_UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_H
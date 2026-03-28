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
 * \file upsample_bicubic2d_aa_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_BICUBIC2D_AA_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_BICUBIC2D_AA_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint16_t MAX_CORE_CONT = 50;

struct UpsampleBicubic2dAACompileInfo {
    uint32_t totalCoreNum = 0;
};

BEGIN_TILING_DATA_DEF(UpsampleBicubic2dAATilingData)
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, invscaleW);
TILING_DATA_FIELD_DEF(float, invscaleH);

TILING_DATA_FIELD_DEF(float, supportH);
TILING_DATA_FIELD_DEF(float, supportW);
TILING_DATA_FIELD_DEF(int32_t, maxInterpSizeW);
TILING_DATA_FIELD_DEF(int32_t, maxInterpSizeH);

TILING_DATA_FIELD_DEF(uint32_t, radioMatrixWSize);
TILING_DATA_FIELD_DEF(uint32_t, radioMatrixHSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNumW);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNumH);
TILING_DATA_FIELD_DEF(uint32_t, sliceSize);
TILING_DATA_FIELD_DEF(uint64_t, intermediateMatrixSize);
TILING_DATA_FIELD_DEF_ARR(int32_t, 4, inputShapes);
TILING_DATA_FIELD_DEF_ARR(int32_t, 4, outputShapes);

TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, sliceStartListW);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, sliceEndListW);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailSliceStartListW);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailSliceEndListW);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailRowStartListW);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailRowEndListW);

TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, sliceStartListH);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, sliceEndListH);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailSliceStartListH);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailSliceEndListH);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailBatchStartListH);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailBatchEndListH);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingW);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingH);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleBicubic2dAA, UpsampleBicubic2dAATilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_BICUBIC2D_AA_TILING_H

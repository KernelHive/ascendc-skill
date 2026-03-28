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
 * \file upsample_bilinear2d_aa_tiling.h
 * \brief
 */

#ifndef OPS_UPSAMPLE_BILINEAR2D_AA_TILING_H
#define OPS_UPSAMPLE_BILINEAR2D_AA_TILING_H
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING"
#define OPS_UTILS_LOG_PACKAGE_TYPE "OP_CUST"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {

struct UpsampleBilinear2dAACompileInfo {
    uint32_t coreNum;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(UpsampleBilinearAATilingData)
TILING_DATA_FIELD_DEF(int64_t, slide_size);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(float, scale_w);
TILING_DATA_FIELD_DEF(float, scale_h);
TILING_DATA_FIELD_DEF(float, invscale_w);
TILING_DATA_FIELD_DEF(float, invscale_h);

TILING_DATA_FIELD_DEF(float, support_w);
TILING_DATA_FIELD_DEF(int16_t, max_interp_size_w);
TILING_DATA_FIELD_DEF(float, support_h);
TILING_DATA_FIELD_DEF(int16_t, max_interp_size_h);
TILING_DATA_FIELD_DEF(uint64_t, intermediate_matrix_size);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size_w);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size_h);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_w);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_h);

TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNumW);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNumW);
TILING_DATA_FIELD_DEF(int64_t, slideNumW);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNumW);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRowsW);
TILING_DATA_FIELD_DEF(int64_t, remainderW);

TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNumH);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNumH);
TILING_DATA_FIELD_DEF(int64_t, slideNumH);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNumH);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRowsH);
TILING_DATA_FIELD_DEF(int64_t, remainderH);

TILING_DATA_FIELD_DEF_ARR(int64_t, 4, input_shapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, output_shapes);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_w);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_h);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleBilinear2dAA, UpsampleBilinearAATilingData)
}  // namespace optiling

#endif  // OPS_UPSAMPLE_BILINEAR2D_AA_TILING_H

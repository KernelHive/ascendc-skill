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
 * \file upsample_nearest_exact3d_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEARESTEXACT3D_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEARESTEXACT3D_TILING_H
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING"
#define OPS_UTILS_LOG_PACKAGE_TYPE "OP_CUST"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {

struct UpsampleNearest3dCompileInfo {
    int64_t coreNum;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(UpsampleNearestExact3dTilingData)
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, batches);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, inputShapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, outputShapes);

TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleD);
TILING_DATA_FIELD_DEF(int64_t, slideSizeW);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeW);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeH);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeD);

TILING_DATA_FIELD_DEF(int64_t, slideNumH);
TILING_DATA_FIELD_DEF(int64_t, slideNumD);
TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNum);
TILING_DATA_FIELD_DEF(int64_t, remainder);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNum);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inputRow);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRow);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleNearestExact3d, UpsampleNearestExact3dTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEARESTEXACT3D_TILING_H

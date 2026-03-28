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
 * \file upsample_nearest3d_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEAREST3D_GRAD_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEAREST3D_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct UpsampleNearest3dGradCompileInfo {
    int64_t coreNum;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(UpsampleNearest3dGradTilingData)
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, batches);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, gradInputShapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, gradOutputShapes);

TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleD);
TILING_DATA_FIELD_DEF(bool, needResizeW);
TILING_DATA_FIELD_DEF(bool, needResizeH);
TILING_DATA_FIELD_DEF(bool, needResizeD);
TILING_DATA_FIELD_DEF(int64_t, slideSize);
TILING_DATA_FIELD_DEF(int64_t, tensorSize);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeMapping);
TILING_DATA_FIELD_DEF(int64_t, radioMatrixSize);
TILING_DATA_FIELD_DEF(int64_t, intermediateMatrixSizeW);
TILING_DATA_FIELD_DEF(int64_t, intermediateMatrixSizeH);

TILING_DATA_FIELD_DEF_ARR(int64_t, 3, eachCoreSlideNums);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, remainders);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, tailStartSlideNums);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, groupCoreNums);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, inputRows);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, tailAvergingRows);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, needCoreNums);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingW);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingH);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingD);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleNearest3dGrad, UpsampleNearest3dGradTilingData);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEAREST3D_GRAD_TILING_H

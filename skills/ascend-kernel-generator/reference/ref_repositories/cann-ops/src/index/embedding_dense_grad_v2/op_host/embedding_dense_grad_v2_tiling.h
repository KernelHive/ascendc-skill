/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file embedding_dense_grad_v2_tiling.h
 */
#ifndef EMBEDDING_DENSE_GRAD_V2_TILING_H
#define EMBEDDING_DENSE_GRAD_V2_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingDenseGradV2TilingParam)
    TILING_DATA_FIELD_DEF(uint64_t, tailRowNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerRowNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerRowRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, computeMask)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeFormerNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeTailNum)
    TILING_DATA_FIELD_DEF(uint64_t, numWeights)
    TILING_DATA_FIELD_DEF(uint64_t, embeddingDim)
    TILING_DATA_FIELD_DEF(uint64_t, paddingIdx)
    TILING_DATA_FIELD_DEF(uint64_t, scaleGradByFreq)
    TILING_DATA_FIELD_DEF(uint64_t, formerDimRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, formerEmbeddingDim)
    TILING_DATA_FIELD_DEF(uint64_t, tailEmbeddingDim)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeFormerNum)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeTailNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGradV2TilingParamOp, EmbeddingDenseGradV2TilingParam)

BEGIN_TILING_DATA_DEF(EmbeddingDenseGradV2ScaleTiling)
    TILING_DATA_FIELD_DEF(uint64_t, tailCoreRowNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerCoreRowNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerCoreRowRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, mask)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeFormerNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeTailNum)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeFormerNum)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeTailNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGradV2ScaleTilingOp, EmbeddingDenseGradV2ScaleTiling)

BEGIN_TILING_DATA_DEF(EmbeddingDenseGradV2DeterminTiling)
    TILING_DATA_FIELD_DEF(uint64_t, gradRow)
    TILING_DATA_FIELD_DEF(uint64_t, tailRowNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerRowNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerRowNumRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, computeMask)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeFormerNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerComputeTailNum)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeRepTime)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeFormerNum)
    TILING_DATA_FIELD_DEF(uint64_t, tailComputeTailNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGradV2DeterminTilingOp, EmbeddingDenseGradV2DeterminTiling)

BEGIN_TILING_DATA_DEF(EmbeddingDenseGradV2SmallDimTiling)
    TILING_DATA_FIELD_DEF(uint64_t, partNum)
    TILING_DATA_FIELD_DEF(uint64_t, formerCopyRow)
    TILING_DATA_FIELD_DEF(uint64_t, tailCopyRow)
    TILING_DATA_FIELD_DEF(uint64_t, formerCopyTime)
    TILING_DATA_FIELD_DEF(uint64_t, tailCopyTime)
    TILING_DATA_FIELD_DEF(uint64_t, maxRowInUb)
    TILING_DATA_FIELD_DEF(uint64_t, formerLastRow)
    TILING_DATA_FIELD_DEF(uint64_t, tailLastRow)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGradV2SmallDimTilingOp, EmbeddingDenseGradV2SmallDimTiling)

BEGIN_TILING_DATA_DEF(EmbeddingDenseGradV2TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(EmbeddingDenseGradV2TilingParam, params)
    TILING_DATA_FIELD_DEF_STRUCT(EmbeddingDenseGradV2ScaleTiling, scaleTiling)
    TILING_DATA_FIELD_DEF_STRUCT(EmbeddingDenseGradV2DeterminTiling, determinTiling)
    TILING_DATA_FIELD_DEF_STRUCT(EmbeddingDenseGradV2SmallDimTiling, smallDimTiling)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGradV2, EmbeddingDenseGradV2TilingData)
REGISTER_TILING_DATA_CLASS(EmbeddingDenseGradV2TilingDataOp, EmbeddingDenseGradV2TilingData)
} // namespace optiling
#endif // EMBEDDING_DENSE_GRAD_V2_TILING_H
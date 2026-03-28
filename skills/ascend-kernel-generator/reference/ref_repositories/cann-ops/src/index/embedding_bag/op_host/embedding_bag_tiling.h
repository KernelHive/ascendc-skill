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
 * \file embedding_bag_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingBagTilingData)
    TILING_DATA_FIELD_DEF(int64_t, formerOffsetNum);
    TILING_DATA_FIELD_DEF(int64_t, tailOffsetNum);
    TILING_DATA_FIELD_DEF(int64_t, numEmbeddings);
    TILING_DATA_FIELD_DEF(int64_t, computeRepTime);
    TILING_DATA_FIELD_DEF(int64_t, indicesMaxMoveLength);
    TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, paddingIdx);
    TILING_DATA_FIELD_DEF(int64_t, numIndices);
    TILING_DATA_FIELD_DEF(int64_t, mode);
    TILING_DATA_FIELD_DEF(int64_t, tilingKey);
    TILING_DATA_FIELD_DEF(int64_t, hasPerSampleWeights)
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EmbeddingBag, EmbeddingBagTilingData)

ge::graphStatus TilingEmbeddingBag(gert::TilingContext* context);
ge::graphStatus TilingPrepareForEmbeddingBag(gert::TilingParseContext* context);

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
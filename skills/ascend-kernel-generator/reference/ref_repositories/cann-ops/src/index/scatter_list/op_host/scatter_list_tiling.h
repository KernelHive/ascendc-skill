/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_LIST_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_LIST_TILING_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ScatterListTilingData)
TILING_DATA_FIELD_DEF(int64_t, dim0Count);
TILING_DATA_FIELD_DEF(int64_t, dim1Count);
TILING_DATA_FIELD_DEF(int64_t, varDim2Count);
TILING_DATA_FIELD_DEF(int64_t, dim2Count);
TILING_DATA_FIELD_DEF(int64_t, dim3Count);
TILING_DATA_FIELD_DEF(int64_t, dim3CountAlign);
TILING_DATA_FIELD_DEF(int64_t, updatesOneBlock);
TILING_DATA_FIELD_DEF(int64_t, indiceDims);
TILING_DATA_FIELD_DEF(int64_t, indiceCount);
TILING_DATA_FIELD_DEF(int64_t, indiceUbSize);
TILING_DATA_FIELD_DEF(int64_t, maskCount);
TILING_DATA_FIELD_DEF(int64_t, maskUbSize);
TILING_DATA_FIELD_DEF(int64_t, srcBatchStride);
TILING_DATA_FIELD_DEF(int64_t, srcBatchStrideAlign);
TILING_DATA_FIELD_DEF(int64_t, dstBatchStride);
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preCoreBatchNum);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBatchNum);
TILING_DATA_FIELD_DEF(int64_t, eachLoopNum);
TILING_DATA_FIELD_DEF(int64_t, eachPreLoopEle);
TILING_DATA_FIELD_DEF(int64_t, eachLastLoopEle);
TILING_DATA_FIELD_DEF(int64_t, eachLastLoopEleAlign);
TILING_DATA_FIELD_DEF(int64_t, updatesCount);
TILING_DATA_FIELD_DEF(int64_t, updatesUbSize);
TILING_DATA_FIELD_DEF(int64_t, dataUbSize);
TILING_DATA_FIELD_DEF(int64_t, transposeUbSize);
TILING_DATA_FIELD_DEF(int64_t, transRepeatTimes);
TILING_DATA_FIELD_DEF(int64_t, transRepeatTimesTail);
TILING_DATA_FIELD_DEF(int64_t, updateDim23Align);
TILING_DATA_FIELD_DEF(int64_t, preCoreUpdateDim23);
TILING_DATA_FIELD_DEF(int64_t, varDim3Stride);
TILING_DATA_FIELD_DEF(int64_t, varDim3Count);
TILING_DATA_FIELD_DEF(int64_t, dim3CountSize);
TILING_DATA_FIELD_DEF(int64_t, eachLastSize);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterList, ScatterListTilingData)

enum class ScatterListTilingKey : int64_t {
    TILINGKEY_TSMALL = 100,
    TILINGKEY_TMORE = 101,
    TILINGKEY_TLARGE = 102,
    TILINGKEY_PSMALL = 103,
    TILINGKEY_PMORE = 104,
    TILINGKEY_PLARGE = 105,
    TILINGKEY_PLE = 106,
    TILINGKEY_PLEDIM2 = 107,
    TILINGKEY_RSBSE = 200,
    TILINGKEY_RLBSE = 210,
    TILINGKEY_RLBSE_PAD = 211,
    TILINGKEY_RSBLE = 220,
    TILINGKEY_RLBLE = 230,
    TILINGKEY_RLBLE_PAD = 231,
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_LIST_TILING_H_
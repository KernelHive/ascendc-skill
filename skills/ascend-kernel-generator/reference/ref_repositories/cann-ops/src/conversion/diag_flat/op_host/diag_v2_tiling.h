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
 * \file diag_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
namespace optiling {

const int64_t SCALAR_UNEQUAL_SIZE16 = 101;
const int64_t SCALAR_EQUAL_SIZE16 = 102;
const int64_t ASSIST_UNEQUAL_SIZE16 = 103;
const int64_t ASSIST_EQUAL_SIZE16 = 104;

const int64_t SCALAR_THRESHOLD_NUM = 64;
const int64_t SCALAR_MAX_WIDTH = 1024;
const int64_t LEAST_NUM_PER_CORE = 64;
const int64_t BLOCK_SIZE = 32;

enum class DiagV2TilingKey : int64_t {
    ASSIST_SIZE_1 = 2101,
    ASSIST_SIZE_2 = 2102,
    ASSIST_SIZE_4 = 2103,
    ASSIST_SIZE_8 = 2104,
    ASSIST_SIZE_16 = 2105,
    SCALAR_SIZE_1 = 2401,
    SCALAR_SIZE_2 = 2402,
    SCALAR_SIZE_4 = 2403,
    SCALAR_SIZE_8 = 2404
};

BEGIN_TILING_DATA_DEF(DiagV2TilingData)
    TILING_DATA_FIELD_DEF(int64_t, xWidth);
    TILING_DATA_FIELD_DEF(int64_t, xHeight);
    TILING_DATA_FIELD_DEF(int64_t, gmOffset);
    TILING_DATA_FIELD_DEF(int64_t, numOut);
    TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, numPerCore);
    TILING_DATA_FIELD_DEF(int64_t, tailNum);
    TILING_DATA_FIELD_DEF(int64_t, tilingKey);
    TILING_DATA_FIELD_DEF(int64_t, matrixRowLength);
    TILING_DATA_FIELD_DEF(int64_t, inputNum);  // form here for DiagFlat
    TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNum);
    TILING_DATA_FIELD_DEF(int64_t, lastCoreHandleNum);
    TILING_DATA_FIELD_DEF(int64_t, ubInputSize);
    TILING_DATA_FIELD_DEF(int64_t, ubOutputSize);
    TILING_DATA_FIELD_DEF(int64_t, diagonal);
    TILING_DATA_FIELD_DEF(int64_t, workspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DiagV2, DiagV2TilingData)
REGISTER_TILING_DATA_CLASS(DiagFlat, DiagV2TilingData)

struct DiagV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

ge::graphStatus TilingDiagFlat(gert::TilingContext* context);

inline int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_

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
 * \file reverse_sequence_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_SEQUENCE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_SEQUENCE_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ReverseSequenceTilingData)
    TILING_DATA_FIELD_DEF(int64_t, tilingKey);
    TILING_DATA_FIELD_DEF(int64_t, batchDimValue);
    TILING_DATA_FIELD_DEF(int64_t, seqDimValue);
    TILING_DATA_FIELD_DEF(int64_t, xDtypeSize);
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, seqSize);
    TILING_DATA_FIELD_DEF(int64_t, cSize);
    TILING_DATA_FIELD_DEF(int64_t, maxProcCount);
    TILING_DATA_FIELD_DEF(int64_t, loopTimePerCore);
    TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, innerLoopTime);
    TILING_DATA_FIELD_DEF(int64_t, innerTailCount);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ReverseSequence, ReverseSequenceTilingData)

struct ReverseSequenceCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

enum class ReverseSequenceTilingKey : uint64_t {
    BATCH_DIM_0_C_SMALL = 101,
    BATCH_DIM_0_C_BIG = 201,
    BATCH_DIM_1_C_SMALL = 301,
    BATCH_DIM_1_C_BIG = 401
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_SEQUENCE_TILING_H_

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
 * \file strided_slice_assign_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_STRIDED_SLICE_ASSIGN_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_STRIDED_SLICE_ASSIGN_V2_H

#include "register/tilingdata_base.h"

namespace optiling {
constexpr size_t MAX_DIM_NUM = 8;
BEGIN_TILING_DATA_DEF(StridedSliceAssignV2TilingData)
    TILING_DATA_FIELD_DEF(int64_t, dimNum);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, varDim);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, inputValueDim);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, begin);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, strides);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, varCumShape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, inputCumShape);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StridedSliceAssignV2, StridedSliceAssignV2TilingData)

struct StridedSliceAssignV2CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_STRIDED_SLICE_ASSIGN_V2_H
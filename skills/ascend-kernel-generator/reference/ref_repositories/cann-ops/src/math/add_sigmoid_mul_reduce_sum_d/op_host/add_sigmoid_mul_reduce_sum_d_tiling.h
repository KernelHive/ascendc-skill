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
 * @file add_sigmoid_mul_reduce_sum_d_tiling.h
 */

#ifndef ADD_SIGMOID_MUL_REDUCE_SUM_D_TILING_H
#define ADD_SIGMOID_MUL_REDUCE_SUM_D_TILING_H
#include "register/tilingdata_base.h"

namespace optiling{
    BEGIN_TILING_DATA_DEF(AddSigmoidMulReduceSumDTilingData)
    TILING_DATA_FIELD_DEF(int32_t, formerCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, formerCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, formerTileNum);
    TILING_DATA_FIELD_DEF(int32_t, formerTileLength);
    TILING_DATA_FIELD_DEF(int32_t, formerLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, tailTileNum);
    TILING_DATA_FIELD_DEF(int32_t, tailTileLength);
    TILING_DATA_FIELD_DEF(int32_t, tailLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim1234Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim14Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim23Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim1Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim234Length);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(AddSigmoidMulReduceSumD, AddSigmoidMulReduceSumDTilingData)
}
#endif // ADD_SIGMOID_MUL_REDUCE_SUM_D_TILING_H
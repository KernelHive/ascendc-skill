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
 * \file dua_quantize_add_layer_norm.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DUA_QUANTIZE_ADD_LAYER_NORM_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DUA_QUANTIZE_ADD_LAYER_NORM_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DuaQuantizeAddLayerNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(uint32_t, nlFirstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, lFirstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(uint32_t, lastDimPerTime);
TILING_DATA_FIELD_DEF(float, aveNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, colMoveCnt);
TILING_DATA_FIELD_DEF(uint32_t, colTail);
TILING_DATA_FIELD_DEF(uint32_t, isZeroPoint1Exist);
TILING_DATA_FIELD_DEF(uint32_t, isZeroPoint2Exist);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DuaQuantizeAddLayerNorm, DuaQuantizeAddLayerNormTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_DUA_QUANTIZE_ADD_LAYER_NORM_H

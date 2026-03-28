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
 * \file dynamic_quant_update_scatter.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_each_core.h"
#include "dynamic_quant_update_scatter_each_core_large_batch.h"
#include "dynamic_quant_update_scatter_each_core_large_ele_little_quant.h"
#include "dynamic_quant_update_scatter_each_core_large_ele_large_quant.h"
#include "dynamic_quant_update_scatter_each_core_large_batch_little_quant.h"
#include "dynamic_quant_update_scatter_each_core_large_batch_large_quant.h"
#include "dynamic_quant_update_scatter_each_core_large_ele_little_quant_mod.h"

using namespace AscendC;
using namespace DynamicQuantUpdateScatterND;

constexpr int TILING_MODE_AXIS_NEG_2 = 100;
constexpr int TILING_MODE_AXIS_NEG_2_LARGE_BATCH = 101;
constexpr int TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_QUANT = 102;
constexpr int TILING_MODE_AXIS_NEG_2_LARGE_ELE_LARGE_QUANT = 103;
constexpr int TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LITTLE_QUANT = 104;
constexpr int TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LARGE_QUANT = 105;
constexpr int TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_MOD_64 = 106;

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices,
                                                                   GM_ADDR updates, GM_ADDR smoothScales,
                                                                   GM_ADDR varOut, GM_ADDR varScaleOut,
                                                                   GM_ADDR workSpace, GM_ADDR tiling) {
  if (var == nullptr || varScale == nullptr || indices == nullptr || updates == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2)) {
    DynamicQuantUpdateScatterEachCore<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2_LARGE_BATCH)) {
    DynamicQuantUpdateScatterEachCoreLargeBatch<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_QUANT)) {
    DynamicQuantUpdateScatterEachCoreLargeEleLittleQuant<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2_LARGE_ELE_LARGE_QUANT)) {
    DynamicQuantUpdateScatterEachCoreLargeEleLargeQuant<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LITTLE_QUANT)) {
    DynamicQuantUpdateScatterEachCoreLargeBatchLittleQuant<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LARGE_QUANT)) {
    DynamicQuantUpdateScatterEachCoreLargeBatchLargeQuant<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_MOD_64)) {
    DynamicQuantUpdateScatterEachCoreLargeEleLittleQuantMod<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
    op.Process(&tilingData);
  } else {
  }
}
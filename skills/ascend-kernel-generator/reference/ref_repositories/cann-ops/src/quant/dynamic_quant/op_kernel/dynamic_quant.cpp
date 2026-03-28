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
 * \file dynamic_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dynamic_quant.h"
#include "dynamic_quant_db.h"
#include "dynamic_quant_large_shape_opt.h"
#include "dynamic_quant_moe.h"
#include "dynamic_quant_moe_large_shape.h"

using namespace AscendC;
using namespace DynamicQuantNDOpt;

extern "C" __global__ __aicore__ void dynamic_quant(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y,
                                                    GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling) {
  if (x == nullptr || y == nullptr || scale == nullptr) {
    return;
  }

  GM_ADDR user1 = GetUserWorkspace(workSpace);
  if (user1 == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  if (GetBlockIdx() >= tilingData.coreNum) {
    return;
  }
  TPipe pipe;
  if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
    DynamicQuant<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, y, scale, nullptr, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
    DynamicQuantDbOpt<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, y, scale, nullptr, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(6)) {
    DynamicQuantLargeShapeOpt<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, y, scale, nullptr, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(7)) {
    DynamicQuantMoe<DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, group_index, y, scale, nullptr, workSpace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(8)) {
    DynamicQuantMoeLargeShape<DTYPE_X, DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
    op.Init(x, smooth_scales, group_index, y, scale, nullptr, workSpace, &tilingData);
    op.Process();
  } else {
  }
}
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
 * \file ascend_quant_v2.cpp
 * \brief
 */

#include "ascend_quant_v2_fp16.h"
#include "ascend_quant_v2_fp32.h"
#include "ascend_quant_v2_bf16.h"

#define KEY_PER_CHANNEL 0
#define KEY_PER_TENSOR 1
#define KEY_PER_HEAD 2

using namespace AscendC;

extern "C" __global__ __aicore__ void ascend_quant_v2(GM_ADDR x, GM_ADDR scale, GM_ADDR offset,
                                                      GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  if constexpr (std::is_same<DTYPE_X, half>::value) {
    if (TILING_KEY_IS(KEY_PER_CHANNEL)) {
      AscendQuantV2::AscendQuantV2PerChannelFP16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    } else if (TILING_KEY_IS(KEY_PER_TENSOR)) {
      AscendQuantV2::AscendQuantV2PerTensorFP16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    }
  } else if constexpr (std::is_same<DTYPE_X, float>::value) {
    if (TILING_KEY_IS(KEY_PER_CHANNEL)) {
      AscendQuantV2::AscendQuantV2PerChannelFP32<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    } else if (TILING_KEY_IS(KEY_PER_TENSOR)) {
      AscendQuantV2::AscendQuantV2PerTensorFP32<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    }
#if __CCE_AICORE__ == 220
  } else if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
    if (TILING_KEY_IS(KEY_PER_CHANNEL)) {
      AscendQuantV2::AscendQuantV2PerChannnelBF16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    } else if (TILING_KEY_IS(KEY_PER_TENSOR)) {
      AscendQuantV2::AscendQuantV2PerTensorBF16<DTYPE_X> op;
      op.Init(x, scale, offset, y, &tilingData);
      op.Process();
    }
  }
  if (TILING_KEY_IS(KEY_PER_HEAD)) {
    AscendQuantV2::AscendQuantV2PerHead<DTYPE_X> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
#endif
  }
}
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

/*!
 * \file top_kv3.cpp
 * \brief
 */
#include "top_kv3.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void top_kv3(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(1)) {
    KernelTopKV3<half> op(&pipe);
    op.Init(x, k, values, indices, &tilingData);
    op.Process();
  }
}

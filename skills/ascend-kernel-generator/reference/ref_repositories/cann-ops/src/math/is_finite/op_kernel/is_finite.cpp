/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file is_finite.cpp
 * \brief
 */

#include "is_finite.h"

using namespace IsFiniteNs;

// kernel function
extern "C" __global__ __aicore__ void is_finite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  GM_ADDR userWS = nullptr;

  const int16_t HALF_TYPE_MASK = 0x7c00;      // 0111 1100 0000 0000
  const int32_t FLOAT_TYPE_MASK = 0x7f800000; // 0111 1111 1000 0000 0000 0000 0000 0000
  const int16_t BF16_TYPE_MASK = 0x7f80;      // 0111 1111 1000 0000

  if (TILING_KEY_IS(1)) {
    IsFinite<half, HALF_TYPE_MASK> op;
    op.Init(x, y, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    IsFinite<float, BF16_TYPE_MASK> op;
    op.Init(x, y, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(3)) {
    IsFinite<half, BF16_TYPE_MASK> op;
    op.Init(x, y, userWS, &tilingData);
    op.Process();
  } 
}
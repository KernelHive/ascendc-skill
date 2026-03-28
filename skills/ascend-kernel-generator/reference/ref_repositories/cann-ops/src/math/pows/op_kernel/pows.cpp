/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file pows.cpp
 * \brief
 */

#include "pows_fp16.h"
#include "pows_fp32.h"
#include "pows_bf16.h"


using namespace Pows;

extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 100
  if (TILING_KEY_IS(101)) {
    Pows::PowsFp16<half> op;
    op.Init(x1, x2, y, workspace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(301)) {
    Pows::PowsFp32<float> op;
    op.Init(x1, x2, y, workspace, &tilingData);
    op.Process();
  } 
#else
  if (TILING_KEY_IS(101)) {
    Pows::PowsFp16<half> op;
    op.Init(x1, x2, y, workspace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(201)) {
    Pows::PowsBfp16<bfloat16_t> op;
    op.Init(x1, x2, y, workspace, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(301)) {
    Pows::PowsFp32<float> op;
    op.Init(x1, x2, y, workspace, &tilingData);
    op.Process();
  } 
#endif
}

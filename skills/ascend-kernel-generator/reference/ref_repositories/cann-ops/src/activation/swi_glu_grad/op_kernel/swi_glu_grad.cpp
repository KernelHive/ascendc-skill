/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file swi_glu_grad.cpp
 * \brief
 */
#include "swi_glu_grad_float.hpp"
#include "swi_glu_grad_bf16.hpp"
#include "swi_glu_grad_single.hpp"

using namespace AscendC;
extern "C" __global__ __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    // DT_FLOAT = 0,            // float type
    // DT_FLOAT16 = 1,          // fp16 type
    // DT_BF16 = 27,            // bf16 type
    GET_TILING_DATA(tempTilingGm, tiling);
    if (tempTilingGm.isDoubleBuffer == 1) {
      if (TILING_KEY_IS(1)) {
        SwiGluGradSingle <SwiGluGradBF16<half, float, half, 2>, half, half> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } else if (TILING_KEY_IS(0)) {
        SwiGluGradSingle<SwiGluGradVector<float, float, float, float, float, 2>, float, float> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } 
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
      else if (TILING_KEY_IS(27)) {
        SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
#endif
    } else {
      if (TILING_KEY_IS(1)) {
        SwiGluGradSingle <SwiGluGradBF16<half, float, half, 1>, half, half> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } else if (TILING_KEY_IS(0)) {
        SwiGluGradSingle<SwiGluGradVector<float, float, float, float, float, 1>, float, float> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } 
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220   
      else if (TILING_KEY_IS(27)) {
        SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
#endif
  }
}

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
 * \file ge_glu_v2.cpp
 * \brief
 */

#include "ge_glu_v2_fp32_stride_310p.h"
#include "ge_glu_v2_fp16_stride_310p.h"
#include "ge_glu_v2_fp32_align_last_axis_big_without_pad.h"
#include "ge_glu_v2_fp16_align_last_axis_big_without_pad.h"

#include "tanh/ge_glu_v2_bf16_align.h"
#include "tanh/ge_glu_v2_bf16_align_last_axis_big.h"
#include "tanh/ge_glu_v2_bf16_vreduce.h"
#include "tanh/ge_glu_v2_fp16_align.h"
#include "tanh/ge_glu_v2_fp16_align_last_axis_big.h"
#include "tanh/ge_glu_v2_fp16_vreduce.h"
#include "tanh/ge_glu_v2_fp32_align.h"
#include "tanh/ge_glu_v2_fp32_vreduce.h"
#include "tanh/ge_glu_v2_fp32_align_last_axis_big.h"

#include "erf/ge_glu_v2_bf16_align_erf.h"
#include "erf/ge_glu_v2_bf16_align_last_axis_big_erf.h"
#include "erf/ge_glu_v2_bf16_vreduce_erf.h"
#include "erf/ge_glu_v2_fp16_align_erf.h"
#include "erf/ge_glu_v2_fp16_align_last_axis_big_erf.h"
#include "erf/ge_glu_v2_fp16_vreduce_erf.h"
#include "erf/ge_glu_v2_fp32_align_erf.h"
#include "erf/ge_glu_v2_fp32_vreduce_erf.h"
#include "erf/ge_glu_v2_fp32_align_last_axis_big_erf.h"


using namespace GeGluV2;

extern "C" __global__ __aicore__ void ge_glu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR gelu,
                                                GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
  if (workspace == nullptr) {
    return;
  }
  
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

#if __CCE_AICORE__ == 200
  if (TILING_KEY_IS(101)) {
    GeGluV2::GeGluV2Fp16Stride310P<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(103)) {
    GeGluV2::GeGluV2Fp16AlignLastAxisBigWithoutPad<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(301)) {
    GeGluV2::GeGluV2Fp32Stride310P<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(303)) {
    GeGluV2::GeGluV2Fp32AlignLastAxisBigWithoutPad<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  }
#else
  if (TILING_KEY_IS(101)) {
    GeGluV2::GeGluV2Fp16Align<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(102)) {
    GeGluV2::GeGluV2Fp16VReduce<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(103)) {
    GeGluV2::GeGluV2Fp16AlignLastAxisBig<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(201)) {
    GeGluV2::GeGluV2Bf16Align<bfloat16_t> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(202)) {
    GeGluV2::GeGluV2Bf16VReduce<bfloat16_t> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(203)) {
    GeGluV2::GeGluV2Bf16AlignLastAxisBig<bfloat16_t> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(301)) {
    GeGluV2::GeGluV2Fp32Align<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(302)) {
    GeGluV2::GeGluV2Fp32VReduce<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(303)) {
    GeGluV2::GeGluV2Fp32AlignLastAxisBig<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  }  else if (TILING_KEY_IS(111)) {
    GeGluV2::GeGluV2Fp16AlignErf<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(112)) {
    GeGluV2::GeGluV2Fp16VReduceErf<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(113)) {
    GeGluV2::GeGluV2Fp16AlignLastAxisBigErf<half> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(211)) {
    GeGluV2::GeGluV2Bf16AlignErf<bfloat16_t> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(212)) {
    GeGluV2::GeGluV2Bf16VReduceErf<bfloat16_t> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(213)) {
    GeGluV2::GeGluV2Bf16AlignLastAxisBigErf<bfloat16_t> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(311)) {
    GeGluV2::GeGluV2Fp32AlignErf<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(312)) {
    GeGluV2::GeGluV2Fp32VReduceErf<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(313)) {
    GeGluV2::GeGluV2Fp32AlignLastAxisBigErf<float> op;
    op.Init(x, y, gelu, userWS, &tilingData);
    op.Process();
  }
#endif
}

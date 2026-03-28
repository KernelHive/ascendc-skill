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
 * \file resize_upsample_trilinear.cpp
 * \brief
 */
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
#include "resize_upsample_trilinear_310p.h"
#else
#include "resize_upsample_trilinear.h"
#endif

using namespace UpsampleTrilinearNs;

extern "C" __global__ __aicore__ void resize_upsample_trilinear(
  GM_ADDR input,
  GM_ADDR output,
  GM_ADDR workspace,
  GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data, tiling);
  const UpsampleTrilinearTilingData* __restrict tilingData = &tiling_data;
  
  GM_ADDR userWs = GetUserWorkspace(workspace);

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
  #define INIT_310P_PROCESS                                                                                    \
    op.Init(input, output, userWs, &tiling_data);                                                             \
    op.Process()
  int64_t outw = tilingData->output_w;
  int64_t outh = tilingData->output_h;
  int64_t outd = tilingData->output_d;
  if (TILING_KEY_IS(1000)) {
    KernelUpsampleTrilinear310p<half> op;
    INIT_310P_PROCESS;
  } else if (TILING_KEY_IS(3000)) {
    KernelUpsampleTrilinear310p<float> op;
    INIT_310P_PROCESS;
  }
#else
  const TCubeTiling* __restrict matmulTilingW = &(tilingData->matmul_tiling_w);
  const TCubeTiling* __restrict matmulTilingH = &(tilingData->matmul_tiling_h);
  const TCubeTiling* __restrict matmulTilingD = &(tilingData->matmul_tiling_d);
  #define INIT_AND_PROCESS                                                                                    \
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmul_w, matmulTilingW, op.matmul_h, matmulTilingH, \
                      op.matmul_d, matmulTilingD);                                                            \
    op.Init(input, output, userWs, &tiling_data);                                                             \
    op.Process()

    if (TILING_KEY_IS(1000)) {
      KernelUpsampleTrilinear<half> op;
      INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
      KernelUpsampleTrilinear<bfloat16_t> op;
      INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3000)) {
      KernelUpsampleTrilinear<float> op;
      INIT_AND_PROCESS;
    }
#endif
}
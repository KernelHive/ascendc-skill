/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
   
#include "kernel_operator.h"
#include "moe_v2_gather_out.h"
#include "moe_v2_mrgsort_out.h"
#include "moe_v2_mrgsort.h"
#include "moe_v2_sort_multi_core.h"
#include "moe_v2_sort_one_core.h"
#include "moe_v2_src_to_dst_op.h"
#include "moe_v2_gather_out.h"
#include "moe_v2_expert_token_out.h"
#include "moe_v2_src_to_dst_with_capacity.h"
#include "moe_v2_init_routing_fullload.h"

using namespace AscendC;
using namespace MoeInitRoutingV2;
extern "C" __global__ __aicore__ void moe_init_routing_v2(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                                          GM_ADDR tiling) {
 if (g_coreType == AIC) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  auto t = &tilingData;
  if (TILING_KEY_IS(20000)) {
    TPipe sortPipe;
    MoeV2FullLoad<DTYPE_X> op;
    op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &sortPipe);
    op.Process();
    sortPipe.Destroy();
    return;
  }

  // sort
  if (TILING_KEY_IS(10001) || TILING_KEY_IS(10011)) {
    TPipe sortPipe;
    MoeV2SortOneCore op;
    op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
    op.Process();
    sortPipe.Destroy();
  } else if (TILING_KEY_IS(10002) ||
             TILING_KEY_IS(10012)) {
    TPipe sortPipe;
    MoeV2SortMultiCore op;
    op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
    op.Process();
    sortPipe.Destroy();
  }

  if (TILING_KEY_IS(10001) || TILING_KEY_IS(10002)) {
    if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
      TPipe expertTokenOutPipe;
      MoeV2ExpertTokenOut expertTokenOutOp;
      expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t,
                            &expertTokenOutPipe);
      expertTokenOutOp.Process();
      expertTokenOutPipe.Destroy();
    }
    TPipe srcToDstPipe;
    MoeV2SrcToDstOp srcToDstOp;
    srcToDstOp.Init<MoeInitRoutingV2TilingData>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();
  } else if (TILING_KEY_IS(10011) ||
             TILING_KEY_IS(10012)) {
    TPipe expertTokenOutPipe;
    MoeV2ExpertTokenOut expertTokenOutOp;
    expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t,
                          &expertTokenOutPipe);
    expertTokenOutOp.Process();
    expertTokenOutPipe.Destroy();

    TPipe srcToDstPipe;
    MoeV2SrcToDstWithCapacity<DTYPE_X, MoeInitRoutingV2TilingData> srcToDstWithCapacityOp;
    srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, userWS, t, &srcToDstPipe);
    srcToDstWithCapacityOp.Process();
    srcToDstPipe.Destroy();
  }

  TPipe gatherPipe;
  MoeV2GatherOut<DTYPE_X> gatherOp;
  gatherOp.Init(x, expandedRowIdx, expertTokensCountOrCumsum, expandedX, userWS, t, &gatherPipe);
  gatherOp.Process();
  gatherPipe.Destroy();
}
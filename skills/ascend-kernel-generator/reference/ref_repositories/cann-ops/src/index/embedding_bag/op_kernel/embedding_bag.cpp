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
 * \file embedding_bag.cpp
 * \brief
 */

#include "embedding_bag.h"
#include "embedding_bag_fp16.h"


using namespace AscendC;
    // kernel function
extern "C" __global__ __aicore__ void embedding_bag(GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets,
                                GM_ADDR per_sample_weights, GM_ADDR y, GM_ADDR offset2bag,
                                GM_ADDR bag_size, GM_ADDR max_indices, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[8] = {weight, indices, offsets, per_sample_weights, y, offset2bag, bag_size, max_indices};
    if (TILING_KEY_IS(1)) {
        EmbeddingBag<float,int> op(gmTensor, tilingData, pipe);
        op.Process();
    } 
    if (TILING_KEY_IS(2)) {
        EmbeddingBagFP16<half,int> op(gmTensor, tilingData, pipe);
        op.Process();
    }  
#if __CCE_AICORE__ >= 220
    if (TILING_KEY_IS(3)) {
        EmbeddingBagFP16<bfloat16_t,int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
#endif
}

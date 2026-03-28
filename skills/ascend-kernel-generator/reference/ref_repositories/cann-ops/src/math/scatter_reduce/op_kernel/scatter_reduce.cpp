/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file scatter_reduce.cpp
 */
#include "kernel_operator.h"
#include "scatter_reduce_sca.h"
#include "scatter_reduce_spec1.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void scatter_reduce(GM_ADDR x, GM_ADDR index, GM_ADDR src,
                                                     GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(1)) {

        ScatterDeduceSpec1 spec1;
        spec1.Init(x, index, src, y, tiling_data.batchSize, tiling_data.dimSizeX,
                   tiling_data.dimSizeSrc, tiling_data.strideSize, tiling_data.reduction,
                   tiling_data.includeSelf, &pipe);
        spec1.Process();
    } else if (TILING_KEY_IS(2)) {
        ScatterDeduceSca sca;
        sca.Init(x, index, src, y, tiling_data.batchSize, tiling_data.dimSizeX,
                 tiling_data.dimSizeSrc, tiling_data.strideSize, tiling_data.reduction,
                 tiling_data.includeSelf, &pipe);
        sca.Process();
    }
}
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
 * @file matmul_reduce_scatter_tiling.h
 */

#ifndef __MATMUL_REDUCE_SCATTER_TILING_H__
#define __MATMUL_REDUCE_SCATTER_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ReduceScatterRCSTiling {
    uint32_t rankDim;
    uint32_t rankM;
    uint32_t rankK;
    uint32_t rankN;
    uint32_t isTransposeA;
    uint32_t isTransposeB;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint8_t determinism;
    uint32_t dataType;
};

class MatmulReduceScatterTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    TCubeTiling matmulTiling;
    TCubeTiling tailTiling;
    ReduceScatterRCSTiling param;
};

#endif //__MATMUL_REDUCE_SCATTER_TILING_H__
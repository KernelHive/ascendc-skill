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
 * \file quant_batch_matmul_v3.cpp
 * \brief
 */
#include "quant_batch_matmul_v3.h"
#include "quant_batch_matmul_v3_init_output.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "quant_batch_matmul_v3_cube_basic.h"
#if (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_SCALE == DT_FLOAT)
#include "quant_batch_matmul_v3_bf16_basic.h"
#include "quant_batch_matmul_v3_bf16.h"
#include "quant_batch_matmul_v3_bf16_opt.h"
#include "quant_batch_matmul_v3_pertoken.h"
#include "quant_batch_matmul_v3_pertoken_basic.h"
#include "quant_batch_matmul_v3_pertoken_opt.h"
#endif
#endif

using namespace AscendC;
using namespace matmul;

#define INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(transposeX1, transposeX2)                                          \
    do {                                                                                                          \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                       \
        QuantBatchMatmulV3BaseKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, \
                                     transposeX2, QuantBatchMatmulV3Update> op;                                   \
        op.Init(x1, x2, scale, bias, y, user1, qBmmV3TilingData, &tPipe);                                         \
        op.Process();                                                                                             \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(transposeX1, transposeX2)                             \
    do {                                                                                                  \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                               \
        const TCubeTiling *mmTiling = &(qBmmV3TilingData->matmulTiling);                                  \
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, mmTiling);                                 \
        op.Init(x1, x2, bias, scale, y, user1, qBmmV3TilingData, &tPipe);                                 \
        op.Process();                                                                                     \
        tPipe.Destroy();                                                                                  \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(transposeX1, transposeX2)                            \
    do {                                                                                                     \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                  \
        BmmDequantBf16Opt<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        op.Init(x1, x2, bias, scale, y, user1, qBmmV3TilingData, &tPipe);                                    \
        op.Process();                                                                                        \
        tPipe.Destroy();                                                                                     \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(transposeX1, transposeX2)                             \
    do {                                                                                                      \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                   \
        const TCubeTiling *mmTiling = &(qBmmV3TilingData->matmulTiling);                                      \
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, mmTiling);                                     \
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, qBmmV3TilingData, &tPipe);                      \
        op.Process();                                                                                         \
        tPipe.Destroy();                                                                                      \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(transposeX1, transposeX2)                            \
    do {                                                                                                         \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                      \
        BmmDequantPertokenOpt<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, qBmmV3TilingData, &tPipe);                         \
        op.Process();                                                                                            \
        tPipe.Destroy();                                                                                         \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(transposeX1, transposeX2)                      \
    do {                                                                                                     \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                  \
        BmmDequantPertokenBasic<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, \
                                transposeX2, QuantBatchMatmulV3Update> op;                                   \
        op.Init(x1, x2, scale, bias, pertokenScale, y, user1, qBmmV3TilingData, &tPipe);                     \
        op.Process();                                                                                        \
        tPipe.Destroy();                                                                                     \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(transposeX1, transposeX2)                                  \
    do {                                                                                                              \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                           \
        BmmBasicDequantBf16<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, transposeX2, \
                            QuantBatchMatmulV3Update>  op;                                                            \
        op.Init(x1, x2, scale, bias, y, user1, qBmmV3TilingData, &tPipe);                                             \
        op.Process();                                                                                                 \
        tPipe.Destroy();                                                                                              \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(transposeX1, transposeX2)                         \
    do {                                                                                                \
        BmmDequantInitOutput<DTYPE_Y> clearOp;                                                          \
        clearOp.Init(y, user1, &tilingData, &tPipe);                                                    \
        clearOp.Process();                                                                              \
        tPipe.Destroy();                                                                                \
        TPipe tPipeOp;                                                                                  \
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, transposeX1, transposeX2, \
                   BMM_DEQUANT_PRELOAD_CFG> op;                                                         \
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipeOp);                                  \
        op.Process(true);                                                                               \
    } while (0)

extern "C" __global__ __aicore__ void quant_batch_matmul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
                                                            GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
                                                            GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

// 6bit from hight to low: needClean, pertoken, opt, basic, transX1, transX2
#if (ORIG_DTYPE_Y == DT_FLOAT16 || ORIG_DTYPE_Y == DT_INT8 || ORIG_DTYPE_Y == DT_INT32)  // fp16, int8, int32
#if (ORIG_DTYPE_SCALE != DT_FLOAT || ORIG_DTYPE_Y == DT_INT32)
    if (TILING_KEY_IS(1)) {  // false true
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    if (TILING_KEY_IS(100001)) {  // false true
        BmmDequantInitOutput<DTYPE_Y> clearOp;
        clearOp.Init(y, user1, &tilingData, &tPipe);
        clearOp.Process();
        tPipe.Destroy();

        TPipe tPipeOp;
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipeOp);
        op.Process();
    }
#endif
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#if (ORIG_DTYPE_SCALE != DT_FLOAT || ORIG_DTYPE_Y == DT_INT32)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (TILING_KEY_IS(0)) {  // false false
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, false> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(10)) {  // true false
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, true, false> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(11)) {  // true true
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, true, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
#if ORIG_DTYPE_Y == DT_INT32
    } else if (TILING_KEY_IS(100000)) {
        KERNEL_TASK_TYPE(100000, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(false, false);
    } else if (TILING_KEY_IS(100001)) {
        KERNEL_TASK_TYPE(100001, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(false, true);
    } else if (TILING_KEY_IS(100010)) {
        KERNEL_TASK_TYPE(100010, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(true, false);
    } else if (TILING_KEY_IS(100011)) {
        KERNEL_TASK_TYPE(100011, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(true, true);
#endif
    } else {  // matmul with basic tiling
        if (TILING_KEY_IS(100)) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(false, false);
        } else if (TILING_KEY_IS(101)) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(false, true);
        } else if (TILING_KEY_IS(110)) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(true, false);
        } else if (TILING_KEY_IS(111)) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(true, true);
        }
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if (TILING_KEY_IS(10000)) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, false);
    } else if (TILING_KEY_IS(10001)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, true);
    } else if (TILING_KEY_IS(10010)) {  // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, false);
    } else if (TILING_KEY_IS(10011)) {  // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, true);
    } else if (TILING_KEY_IS(11000)) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, false);
    } else if (TILING_KEY_IS(11001)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, true);
    } else if (TILING_KEY_IS(11010)) {  // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, false);
    } else if (TILING_KEY_IS(11011)) {  // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, true);
    } else if (TILING_KEY_IS(10100)) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else if (TILING_KEY_IS(10101)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true);
    } else if (TILING_KEY_IS(10110)) {  // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false);
    } else if (TILING_KEY_IS(10111)) {  // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true);
    }

#endif
#endif
#else  // bf16
    // 5bit from hight to low: pertoken, opt, basic, transX1, transX2
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    if (TILING_KEY_IS(0)) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(false, false);
    } else if (TILING_KEY_IS(1)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(false, true);
    } else if (TILING_KEY_IS(10)) {  // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(true, false);
    } else if (TILING_KEY_IS(11)) {  // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(true, true);
    } else if (TILING_KEY_IS(1000)) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(false, false);
    } else if (TILING_KEY_IS(1001)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(false, true);
    } else if (TILING_KEY_IS(1010)) {  // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(true, false);
    } else if (TILING_KEY_IS(1011)) {  // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(true, true);
    } else if (TILING_KEY_IS(10000)) {  // false false pertoken begin
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, false);
    } else if (TILING_KEY_IS(10001)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, true);
    } else if (TILING_KEY_IS(10010)) {  // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, false);
    } else if (TILING_KEY_IS(10011)) {  // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, true);
    } else if (TILING_KEY_IS(11000)) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, false);
    } else if (TILING_KEY_IS(11001)) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, true);
    } else if (TILING_KEY_IS(11010)) {  // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, false);
    } else if (TILING_KEY_IS(11011)) {  // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, true);
    } else if (TILING_KEY_IS(10100)) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else if (TILING_KEY_IS(10101)) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true);
    } else if (TILING_KEY_IS(10110)) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false);
    } else if (TILING_KEY_IS(10111)) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true);
    } else {  // matmul with basic tiling no pertoken
        if (TILING_KEY_IS(100)) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(false, false);
        } else if (TILING_KEY_IS(101)) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(false, true);
        } else if (TILING_KEY_IS(110)) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(true, false);
        } else if (TILING_KEY_IS(111)) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(true, true);
        }
    }
#endif
}
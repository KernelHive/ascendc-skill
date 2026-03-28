/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    #include "weight_quant_batch_matmul_v2_constant.h"
    #include "tool.h"
    #if (defined(ORIG_DTYPE_ANTIQUANT_SCALE) && \
         ((ORIG_DTYPE_ANTIQUANT_SCALE == DT_UINT64) || (ORIG_DTYPE_ANTIQUANT_SCALE == DT_INT64)))
        #include "fixpipe/weight_quant_batch_matmul_v2_fixpipe.h"
    #else
        #include "weight_quant_batch_matmul_v2_custom.h"
        #if (defined(ORIG_DTYPE_Y) && ORIG_DTYPE_Y != DT_INT8)
            #include "weight_quant_batch_matmul_v2_msd_multicore.h"
            #include "weight_quant_batch_matmul_v2_msd_group.h"
            #include "weight_quant_batch_matmul_v2_msd_split_k.h"
            #include "weight_quant_batch_matmul_v2_custom_mix_splitk.h"
            #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT == FORMAT_FRACTAL_NZ))
                #include "weight_quant_batch_matmul_v2_custom_weight_nz.h"
                #include "weight_quant_batch_matmul_v2_custom_nz_splitk.h"
            #endif
            using WeightQuantBatchMatmulV2Msd::WeightQuantBatchMatmulV2MsdMultiCoreKernel;
        #endif
    #endif
#else
    #include "weight_quant_batch_matmul_v2_weight_nz_performance.h"
#endif
using namespace WeightQuantBatchMatmulV2;

// if run with ttk without bias, can't get DTYPE_BIAS macro
#ifndef DTYPE_BIAS
    #if defined(ORIG_DTYPE_X) && defined(DT_FLOAT16) && ORIG_DTYPE_X == DT_FLOAT16
        #define DTYPE_BIAS DTYPE_X
    #else
        #define DTYPE_BIAS float
    #endif
#endif

#ifndef DTYPE_ANTIQUANT_OFFSET
    #if defined(ORIG_DTYPE_ANTIQUANT_SCALE) && defined(DT_UINT64) && \
        ORIG_DTYPE_ANTIQUANT_SCALE != DT_UINT64 && ORIG_DTYPE_ANTIQUANT_SCALE != DT_INT64
        #define DTYPE_ANTIQUANT_OFFSET DTYPE_ANTIQUANT_SCALE
    #else
       #define DTYPE_ANTIQUANT_OFFSET int32_t
    #endif
#endif

#if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT32) && ORIG_DTYPE_WEIGHT == DT_INT32
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT AscendC::int4b_t
#endif

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                       \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2ASTilingData, tilingDataIn, tiling);                     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(templateClass, ...)                                                      \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdTilingData, tilingDataIn, tiling);                    \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(templateClass, ...)                                                  \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2FixpipeTilingData, tilingDataIn, tiling);                \
        templateClass<DTYPE_ANTIQUANT_OFFSET, DTYPE_BIAS, __VA_ARGS__> op;                                           \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_DTYPE(templateClass, ...)                                                    \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                       \
        templateClass<__VA_ARGS__> op;                                                                               \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(templateClass, ...)                                                       \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2NzTilingData, tilingDataIn, tiling);                     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(templateClass, ...)                                            \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2CustomNzSplitKTilingData, tilingDataIn, tiling);         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void weight_quant_batch_matmul_v2(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                                                   GM_ADDR antiquantOffset, GM_ADDR quantScale,
                                                                   GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    AscendC::TPipe tPipe;
    #if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220)
        #if (defined(ORIG_DTYPE_ANTIQUANT_SCALE) && \
             ((ORIG_DTYPE_ANTIQUANT_SCALE == DT_UINT64) || (ORIG_DTYPE_ANTIQUANT_SCALE == DT_INT64)))
            // fixp方案
            #if ((ORIG_DTYPE_X == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT16) && (ORIG_DTYPE_WEIGHT == DT_INT8))
                KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
                if ASCEND_IS_AIC {
                    // 模板参数为Trans bTrans antiquantType quantType hasAntiquantOffset hasBias weightFormat aFullLoad
                    if (TILING_KEY_IS(1000200000000012000UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, false, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012000UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, false, CubeFormat::ND, true);
                    } else if (TILING_KEY_IS(1000200000000012010UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, true, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012010UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, true, CubeFormat::ND, true);
                    } else if (TILING_KEY_IS(1000200000000012020UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, false, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012020UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, false, CubeFormat::ND, true);
                    } else if (TILING_KEY_IS(1000200000000012030UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, true, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012030UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, true, CubeFormat::ND, true);
                    }
                }
            #endif
        #elif (ORIG_DTYPE_Y == DT_INT8)
            if (TILING_KEY_IS(310100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(320300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            }
        #else
            #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
                if (TILING_KEY_IS(310100UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311100UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310110UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311110UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310101UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311101UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310111UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311111UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(320300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(320310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(320301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(320311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10611300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::ND);
                } else if (TILING_KEY_IS(10610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10610300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::ND);
                } else if (TILING_KEY_IS(1000111000000003020UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::ND, HighPerformanceType);
                } else if (TILING_KEY_IS(1000111000000003000UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::ND, HighPerformanceType);
                } else if (TILING_KEY_IS(10611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(20611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(20610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(711300UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::ND, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(710300UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::ND, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003020UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::ND, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003000UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::ND, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(911300UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatmulV2MixSplitKKernel<bfloat16_t, int8_t, float, bfloat16_t, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                }
            #else
                if (TILING_KEY_IS(810200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(8611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(8610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(8611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(8610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(1000111000000003021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::NZ, HighPerformanceType);
                } else if (TILING_KEY_IS(1000110000000003021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::NZ, HighPreciseType);
                } else if (TILING_KEY_IS(1000111000000003001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::NZ, HighPerformanceType);
                } else if (TILING_KEY_IS(1000110000000003001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::NZ, HighPreciseType);
                } else if (TILING_KEY_IS(28611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(28610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(18611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(18610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(18611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(18610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(1000100000000003021UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::NZ, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000100000000003001UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::NZ, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003021UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::NZ, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003001UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::NZ, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000010000000012001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, false);
                } else if (TILING_KEY_IS(1000010000000012021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, false);
                } else if (TILING_KEY_IS(1000010001000012001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, true);
                } else if (TILING_KEY_IS(1000010001000012021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, true);
                }
            #endif
        #endif
    #else
        if (TILING_KEY_IS(80010)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_TENSOR, false);
        } else if (TILING_KEY_IS(80011)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_TENSOR, false);
        } else if (TILING_KEY_IS(80020)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_CHANNEL, false);
        } else if (TILING_KEY_IS(80021)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_CHANNEL, false);
        } else if (TILING_KEY_IS(80110)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_TENSOR, true);
        } else if (TILING_KEY_IS(80111)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_TENSOR, true);
        } else if (TILING_KEY_IS(80120)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_CHANNEL, true);
        } else if (TILING_KEY_IS(80121)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_CHANNEL, true);
        }
    #endif
}
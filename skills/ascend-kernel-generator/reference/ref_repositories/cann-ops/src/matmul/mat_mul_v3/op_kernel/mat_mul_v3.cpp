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
 * \file mat_mul_v3.cpp
 * \brief
 */
#include "mat_mul_v3_common.h"
#include "mat_mul_base_kernel.h"
#include "mat_mul_deterministic_splitk_kernel.h"
#include "mat_mul_sc_splitk_kernel.h"
#include "mat_mul_unaligned_base_kernel.h"
#include "mat_mul_unaligned_deterministic_splitk_kernel.h"
#include "mat_mul_unaligned_sc_splitk_kernel.h"
#include "mat_mul_optimized_fixpipe_algorithm.h"
#include "mat_mul_l1_full_load.h"

using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#define MMV3_IMPL(templateFunc, ...)                                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        }                                                                                                            \
    } while(0)


#define MMV3_IMPL_CLASS(templateClass, aFormat, ...)                                                                          \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

// cFormat is for tempCGlobal Nz out, not cTensor out
#define MMV3_IMPL_C_CLASS(templateClass, aFormat, cFormat, ...)                                                                          \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

extern "C" __global__ __aicore__ void mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    // 第一个模板使用mix类型的，使得整个算子的coreType在dyn场景都为mix，静态则根据选择的tilingkey决定coreType
    if (TILING_KEY_IS(10000000000000000000UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernel, format_x1, MatmulBaseBlock, MM_CFG_VEC_ND2NZ);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        MMV3_IMPL_CLASS(MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_VEC_ND2NZ);
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000000UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000021UL)) {
        MMV3_IMPL_CLASS(MatMulSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK);
    } else if (TILING_KEY_IS(10000000000000000020UL)) {
        MMV3_IMPL_CLASS(MatMulUnAlignedSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock,
                        MM_CFG_PRELOAD_MK);
    } else if (TILING_KEY_IS(10000000000000000031UL)) {
        MMV3_IMPL(MatMulKernelDeterministicSplitK, aGM, bGM, cGM, biasGM, tilingData, user);
    } else if (TILING_KEY_IS(10000000000000000030UL)) {
        MMV3_IMPL(MatMulUnAlignedKernelDeterministicSplitK, aGM, bGM, cGM, biasGM, tilingData, user);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        KERNEL_TASK_TYPE(10000000000000000001UL, KERNEL_TYPE_AIC_ONLY);
        MMV3_IMPL_CLASS(MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000101UL)) {
        KERNEL_TASK_TYPE(10000000000000000101UL, KERNEL_TYPE_AIC_ONLY);
        MMV3_IMPL_CLASS(MatmulBaseKernelAL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_MDL);
    } else if (TILING_KEY_IS(10000000000000000201UL)) {
        KERNEL_TASK_TYPE(10000000000000000201UL, KERNEL_TYPE_AIC_ONLY);
        MMV3_IMPL_CLASS(MatmulBaseKernelBL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000010201UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnalignedNKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000000200UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernelBL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000010200UL)) {
        MMV3_IMPL_CLASS(MatmulBaseAToNZWithBL1FixpipeKernel, CubeFormat::NZ, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000020201UL)) {
        MMV3_IMPL_C_CLASS(MatmulBaseUnalignedNKernel, format_x1, CubeFormat::NZ, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT);
    }
#endif
}

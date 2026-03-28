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
 * \file gemm_v2.cpp
 * \brief
 */
#include "mat_mul_base_kernel.h"
using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_A) && FORMAT_A == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_a = CubeFormat::NZ;
#else
constexpr CubeFormat format_a = CubeFormat::ND;
#endif

#if defined(FORMAT_B) && FORMAT_B == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_b = CubeFormat::NZ;
#else
constexpr CubeFormat format_b = CubeFormat::ND;
#endif

#if defined(FORMAT_C) && FORMAT_C == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_c = CubeFormat::NZ;
#else
constexpr CubeFormat format_c = CubeFormat::ND;
#endif

#define MMV3_IMPL_CLASS(templateClass, ...)                                                                                \
   do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_c, DTYPE_C>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        }                                                                                                            \
    } while (0)

extern "C" __global__ __aicore__ void gemm_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR alpha, GM_ADDR beta, GM_ADDR ref_c,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    SetSysWorkspace(workspaceGM);
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    GM_ADDR biasGM = nullptr;
    GM_ADDR offsetWGM = nullptr;
    // 第一个模板使用mix类型的，使得整个算子的coreType在dyn场景都为mix，静态则根据选择的tilingkey决定coreType
    if (TILING_KEY_IS(10000000000000000001UL)) {
        MMV3_IMPL_CLASS(MatmulBaseKernel, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    }
}

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
 * @file matmul_api_constant.cpp
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    if(b == 0) {
        return 0;
    } 
    return (a + b - 1) / b;
}

// The specified value remains consistent with the runtime tiling paramters.
constexpr int32_t singleCoreM = 512;
constexpr int32_t singleCoreN = 640;
constexpr int32_t singleCoreK = 256;
constexpr int32_t baseM = 128;
constexpr int32_t baseN = 128;
constexpr int32_t baseK = 64;

template <typename aType, typename bType, typename cType, typename biasType> class MatmulApiConstantKernel {
public:
    __aicore__ inline MatmulApiConstantKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    using aMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>;
    using bMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>;
    using cMatmulType =  MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>;
    using biasMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;

    constexpr static MatmulShapeParams shapeParams = {singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK};
    constexpr static MatmulConfig mmConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // Get the fully constant template parameters.
    constexpr static MatmulApiStaticTiling staticConfig = GetMatmulApiTiling<aMatmulType, bMatmulType, cMatmulType, biasMatmulType>(mmConfig);

    Matmul<aMatmulType, bMatmulType, cMatmulType, biasMatmulType, staticConfig> matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulApiConstantKernel<aType, bType, cType, biasType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulApiConstantKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulApiConstantKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA,
                                                        int32_t &offsetB, int32_t &offsetC, int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_api_constant(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulApiConstantKernel<half, half, float, float> matmulApiConstantKernel;
    AscendC::TPipe pipe;

    // With the fully constant template parameters, nullptr can be passed into REGIST_MATMUL_OBJ to replace tiling.
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulApiConstantKernel.matmulObj, &tilingData.cubeTilingData);
    matmulApiConstantKernel.Init(a, b, bias, c, workspace, tilingData.cubeTilingData);
    matmulApiConstantKernel.Process(&pipe);
}
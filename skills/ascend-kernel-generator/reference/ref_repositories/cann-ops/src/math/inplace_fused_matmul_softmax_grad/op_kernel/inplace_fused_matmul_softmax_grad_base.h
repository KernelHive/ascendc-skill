/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BASE_H
#define INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace InplaceFusedMatmulSoftmaxGradOpt {
using namespace AscendC;
using namespace matmul;

template <typename cubeDataType, const MatmulConfig &MM_CFG = CFG_NORM>
struct MMType {
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, cubeDataType>;
    using BT = MatmulType<TPosition::GM, CubeFormat::ND, cubeDataType, true>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, cubeDataType>;
    using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ELE_NUM_FP32 = 8;

template <typename mmType, typename dataType, bool isAligned = false, bool isCast = false>
class InplaceFusedMatmulSoftmaxGradBase {
public:
    __aicore__ inline InplaceFusedMatmulSoftmaxGradBase(mmType &mm) : mm1_(mm)
    {
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

public:
    TPipe *pPipe_ = nullptr;
    mmType &mm1_;
    /* tiling data */
    BaseTiling baseTilingData_;
    SoftMaxTiling headSoftMaxGradTilingData_;
    TCubeTiling cubeTilingData_;
    
    /* variable */
    AscendC::GlobalTensor<dataType> softmaxOutputGm_;   // 输入A，兼输出
    AscendC::GlobalTensor<dataType> gradOutputGm_;      // 输入B
    AscendC::GlobalTensor<dataType> valuesGm_;          // 输入C
    AscendC::GlobalTensor<float> gradSoftmaxGm_;     // 中间值D: D = matmul(B, transpose(C));
    AscendC::GlobalTensor<dataType> gradXGm_;           // 输出E，后续改输出到A上
    AscendC::LocalTensor<float> softmaxTmpUb1_;                  // big shape 非bf16场景临时UB空间
    AscendC::LocalTensor<float> softmaxTmpUb2_;                  // big shape 非bf16场景临时UB空间
    AscendC::LocalTensor<float> softmaxOutputUB32Temp;           // bf16 场景临时UB空间
    AscendC::LocalTensor<float> outGradX32Temp;                  // bf16 场景临时UB空间

    uint32_t blockIdx_{0};
    uint32_t rowLen_{0};

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSoftmaxOutput_; // 输入A
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradSoftmax_;   // 中间值D: D = matmul(B, transpose(C));
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradX_;       // 输出E
    TBuf<TPosition::VECCALC> SoftmaxOutput32Temp_; // 输入A中间值
    TBuf<TPosition::VECCALC> outGradX32Temp_;  //输出中间值
    TBuf<TPosition::VECCALC> sharedTempBuf1_;                    //  vector计算值中间值
    TBuf<TPosition::VECCALC> sharedTempBuf2_;                    //  vector计算值中间值
};
}  // namespace InplaceFusedMatmulSoftmaxGradOpt
#endif  // INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BASE_H
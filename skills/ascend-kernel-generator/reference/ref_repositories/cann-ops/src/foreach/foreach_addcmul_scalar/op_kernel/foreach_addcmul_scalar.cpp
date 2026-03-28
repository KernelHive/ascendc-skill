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
 * \file foreach_addcmul_scalar.cpp
 * \brief
 */

 #include "kernel_operator.h"

 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
 #include "foreach_one_scalar_quaternary_implict_output.h"
 
 using namespace AscendC;
 using namespace Common::OpKernel;
 
template <typename T>
__aicore__ void AddcMulScalarAdapterForFloat(
    const LocalTensor<T>& tensorLocal1, 
    const LocalTensor<T>& tensorLocal2,
    const LocalTensor<T>& tensorLocal3,
    const LocalTensor<T>& float32Tensor,
    const T scalarValue,
    const uint32_t maxCastDataCount,
    const int64_t dataCount) {
    Mul(tensorLocal2, tensorLocal2, tensorLocal3, dataCount);
    PipeBarrier<PIPE_V>();
    Axpy<T, T>(tensorLocal1, tensorLocal2, scalarValue, dataCount);
}
 
template <typename T>
__aicore__ void AddcMulScalarAdapterForInt(
    const LocalTensor<T>& tensorLocal1, 
    const LocalTensor<T>& tensorLocal2,
    const LocalTensor<T>& tensorLocal3,
    const LocalTensor<float>& float32Tensor,
    const float scalarValue,
    const uint32_t maxCastDataCount,
    const int64_t dataCount) {
    Mul(tensorLocal2, tensorLocal2, tensorLocal3, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(tensorLocal2, tensorLocal2, (int32_t)scalarValue, dataCount);
    PipeBarrier<PIPE_V>(); 
    Add(tensorLocal1, tensorLocal1, tensorLocal2, dataCount);
}
 
 template <typename T>
 __aicore__ void ComputerPerCastForAddcMulScalar(
         const LocalTensor<T> &tensorLocal1, 
         const LocalTensor<T> &tensorLocal2, 
         const LocalTensor<T> &tensorLocal3, 
         const LocalTensor<float> &float32Tensor,
         const float scalarValue, const uint32_t maxCastDataCount, 
         const uint32_t index, const int64_t dataCount) {
     
     Cast(float32Tensor, tensorLocal2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
     PipeBarrier<PIPE_V>();
     Cast(float32Tensor[maxCastDataCount], tensorLocal3[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
     PipeBarrier<PIPE_V>();
     Cast(float32Tensor[maxCastDataCount * 2], tensorLocal1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
     PipeBarrier<PIPE_V>();
     // input + scalar_tensor * (tensor1 / tensor2)
     PipeBarrier<PIPE_V>();
     Mul(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], dataCount);
     PipeBarrier<PIPE_V>();
     Axpy<float, float>(float32Tensor[maxCastDataCount * 2], float32Tensor, scalarValue, dataCount);
     PipeBarrier<PIPE_V>();
     Cast(tensorLocal1[index * maxCastDataCount], float32Tensor[maxCastDataCount * 2], RoundMode::CAST_RINT, dataCount);
 }
 
 template <typename T>
 __aicore__ void AddcMulScalarAdapter(
     const LocalTensor<T>& tensorLocal1, 
     const LocalTensor<T>& tensorLocal2,
     const LocalTensor<T>& tensorLocal3,
     const LocalTensor<float>& float32Tensor,
     const float scalarValue,
     const uint32_t maxCastDataCount,
     const int64_t dataCount) {
     
     uint32_t castTimes = dataCount / maxCastDataCount;
     uint32_t castTimesRemainder = dataCount % maxCastDataCount;
     for (uint32_t i = 0; i < castTimes; i++) {
         ComputerPerCastForAddcMulScalar<T>(
             tensorLocal1, tensorLocal2, tensorLocal3,
             float32Tensor, scalarValue, maxCastDataCount, i, maxCastDataCount);
     }
     if (castTimesRemainder > 0) {
         ComputerPerCastForAddcMulScalar<T>(
             tensorLocal1, tensorLocal2, tensorLocal3,
             float32Tensor, scalarValue, maxCastDataCount, castTimes, castTimesRemainder);
     }
 }
 
 extern "C" __global__ __aicore__ void foreach_addcmul_scalar(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3, GM_ADDR scalar,
     GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
     GET_TILING_DATA(tilingData, tiling);
 
     //foreach(vector) not need workspace
     GM_ADDR userWS = nullptr;
 
     if (TILING_KEY_IS(1)) {
         ForeachOneScalarQuaternaryImplictOutput<half, AddcMulScalarAdapter<half>, 2, 3> op;
         op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(2)) {
         ForeachOneScalarQuaternaryImplictOutput<float, AddcMulScalarAdapterForFloat<float>, 2, 3> op;
         op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
         op.Process();
     } 
     #if __CCE_AICORE__ == 220
     else if (TILING_KEY_IS(3)) {
         ForeachOneScalarQuaternaryImplictOutput<int, AddcMulScalarAdapterForInt<int>, 2, 3> op;
         op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(4)) {
         ForeachOneScalarQuaternaryImplictOutput<bfloat16_t, AddcMulScalarAdapter<bfloat16_t>, 2, 3> op;
         op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
         op.Process();
     }
     #endif
 }
 
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
 * @file mul_mul_reduce_mean_d_twice.cpp
 */

#include "kernel_operator.h"
using namespace AscendC;
namespace AscendC {
template <typename T>
class MulMulReduceMeanDTwice {
public:
    __aicore__ inline MulMulReduceMeanDTwice(){}

    __aicore__ inline void Init(GM_ADDR mul0_input0, GM_ADDR mul0_input1, GM_ADDR mul1_input0, GM_ADDR add_y, GM_ADDR gamma, GM_ADDR beta, 
    GM_ADDR output, const MulMulReduceMeanDTwiceTilingData &tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        former_num = tiling_data.formerNum;
        tail_num = tiling_data.tailNum;
        former_length = tiling_data.formerLength;
        tail_length = tiling_data.tailLength;
        tile_length = tiling_data.tileLength;
        share_size = tiling_data.shareSize;
        core_id = GetBlockIdx();

        meanParams.outter = 1; 
        meanParams.inner = tile_length;
        meanParams.n = tile_length;
        uint32_t block_offset, block_length;
        if (core_id < former_num) {
            block_offset = core_id * former_length * tile_length;
            block_length = former_length * tile_length;
        } else {
            uint32_t total_former_length = former_num * former_length * tile_length;
            block_offset = (core_id - former_num) * tail_length * tile_length + total_former_length;
            block_length = tail_length * tile_length;
        }
        mul0_input0_global.SetGlobalBuffer((__gm__ T*)mul0_input0 + block_offset, block_length);
        mul0_input1_global.SetGlobalBuffer((__gm__ T*)mul0_input1 + block_offset, block_length);
        output_global.SetGlobalBuffer((__gm__ T*)output + block_offset, block_length);
        gamma_global.SetGlobalBuffer((__gm__ T*)gamma, tile_length);
        beta_global.SetGlobalBuffer((__gm__ T*)beta, tile_length);
        
        pipe->InitBuffer(mul0_input0_queue, 1, tile_length * sizeof(T));
        pipe->InitBuffer(mul0_input1_queue, 1, tile_length * sizeof(T));
        pipe->InitBuffer(output_queue, 1, tile_length * sizeof(T));

        pipe->InitBuffer(gamma_buf, tile_length * sizeof(T));
        pipe->InitBuffer(beta_buf, tile_length * sizeof(T));
        pipe->InitBuffer(mul_result_buf, tile_length * sizeof(T));
        pipe->InitBuffer(reduceMeanD_buf0, tile_length * sizeof(T));
        pipe->InitBuffer(reduceMeanD_buf1, tile_length * sizeof(T));
        pipe->InitBuffer(reduce_worklocal_buf, share_size);

        mul1_input0_scalar = *((__gm__ T*)mul1_input0);
        add_y_scalar = *((__gm__ T*)add_y);
    }

     __aicore__ inline void Process()
    {
        gamma_local = gamma_buf.Get<T>();
        beta_local = beta_buf.Get<T>();
        DataCopy(gamma_local, gamma_global, tile_length);
        DataCopy(beta_local, beta_global, tile_length);
        if (core_id < former_num) {  
            for (int rowIdx = 0; rowIdx < former_length; rowIdx++) {
                CopyIn(rowIdx);
                Compute(rowIdx);
                CopyOut(rowIdx);
            }
        } else {
            for (int rowIdx = 0; rowIdx < tail_length; rowIdx++) {
                CopyIn(rowIdx);
                Compute(rowIdx);
                CopyOut(rowIdx);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T>  mul0_input0_local = mul0_input0_queue.AllocTensor<T>();
        DataCopy(mul0_input0_local, mul0_input0_global[progress * tile_length], tile_length);
        LocalTensor<T>  mul0_input1_local = mul0_input1_queue.AllocTensor<T>();
        DataCopy(mul0_input1_local, mul0_input1_global[progress * tile_length], tile_length);
        mul0_input0_queue.EnQue(mul0_input0_local);
        mul0_input1_queue.EnQue(mul0_input1_local);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T>  mul0_input0_local = mul0_input0_queue.DeQue<T>();
        LocalTensor<T>  mul0_input1_local = mul0_input1_queue.DeQue<T>();
        LocalTensor<T> output_Local = output_queue.AllocTensor<T>();
	   	Mul(mul0_input0_local, mul0_input0_local, mul0_input1_local, tile_length);
        Muls(mul0_input0_local, mul0_input0_local, mul1_input0_scalar, tile_length);

        // first reduceMeanD
        LocalTensor<T>  reduceMeanD_buf0_local = reduceMeanD_buf0.Get<T>();
        LocalTensor<uint8_t>  reduce_worklocal_buf_local = reduce_worklocal_buf.Get<uint8_t>();
        Mean<T>(reduceMeanD_buf0_local, mul0_input0_local, reduce_worklocal_buf_local, meanParams);
        T reduce_mean0 = reduceMeanD_buf0_local.GetValue(0);

        // squredDifference
        LocalTensor<T>  mul_result_local = mul_result_buf.Get<T>();
        Duplicate(reduceMeanD_buf0_local, reduce_mean0, tile_length);
		Sub(mul0_input1_local, mul0_input0_local, reduceMeanD_buf0_local, tile_length);
		Mul(mul_result_local, mul0_input1_local, mul0_input1_local, tile_length); // get (x - y)(x - y)

        // second reduceMeanD
        LocalTensor<T>  reduceMeanD_buf1_local = reduceMeanD_buf1.Get<T>();
        Mean<T>(reduceMeanD_buf1_local, mul_result_local, reduce_worklocal_buf_local, meanParams);
		Adds(reduceMeanD_buf1_local, reduceMeanD_buf1_local, add_y_scalar, 16); // 取前32Byte计算即可, 只需要第一个值
		Rsqrt(reduceMeanD_buf1_local, reduceMeanD_buf1_local, 16);
		T rsqrt_result = reduceMeanD_buf1_local.GetValue(0);
        Muls(reduceMeanD_buf1_local, gamma_local, rsqrt_result, tile_length);

        // Get output
        Mul(reduceMeanD_buf0_local, reduceMeanD_buf1_local, reduceMeanD_buf0_local, tile_length);
		Mul(reduceMeanD_buf1_local, reduceMeanD_buf1_local, mul0_input0_local, tile_length);
        Sub(reduceMeanD_buf0_local, beta_local, reduceMeanD_buf0_local, tile_length);
		Add(output_Local, reduceMeanD_buf0_local, reduceMeanD_buf1_local, tile_length);
        mul0_input0_queue.FreeTensor(mul0_input0_local);
        mul0_input1_queue.FreeTensor(mul0_input1_local);
        output_queue.EnQue<T>(output_Local);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> output_Local = output_queue.DeQue<T>();
        DataCopy(output_global[progress * tile_length], output_Local, tile_length);
        output_queue.FreeTensor(output_Local);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> mul0_input0_queue, mul0_input1_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;

    TBuf<QuePosition::VECIN> gamma_buf, beta_buf;
    TBuf<QuePosition::VECCALC> reduceMeanD_buf0, reduceMeanD_buf1, reduce_worklocal_buf, mul_result_buf; 
    GlobalTensor<T> mul0_input0_global;
    GlobalTensor<T> mul0_input1_global;
    GlobalTensor<T> gamma_global;
    GlobalTensor<T> beta_global;
    GlobalTensor<T> output_global;
    LocalTensor<T> gamma_local;
    LocalTensor<T> beta_local;
    MeanParams meanParams;

    uint32_t core_id;
    uint32_t former_num; 
    uint32_t tail_num; 
    uint32_t former_length;
    uint32_t tail_length; 
    uint32_t tile_length; 
    uint32_t share_size;    

    T mul1_input0_scalar; 
    T add_y_scalar; 
};
}

extern "C" __global__ __aicore__ void mul_mul_reduce_mean_d_twice(GM_ADDR mul0_input0, GM_ADDR mul0_input1, GM_ADDR mul1_input0, GM_ADDR add_y, GM_ADDR gamma, GM_ADDR beta, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
    GET_TILING_DATA(tiling_data, tiling);
    MulMulReduceMeanDTwice<half> op;
    TPipe pipe;
    op.Init(mul0_input0, mul0_input1, mul1_input0, add_y, gamma, beta, output, tiling_data, &pipe);
    op.Process();
}
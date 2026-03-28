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
 * @file mul_sigmoid.cpp
 */

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace Ascend
{
    class MulSigmoid {
        public:
            __aicore__ MulSigmoid(){};
        
            __aicore__ inline void init(GM_ADDR x1, GM_ADDR x2, GM_ADDR out_buf,
                                            GM_ADDR workspace, const MulSigmoidTilingData& tiling) {
                ASSERT(AscendC::GetBlockNum() != 0 && "useful core num can not be zero!!!")
                uint32_t BUFFER_NUM = 1;
                this->core_id = AscendC::GetBlockIdx();
        
                this->former_num = tiling.formerCoreNum; // 大块核使用的核数
                this->former_length = tiling.formerCoreRowLen; // 每个大块核负责的行数
                this->tail_num = tiling.tailCoreNum; // 小块核使用的核数
                this->tail_length = tiling.tailCoreRowLen; // 每个小块核负责的行数
        
                this->tile_num = tiling.tileNum; // 每个核每次循环的次数，tile_num * tile_length == 8192
                this->tile_length = tiling.tileLen; // 每个核每次循环的列数，先暂定1024
                
                this->t1 = static_cast<half>(tiling.t1);
                this->t2 = static_cast<half>(tiling.t2);
                this->t3 = static_cast<half>(tiling.t3);
                
                if (this->core_id < this->former_num * this->tile_num) { // 大块核
                    this->start_row_idx = this->core_id / this->tile_num * this->former_length * this->tile_num + this->core_id % this->tile_num;
                    this->end_row_idx = this->start_row_idx + this->former_length;
                } else { // 小块核
                    this->start_row_idx = (this->former_num * this->former_length + (this->core_id / this->tile_num - this->former_num) * this->tail_length) * this->tile_num + this->core_id % this->tile_num;
                    this->end_row_idx = this->start_row_idx + this->tail_length;
                }
                this->x1_buf_global_.SetGlobalBuffer((__gm__ half*)x1 + this->start_row_idx * this->tile_length); // 每个核负责的行
                this->out_buf_global_.SetGlobalBuffer((__gm__ half*)out_buf + this->start_row_idx * this->tile_length); // 每个核负责的行
                this->x2_buf_global_.SetGlobalBuffer((__gm__ half*)x2 + this->core_id %  this->tile_num * this->tile_length); // 每个核负责的行
        
                pipe_.InitBuffer(this->x1_queue, BUFFER_NUM, this->tile_length * sizeof(half)); // 存放x1，每次放tile_length这么长
                pipe_.InitBuffer(this->x2_buf, this->tile_length * sizeof(half)); // 每次都把x2全部拷贝进来
                pipe_.InitBuffer(this->out_queue, BUFFER_NUM, this->tile_length * sizeof(half));
        
                pipe_.InitBuffer(less_local_buf, this->tile_length * sizeof(uint8_t));
        }
        
            __aicore__ inline void process() {
                CopyInX2();
                for (int32_t idx = 0; idx < this->end_row_idx - this->start_row_idx; idx++) {
                    CopyInX1(idx);
                    Compute(idx);
                    CopyOut(idx);
                }
            }
        
            __aicore__ inline void CopyInX2() {
                this->x2_local = this->x2_buf.Get<half>();
                DataCopy(this->x2_local, this->x2_buf_global_, this->tile_length);
            }
        
            __aicore__ inline void CopyInX1(int32_t idx) {
                AscendC::LocalTensor<half>  x1_local = this->x1_queue.AllocTensor<half>();
                DataCopy(x1_local, this->x1_buf_global_[idx * this->tile_length * this->tile_num], this->tile_length);
                this->x1_queue.EnQue(x1_local);
            }
        
            __aicore__ inline void Compute(uint32_t idx) {
                AscendC::LocalTensor<half>  x1_local = this->x1_queue.DeQue<half>();
                AscendC::LocalTensor<half> out_local = this->out_queue.AllocTensor<half>();
        
                /* 计算过程 */
                AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);
        
                AscendC::Sigmoid<half, false>(out_local, x1_local, this->tile_length);
        
                AscendC::LocalTensor<uint8_t> less_local = this->less_local_buf.Get<uint8_t>();
                AscendC::CompareScalar(less_local, out_local, this->t2, AscendC::CMPMODE::LT, this->tile_length);
        
                AscendC::Muls(x1_local, out_local, half_two, this->tile_length);
                
                AscendC::Select(out_local, less_local, out_local, x1_local, AscendC::SELMODE::VSEL_CMPMASK_SPR, this->tile_length);
        
                AscendC::Mul(out_local, out_local, this->x2_local, this->tile_length);
                AscendC::Muls(out_local, out_local, static_cast<half>(this->t3), this->tile_length);
        
                /* 计算过程 */
                this->out_queue.EnQue(out_local);
                this->x1_queue.FreeTensor(x1_local);
            }
        
            __aicore__ inline void CopyOut(uint32_t idx) {
                AscendC::LocalTensor<half>  out_local = this->out_queue.DeQue<half>();
                DataCopy(out_buf_global_[idx * this->tile_length * this->tile_num], out_local, this->tile_length);
                this->out_queue.FreeTensor(out_local);
            }
        
        private:
            AscendC::TPipe pipe_;
            AscendC::TQue<AscendC::QuePosition::VECIN, 1> x1_queue;
            AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_queue;
        
            AscendC::TBuf<AscendC::TPosition::VECIN> x2_buf;
            AscendC::TBuf<AscendC::TPosition::VECCALC> less_local_buf;
        
            AscendC::GlobalTensor<half> x1_buf_global_;
            AscendC::GlobalTensor<half> x2_buf_global_;
            AscendC::GlobalTensor<half> out_buf_global_;
            AscendC::LocalTensor<half> x2_local;
            
            int32_t former_num; // 大块核使用的核数
            int32_t tail_num; // 小块核使用的核数
            int32_t former_length; // 大块核每个核负责的行数
            int32_t tail_length; // 小块核每个核负责的行数
        
            int32_t tile_num; // 每个核每次循环的次数，tile_num * tile_length == 8192
            int32_t tile_length; // 每个核每次循环的列数，先暂定1024
        
            int32_t start_row_idx;
            int32_t end_row_idx;
        
            int32_t core_id;
            half t1;
            half t2;
            half t3; // 接收输入的3个标量
        
            float16_t half_two = 2.0;
    };
}

extern "C" __global__ __aicore__ void mul_sigmoid(GM_ADDR x1, GM_ADDR x2, GM_ADDR out_buf, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data_in, tiling);

  if (TILING_KEY_IS(1)) {
    Ascend::MulSigmoid op;  
    op.init(x1, x2, out_buf, workspace, tiling_data_in);
    op.process();
  }
}
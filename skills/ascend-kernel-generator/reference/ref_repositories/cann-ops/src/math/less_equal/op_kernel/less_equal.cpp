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
 * @file less_equal.cpp
 */
#include <type_traits>
#include "kernel_operator.h"


namespace LessEqualK{

constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
constexpr float NEGATIVE_ONE_FP32 = -1.0F;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t NEGATIVE_ONE_I32 = -1;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;
constexpr uint32_t BLOCK_SIZE = 32;


template <typename typeT>
class KernelLessEqual {
public:
  __aicore__ inline KernelLessEqual() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
      const LessEqualTilingData& tiling_data) {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    ResovleTiling(tiling_data);
    x1_gm.SetGlobalBuffer(
      (__gm__ typeT*)x1 + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    x2_gm.SetGlobalBuffer(
      (__gm__ typeT*)x2 + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    y_gm.SetGlobalBuffer((__gm__ int8_t*)y + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);

    pipe.InitBuffer(x1_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(x2_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(y_outque, BUFFER_NUM,
      this->tile_cache * sizeof(int8_t) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(int8_t));
    pipe.InitBuffer(calc_buf_1, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(calc_buf_2, this->tile_cache * sizeof(half) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_3, this->tile_cache * sizeof(half) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_4, this->tile_cache * sizeof(float) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(float));
  }
  __aicore__ inline void Process() {
    if (this->total_length <= BLOCK_SIZE / sizeof(typeT)) {
      CopyInPad(0);
      Compute(0);
      CopyOutPad(0);
      return;
    }
    int32_t loopCount = this->tile_num;
    for (int32_t i = 0; i < loopCount - 1; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
    if (AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
      CopyInPad(loopCount - 1);
      Compute(loopCount - 1);
      CopyOutPad(loopCount - 1);
    }
    else {
      CopyIn(loopCount - 1);
      Compute(loopCount - 1);
      CopyOut(loopCount - 1);
    }
  }

private:
  __aicore__ inline void ComputeHalf(AscendC::LocalTensor<half> x1_local, AscendC::LocalTensor<half> x2_local, AscendC::LocalTensor<half> y_compute) {
    AscendC::Max(y_compute, x1_local, x2_local, this->tile_cache);
    AscendC::Sub(y_compute, x2_local, y_compute, this->tile_cache);
    AscendC::Abs(y_compute, y_compute, this->tile_cache);
    AscendC::Mins(y_compute, y_compute, static_cast<half>(MIN_ACCURACY_FP16), this->tile_cache);
    AscendC::Muls(y_compute, y_compute, static_cast<half>(MAX_MUL_FP16), this->tile_cache);
    AscendC::Muls(y_compute, y_compute, static_cast<half>(MAX_MUL_FP16), this->tile_cache);
    AscendC::Adds(y_compute, y_compute, static_cast<half>(NEGATIVE_ONE_FP32), this->tile_cache);
    AscendC::Abs(y_compute, y_compute, this->tile_cache);
  }

  __aicore__ inline void ComputeFloat(AscendC::LocalTensor<float> x1_local, AscendC::LocalTensor<float> x2_local, AscendC::LocalTensor<float> y_compute) {
    AscendC::Max(y_compute, x1_local, x2_local, this->tile_cache);
    AscendC::Sub(y_compute, x2_local, y_compute, this->tile_cache);
    AscendC::Abs(y_compute, y_compute, this->tile_cache);
    AscendC::Mins(y_compute, y_compute, static_cast<float>(MIN_ACCURACY_FP32), this->tile_cache);
    AscendC::Muls(y_compute, y_compute, static_cast<float>(MAX_MUL_1_FP32), this->tile_cache);
    AscendC::Muls(y_compute, y_compute, static_cast<float>(MAX_MUL_1_FP32), this->tile_cache);
    AscendC::Muls(y_compute, y_compute, static_cast<float>(MAX_MUL_2_FP32), this->tile_cache);
    AscendC::Adds(y_compute, y_compute, static_cast<float>(NEGATIVE_ONE_FP32), this->tile_cache);
    AscendC::Abs(y_compute, y_compute, this->tile_cache);
  }

  __aicore__ inline void ComputeInt8(AscendC::LocalTensor<half> x1_local_fp16, AscendC::LocalTensor<half> x2_local_fp16, AscendC::LocalTensor<half> y_local_fp16) {
    AscendC::Min(y_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
    AscendC::Sub(y_local_fp16, x2_local_fp16, y_local_fp16, this->tile_cache);
    AscendC::Mins(y_local_fp16, y_local_fp16, (half)POSITIVE_ONE_FP32, this->tile_cache);

    AscendC::Sub(x1_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
    AscendC::Abs(x1_local_fp16, x1_local_fp16, this->tile_cache);
    AscendC::Mins(x1_local_fp16, x1_local_fp16, (half)POSITIVE_ONE_FP32, this->tile_cache);
    AscendC::Duplicate(x2_local_fp16, (half)POSITIVE_ONE_FP32, this->tile_cache);
    AscendC::Sub(x1_local_fp16, x2_local_fp16, x1_local_fp16, this->tile_cache);

    AscendC::Add(y_local_fp16, y_local_fp16, x1_local_fp16, this->tile_cache);
  }

  __aicore__ inline void ComputeInt32(AscendC::LocalTensor<int32_t> x1_local, AscendC::LocalTensor<int32_t> x2_local, AscendC::LocalTensor<int32_t> y_compute) {
    AscendC::Min(y_compute, x1_local, x2_local, this->tile_cache);
    AscendC::Sub(y_compute, x2_local, y_compute, this->tile_cache);
    AscendC::Mins(y_compute, y_compute, static_cast<int32_t>(POSITIVE_ONE_I32), this->tile_cache);

    AscendC::Sub(x1_local, x1_local, x2_local, this->tile_cache);
    AscendC::Mins(x1_local, x1_local, static_cast<int32_t>(POSITIVE_ONE_I32), this->tile_cache);
    AscendC::Maxs(x1_local, x1_local, static_cast<int32_t>(NEGATIVE_ONE_I32), this->tile_cache);
    AscendC::Mul(x1_local, x1_local, x1_local, this->tile_cache);
    AscendC::Duplicate(x2_local, static_cast<int32_t>(POSITIVE_ONE_I32), this->tile_cache);
    AscendC::Sub(x1_local, x2_local, x1_local, this->tile_cache);

    AscendC::Add(y_compute, y_compute, x1_local, this->tile_cache);
  }

  __aicore__ inline void ResovleTiling(const LessEqualTilingData& tiling_data) {
    uint32_t total_length = tiling_data.totalLength;
    uint32_t tile_num_mean = tiling_data.tileNumMean;
    uint32_t tile_num_end = tiling_data.tileNumEnd;
    uint32_t tile_length_mean = tiling_data.tileLengthMean;
    uint32_t tile_length_end = tiling_data.tileLengthEnd;
    uint32_t block_length_mean = tiling_data.blockLengthMean;
    uint32_t block_length_end = tiling_data.blockLengthEnd;

    uint32_t pad32 = BLOCK_SIZE;  // 对齐32B需要的最小数据量
    this->total_length = total_length;
    if (AscendC::GetBlockNum() >= 1 && AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
      this->block_length = block_length_end;
      this->tile_num = tile_num_end;
    }
    else {
      this->block_length = block_length_mean;
      this->tile_num = tile_num_mean;
    }
    this->block_offset = block_length_mean;
    this->tile_length = tile_length_mean;
    this->tile_cache = tile_length_mean;
    this->tile_length_end = tile_length_end;
    if (total_length < pad32) {
      this->block_offset = 0;
      this->tile_cache = pad32;
      this->block_length = pad32;
    }
  }
  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
    AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache], this->tile_cache);
    AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache], this->tile_cache);
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void CopyInPad(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
    AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache],
      ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache],
      ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.DeQue<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.DeQue<typeT>();
    AscendC::LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
    AscendC::LocalTensor<typeT> y_compute = calc_buf_1.Get<typeT>();

    if constexpr (std::is_same_v<typeT, half>) {
      ComputeHalf(x1_local, x2_local, y_compute);

      AscendC::Cast(y_local, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, float>) {
      AscendC::LocalTensor<half> y_fp16 = calc_buf_2.Get<half>();

      ComputeFloat(x1_local, x2_local, y_compute);

      AscendC::Cast(y_fp16, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, int8_t>) {
      AscendC::LocalTensor<half> x1_local_fp16 = calc_buf_2.Get<half>();
      AscendC::LocalTensor<half> x2_local_fp16 = calc_buf_3.Get<half>();
      AscendC::LocalTensor<half> y_local_fp16 = calc_buf_4.Get<half>();

      AscendC::Cast(x1_local_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(x2_local_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      ComputeInt8(x1_local_fp16, x2_local_fp16, y_local_fp16);

      AscendC::Cast(y_local, y_local_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, int32_t>) {
      AscendC::LocalTensor<half> y_fp16 = calc_buf_3.Get<half>();
      AscendC::LocalTensor<float> y_fp32 = calc_buf_4.Get<float>();

      ComputeInt32(x1_local, x2_local, y_compute);

      AscendC::Cast(y_fp32, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_fp16, y_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }

    y_outque.EnQue<int8_t>(y_local);
    x1_inque.FreeTensor(x1_local);
    x2_inque.FreeTensor(x2_local);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    AscendC::DataCopy(y_gm[progress * this->tile_cache], y_local,
      this->tile_cache);
    y_outque.FreeTensor(y_local);
  }
  __aicore__ inline void CopyOutPad(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    AscendC::DataCopy(y_gm[progress * this->tile_cache], y_local,
      (this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    y_outque.FreeTensor(y_local);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_1; 
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_2; 
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_3; 
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_4;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1_inque;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x2_inque;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> y_outque;
  AscendC::GlobalTensor<typeT> x1_gm;
  AscendC::GlobalTensor<typeT> x2_gm;
  AscendC::GlobalTensor<int8_t> y_gm;
  uint32_t total_length;
  uint32_t block_length;
  uint32_t block_offset;
  uint32_t tile_num;
  uint32_t tile_cache;
  uint32_t tile_length;
  uint32_t tile_length_end;
};
}

extern"C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  LessEqualK::KernelLessEqual<DTYPE_X1> op;
  op.Init(x1, x2, y, tiling_data);
  op.Process();
}
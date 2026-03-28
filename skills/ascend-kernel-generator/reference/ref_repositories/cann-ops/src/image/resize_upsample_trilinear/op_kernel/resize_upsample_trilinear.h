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
 * \file resize_upsample_trilinear.h
 * \brief
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_H
#define RESIZE_UPSAMPLE_TRILINEAR_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleTrilinearNs {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class KernelUpsampleTrilinear {
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

 public:
  TPipe pipe;
  matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
      matmul_w;
  matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
      matmul_h;
  matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
      matmul_d;

  __aicore__ inline KernelUpsampleTrilinear(){};

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const UpsampleTrilinearTilingData* tilingData);

  __aicore__ inline void Process();

  constexpr static uint8_t BUFFER_NUM = 2;
  constexpr static int32_t SLIDE_SIZE = 16;
  constexpr static int32_t SINGLE_N_MAX = 65534;

 private:
  __aicore__ inline bool FloatEqual(float a, float b) {
    float closeTo0 = float(1e-6);
    if (a > b) {
      return a - b < closeTo0;
    } else {
      return b - a < closeTo0;
    }
  };
  __aicore__ inline void CalcInputOffsetW();

  __aicore__ inline void CalcInputOffsetH();

  __aicore__ inline void CalcInputOffsetD();

  __aicore__ inline void CalcRatioMetrixInRight(const int32_t output_start_indx, const int32_t output_end_indx,
                                                const int32_t omega_metrix_w, const int32_t omega_metrix_h, float scale,
                                                const int32_t input_start_indx, const int32_t input_end_indx);

  __aicore__ inline void CalcRatioMetrixInLeft(const int32_t output_start_indx, const int32_t output_end_indx,
                                               const int32_t omega_metrix_w, const int32_t omega_metrix_h, float scale,
                                               const int32_t input_start_indx, const int32_t input_end_indx);

  __aicore__ inline void CalcMatMulInW(int32_t input_col_start, int32_t output_col_start, int32_t row_start,
                                       int32_t row_end, int32_t k, int32_t n);

  __aicore__ inline void CalcMatMulInH(int32_t input_col_start, int32_t output_col_start, int32_t batch_index,
                                       int32_t m, int32_t n, int32_t k);

  __aicore__ inline void CalcMatMulInD(int32_t input_col_start, int32_t output_col_start, int32_t batch_index,
                                       int32_t m, int32_t n, int32_t k);

  __aicore__ inline void CopyRatioMetrix2Gm();

  __aicore__ inline float AreaPixelComputeSourceIndex(float scale, int32_t dst_index);

 private:
  TQue<QuePosition::VECOUT, BUFFER_NUM> omega_metrix_que;
  TQue<QuePosition::VECOUT, BUFFER_NUM> omega_metrix_cast_que;
  GlobalTensor<T> input_gm;
  GlobalTensor<T> output_gm;
  GlobalTensor<T> intermediate_gm;

  const TCubeTiling* __restrict matmul_tiling_w;
  const TCubeTiling* __restrict matmul_tiling_h;
  const TCubeTiling* __restrict matmul_tiling_d;

  float scale_w;
  float scale_h;
  float scale_d;
  int64_t blockIdx;
  uint16_t total_core_num;
  bool widthZoom;
  bool heightZoom;
  bool depthZoom;
  uint64_t temp_result_w_size;
  uint64_t temp_result_h_size;
  uint64_t ratio_metrix_size;
  int32_t align_corners;
  int64_t output_w;
  int64_t output_h;
  int64_t output_d;
  int64_t batches;
  int64_t input_w;
  int64_t input_h;
  int64_t input_d;
  int32_t slide_size;
  uint32_t slide_num_w;
  uint32_t slide_num_h;
  uint32_t slide_num_d;
  int64_t slide_start_indx_w;
  int64_t slide_end_indx_w;
  int64_t tail_group_block_start_indx_w;
  int64_t tail_group_block_end_indx_w;
  int64_t tail_group_slide_start_inx_w;
  int64_t tail_group_slide_end_inx_w;
  int64_t slide_start_indx_h;
  int64_t slide_end_indx_h;
  int64_t tail_group_block_start_indx_h;
  int64_t tail_group_block_end_indx_h;
  int64_t tail_group_batch_start_inx_h;
  int64_t tail_group_batch_end_inx_h;
  int64_t slide_start_indx_d;
  int64_t slide_end_indx_d;
  int64_t tail_group_block_start_indx_d;
  int64_t tail_group_block_end_indx_d;
  int64_t tail_group_batch_start_inx_d;
  int64_t tail_group_batch_end_inx_d;

  uint64_t workspace_offset;
};

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                        const UpsampleTrilinearTilingData* tilingData) {
  input_gm.SetGlobalBuffer((__gm__ T*)input);
  output_gm.SetGlobalBuffer((__gm__ T*)output);
  intermediate_gm.SetGlobalBuffer((__gm__ T*)workspace);

  // parse tilingdata
  blockIdx = GetBlockIdx() / 2;
  scale_w = tilingData->scale_w;
  scale_h = tilingData->scale_h;
  scale_d = tilingData->scale_d;
  output_w = tilingData->output_w;
  output_h = tilingData->output_h;
  output_d = tilingData->output_d;
  input_w = tilingData->input_w;
  input_h = tilingData->input_h;
  input_d = tilingData->input_d;
  batches = tilingData->batches;
  align_corners = tilingData->align_corners;
  total_core_num = tilingData->total_core_num;

  widthZoom = FloatEqual(scale_w, 1.0);
  heightZoom = FloatEqual(scale_h, 1.0);
  depthZoom = FloatEqual(scale_d, 1.0);
  slide_size = SLIDE_SIZE;

  matmul_tiling_w = &tilingData->matmul_tiling_w;
  matmul_tiling_h = &tilingData->matmul_tiling_h;
  matmul_tiling_d = &tilingData->matmul_tiling_d;

  temp_result_w_size = widthZoom ? 0 : batches * input_d * input_h * output_w;
  temp_result_h_size = heightZoom ? 0 : batches * input_d * output_h * output_w;
  ratio_metrix_size = tilingData->ratio_metrix_size;

  slide_num_w = output_w / (slide_size * total_core_num);
  if (slide_num_w > 0) {
    slide_start_indx_w = blockIdx * slide_num_w * slide_size;
    slide_end_indx_w = slide_start_indx_w + slide_num_w * slide_size - 1;
  } else {
    slide_start_indx_w = 0;
    slide_end_indx_w = 0;
  }
  slide_num_h = output_h / (slide_size * total_core_num);
  if (slide_num_h > 0) {
    slide_start_indx_h = blockIdx * slide_num_h * slide_size;
    slide_end_indx_h = slide_start_indx_h + slide_num_h * slide_size - 1;
  } else {
    slide_start_indx_h = 0;
    slide_end_indx_h = 0;
  }
  slide_num_d = output_d / (slide_size * total_core_num);
  if (slide_num_d > 0) {
    slide_start_indx_d = blockIdx * slide_num_d * slide_size;
    slide_end_indx_d = slide_start_indx_d + slide_num_d * slide_size - 1;
  } else {
    slide_start_indx_d = 0;
    slide_end_indx_d = 0;
  }

  // 按照滑块分组后，每个核在h方向上面分到的起始和结束位置
  tail_group_block_start_indx_w = tilingData->tail_group_start_inx_w_list[blockIdx];
  tail_group_block_end_indx_w = tilingData->tail_group_end_inx_w_list[blockIdx];
  tail_group_slide_start_inx_w = tilingData->tail_group_slide_start_inx_w_list[blockIdx];
  tail_group_slide_end_inx_w = tilingData->tail_group_slide_end_inx_w_list[blockIdx];

  tail_group_block_start_indx_h = tilingData->tail_group_start_inx_h_list[blockIdx];
  tail_group_block_end_indx_h = tilingData->tail_group_end_inx_h_list[blockIdx];
  tail_group_batch_start_inx_h = tilingData->tail_group_batch_start_inx_h_list[blockIdx];
  tail_group_batch_end_inx_h = tilingData->tail_group_batch_end_inx_h_list[blockIdx];

  tail_group_block_start_indx_d = tilingData->tail_group_start_inx_d_list[blockIdx];
  tail_group_block_end_indx_d = tilingData->tail_group_end_inx_d_list[blockIdx];
  tail_group_batch_start_inx_d = tilingData->tail_group_batch_start_inx_d_list[blockIdx];
  tail_group_batch_end_inx_d = tilingData->tail_group_batch_end_inx_d_list[blockIdx];

  pipe.InitBuffer(omega_metrix_que, BUFFER_NUM, (ratio_metrix_size * sizeof(float) + 31) / 32 * 32);
  pipe.InitBuffer(omega_metrix_cast_que, BUFFER_NUM, (ratio_metrix_size * sizeof(T) + 31) / 32 * 32);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::Process() {
  if (GetSubBlockIdx() == 1) {
    SyncAll();
    SyncAll();
    return;
  }
  // 横向缩放
  if (!widthZoom) {
    CalcInputOffsetW();
  }
  SyncAll();
  // 纵向缩放
  if (!heightZoom) {
    CalcInputOffsetH();
  }
  SyncAll();
  // 深度缩放
  if (!depthZoom || (widthZoom && heightZoom && depthZoom)) {
    CalcInputOffsetD();
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcInputOffsetD() {
  for (size_t i = 0; i < slide_num_d + 1; i++) {
    int32_t omega_metrix_w, omega_metrix_h;
    int32_t input_start_idx, input_end_idx, batch_start, batch_end;
    int32_t output_start_idx, output_end_idx;
    batch_start = 0;
    batch_end = batches;
    if (i == slide_num_d) {
      // 处理尾块
      batch_start = tail_group_batch_start_inx_d;
      batch_end = tail_group_batch_end_inx_d;
      if (batch_start >= batch_end && output_d != 1) {
        continue;
      }
      output_start_idx = tail_group_block_start_indx_d;
      output_end_idx = tail_group_block_end_indx_d;
      input_start_idx = static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_d, output_start_idx));
      input_end_idx = MIN(static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_d, output_end_idx)) + 1, input_d - 1);
      omega_metrix_h = output_end_idx - output_start_idx + 1;
    } else {
      output_start_idx = slide_start_indx_d + slide_size * i;
      output_end_idx = output_start_idx + slide_size - 1;
      input_start_idx = static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_d, output_start_idx));
      input_end_idx = MIN(static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_d, output_end_idx)) + 1, input_d - 1);
      omega_metrix_h = slide_size;
    }
    // singleK
    omega_metrix_w = input_end_idx - input_start_idx + 1;
    // calc metrix
    CalcRatioMetrixInLeft(output_start_idx, output_end_idx, omega_metrix_w, omega_metrix_h, scale_d, input_start_idx,
                          input_end_idx);
    CopyRatioMetrix2Gm();
    for (size_t j = batch_start; j < batch_end; j++) {
      CalcMatMulInD(input_start_idx, output_start_idx, j, omega_metrix_h, output_w * output_h, omega_metrix_w);
    }
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcInputOffsetH() {
  for (size_t i = 0; i < slide_num_h + 1; i++) {
    int32_t omega_metrix_w, omega_metrix_h;
    int32_t input_start_idx, input_end_idx, batch_start, batch_end;
    int32_t output_start_idx, output_end_idx;
    batch_start = 0;
    batch_end = batches * input_d;
    if (i == slide_num_h) {
      // 处理尾块
      batch_start = tail_group_batch_start_inx_h;
      batch_end = tail_group_batch_end_inx_h;
      if (batch_start >= batch_end && output_h != 1) {
        continue;
      }
      output_start_idx = tail_group_block_start_indx_h;
      output_end_idx = tail_group_block_end_indx_h;
      input_start_idx = static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_h, output_start_idx));
      input_end_idx = MIN(static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_h, output_end_idx)) + 1, input_h - 1);
      omega_metrix_h = output_end_idx - output_start_idx + 1;
    } else {
      output_start_idx = slide_start_indx_h + slide_size * i;
      output_end_idx = output_start_idx + slide_size - 1;
      input_start_idx = static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_h, output_start_idx));
      input_end_idx = MIN(static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_h, output_end_idx)) + 1, input_h - 1);
      omega_metrix_h = slide_size;
    }
    // singleK
    omega_metrix_w = input_end_idx - input_start_idx + 1;
    // calc metrix
    CalcRatioMetrixInLeft(output_start_idx, output_end_idx, omega_metrix_w, omega_metrix_h, scale_h, input_start_idx,
                          input_end_idx);
    CopyRatioMetrix2Gm();
    for (size_t j = batch_start; j < batch_end; j++) {
      CalcMatMulInH(input_start_idx, output_start_idx, j, omega_metrix_h, output_w, omega_metrix_w);
    }
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcInputOffsetW() {
  for (size_t i = 0; i < slide_num_w + 1; i++) {
    int32_t omega_metrix_w, omega_metrix_h;
    int32_t input_start_idx, input_end_idx, row_start, row_end;
    int32_t output_start_idx, output_end_idx;
    row_start = 0;
    row_end = 0;
    if (i == slide_num_w) {
      // tail block process
      output_start_idx = tail_group_slide_start_inx_w;
      output_end_idx = tail_group_slide_end_inx_w;
      if (output_start_idx == 0 && output_start_idx >= output_end_idx && output_w != 1) {
        continue;
      }
      input_start_idx = static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_w, output_start_idx));
      omega_metrix_w = output_end_idx - output_start_idx + 1;
      input_end_idx = MIN(static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_w, output_end_idx)) + 1, input_w - 1);
      row_start = tail_group_block_start_indx_w;
      row_end = tail_group_block_end_indx_w;
    } else {
      output_start_idx = slide_start_indx_w + slide_size * i;
      output_end_idx = output_start_idx + slide_size - 1;
      input_start_idx = static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_w, output_start_idx));
      input_end_idx = MIN(static_cast<int32_t>(AreaPixelComputeSourceIndex(scale_w, output_end_idx)) + 1, input_w - 1);
      omega_metrix_w = slide_size;
    }
    // singleK
    omega_metrix_h = input_end_idx - input_start_idx + 1;
    // calc metrix
    CalcRatioMetrixInRight(output_start_idx, output_end_idx, omega_metrix_w, omega_metrix_h, scale_w, input_start_idx,
                           input_end_idx);
    CopyRatioMetrix2Gm();
    // matmul算完之后，下一个系数矩阵才能往gm上面copy要看下这块后续是否有性能问题
    CalcMatMulInW(input_start_idx, output_start_idx, row_start, row_end, omega_metrix_h, omega_metrix_w);
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcMatMulInD(int32_t input_col_start, int32_t output_col_start,
                                                                 int32_t batch_index, int32_t m, int32_t n, int32_t k) {
  int32_t singleCoreM = m;
  int32_t singleCoreN = n;
  int64_t input_index, output_index;
  input_index = (output_w * output_h * input_d) * batch_index + input_col_start * (output_w * output_h);
  output_index = (output_w * output_h * output_d) * batch_index + output_col_start * (output_w * output_h);

  // m本次范围，n本次范围(B的原始n方向大小)，k原始范围，k本次范围，n原始范围（结果原始n方向大小）
  // SetOrgShape要放最上面
  matmul_d.SetOrgShape(singleCoreM, output_w * output_h, k, input_d, output_w * output_h);
  matmul_d.SetSingleShape(singleCoreM, singleCoreN, k);
  matmul_d.SetTensorA(intermediate_gm[workspace_offset], false);
  if (widthZoom && heightZoom) {
    matmul_d.SetTensorB(input_gm[input_index], false);
  } else if (heightZoom) {
    matmul_d.SetTensorB(intermediate_gm[input_index], false);
  } else {
    matmul_d.SetTensorB(intermediate_gm[temp_result_w_size + input_index], false);
  }
  matmul_d.IterateAll(output_gm[output_index], false);
  matmul_d.End();

  event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
  SetFlag<HardEvent::MTE3_MTE2>(eventID3);
  WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcMatMulInH(int32_t input_col_start, int32_t output_col_start,
                                                                 int32_t batch_index, int32_t m, int32_t n, int32_t k) {
  int32_t singleCoreM = m;
  int32_t singleCoreN = n;
  int64_t input_index, output_index;
  input_index = (output_w * input_h) * batch_index + input_col_start * output_w;
  output_index = (output_w * output_h) * batch_index + output_col_start * output_w;

  matmul_h.SetOrgShape(singleCoreM, output_w, k, input_h, output_w);
  matmul_h.SetSingleShape(singleCoreM, singleCoreN, k);
  matmul_h.SetTensorA(intermediate_gm[workspace_offset], false);
  if (!widthZoom) {
    matmul_h.SetTensorB(intermediate_gm[input_index], false);
  } else {
    matmul_h.SetTensorB(input_gm[input_index], false);
  }
  if (depthZoom) {
    matmul_h.IterateAll(output_gm[output_index], false);
  } else {
    matmul_h.IterateAll(intermediate_gm[temp_result_w_size + output_index], false);
  }
  matmul_h.End();

  event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
  SetFlag<HardEvent::MTE3_MTE2>(eventID3);
  WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcMatMulInW(int32_t input_col_start, int32_t output_col_start,
                                                                 int32_t row_start, int32_t row_end, int32_t k,
                                                                 int32_t n) {
  int32_t singleCoreM = matmul_tiling_w->singleCoreM;
  int32_t singleCoreN = n;
  int64_t input_index, output_index;
  if (row_end != 0) {
    singleCoreM = row_end - row_start + 1;
    input_index = row_start * input_w + input_col_start;
    output_index = row_start * output_w + output_col_start;
  } else {
    input_index = input_col_start;
    output_index = output_col_start;
  }
  matmul_w.SetOrgShape(singleCoreM, singleCoreN, input_w, k, output_w);
  matmul_w.SetSingleShape(singleCoreM, singleCoreN, k);
  matmul_w.SetTensorA(input_gm[input_index], false);
  matmul_w.SetTensorB(intermediate_gm[workspace_offset], false);
  if (heightZoom && depthZoom) {
    matmul_w.IterateAll(output_gm[output_index], false);
  } else {
    matmul_w.IterateAll(intermediate_gm[output_index], false);
  }
  matmul_w.End();

  event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
  SetFlag<HardEvent::MTE3_MTE2>(eventID3);
  WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CopyRatioMetrix2Gm() {
  workspace_offset = temp_result_w_size + temp_result_h_size + ratio_metrix_size * blockIdx;
  if (sizeof(T) == 2) {
    LocalTensor<T> omega_metrix_cast_lc = omega_metrix_cast_que.DeQue<T>();
    DataCopy(intermediate_gm[workspace_offset], omega_metrix_cast_lc, omega_metrix_cast_lc.GetSize());
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    omega_metrix_cast_que.FreeTensor(omega_metrix_cast_lc);
  } else {
    LocalTensor<T> omega_metrix_lc = omega_metrix_que.DeQue<T>();
    DataCopy(intermediate_gm[workspace_offset], omega_metrix_lc, omega_metrix_lc.GetSize());
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    omega_metrix_que.FreeTensor(omega_metrix_lc);
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcRatioMetrixInLeft(
    const int32_t output_start_indx, const int32_t output_end_indx, const int32_t omega_metrix_w,
    const int32_t omega_metrix_h, float scale, const int32_t input_start_indx, const int32_t input_end_indx) {
  LocalTensor<float> omega_metrix_lc = omega_metrix_que.AllocTensor<float>();
  Duplicate(omega_metrix_lc, (float)0., omega_metrix_lc.GetSize());
  float real_index;
  int32_t output_index = output_start_indx;
  for (size_t i = 0; i < omega_metrix_h; i++) {
    real_index = AreaPixelComputeSourceIndex(scale, output_index);
    int32_t input_idx_0 = MIN(static_cast<int32_t>(real_index), input_end_indx);
    float lambda0 = MIN(MAX(real_index - (float)input_idx_0, static_cast<float>(0.0)), static_cast<float>(1.0));
    int32_t offset = (input_idx_0 < input_end_indx) ? 1 : 0;
    int32_t input_idx_1 = input_idx_0 + offset;
    float lambda1 = static_cast<float>(1.) - lambda0;
    omega_metrix_lc.SetValue((input_idx_0 - input_start_indx) + omega_metrix_w * i, lambda1);
    omega_metrix_lc.SetValue((input_idx_1 - input_start_indx) + omega_metrix_w * i, lambda0);
    if (input_idx_0 == input_idx_1) {
      omega_metrix_lc.SetValue((input_idx_1 - input_start_indx) + omega_metrix_w * i, static_cast<float>(1.0));
    }
    output_index++;
  }
  if (sizeof(T) == 2) {
    LocalTensor<T> omega_metrix_cast_lc = omega_metrix_cast_que.AllocTensor<T>();
    Cast(omega_metrix_cast_lc, omega_metrix_lc, RoundMode::CAST_RINT, omega_metrix_lc.GetSize());
    omega_metrix_cast_que.EnQue(omega_metrix_cast_lc);
    omega_metrix_que.FreeTensor(omega_metrix_lc);
  } else {
    omega_metrix_que.EnQue(omega_metrix_lc);
  }
}

template <typename T>
__aicore__ inline void KernelUpsampleTrilinear<T>::CalcRatioMetrixInRight(
    const int32_t output_start_indx, const int32_t output_end_indx, const int32_t omega_metrix_w,
    const int32_t omega_metrix_h, float scale, const int32_t input_start_indx, const int32_t input_end_indx) {
  LocalTensor<float> omega_metrix_lc = omega_metrix_que.AllocTensor<float>();
  Duplicate(omega_metrix_lc, (float)0., omega_metrix_lc.GetSize());
  float real_index;
  int32_t output_index = output_start_indx;
  for (size_t i = 0; i < omega_metrix_w; i++) {
    real_index = AreaPixelComputeSourceIndex(scale, output_index);
    int32_t input_idx_0 = MIN(static_cast<int32_t>(real_index), input_end_indx);
    float lambda0 = MIN(MAX(real_index - (float)input_idx_0, static_cast<float>(0.0)), static_cast<float>(1.0));
    int32_t offset = (input_idx_0 < input_end_indx) ? 1 : 0;
    int32_t input_idx_1 = input_idx_0 + offset;
    float lambda1 = static_cast<float>(1.) - lambda0;
    omega_metrix_lc.SetValue((input_idx_0 - input_start_indx) * omega_metrix_w + i, lambda1);
    omega_metrix_lc.SetValue((input_idx_1 - input_start_indx) * omega_metrix_w + i, lambda0);
    if (input_idx_0 == input_idx_1) {
      omega_metrix_lc.SetValue((input_idx_1 - input_start_indx) * omega_metrix_w + i, static_cast<float>(1.0));
    }
    output_index++;
  }
  if (sizeof(T) == 2) {
    LocalTensor<T> omega_metrix_cast = omega_metrix_cast_que.AllocTensor<T>();
    Cast(omega_metrix_cast, omega_metrix_lc, RoundMode::CAST_RINT, omega_metrix_lc.GetSize());
    omega_metrix_cast_que.EnQue(omega_metrix_cast);
    omega_metrix_que.FreeTensor(omega_metrix_lc);
  } else {
    omega_metrix_que.EnQue(omega_metrix_lc);
  }
}

template <typename T>
__aicore__ inline float KernelUpsampleTrilinear<T>::AreaPixelComputeSourceIndex(float scale, int32_t dst_index) {
  // calc coordinate range with group
  float result;
  if (align_corners == 1) {
    result = scale * (float)dst_index;
  } else {
    auto zero = static_cast<float>(0.);
    float src_idx = static_cast<float>(scale * ((float)dst_index + (float)0.5) - (float)0.5);
    result = (src_idx < zero) ? float(0.) : src_idx;
  }
  return result;
}

}  // namespace UpsampleTrilinearNs

#endif  // RESIZE_UPSAMPLE_TRILINEAR_H
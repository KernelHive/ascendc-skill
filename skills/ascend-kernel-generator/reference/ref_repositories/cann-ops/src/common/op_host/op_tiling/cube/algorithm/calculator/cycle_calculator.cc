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
 * \file cycle_calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/cycle_calculator.h"

#include <algorithm>

#include "cube/util/cube_util.h"
#include "cube/util/math_util.h"

namespace optiling {
namespace cachetiling {
CycleCalculator::CycleCalculator(SingleCoreStatus &core_status)
    : Calculator(core_status), orig_shape_(core_status.orig_shape()) {}

bool CycleCalculator::Init(const CubeTilingParam &params) {
  (void)Calculator::Init(params);
  shape_.Init();
  block_dims_.Init();
  l1_status_.Init();
  l0_status_.Init();
  mapped_shape_.Init();
  core_num_ = params_->platform_info.core_num();
  int32_t vec_size = params_->linear_embedding_opti_flag ? KBatchBindCoreLimit : core_num_;
  batch_dims_factors_.clear();
  m_dims_factors_.clear();
  n_dims_factors_.clear();
  g_dims_factors_.clear();
  batch_dims_factors_.reserve(static_cast<size_t>(vec_size));
  m_dims_factors_.reserve(static_cast<size_t>(vec_size));
  n_dims_factors_.reserve(static_cast<size_t>(vec_size));
  g_dims_factors_.reserve(static_cast<size_t>(vec_size));
  shape_m_vec_.resize(vec_size, 0);
  std::fill(shape_m_vec_.begin(), shape_m_vec_.end(), 0);
  shape_n_vec_.resize(vec_size, 0);
  std::fill(shape_n_vec_.begin(), shape_n_vec_.end(), 0);
  tiling_pattern_flag_ = false;
  // min kl0：FP16（1），FP32（2）
  kl0_min_size_ = params_->Fp32Input() ? 2 : 1;
  kl0_offset_ = params_->Fp32Input() ? 1 : 0;
  final_cycle_ = INT64_MAX;
  load2d_mode_ = params_->load2d_flag;
  load3d_mode_ = params_->load3d_flag;

  return true;
}

bool CycleCalculator::CheckL1Size() const {
  return IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
}

bool CycleCalculator::CheckAl1FullLoad() {
  // The w-split scene can not be full loaded,
  // If it can be fully loaded, then a non-w split load3d is available
  // dma scene, only linear_embedding_opti_flag scene has AL1 full-load template, reduce the number of templates
  if (params_->groups * params_->a_shape.batch > core_num_ || params_->split_w_flag ||
      (!params_->linear_embedding_opti_flag && params_->dma_flag)) {
    return false;
  }

  if (orig_shape_.m > INT32_MAX || orig_shape_.k > INT32_MAX) {
    return false;
  }

  shape_.Init();
  l1_status_.Init();
  l0_status_.Init();

  shape_.group = 1;
  shape_.batch = 1;
  // all cores are acting on the m-axis, check if it can be fully loaded under extreme conditions
  shape_.m = MathUtil::CeilDivision(static_cast<int64_t>(orig_shape_.m),
                                    core_num_ / (params_->groups * params_->a_shape.batch));
  shape_.k = orig_shape_.k;
  shape_.n = orig_shape_.n;

  l0_status_.k = kl0_min_size_;
  l1_status_.m_al1 = shape_.m;
  l1_status_.k_al1 = shape_.k;
  l1_status_.k_bl1 = kl0_min_size_;
  l1_status_.n_bl1 = 1;

  return CheckL1Size();
}

bool CycleCalculator::CheckBl1FullLoad() {
  // The w-split scene can not be full loaded,
  // If it can be fully loaded, then a non-w split load3d is available
  // DMA, do not consider the full load, reduce the number of templates
  // If real_g > 1, then kd can only be 1 for conv3d dw
  if (params_->groups * params_->a_shape.batch > core_num_ || params_->split_w_flag || total_n_l1_ != shape_.n ||
      params_->dma_flag) {
    return false;
  }

  if (orig_shape_.n > INT32_MAX || orig_shape_.k > INT32_MAX) {
    return false;
  }

  shape_.Init();
  l1_status_.Init();
  l0_status_.Init();

  shape_.group = 1;
  shape_.batch = 1;
  shape_.m = orig_shape_.m;
  shape_.k = orig_shape_.k;
  shape_.n = MathUtil::CeilDivision(static_cast<int64_t>(orig_shape_.n),
                                    core_num_ / (params_->groups * params_->a_shape.batch));

  l0_status_.k = kl0_min_size_;
  l1_status_.m_al1 = 1;
  l1_status_.k_al1 = kl0_min_size_;
  l1_status_.k_bl1 = shape_.k;
  l1_status_.n_bl1 = shape_.n;

  return CheckL1Size();
}

int64_t CycleCalculator::GetFixpCycle() const {
  int64_t l0c_bytes = l0_status_.m * l0_status_.n * kFractalSize * kFp32Bytes;
  int64_t cycle_l0c_2_fixp = l0c_bytes / k1971L0CBandWidth;
  int64_t cycle_fixp_2_out = l0c_bytes / k1971FixpBandWidth;
  return cycle_l0c_2_fixp + cycle_fixp_2_out;
}

int64_t CycleCalculator::GetMadCycle() const {
  int64_t mad_cycle = l0_status_.m * l0_status_.k * l0_status_.n;
  if (params_->Fp32Input() && params_->hf32_flag == 0) {
    mad_cycle = mad_cycle << 1;
  }
  return mad_cycle + k1971MadCost;
}

int64_t CycleCalculator::GetMte1ACycle() const {
  int64_t cycle = 0;
  if (params_->Fp32Input()) {
    // fp32 use load3d
    int64_t actual_l0a_band = CubeUtil::GetLoad3dThroughput(true, 0, 1);
    cycle = k1971Mte1Load3dCost + l0_status_.k * l0_status_.m * kFractalSize * kFp16Bytes / actual_l0a_band;
  } else {
    int64_t read_cycle = l0_status_.k * kFractalSize * kFp16Bytes / k1971L1ReadBandwidth;
    int64_t write_cycle = l0_status_.m * l0_status_.k * kFractalSize * kFp16Bytes / k1971Mte1L0ABandWidth;
    cycle = k1971Mte1Load2dCost + read_cycle + write_cycle;
  }

  return cycle;
}

int64_t CycleCalculator::GetMte1BCycle() const {
  int64_t cycle = 0;
  if (load2d_mode_) {
    int64_t read_cycle = l0_status_.n * kFractalSize * kFp16Bytes / k1971L1ReadBandwidth;
    int64_t write_cycle = l0_status_.k * l0_status_.n * kFractalSize * kFp16Bytes / k1971Mte1L0BBandWidth;
    cycle = k1971Mte1Load2dCost + read_cycle + write_cycle;
  } else if (params_->dma_flag) {
    int64_t actual_l0b_band = CubeUtil::GetLoad3dThroughput(false, 0, 1);
    cycle = k1971Mte1Load3dCost + l0_status_.k * l0_status_.n * kFractalSize * kFp16Bytes / actual_l0b_band;
  } else {
    int32_t row_switch_times = kBlockSize / params_->b_shape.w;
    if (kBlockSize % params_->a_shape.w == 0) {
      row_switch_times = (kBlockSize - 1) / params_->b_shape.w;
    }
    int64_t actual_l0b_band = CubeUtil::GetLoad3dThroughput(false, row_switch_times, params_->stride_w);
    cycle = k1971Mte1Load3dCost + l0_status_.k * l0_status_.n * kFractalSize * kFp16Bytes / actual_l0b_band;
  }

  return cycle;
}

int64_t CycleCalculator::GetMte2ACycle() const {
  int64_t burst_len = l1_status_.k_al1 * kBlockSize;
  int64_t num_burst = l1_status_.m_al1 * l0_status_.m;
  return k1971Mte2Latency + num_burst * burst_len * kBlockSize * kFp16Bytes / k1971HbmBandwidth;
}

int64_t CycleCalculator::GetMte2BCycle() const {
  int64_t cycle = 0;
  if (load2d_mode_) {
    int64_t burst_len = l1_status_.k_bl1 * kBlockSize;
    int64_t num_burst = l1_status_.n_bl1 * l0_status_.n;
    cycle = k1971Mte2Latency + num_burst * burst_len * kBlockSize * kFp16Bytes / k1971HbmBandwidth;
  } else if (params_->dma_flag) {
    int64_t num_insn = l1_status_.k_bl1 * kBlockSize * l1_status_.n_bl1 * l0_status_.n;
    cycle = k1971Mte2Latency + num_insn * kBlockSize * kBlockSize * kFp16Bytes / k1971HbmBandwidth;
  } else {
    int32_t ho = CubeUtil::CalcHo(l1_status_.k_bl1 * params_->k0, params_->a_shape.w, params_->op_type);
    int32_t hi = CubeUtil::CalcHi(ho, params_->stride_h, params_->kernel_h_dilation, params_->b_shape.h);
    int32_t c1 = MathUtil::CeilDivision(l1_status_.n_bl1 * l0_status_.n, params_->kernel_h * params_->kernel_w);
    int32_t num_burst = std::max(c1, hi);
    int32_t num_insn = std::min(c1, hi);
    int32_t wo = params_->split_w_flag ? l0_status_.k * params_->k0 : params_->a_shape.w;
    int32_t burstLen = CubeUtil::CalcWi(wo, params_->stride_w, params_->kernel_w_dilation, params_->b_shape.w);

    cycle = k1971Mte2Latency + num_insn * num_burst * CubeUtil::GetMilanPkgNumByCacheline(burstLen) * kFractalSize *
                                   kFp16Bytes / k1971HbmBandwidth;
  }
  return cycle;
}

void CycleCalculator::GetAttachFlag(const TilingShape &status) {
  auto attach_l1_helper_func = [&status](int32_t k_l1, bool is_full_load_non_k_axis, bool is_split_w) -> int32_t {
    const TilingShape &shape = status;
    bool k_full_load = shape.batch == 1 && k_l1 == shape.k && !is_split_w;
    bool full_load = (shape.batch == 1 && shape.group == 1) && k_full_load && is_full_load_non_k_axis;
    if (full_load) {
      return kFullLoadFlag;
    } else if (k_full_load) {
      return kKFullLoadFlag;
    } else {
      return kKPartLoadFlag;
    }
  };
  int32_t al1_attach_flag = attach_l1_helper_func(l1_status_.k_al1, (l1_status_.m_al1 * l0_status_.m == shape_.m),
                                                  params_->split_w_flag);
  int32_t bl1_attach_flag = attach_l1_helper_func(l1_status_.k_bl1, (l1_status_.n_bl1 * l0_status_.n == shape_.n),
                                                  params_->split_w_flag);
  int32_t abkl1_attach_flag = l1_status_.k_al1 == l1_status_.k_bl1 ? 0 : l1_status_.k_al1 > l1_status_.k_bl1 ? 1 : 2;
  // Scenario linear_embedding_opti_flag, which can enable A to fully load
  // Scenario dma_flag and not linear_embedding_opti_flag, the full load template is not enabled
  if (params_->linear_embedding_opti_flag) {
    bl1_attach_flag = kKPartLoadFlag;
  } else if (params_->dma_flag) {
    al1_attach_flag = kKPartLoadFlag;
    bl1_attach_flag = kKPartLoadFlag;
  }
  attach_flags_ = AttachFlags(abkl1_attach_flag, al1_attach_flag, bl1_attach_flag);
  min_kl1_cmp_kl0_ = std::min(l1_status_.k_al1, l1_status_.k_bl1) == l0_status_.k ? 0 : 1;
  al1_full_load_ = al1_attach_flag == kFullLoadFlag;
  bl1_full_load_ = bl1_attach_flag == kFullLoadFlag;
}

void CycleCalculator::SetDoubleBuffer() {
  if (static_cast<uint64_t>(l0_status_.m * l0_status_.n * kCubeTileNumSize * params_->c_dtype_bytes * kDbOn) <=
      params_->platform_info.l0c_size()) {
    l0_status_.db_l0c = kDbOn;
  }

  int64_t al1_size = CubeUtil::CalcAL1Size(params_, l1_status_, l0_status_);
  int64_t bl1_size = CubeUtil::CalcBL1Size(params_, l1_status_, l0_status_);
  if (!al1_full_load_) {
    l1_status_.db_al1 = kDbOn;
    l1_status_.db_al1 = IsL1SizeValid(kDbOn * al1_size + bl1_size) ? kDbOn : kDbOff;
    al1_size *= l1_status_.db_al1;
  }

  if (!bl1_full_load_) {
    l1_status_.db_bl1 = kDbOn;
    l1_status_.db_bl1 = IsL1SizeValid(al1_size + kDbOn * bl1_size) ? kDbOn : kDbOff;
  }
}

void CycleCalculator::CalCycleBothFullLoad(int64_t &cycle) {
  int64_t db_cycle = 0;
  int32_t al1_attach_flag, bl1_attach_flag;
  std::tie(std::ignore, al1_attach_flag, bl1_attach_flag) = attach_flags_;
  std::tuple<int32_t, int32_t> relate_flags(al1_attach_flag, bl1_attach_flag);
  int64_t mte1_a_db_cycle = std::max(cycle_.mte1_a, shape_.batch * GetMadCycle());  // L0A enable db by default
  int64_t core_cycle = min_kl1_cmp_kl0_ == 0
              ? l1_status_.n_bl1 * (cycle_.mte1_b + l1_status_.m_al1 * (mte1_a_db_cycle + cycle_.fixp))
              : l1_status_.n_bl1 * l1_status_.m_al1 * (shape_.batch * tiling_.k_al0_factor * cycle_.mad + cycle_.fixp);

  if (relate_flags == std::make_tuple(kFullLoadFlag, kFullLoadFlag)) { // both AL1 and BL1 are fullload
    cycle = cycle_.mte2_a + cycle_.mte2_b + shape_.group * core_cycle;
  } else if (relate_flags == std::make_tuple(kFullLoadFlag, kKFullLoadFlag)) { // AL1 is fullload, BL1'K is fullload
    db_cycle = l1_status_.db_bl1 == kDbOn ? std::max(cycle_.mte2_b, core_cycle) : cycle_.mte2_b + core_cycle;
    cycle = cycle_.mte2_a + shape_.group * tiling_.n_single_core * db_cycle;
  } else if (relate_flags == std::make_tuple(kKFullLoadFlag, kFullLoadFlag)) { // AL1'K is fullload, BL1 is fullload
    db_cycle = l1_status_.db_al1 == kDbOn ? std::max(cycle_.mte2_a, core_cycle) : cycle_.mte2_a + core_cycle;
    cycle = cycle_.mte2_b + shape_.group * tiling_.m_single_core * db_cycle;
  } else if (relate_flags == std::make_tuple(kKFullLoadFlag, kKFullLoadFlag)) { // both AL1'K and BL1'K are fullload
    db_cycle = l1_status_.db_al1 == kDbOn ? std::max(cycle_.mte2_a, core_cycle) : cycle_.mte2_a + core_cycle;
    cycle = shape_.group * tiling_.n_single_core * (cycle_.mte2_b + tiling_.m_single_core * db_cycle);
  }
}

void CycleCalculator::CalCycleBL1FullLoad(int64_t &cycle) {
  int32_t al1_attach_flag, bl1_attach_flag;
  std::tie(std::ignore, al1_attach_flag, bl1_attach_flag) = attach_flags_;

  if (cycle != 0 || al1_attach_flag != kKPartLoadFlag) {
    return;
  }
  int64_t db_cycle = l1_status_.db_al1 == kDbOn ? std::max(cycle_.mte2_a, tiling_.k_al0_factor * cycle_.mad)
                                                : cycle_.mte2_a + tiling_.k_al0_factor * cycle_.mad;
  if (bl1_attach_flag == kKFullLoadFlag) {
    cycle = shape_.group * tiling_.n_single_core *
            (cycle_.mte2_b + tiling_.m_single_core * l1_status_.n_bl1 *
                                 (shape_.batch * tiling_.kl1_times * db_cycle + cycle_.fixp));
  } else if (bl1_attach_flag == kFullLoadFlag) {
    cycle = cycle_.mte2_b + shape_.group * tiling_.m_single_core * l1_status_.n_bl1 *
                                (shape_.batch * tiling_.kl1_times * db_cycle + cycle_.fixp);
  }
}

void CycleCalculator::CalCycleAL1FullLoad(int64_t &cycle) {
  int32_t al1_attach_flag, bl1_attach_flag;
  std::tie(std::ignore, al1_attach_flag, bl1_attach_flag) = attach_flags_;

  if (cycle != 0 || bl1_attach_flag != kKPartLoadFlag) {
    return;
  }
  int64_t db_cycle = l1_status_.db_bl1 == kDbOn ? std::max(cycle_.mte2_b, tiling_.k_bl0_factor * cycle_.mad)
                                                : cycle_.mte2_b + tiling_.k_bl0_factor * cycle_.mad;
  if (al1_attach_flag == kKFullLoadFlag) {
    cycle = shape_.group * tiling_.n_single_core * tiling_.m_single_core *
            (cycle_.mte2_a + l1_status_.m_al1 * (shape_.batch * tiling_.kl1_times * db_cycle + cycle_.fixp));
  } else if (al1_attach_flag == kFullLoadFlag) {
    cycle = cycle_.mte2_a + shape_.group * tiling_.n_single_core * l1_status_.m_al1 *
                                (shape_.batch * tiling_.kl1_times * db_cycle + cycle_.fixp);
  }
}

void CycleCalculator::CalCycleNeitherFullLoad(int64_t &cycle) {
  int64_t db_cycle = 0;
  int32_t abkl1_attach_flag, al1_attach_flag, bl1_attach_flag;
  std::tie(abkl1_attach_flag, al1_attach_flag, bl1_attach_flag) = attach_flags_;
  bool neither_full_load_flag = al1_attach_flag == kKPartLoadFlag && bl1_attach_flag == kKPartLoadFlag;
  if (cycle != 0 || !neither_full_load_flag) {
    return;
  }

  const int32_t kMaxLoad2dInsn = 60;
  // the number of load2d instructions is equal to ml0, when it reaches kMaxLoad2dInsn, L1 will not enable DB
  int32_t load2d_insn = tiling_.min_kl1_div_kl0 * l0_status_.m;
  int32_t wo_single_core =
      params_->split_w_flag
          ? MathUtil::CeilDivision(params_->a_shape.w, static_cast<int64_t>(l0_status_.k) * params_->k0)
          : 1;
  if (abkl1_attach_flag == kKal1EqualKbl1Flag) {
    db_cycle = l1_status_.db_al1 == kDbOn && l1_status_.db_bl1 == kDbOn && load2d_insn < kMaxLoad2dInsn
                   ? std::max(cycle_.mte2_a + cycle_.mte2_b, tiling_.k_al0_factor * cycle_.mad)
                   : cycle_.mte2_a + cycle_.mte2_b + tiling_.k_al0_factor * cycle_.mad;
    cycle = shape_.group * tiling_.n_single_core * tiling_.m_single_core *
            (shape_.batch * wo_single_core * tiling_.k_al1_factor * db_cycle + cycle_.fixp);
  } else if (abkl1_attach_flag == kKal1LargerThanKbl1Flag) {
    db_cycle = l1_status_.db_bl1 == kDbOn && load2d_insn < kMaxLoad2dInsn
                   ? std::max(cycle_.mte2_b, tiling_.k_bl0_factor * cycle_.mad)
                   : cycle_.mte2_b + tiling_.k_bl0_factor * cycle_.mad;
    int64_t mte2_a_db_cycle = l1_status_.db_al1 == kDbOn && load2d_insn * tiling_.kl1_times < kMaxLoad2dInsn
                                  ? std::max(cycle_.mte2_a, tiling_.kl1_times * db_cycle)
                                  : cycle_.mte2_a + tiling_.kl1_times * db_cycle;
    cycle = shape_.group * tiling_.n_single_core * tiling_.m_single_core *
            (shape_.batch * wo_single_core * tiling_.k_al1_factor * mte2_a_db_cycle + cycle_.fixp);
  } else if (abkl1_attach_flag == kKal1SmallerThanKbl1Flag) {
    db_cycle = l1_status_.db_al1 == kDbOn && load2d_insn < kMaxLoad2dInsn
                   ? std::max(cycle_.mte2_a, tiling_.k_al0_factor * cycle_.mad)
                   : cycle_.mte2_a + tiling_.k_al0_factor * cycle_.mad;
    int64_t mte2_b_db_cycle = l1_status_.db_bl1 == kDbOn && load2d_insn * tiling_.kl1_times < kMaxLoad2dInsn
                                  ? std::max(cycle_.mte2_b, tiling_.kl1_times * db_cycle)
                                  : cycle_.mte2_b + tiling_.kl1_times * db_cycle;
    cycle = shape_.group * tiling_.n_single_core * tiling_.m_single_core *
            (shape_.batch * wo_single_core * tiling_.k_bl1_factor * mte2_b_db_cycle + cycle_.fixp);
  }
}

int64_t CycleCalculator::GetCycleByModel() {
  int64_t cycle = 0;
  cycle_.mte1_a = GetMte1ACycle();
  cycle_.mte1_b = GetMte1BCycle();
  cycle_.mte2_a = GetMte2ACycle();
  cycle_.mte2_b = GetMte2BCycle();
  cycle_.mad = std::max(GetMadCycle(), cycle_.mte1_a + cycle_.mte1_b);
  cycle_.fixp = GetFixpCycle();

  int64_t single_core_k = MathUtil::CeilDivision(
      MathUtil::Align(params_->a_shape.h * params_->a_shape.w, params_->b_shape.c0) / params_->k0, block_dims_.k);
  if (params_->split_w_flag) {
    // In split-w-load3d, multi-core split h, L0 split w
    // single_core_k = h_single_core * w_l0
    single_core_k = MathUtil::CeilDivision(params_->a_shape.h, static_cast<int64_t>(block_dims_.k)) * l0_status_.k;
  }
  tiling_.m_single_core =
      MathUtil::CeilDivision(MathUtil::CeilDivision(params_->a_shape.c1, static_cast<int64_t>(block_dims_.m)),
                             static_cast<int64_t>(l1_status_.m_al1) * l0_status_.m);
  int32_t cin1_g = params_->b_shape.c1;
  if (params_->Fp32Input()) {
    cin1_g =
        MathUtil::Align(params_->mag_factor * params_->b_shape.c / params_->groups, params_->b_shape.c0) / params_->k0;
  }

  int64_t kd_split_num = std::max(MathUtil::GetGcd(params_->c_shape.d, block_dims_.n), 1);
  tiling_.n_single_core =
      MathUtil::CeilDivision(MathUtil::CeilDivision(cin1_g, block_dims_.n / kd_split_num) *
                                 (params_->c_shape.d / kd_split_num) * params_->kernel_h * params_->kernel_w,
                             static_cast<int64_t>(l1_status_.n_bl1) * l0_status_.n);
  tiling_.k_al1_factor = single_core_k / l1_status_.k_al1;
  tiling_.k_bl1_factor = single_core_k / l1_status_.k_bl1;
  tiling_.k_al0_factor = l1_status_.k_al1 / l0_status_.k;
  tiling_.k_bl0_factor = l1_status_.k_bl1 / l0_status_.k;
  tiling_.kl1_times = std::max(l1_status_.k_al1, l1_status_.k_bl1) / std::min(l1_status_.k_al1, l1_status_.k_bl1);
  tiling_.min_kl1_div_kl0 = MathUtil::CeilDivision(std::min(l1_status_.k_al1, l1_status_.k_bl1), l0_status_.k);

  GetAttachFlag(shape_);
  SetDoubleBuffer();
  CalCycleBothFullLoad(cycle);
  CalCycleBL1FullLoad(cycle);
  CalCycleAL1FullLoad(cycle);
  CalCycleNeitherFullLoad(cycle);

  return cycle;
}

void CycleCalculator::UpdateSingleCoreStatus() {
  int64_t cur_cycle = GetCycleByModel();
  OP_LOGD(params_->op_type,
          "[Pattern Params] (block_dim: %s), (single_core: %s), (l0_status: %s), (l1_status: %s), (cycle: %ld)",
          block_dims_.ToString().c_str(), shape_.ToString().c_str(), l0_status_.ToString().c_str(),
          l1_status_.ToString().c_str(), cur_cycle);
  if (cur_cycle < final_cycle_) {
    OP_LOGD(params_->op_type, "[Update Success](old_cycle: %ld, new_cycle: %ld)", final_cycle_, cur_cycle);
    final_cycle_ = cur_cycle;
    CubeUtil::CalcL1RemainStatus(params_, l0_status_, l1_status_);
    single_core_status_.UpdateShape(shape_);
    single_core_status_.UpdateBlockDims(block_dims_);
    single_core_status_.UpdateL0Status(l0_status_);
    single_core_status_.UpdateL1Status(l1_status_);
  }
}

bool CycleCalculator::FastFindAl1FullLoad() {
  l0_status_.Init();
  l1_status_.Init();

  l1_status_.m_al1 = shape_.m;
  l1_status_.k_al1 = shape_.k;
  l1_status_.k_bl1 = kl0_min_size_;
  l1_status_.n_bl1 = 1;
  l0_status_.k = kl0_min_size_;
  // CheckAl1FullLoad() returning true indicates that AL1 can be fully loaded under extreme conditions,
  // but it can not be fully loaded in some cases, so need to checkL1 here.
  if (!CheckL1Size()) {
    return true;
  }

  l0_status_.m = (params_->linear_embedding_opti_flag && !params_->Fp32Input())
                     ? MathUtil::Min(kL0aNzSize, shape_.m)
                     : MathUtil::Min(32, shape_.m);  // ml0<=32，ensure that nl0 is smaller than ml0
  while (shape_.m % l0_status_.m != 0) {
    l0_status_.m--;
  }
  l1_status_.m_al1 = MathUtil::CeilDivision(shape_.m, l0_status_.m);
  l0_status_.k = MathUtil::Min(shape_.k, kL0aNzSize / l0_status_.m);
  OP_LOGE_IF(l0_status_.k == 0, false, params_->op_type, "l0_status_.k == 0");
  l1_status_.k_bl1 = l0_status_.k;
  while ((shape_.k % l0_status_.k != 0) || !CheckL1Size() || (params_->Fp32Input() && ((l0_status_.k & 0x1) != 0))) {
    l0_status_.k = MathUtil::NearestFactor(shape_.k, l0_status_.k - 1, params_->Fp32Input());
    OP_LOGE_IF(l0_status_.k == 0, false, params_->op_type, "l0_status_.k == 0");
    l1_status_.k_bl1 = l0_status_.k;
  }
  // nl0 needs to be less than 16
  l0_status_.n = GetNl0(std::min(16, std::min(kL0aNzSize / l0_status_.k, kL0cNzSize / l0_status_.m)));
  CubeUtil::CalcKbl1(params_, shape_.k, l0_status_, l1_status_);
  OP_LOGE_IF(l1_status_.k_bl1 <= 0, false, params_->op_type, "invalid kbl1: %d", l1_status_.k_bl1);
  CubeUtil::CalcNbl1(params_, shape_, l0_status_, total_n_l1_, l1_status_);
  OP_LOGE_IF(l1_status_.n_bl1 <= 0, false, params_->op_type, "invalid nbl1: %d", l1_status_.n_bl1);
  UpdateSingleCoreStatus();
  tiling_pattern_flag_ = true;
  OP_LOGD(params_->op_type, "[Al1 FullLoad Pattern]");
  return true;
}

bool CycleCalculator::FastFindBl1FullLoad() {
  l0_status_.Init();
  l1_status_.Init();

  l1_status_.m_al1 = 1;
  l1_status_.k_al1 = kl0_min_size_;
  l1_status_.k_bl1 = shape_.k;
  l1_status_.n_bl1 = shape_.n;
  l0_status_.k = kl0_min_size_;
  // CheckBl1FullLoad() returning true indicates that BL1 can be fully loaded under extreme conditions,
  // but it can not be fully loaded in some cases, so need to checkL1 here.
  if (!CheckL1Size()) {
    return true;
  }
  // check the limit of load3dv2's k_pos.
  if (load3d_mode_ && shape_.n * params_->kernel_h * params_->kernel_w > kMaxLoad3dV2Kstart) {
    return true;
  }

  // nl0<=32，ensure that ml0 is smaller than nl0
  l0_status_.n = MathUtil::Min(std::min(32, kL0aNzSize / (l0_status_.k + kl0_offset_)), shape_.n);
  while (shape_.n % l0_status_.n != 0) {
    l0_status_.n--;
  }
  l1_status_.n_bl1 = MathUtil::CeilDivision(shape_.n, l0_status_.n);
  l0_status_.k = MathUtil::Min(shape_.k, kL0aNzSize / l0_status_.n);
  OP_LOGE_IF(l0_status_.k == 0, false, params_->op_type, "l0_status_.k == 0");
  l1_status_.k_al1 = l0_status_.k;
  while ((shape_.k % l0_status_.k != 0) || !CheckL1Size() || (params_->Fp32Input() && ((l0_status_.k & 0x1) != 0))) {
    l0_status_.k = MathUtil::NearestFactor(shape_.k, l0_status_.k - 1, params_->Fp32Input());
    OP_LOGE_IF(l0_status_.k == 0, false, params_->op_type, "l0_status_.k == 0");
    l1_status_.k_al1 = l0_status_.k;
  }
  // ml0 needs to be less than 16
  l0_status_.m = GetMl0(std::min(std::min(kL0aNzSize / l0_status_.k, kL0cNzSize / l0_status_.n), 16));
  CubeUtil::CalcKal1(params_, shape_, shape_.k, l0_status_, l1_status_);
  OP_LOGE_IF(l1_status_.k_al1 <= 0, false, params_->op_type, "invalid kal1: %d", l1_status_.k_al1);
  CubeUtil::CalcMal1(params_, shape_, shape_.k, l0_status_, l1_status_);
  OP_LOGE_IF(l1_status_.m_al1 <= 0, false, params_->op_type, "invalid mal1: %d", l1_status_.m_al1);
  UpdateSingleCoreStatus();
  tiling_pattern_flag_ = true;
  OP_LOGD(params_->op_type, "[Bl1 FullLoad Pattern]");
  return true;
}

int32_t CycleCalculator::GetNl0(int32_t max_nl0) {
  l0_status_.n = MathUtil::Min(max_nl0, total_n_l0_);
  int64_t al1_size = CubeUtil::CalcAL1Size(params_, l1_status_, l0_status_);
  while (l0_status_.n != 0 && (total_n_l0_ % l0_status_.n != 0 ||
                               !IsL1SizeValid(al1_size + CubeUtil::CalcBL1Size(params_, l1_status_, l0_status_)))) {
    l0_status_.n--;
  }

  return l0_status_.n;
}

int32_t CycleCalculator::GetMl0(int32_t max_ml0) {
  l0_status_.m = MathUtil::Min(max_ml0, shape_.m);
  int64_t bl1_size = CubeUtil::CalcBL1Size(params_, l1_status_, l0_status_);
  while (l0_status_.m != 0 && (shape_.m % l0_status_.m != 0 ||
                               !IsL1SizeValid(CubeUtil::CalcAL1Size(params_, l1_status_, l0_status_) + bl1_size))) {
    l0_status_.m--;
  }

  return l0_status_.m;
}

bool CycleCalculator::GetMN0() {
  L0Status l0_status = l0_status_;
  // we need to limit nl0 and ml0 to 8~16 to ensure that mad will not be interrupted by mte1
  int32_t nl0_1 = GetNl0(16);
  OP_LOGE_IF(nl0_1 <= 0, false, params_->op_type, "nl0_1(%d) <= 0", nl0_1);
  int32_t ml0_1 = GetMl0(std::min(kL0cNzSize / nl0_1, kL0aNzSize / kl0_min_size_));
  OP_LOGE_IF(ml0_1 <= 0, false, params_->op_type, "ml0_1(%d) <= 0", ml0_1);

  l0_status_ = l0_status;
  int32_t ml0_2 = GetMl0(16);
  OP_LOGE_IF(ml0_2 <= 0, false, params_->op_type, "ml0_2(%d) <= 0", ml0_2);
  int32_t nl0_2 = GetNl0(std::min(kL0cNzSize / ml0_2, kL0aNzSize / kl0_min_size_));
  OP_LOGE_IF(nl0_2 <= 0, false, params_->op_type, "nl0_2(%d) <= 0", nl0_2);
  int32_t mnl0_1 = ml0_1 * nl0_1;
  int32_t mnl0_2 = ml0_2 * nl0_2;
  if (mnl0_1 > mnl0_2) {
    l0_status_.m = ml0_1;
    l0_status_.n = nl0_1;
  } else if (mnl0_1 < mnl0_2) {
    l0_status_.m = ml0_2;
    l0_status_.n = nl0_2;
  } else {
    if (std::abs(ml0_1 - nl0_1) < std::abs(ml0_2 - nl0_2)) {
      l0_status_.m = ml0_1;
      l0_status_.n = nl0_1;
    } else if (std::abs(ml0_1 - nl0_1) > std::abs(ml0_2 - nl0_2)) {
      l0_status_.m = ml0_2;
      l0_status_.n = nl0_2;
    } else {
      l0_status_.m = std::min(ml0_1, ml0_2);
      l0_status_.n = mnl0_1 / l0_status_.m;
    }
  }
  return true;
}

bool CycleCalculator::FastFindNotFullLoad() {
  l0_status_.Init();
  l1_status_.Init();
  l0_status_.k = kl0_min_size_;
  l1_status_.m_al1 = 1;
  l1_status_.k_al1 = l0_status_.k;
  l1_status_.k_bl1 = l0_status_.k;
  l1_status_.n_bl1 = 1;

  OP_LOGE_IF(!GetMN0(), false, params_->op_type, "GetMN0 failed");
  if ((shape_.k * l0_status_.m > kL0aNzSize || shape_.k * l0_status_.n > kL0aNzSize) && final_cycle_ != INT64_MAX &&
      MathUtil::IsPrime(shape_.k)) {
    return true;
  }
  int64_t max_kl0 = params_->split_w_flag
                        ? std::max(MathUtil::CeilDivision(params_->a_shape.w, static_cast<int64_t>(params_->k0)),
                                   static_cast<int64_t>(kl0_min_size_))
                        : shape_.k;
  l0_status_.k = std::min(MathUtil::Min(max_kl0, kL0aNzSize / l0_status_.m), kL0aNzSize / l0_status_.n);
  OP_LOGE_IF(l0_status_.k == 0, false, params_->op_type, "l0_status_.k == 0");
  l1_status_.k_al1 = l0_status_.k;
  l1_status_.k_bl1 = l0_status_.k;

  // The w-split scene does not need to satisfy l0 factor
  while ((max_kl0 % l0_status_.k != 0 && !params_->split_w_flag) ||
         (params_->Fp32Input() && ((l0_status_.k & 0x1) != 0)) ||
         !CheckL1Size()) {
    if (params_->split_w_flag) {
      l0_status_.k --;
    } else {
      l0_status_.k = MathUtil::NearestFactor(max_kl0, l0_status_.k - 1, params_->Fp32Input());
    }
    OP_LOGE_IF(l0_status_.k == 0, false, params_->op_type, "l0_status_.k == 0");
    l1_status_.k_al1 = l0_status_.k;
    l1_status_.k_bl1 = l0_status_.k;
  }
  // actual_k/l1_status_.k_al1/l1_status_.k_bl1 Contains l0_k
  int64_t actual_k = params_->split_w_flag ? shape_.k * l0_status_.k : shape_.k;
  CubeUtil::CalcKbl1(params_, actual_k, l0_status_, l1_status_);
  OP_LOGE_IF(l1_status_.k_bl1 <= 0, false, params_->op_type, "invalid kbl1: %d", l1_status_.k_bl1);
  CubeUtil::CalcKal1(params_, shape_, actual_k, l0_status_, l1_status_);
  OP_LOGE_IF(l1_status_.k_al1 <= 0, false, params_->op_type, "invalid kal1: %d", l1_status_.k_al1);

  if (l1_status_.k_al1 > l1_status_.k_bl1 && l1_status_.k_al1 % l1_status_.k_bl1 != 0) {
    while ((l1_status_.k_al1 % l1_status_.k_bl1 != 0 && l1_status_.k_bl1 % l1_status_.k_al1 != 0) ||
           actual_k % l1_status_.k_al1 != 0 || !CheckL1Size()) {
      l1_status_.k_al1 -= l0_status_.k;
    }
  } else if (l1_status_.k_al1 < l1_status_.k_bl1 && l1_status_.k_bl1 % l1_status_.k_al1 != 0) {
    while ((l1_status_.k_al1 % l1_status_.k_bl1 != 0 && l1_status_.k_bl1 % l1_status_.k_al1 != 0) ||
           actual_k % l1_status_.k_bl1 != 0 || !CheckL1Size()) {
      l1_status_.k_bl1 -= l0_status_.k;
    }
  }

  UpdateSingleCoreStatus();
  tiling_pattern_flag_ = true;
  OP_LOGD(params_->op_type, "[Not FullLoad Pattern]");
  return true;
}

void CycleCalculator::UpdateTilingShape() {
  shape_.batch = MathUtil::CeilDivision(orig_shape_.batch, block_dims_.batch);
  shape_.m = shape_m_vec_[block_dims_.m - 1];
  if (shape_.m == 0) {
    shape_.m = MathUtil::FindBestSingleCore(orig_shape_.m, mapped_shape_.m, block_dims_.m, false);
    shape_m_vec_[block_dims_.m - 1] = shape_.m;
  }
  int64_t ci_n = shape_n_vec_[block_dims_.n - 1];
  if (ci_n == 0) {
    if (params_->c_shape.d == 1) {
      ci_n = MathUtil::FindBestSingleCore(params_->b_shape.c1, mapped_shape_.n, block_dims_.n, false);
    } else {
      ci_n = params_->b_shape.c1 / block_dims_.n;
    }
    shape_n_vec_[block_dims_.n - 1] = ci_n;
  }
  shape_.n = ci_n * params_->kernel_h * params_->kernel_w;
  shape_.k = MathUtil::CeilDivision(orig_shape_.k, block_dims_.k);
  shape_.group = orig_shape_.group / block_dims_.group;

  total_n_l0_ = shape_.n;
  total_n_l1_ = shape_.n;
}

bool CycleCalculator::FastFindPatternTiling(const DimFactor &block_dims) {
  block_dims_ = block_dims;
  UpdateTilingShape();
  if (shape_.group == 1 && shape_.batch == 1) {
    if (is_al1_full_load_pattern_) {
      OP_LOGE_IF(!FastFindAl1FullLoad(), false, params_->op_type, "FastFindAl1FullLoad failed");
    }
    if (is_bl1_full_load_pattern_) {
      OP_LOGE_IF(!FastFindBl1FullLoad(), false, params_->op_type, "FastFindBl1FullLoad failed");
    }
  }
  OP_LOGE_IF(!FastFindNotFullLoad(), false, params_->op_type, "FastFindNotFullLoad failed");
  return true;
}

void CycleCalculator::GenBlockDimsMapFactors() {
  mapped_shape_.batch = MathUtil::GetNonFactorMap(batch_dims_factors_, orig_shape_.batch, core_num_);
  MathUtil::AddCoreFactor(orig_shape_.batch, core_num_, batch_dims_factors_);
  // m = co1 * co0, only cut co1
  mapped_shape_.m = MathUtil::GetNonFactorMap(m_dims_factors_, params_->a_shape.c1, core_num_);
  MathUtil::AddCoreFactor(params_->a_shape.c1, core_num_, m_dims_factors_);

  // n = ci1 * kh * kw * ci0, only cut ci1
  mapped_shape_.n = MathUtil::GetNonFactorMap(n_dims_factors_, params_->b_shape.c1, core_num_);
  MathUtil::AddCoreFactor(params_->b_shape.c1, core_num_, n_dims_factors_);
  MathUtil::GetFactors(g_dims_factors_, params_->real_g, core_num_);
}

bool CycleCalculator::PruneBlockDim(int32_t used_core) const {
  return used_core < min_use_core_;
}

bool CycleCalculator::Exec() {
  if (params_->linear_embedding_opti_flag) {
    core_num_ = MathUtil::Min(std::max(orig_shape_.batch, static_cast<int64_t>(core_num_)), KBatchBindCoreLimit);
  }
  min_use_core_ = static_cast<int32_t>(ceil(core_num_ * 0.8));  // 0.8: default prune policy is to use 80% core at least
  is_al1_full_load_pattern_ = CheckAl1FullLoad();
  is_bl1_full_load_pattern_ = CheckBl1FullLoad();
  GenBlockDimsMapFactors();
  // if prune all block dims, then try again without prune
  bool prune = orig_shape_.batch * params_->a_shape.d >= core_num_;
  return LoopBlockDims(prune) && (tiling_pattern_flag_ || (prune && LoopBlockDims(false) && tiling_pattern_flag_));
}

bool CycleCalculator::LoopBlockDims(bool prune, int32_t used_core) {
  for (auto g_factor : g_dims_factors_) {
    for (auto batch_factor : batch_dims_factors_) {
      if (IsInvalidFactor(used_core * g_factor * batch_factor)) {
        break;
      }
      if (!IsValidBatchDim(batch_factor)) {
        continue;
      }

      for (auto n_factor : n_dims_factors_) {
        if (IsInvalidFactor(used_core * g_factor * batch_factor * n_factor)) {
          break;
        }
        for (auto m_factor : m_dims_factors_) {
          if (IsInvalidFactor(used_core * g_factor * batch_factor * n_factor * m_factor)) {
            break;
          }
          int32_t k_factor =
              CubeUtil::GetKFactor(params_, orig_shape_.k, used_core * g_factor * batch_factor * m_factor * n_factor);
          if (k_factor == 0) {
            continue;
          }
          if (prune && PruneBlockDim(used_core * batch_factor * m_factor * k_factor * n_factor * g_factor)) {
            continue;
          }
          DimFactor block_dims(batch_factor, m_factor, k_factor, n_factor, g_factor);
          OP_LOGE_IF(!FastFindPatternTiling(block_dims), false, params_->op_type, "FastFindPatternTiling failed");
        }
      }
    }
  }
  return true;
}

void CycleCalculator::Clear() { params_ = nullptr; }
}  // namespace cachetiling
}  // namespace optiling

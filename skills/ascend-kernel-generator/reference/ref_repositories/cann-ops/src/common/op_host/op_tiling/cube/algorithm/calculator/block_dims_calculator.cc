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
 * \file block_dims_calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/block_dims_calculator.h"

namespace optiling {
namespace cachetiling {
#define OP_LOGW(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
BlockDimsCalculator::BlockDimsCalculator(SingleCoreStatus &core_status) : Calculator(core_status) {}

bool BlockDimsCalculator::Init(const CubeTilingParam &params) {
  (void)Calculator::Init(params);
  orig_shape_.Init();
  shape_.Init();
  block_dims_.Init();
  al1_min_load_size_ = 0;
  bl1_min_load_size_ = 0;
  extend_shape_k_ = 0;
  load_size_ = INT64_MAX;
  core_used_ = 0;
  loop_num_ = 0;
  min_single_core_k_ = 0;
  batch_dims_factors_.clear();
  m_dims_factors_.clear();
  n_dims_factors_.clear();
  g_dims_factors_.clear();
  int32_t core_num = params_->platform_info.core_num();
  batch_dims_factors_.reserve(static_cast<size_t>(core_num));
  m_dims_factors_.reserve(static_cast<size_t>(core_num));
  n_dims_factors_.reserve(static_cast<size_t>(core_num));
  g_dims_factors_.reserve(static_cast<size_t>(core_num));
  shape_m_vec_.resize(core_num, 0);
  std::fill(shape_m_vec_.begin(), shape_m_vec_.end(), 0);
  shape_n_vec_.resize(core_num, 0);
  std::fill(shape_n_vec_.begin(), shape_n_vec_.end(), 0);
  max_mk_ = params_->platform_info.l0a_size() / (kBlockSize * params_->k0) / params_->a_dtype_bytes / kDbOn;
  max_nk_ = params_->platform_info.l0b_size() / (kBlockSize * params_->k0) / params_->b_dtype_bytes / kDbOn;

  return true;
}

bool BlockDimsCalculator::Exec() {
  orig_shape_ = single_core_status_.orig_shape();
  CalcL1MinLoadSize();

  // fp32 scene not use non factor for now
  if (params_->binary_mode != kBinaryModeNC1HWC0) {
    GenBlockDimsFactors();
  } else {
    GenBlockDimsMapFactors();
  }

  for (auto g_factor: g_dims_factors_) {
    for (auto batch_factor : batch_dims_factors_) {
      if (IsInvalidFactor(g_factor * batch_factor)) {
        break;
      }

      for (auto n_factor : n_dims_factors_) {
        if (IsInvalidFactor(g_factor * batch_factor * n_factor)) {
          break;
        }
        for (auto m_factor : m_dims_factors_) {
          if (IsInvalidFactor(g_factor * batch_factor * n_factor * m_factor)) {
            break;
          }

          int32_t k_factor =
              CubeUtil::GetKFactor(params_, orig_shape_.k, g_factor * batch_factor * m_factor * n_factor);
          if (k_factor == 0) {
            continue;
          }
          DimFactor block_dims(batch_factor, m_factor, k_factor, n_factor, g_factor);
          CalcBlockDims(block_dims);
        }
      }
    }
  }
  UpdateSingleCoreStatus();
  return true;
}

void BlockDimsCalculator::CalcL1MinLoadSize() {
  int32_t min_k_l0 = 1;
  if (params_->b_dtype == ge::DT_FLOAT) {
    // in fp32 scene, k axis aligned to 8 in L0 but aligned to 16 in L1, so k_l0 must be even number, min value is 2
    min_k_l0 = 2;
  }

  int32_t hi = 1;
  int32_t wi = 1;
  CubeUtil::CalcMinHiWi(params_, min_k_l0, hi, wi);
  bl1_min_load_size_ = hi * wi * params_->b_shape.c0;
  if (params_->dma_flag) {
    bl1_min_load_size_ = params_->k0 * params_->b_shape.c0;
  }
  al1_min_load_size_ = min_k_l0 * params_->k0 * params_->a_shape.c0 * params_->load3d_special;
  OP_LOGD(params_->op_type, "al1_min_load_size_: %d bl1_min_load_size_: %d", al1_min_load_size_, bl1_min_load_size_);
}

int64_t BlockDimsCalculator::FullLoadSize() const {
  // wo is big enough then have to cut wo in L0, so (k)full load is impossible when split w
  if (shape_.batch != 1 || shape_.group != 1 || params_->split_w_flag || params_->dma_flag) {
    return INT64_MAX;
  }
  int64_t al1_k_full_load_size = extend_shape_k_ * params_->k0 * params_->a_shape.c0;
  int64_t al1_full_load_size = shape_.m * al1_k_full_load_size;
  int32_t ho = CubeUtil::CalcHo(shape_.k * params_->k0, params_->a_shape.w, params_->op_type);
  int32_t hi = CubeUtil::CalcHi(ho, params_->stride_h, params_->kernel_h_dilation, params_->b_shape.h);
  int64_t bl1_k_full_load_size = hi * params_->b_shape.w * params_->b_shape.c0;
  int64_t bl1_full_load_size = params_->b_shape.c1 * bl1_k_full_load_size;
  if (params_->dma_flag) {
    bl1_full_load_size = shape_.k * params_->k0 * shape_.n * params_->b_shape.c0;
  }
  bool is_al1_full_load = IsL1SizeValid((al1_full_load_size + bl1_min_load_size_) * params_->a_dtype_bytes);
  bool is_bl1_full_load = IsL1SizeValid((bl1_full_load_size + al1_min_load_size_) * params_->b_dtype_bytes);

  int64_t load_size = INT64_MAX;
  if (is_al1_full_load || is_bl1_full_load) {
    load_size = static_cast<int64_t>(shape_.m * extend_shape_k_ + shape_.n * shape_.k) * shape_.batch * shape_.group;
  }
  OP_LOGD(params_->op_type, "load_size: %ld al1_full_load_size: %ld bl1_full_load_size: %ld", load_size,
          al1_full_load_size, bl1_full_load_size);
  return load_size;
}

int64_t BlockDimsCalculator::SingleKFullLoadSize(KLoadType load_type, int64_t k_full_load_size) const {
  // Default is Al1KfullLoad
  int32_t min_load_size = bl1_min_load_size_;
  int64_t axis = shape_.m;
  int64_t pair_axis = shape_.n;
  int32_t dtype_bytes = params_->a_dtype_bytes;
  if (load_type == kBl1KFullLoad) {
    min_load_size = al1_min_load_size_;
    axis = shape_.n;
    pair_axis = shape_.m;
    dtype_bytes = params_->b_dtype_bytes;
  }
  int32_t l1_remain_size = params_->platform_info.l1_size() / dtype_bytes - min_load_size;
  int32_t min_l1 = std::min(l1_remain_size / k_full_load_size, axis);
  if (min_l1 >= 1) {
    while (axis % min_l1 != 0) {
      min_l1--;
    }
  }
  int64_t axis_outer = axis / min_l1;
  // wout=1 scene k_al1 is 2 * shape_.k
  int64_t mn_load_size = axis_outer * pair_axis * extend_shape_k_ + axis * shape_.k;
  if (load_type == kAl1FullLoad) {
    mn_load_size = axis_outer * pair_axis * shape_.k + axis * extend_shape_k_;
  }
  return mn_load_size * shape_.batch * shape_.group;
}

int64_t BlockDimsCalculator::KFullLoadSize() const {
  // wo is big enough then have to cut wo in L0, so (k)full load is impossible when split w
  if (shape_.batch != 1 || params_->split_w_flag || params_->dma_flag) {
    return INT64_MAX;
  }

  int64_t al1_k_full_load_size = extend_shape_k_ * params_->k0 * params_->a_shape.c0;
  int32_t ho = CubeUtil::CalcHo(shape_.k * params_->k0, params_->a_shape.w, params_->op_type);
  int32_t hi = CubeUtil::CalcHi(ho, params_->stride_h, params_->kernel_h_dilation, params_->b_shape.h);
  int64_t bl1_k_full_load_size = hi * params_->b_shape.w * params_->b_shape.c0;
  if (params_->dma_flag) {
    bl1_k_full_load_size = shape_.k * params_->k0 * params_->b_shape.c0;
  }
  bool is_al1_k_full_load = IsL1SizeValid((al1_k_full_load_size + bl1_min_load_size_) * params_->a_dtype_bytes);
  bool is_bl1_k_full_load = IsL1SizeValid((bl1_k_full_load_size + al1_min_load_size_) * params_->b_dtype_bytes);

  int64_t al1_load_size = 0;
  int64_t bl1_load_size = 0;
  if (is_al1_k_full_load) {
    al1_load_size = SingleKFullLoadSize(kAl1KFullLoad, al1_k_full_load_size);
  }
  if (is_bl1_k_full_load) {
    bl1_load_size = SingleKFullLoadSize(kBl1KFullLoad, bl1_k_full_load_size);
  }

  int64_t load_size = INT64_MAX;
  if (is_al1_k_full_load && is_bl1_k_full_load) {
    load_size = std::min(al1_load_size, bl1_load_size);
  } else if (is_al1_k_full_load) {
    load_size = al1_load_size;
  } else if (is_bl1_k_full_load) {
    load_size = bl1_load_size;
  }

  OP_LOGD(params_->op_type,
          "load_size: %ld is_al1_k_full_load: %d is_bl1_k_full_load: %d al1_load_size: %ld bl1_load_size: %ld",
          load_size, is_al1_k_full_load, is_bl1_k_full_load, al1_load_size, bl1_load_size);
  return load_size;
}

int64_t BlockDimsCalculator::NeitherFullLoadSize() const {
  int32_t l1_size = params_->platform_info.l1_size();
  int32_t max_ml1 =
      MathUtil::Min((l1_size / params_->a_dtype_bytes - bl1_min_load_size_) / al1_min_load_size_, shape_.m);
  int32_t max_nl1 =
      MathUtil::Min((l1_size / params_->b_dtype_bytes - al1_min_load_size_) / bl1_min_load_size_, shape_.n);
  int32_t l1_min_pnt =
      l1_size / (al1_min_load_size_ * params_->a_dtype_bytes + bl1_min_load_size_ * params_->b_dtype_bytes);
  std::array<int32_t, kExtremeNumSize> ml1_factors = {0, 0};
  std::array<int32_t, kExtremeNumSize> nl1_factors = {0, 0};
  std::array<int32_t, kExtremeNumSize> limit_m = {max_ml1, 1};
  std::array<int32_t, kExtremeNumSize> limit_n = {max_nl1, 1};
  size_t ml1_size = MathUtil::GetTwoFactors(ml1_factors, l1_min_pnt, shape_.m, limit_m, 1);
  size_t nl1_size = MathUtil::GetTwoFactors(nl1_factors, l1_min_pnt, shape_.n, limit_n, 1);
  int64_t min_load_size = INT64_MAX;
  int64_t shape_mn = shape_.m * shape_.n;
  for (size_t i = 0; i < ml1_size; i++) {
    int32_t ml1 = ml1_factors[i];
    int32_t l1_remain_size = l1_size - al1_min_load_size_ * ml1 * params_->a_dtype_bytes;
    int32_t nl1 = std::min(l1_remain_size / (bl1_min_load_size_ * params_->b_dtype_bytes), max_nl1);
    OP_TILING_CHECK(nl1 <= 0, OP_LOGW(params_->op_type, "invalid nl1: %d", nl1), return INT64_MAX);
    while (shape_.n % nl1 != 0) {
      nl1--;
    }
    int64_t load_size = (shape_mn / ml1 * shape_.k + extend_shape_k_ * (shape_mn / nl1)) * shape_.batch * shape_.group;
    if (load_size < min_load_size) {
      min_load_size = load_size;
    }
  }
  for (size_t i = 0; i < nl1_size; i++) {
    int32_t nl1 = nl1_factors[i];
    int32_t l1_remain_size = l1_size - bl1_min_load_size_ * nl1 * params_->b_dtype_bytes;
    int32_t ml1 = std::min(l1_remain_size / (al1_min_load_size_ * params_->a_dtype_bytes), max_ml1);
    OP_TILING_CHECK(ml1 <= 0, OP_LOGW(params_->op_type, "invalid ml1: %d", ml1), return INT64_MAX);
    while (shape_.m % ml1 != 0) {
      ml1--;
    }
    int64_t load_size = (shape_mn / ml1 * shape_.k + extend_shape_k_ * (shape_mn / nl1)) * shape_.batch * shape_.group;
    if (load_size < min_load_size) {
      min_load_size = load_size;
    }
  }
  return min_load_size;
}

void BlockDimsCalculator::CalcBlockDims(const DimFactor &block_dims) {
  UpdateTilingShape(block_dims);
  extend_shape_k_ = GetExtendShapeK(shape_.k, params_->load3d_special);

  int64_t load_size = CalcLoadSize(block_dims);
  int32_t core_used = block_dims.ReduceMul();
  int64_t loop_num = CubeUtil::LoopNumFromSingleCoreToL0(orig_shape_, block_dims, params_->platform_info);
  if (loop_num == 0) {
    return;
  }
  OP_LOGD(params_->op_type, "load_size: %ld core_used: %d", load_size, core_used);
  // cut ho as k_dim when split w, maybe ho divided by k_dim is very small, so bypass this filter condition
  min_single_core_k_ = !params_->split_w_flag ? MathUtil::Min(kMinSingleCoreK, orig_shape_.k) : 1;
  if (NeedUpdate(block_dims, load_size, core_used, loop_num)) {
    Update(block_dims, load_size, core_used, loop_num);
  }
}

void BlockDimsCalculator::GenBlockDimsFactors() {
  int32_t core_num = params_->platform_info.core_num();
  MathUtil::GetFactors(batch_dims_factors_, orig_shape_.batch, core_num);
  int32_t max_core_num = MathUtil::Min(core_num, orig_shape_.batch);
  MathUtil::GetFactors(batch_dims_factors_, core_num, max_core_num);
  sort(batch_dims_factors_.begin(), batch_dims_factors_.end());
  (void)batch_dims_factors_.erase(unique(batch_dims_factors_.begin(), batch_dims_factors_.end()),
                                  batch_dims_factors_.cend());
  MathUtil::GetFactors(m_dims_factors_, params_->a_shape.c1, core_num);  // m = co1 * co0, only cut co1
  MathUtil::GetFactors(n_dims_factors_, params_->b_shape.c1, core_num);  // n = ci1 * kh * kw * ci0, only cut ci1
  MathUtil::GetFactors(g_dims_factors_, params_->real_g, core_num);
}

void BlockDimsCalculator::GenBlockDimsMapFactors() {
  int32_t core_num = params_->platform_info.core_num();
  mapped_shape_.batch = MathUtil::GetNonFactorMap(batch_dims_factors_, orig_shape_.batch, core_num);
  MathUtil::AddCoreFactor(orig_shape_.batch, core_num, batch_dims_factors_);
  // m = co1 * co0, only cut co1
  mapped_shape_.m = MathUtil::GetNonFactorMap(m_dims_factors_, params_->a_shape.c1, core_num);
  MathUtil::AddCoreFactor(params_->a_shape.c1, core_num, m_dims_factors_);

  // n = ci1 * kh * kw * ci0, only cut ci1
  mapped_shape_.n = MathUtil::GetNonFactorMap(n_dims_factors_, params_->b_shape.c1, core_num);
  MathUtil::AddCoreFactor(params_->b_shape.c1, core_num, n_dims_factors_);
  MathUtil::GetFactors(g_dims_factors_, params_->real_g, core_num);
}

int64_t BlockDimsCalculator::CalcLoadSize(const DimFactor &block_dims) const {
  OP_LOGD(params_->op_type, "[block_dims][%s]", block_dims.ToString().c_str());
  int64_t load_size = INT64_MAX;
  load_size = FullLoadSize();
  if (load_size != INT64_MAX) {
    return load_size;
  }

  load_size = KFullLoadSize();
  if (load_size != INT64_MAX) {
    return load_size;
  }

  load_size = NeitherFullLoadSize();
  if (load_size != INT64_MAX) {
    return load_size;
  }
  return load_size;
}

void BlockDimsCalculator::UpdateSingleCoreStatus() {
  single_core_status_.UpdateBlockDims(block_dims_);
  UpdateTilingShape();
  single_core_status_.UpdateShape(shape_);
  OP_LOGD(params_->op_type, "Update single core status [block_dims][%s]", block_dims_.ToString().c_str());
  OP_LOGD(params_->op_type, "Update single core status [shape][%s]", shape_.ToString().c_str());
}

void BlockDimsCalculator::UpdateTilingShape(const DimFactor &block_dims) {
  shape_.batch = MathUtil::CeilDivision(orig_shape_.batch, block_dims.batch);
  shape_.m = orig_shape_.m / block_dims.m;
  shape_.k = MathUtil::CeilDivision(orig_shape_.k, block_dims.k);
  shape_.n = orig_shape_.n / block_dims.n;
  shape_.group = orig_shape_.group / block_dims.group;
}

void BlockDimsCalculator::UpdateTilingShape() {
  shape_.batch = MathUtil::CeilDivision(orig_shape_.batch, block_dims_.batch);
  if (params_->binary_mode == kBinaryModeNC1HWC0) {
    shape_.m = shape_m_vec_[block_dims_.m - 1];
    if (shape_.m == 0) {
      shape_.m = MathUtil::FindBestSingleCore(orig_shape_.m, mapped_shape_.m, block_dims_.m, false);
      shape_m_vec_[block_dims_.m - 1] = shape_.m;
    }

    int64_t ci_n = shape_n_vec_[block_dims_.n - 1];
    if (ci_n == 0) {
      ci_n = MathUtil::FindBestSingleCore(params_->b_shape.c1, mapped_shape_.n, block_dims_.n, false);
      shape_n_vec_[block_dims_.n - 1] = ci_n;
    }
    shape_.n = ci_n * params_->kernel_h * params_->kernel_w;
    if (params_->b_shape.c0 == kSmallChannelSize) {
      shape_.n = MathUtil::CeilDivision(shape_.n, kSmallChannelSize);
    }
  } else {
    shape_.m = orig_shape_.m / block_dims_.m;
    shape_.n = orig_shape_.n / block_dims_.n;
  }

  shape_.k = MathUtil::CeilDivision(orig_shape_.k, block_dims_.k);
  shape_.group = orig_shape_.group / block_dims_.group;
}

bool BlockDimsCalculator::NeedUpdate(const DimFactor &block_dims, int64_t load_size, int32_t core_used,
                                     int64_t loop_num) const {
  constexpr float kThreshold = 1.03f;
  constexpr int32_t kLoopThreshold = 70;
  // when calculating L0 loadsize, it will choose large m and n, resulting in a large probability of kl0 being 1
  if (load_size_ != INT64_MAX && (shape_.k * shape_.m > max_mk_ && shape_.k * shape_.n > max_nk_) &&
      MathUtil::IsPrime(shape_.k)) {
    return false;
  }

  if (loop_num > kLoopThreshold && loop_num <= loop_num_ && core_used >= core_used_ &&
      load_size <= static_cast<int64_t>(kThreshold * load_size_) && shape_.k >= min_single_core_k_) {
    return true;
  }

  if (load_size > load_size_) {
    return false;
  }

  // the smaller load_size, a better strategy
  if (load_size < load_size_ && (shape_.k >= min_single_core_k_)) {
    return true;
  }

  // if load_size is the same, a larger value of batch indicates a better strategy
  if (block_dims.batch < block_dims_.batch) {
    return false;
  }

  // if load_size is the same, a larger value of batch indicates a better strategy
  if (block_dims.batch > block_dims_.batch && (shape_.k >= min_single_core_k_)) {
    return true;
  }

  // if load_size and batch are same, a larger value of core_used indicates a better strategy
  if (core_used > core_used_ && (shape_.k >= min_single_core_k_)) {
    return true;
  }

  return false;
}

void BlockDimsCalculator::Update(const DimFactor &block_dims, int64_t load_size, int32_t core_used, int64_t loop_num) {
  block_dims_ = block_dims;
  load_size_ = load_size;
  core_used_ = core_used;
  loop_num_ = loop_num;
  OP_LOGD(params_->op_type, "Find better Blockdims [block_dim][%s] [load_size][%ld] [core_used][%d]",
          block_dims.ToString().c_str(), load_size_, core_used_);
}
}  // namespace cachetiling
}  // namespace optiling

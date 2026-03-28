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
 * \file l0_calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/l0_calculator.h"
#include <sstream>

namespace optiling {
namespace cachetiling {
std::string CalcStatus::ToString() const {
  std::stringstream ss;
  ss << "db_l0a: " << db_l0a
     << " db_l0b: " << db_l0b
     << " db_l0c: " << db_l0c
     << " max_mk: " << max_mk
     << " max_nk: " << max_nk
     << " max_mn: " << max_mn
     << " max_axis_num: " << max_axis_num
     << " max_axis_pnt: " << max_axis_pnt;
  return ss.str();
}

L0Calculator::L0Calculator(SingleCoreStatus &core_status)
    : Calculator(core_status), shape_(single_core_status_.shape()) {}

bool L0Calculator::Init(const CubeTilingParam &params) {
  (void)Calculator::Init(params);
  l0_status_.Init();
  l1_status_.Init();
  extend_shape_k_ = 0;
  load_size_ = INT64_MAX;
  l0c_used_ = 0;
  scope_ = 0;
  best_l0_status_.Init();
  return true;
}

bool L0Calculator::Exec() {
  if (params_->split_w_flag) {
    // when split w ho is 1, and only cut wo in L0 as k_l0
    extend_shape_k_ = MathUtil::Align(params_->a_shape.w, kBlockSize) / params_->k0;
  } else {
    extend_shape_k_ = GetExtendShapeK(shape_.k, params_->load3d_special);
  }
  for (auto l0c_db_type : {kDbOn, kDbOff}) {
    CalcStatus calc_status = InitCalcStatus(l0c_db_type);
    GenL0Factors(calc_status, l0c_db_type);
  }
  UpdateSinleCoreStatus();
  return true;
}

CalcStatus L0Calculator::InitCalcStatus(int32_t type) const {
  CalcStatus calc_status;
  calc_status.db_l0a = kDbOn;
  calc_status.db_l0b = kDbOn;
  calc_status.db_l0c = type;
  calc_status.max_mk =
      params_->platform_info.l0a_size() / (kBlockSize * params_->k0) / params_->a_dtype_bytes / calc_status.db_l0a;
  calc_status.max_nk =
      params_->platform_info.l0b_size() / (kBlockSize * params_->k0) / params_->b_dtype_bytes / calc_status.db_l0b;
  calc_status.max_mn =
      params_->platform_info.l0c_size() / kCubeTileNumSize / params_->c_dtype_bytes / calc_status.db_l0c;

  const UbStatus &ub_status = single_core_status_.ub_status();

  // UB_min_loadsize <= UBSize
  int32_t cub_min_size = (params_->cub_fused_num + 1) * kCubeTileNumSize * params_->c_dtype_bytes * ub_status.db_cub;
  int32_t aub_min_size = 0;
  int32_t bub_min_size = 0;
  if (params_->binary_mode == kBinaryModeNCHW || params_->binary_mode == kBinaryModeNHWC) {
    aub_min_size = (params_->aub_fused_num + 1) * kCubeTileNumSize * params_->a_dtype_bytes * ub_status.db_aub;
    int32_t algined_wi =
        IsLargeWi(params_->b_shape.w) || params_->conv1d_flag || params_->split_w_flag || params_->dma_flag
            ? kBlockSize
            : MathUtil::Align(params_->b_shape.w, kBlockSize);
    bub_min_size = (params_->bub_fused_num + 1) * kBlockSize * algined_wi * ub_status.db_bub * params_->b_dtype_bytes;
  } else if (params_->dma_flag && !params_->platform_info.support_l0c2out()) {
    bub_min_size = KDmaFmapUbSize;
  }
  int32_t max_ml0 = (params_->platform_info.ub_size() - aub_min_size - bub_min_size) / cub_min_size;
  calc_status.max_axis_num = std::min(std::min(calc_status.max_mk, max_ml0), calc_status.max_mn);
  calc_status.max_axis_pnt = type == kDbOn ? 11 : 16;  // 11 & 16 is extremum point
  calc_status.max_axis_pnt = std::min(calc_status.max_axis_pnt, calc_status.max_axis_num);
  OP_LOGD(params_->op_type, "CalcStatus [%s]", calc_status.ToString().c_str());
  return calc_status;
}

void L0Calculator::CalcL0Factors(L0Status &l0_status, L1Status &l1_status, int32_t k0_max, int32_t l0c_db_type,
                                 bool even_k_factor) {
  while (k0_max > 0) {
    if (IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status, l0_status))) {
      DimFactor dim_factor{1, l0_status.m, k0_max, l0_status.n};
      CalcL0Status(dim_factor, l0c_db_type);
      return;
    }
    k0_max--;
    k0_max = MathUtil::NearestFactor(extend_shape_k_, k0_max, even_k_factor);
    if (k0_max == 0) {
      return;
    }

    l0_status.k = k0_max;
    l1_status.k_al1 = k0_max;
    l1_status.k_bl1 = k0_max;
  }
}

void L0Calculator::GenL0Factors(const CalcStatus &calc_status, int32_t l0c_db_type) {
  std::vector<int32_t> m_dim_factors;
  std::vector<int32_t> n_dim_factors;
  MathUtil::GetFactors(m_dim_factors, shape_.m, calc_status.max_mk);
  MathUtil::GetFactors(n_dim_factors, shape_.n, calc_status.max_nk);

  if (params_->b_shape.c0 == kSmallChannelSize) {
    n_dim_factors = {static_cast<int32_t>(shape_.n), };  // only one case in white list
  }
  L0Status l0_status;
  L1Status l1_status;

  // for fp32 scene, k0 is 8 and k_l0 * k0 should align to 16(restricted by load3d), so we need even factor here
  // wout=1 scene, k0 must be even numbers
  bool even_k_factor = params_->b_dtype == ge::DT_FLOAT || params_->load3d_special > 1;
  for (auto m_dim_factor : m_dim_factors) {
    for (auto n_dim_factor : n_dim_factors) {
      if (m_dim_factor * n_dim_factor > calc_status.max_mn) {
        continue;
      }

      int32_t k0_max = std::min(calc_status.max_mk / m_dim_factor, calc_status.max_nk / n_dim_factor);
      k0_max = MathUtil::NearestFactor(extend_shape_k_, k0_max, even_k_factor);
      if (k0_max == 0) {
        continue;
      }

      l0_status.m = m_dim_factor;
      l0_status.n = n_dim_factor;
      l0_status.k = k0_max;
      l1_status.k_al1 = k0_max;
      l1_status.k_bl1 = k0_max;
      CalcL0Factors(l0_status, l1_status, k0_max, l0c_db_type, even_k_factor);
    }
  }
}

void L0Calculator::CalcL0Status(const DimFactor &factor, int32_t l0c_db_type) {
  l0_status_.m = factor.m;
  l0_status_.k = factor.k;
  l0_status_.n = factor.n;
  l0_status_.db_l0a = kDbOn;
  l0_status_.db_l0b = kDbOn;
  l0_status_.db_l0c = l0c_db_type;
  l1_status_.k_al1 = factor.k;
  l1_status_.k_bl1 = factor.k;
  OP_LOGD(params_->op_type, "[l0_status][%s] [l1_status_][%s]", l0_status_.ToString().c_str(),
          l1_status_.ToString().c_str());

  int64_t load_size = CalcLoadSize();
  int64_t scope = static_cast<int64_t>(l0_status_.m) * l0_status_.k * l0_status_.n;
  int64_t l0c_used =
      static_cast<int64_t>(l0_status_.m) * l0_status_.n * l0_status_.db_l0c * kCubeTileNumSize * params_->c_dtype_bytes;
  OP_LOGD(params_->op_type, "load_size: %ld scope: %ld l0c_used: %ld", load_size, scope, l0c_used);

  if (NeedUpdate(load_size, scope, l0c_used)) {
    Update(load_size, scope, l0c_used);
  }
}

int64_t L0Calculator::CalcLoadSize() const {
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

int64_t L0Calculator::FullLoadSize() const {
  // wo is big enough then have to cut wo in L0, so (k)full load is impossible when split w
  if (shape_.batch != 1 || shape_.group != 1 || params_->split_w_flag || params_->dma_flag) {
    return INT64_MAX;
  }

  int64_t al1_full_load_size = static_cast<int64_t>(shape_.m) * extend_shape_k_ * params_->k0 * params_->a_shape.c0;
  int32_t full_ho = CubeUtil::CalcHo(shape_.k * params_->k0, params_->a_shape.w, params_->op_type);
  int32_t full_hi = CubeUtil::CalcHi(full_ho, params_->stride_h, params_->kernel_h_dilation, params_->b_shape.h);
  int64_t bl1_full_load_size =
      static_cast<int64_t>(full_hi) * params_->b_shape.w * params_->b_shape.c1 * params_->b_shape.c0;
  if (params_->dma_flag) {
    bl1_full_load_size = shape_.k * params_->k0 * shape_.n * params_->b_shape.c0;
  }

  int32_t al1_min_load_size = l0_status_.m * l0_status_.k * params_->k0 * params_->a_shape.c0;
  int32_t hi = 1;
  int32_t wi = 1;
  CubeUtil::CalcHiWi(params_, l0_status_.k, l0_status_.k, hi, wi);
  int32_t ci1_factor = MathUtil::CeilDivision(l0_status_.n, params_->kernel_h * params_->kernel_w);
  int64_t bl1_min_load_size = static_cast<int64_t>(hi) * wi * ci1_factor * params_->b_shape.c0;

  bool al1_full_load = IsL1SizeValid((al1_full_load_size + bl1_min_load_size) * params_->a_dtype_bytes);
  bool bl1_full_load = IsL1SizeValid((bl1_full_load_size + al1_min_load_size) * params_->b_dtype_bytes);

  int64_t load_size = INT64_MAX;
  if (al1_full_load || bl1_full_load) {
    load_size = shape_.m + shape_.n;
  }

  OP_LOGD(params_->op_type, "load_size: %ld al1_full_load: %d bl1_full_load: %d", load_size, al1_full_load,
          bl1_full_load);
  return load_size;
}

int64_t L0Calculator::KFullLoadSize() const {
  // wo is big enough then have to cut wo in L0, so (k)full load is impossible when split w
  if (shape_.batch != 1 || params_->split_w_flag || params_->dma_flag) {
    return INT64_MAX;
  }

  int64_t al1_k_full_load_size = static_cast<int64_t>(shape_.k) * params_->k0 * params_->a_shape.c0;
  int32_t k_full_ho = CubeUtil::CalcHo(shape_.k * params_->k0, params_->a_shape.w, params_->op_type);
  int32_t k_full_hi = CubeUtil::CalcHi(k_full_ho, params_->stride_h, params_->kernel_h_dilation, params_->b_shape.h);
  int64_t bl1_k_full_load_size = static_cast<int64_t>(k_full_hi) * params_->b_shape.w * params_->b_shape.c0;
  if (params_->dma_flag) {
    bl1_k_full_load_size = shape_.k * params_->k0 * params_->b_shape.c0;
  }

  int32_t al1_min_load_size = l0_status_.m * l0_status_.k * params_->k0 * params_->a_shape.c0;
  int32_t ho = CubeUtil::CalcHo(l0_status_.k * params_->k0, params_->a_shape.w, params_->op_type);
  int32_t hi = CubeUtil::CalcHi(ho, params_->stride_h, params_->kernel_h_dilation, params_->b_shape.h);
  int32_t ci1_factor = MathUtil::CeilDivision(l0_status_.n, params_->kernel_h * params_->kernel_w);
  int64_t bl1_min_load_size = static_cast<int64_t>(hi) * params_->b_shape.w * ci1_factor * params_->b_shape.c0;
  if (params_->dma_flag) {
    bl1_min_load_size = l0_status_.k * params_->k0 * l1_status_.n_bl1 * l0_status_.n;
  }

  bool is_al1_k_full_load = IsL1SizeValid((al1_k_full_load_size + bl1_min_load_size) * params_->a_dtype_bytes);
  bool is_bl1_k_full_load = IsL1SizeValid((bl1_k_full_load_size + al1_min_load_size) * params_->b_dtype_bytes);

  int64_t al1_load_size = 0;
  int64_t bl1_load_size = 0;
  if (is_al1_k_full_load) {
    al1_load_size = SingleKFullLoadSize(kAl1KFullLoad, al1_k_full_load_size, bl1_min_load_size);
  }
  if (is_bl1_k_full_load) {
    bl1_load_size = SingleKFullLoadSize(kBl1KFullLoad, bl1_k_full_load_size, al1_min_load_size);
  }

  int64_t load_size = INT64_MAX;
  if (is_al1_k_full_load && is_bl1_k_full_load) {
    load_size = std::min(al1_load_size, bl1_load_size);
  } else if (is_al1_k_full_load) {
    load_size = al1_load_size;
  } else if (is_bl1_k_full_load) {
    load_size = bl1_load_size;
  }
  OP_LOGD(params_->op_type, "load_size: %ld is_al1_k_full_load: %d bl1_k_full_load: %d", load_size, is_al1_k_full_load,
          is_bl1_k_full_load);
  return load_size;
}

int64_t L0Calculator::SingleKFullLoadSize(KLoadType load_type, int64_t k_full_load_size, int32_t min_load_size) const {
  // Default is Al1KfullLoad
  int64_t axis = shape_.m;
  int64_t pair_axis = shape_.n;
  int32_t axis_tiling = l0_status_.m;
  int32_t dtype_bytes = params_->a_dtype_bytes;
  if (load_type == kBl1KFullLoad) {
    axis = shape_.n;
    pair_axis = shape_.m;
    axis_tiling = l0_status_.n;
    dtype_bytes = params_->b_dtype_bytes;
  }
  int32_t l1_remain_size = params_->platform_info.l1_size() / dtype_bytes - min_load_size;
  int32_t min_l1 = std::min(l1_remain_size / k_full_load_size, axis);
  if (min_l1 >= axis_tiling) {
    while (axis % min_l1 != 0 || min_l1 % axis_tiling != 0) {
      min_l1--;
    }
  }
  int64_t axis_outer = min_l1 != 0 ? axis / min_l1 : axis;
  return axis_outer * pair_axis + axis;
}

int64_t L0Calculator::NeitherFullLoadSize() const {
  int64_t mn = shape_.m * shape_.n;
  return mn / l0_status_.m + mn / l0_status_.n;
}

void L0Calculator::UpdateSinleCoreStatus() {
  single_core_status_.UpdateL0Status(best_l0_status_);
  single_core_status_.UpdateL1Status(l1_status_);
  OP_LOGD(params_->op_type, "Update single core status [l0_status][%s]", best_l0_status_.ToString().c_str());
  OP_LOGD(params_->op_type, "Update single core status [l1_status][%s]", l1_status_.ToString().c_str());
}

bool L0Calculator::NeedUpdate(int64_t load_size, int64_t scope, int64_t l0c_used) const {
  constexpr float kThreshold = 1.5;

  if (params_->platform_info.support_l0c2out() && load_size < static_cast<int64_t>(load_size_ * kThreshold) &&
      l0c_used >= static_cast<int64_t>(l0c_used_ * kThreshold)) {
    return true;
  }

  if (load_size > load_size_) {
    return false;
  }

  // the smaller load_size, a better strategy
  if (load_size < load_size_) {
    // if the utilization rate of L0C is too low, discard it
    if (params_->platform_info.support_l0c2out() && l0c_used_ > static_cast<int64_t>(l0c_used * kThreshold)) {
      return true;
    }
    return true;
  }

  // if load_size is the same, a larger value of scope indicates a better strategy
  if (scope <= scope_) {
    return false;
  }

  // if scope is the larger, a larger value of scope * l0c_used indicates a better strategy
  if (scope * l0c_used >= scope_ * l0c_used_) {
    return true;
  }

  return false;
}

void L0Calculator::Update(int64_t load_size, int64_t scope, int64_t l0c_used) {
  best_l0_status_ = l0_status_;
  load_size_ = load_size;
  scope_ = scope;
  l0c_used_ = l0c_used;
  OP_LOGD(params_->op_type,
          "Find better l0_status [best_l0_status_][%s] [load_size_][%ld] [scope_][%ld] [l0c_used_][%ld]",
          best_l0_status_.ToString().c_str(), load_size_, scope_, l0c_used_);
}
}  // namespace cachetiling
}  // namespace optiling
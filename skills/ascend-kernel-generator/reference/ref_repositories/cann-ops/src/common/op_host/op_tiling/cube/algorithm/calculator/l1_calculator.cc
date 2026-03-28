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
 * \file l1_calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/l1_calculator.h"

namespace optiling {
namespace cachetiling {
L1Calculator::L1Calculator(SingleCoreStatus &core_status)
    : Calculator(core_status), shape_(single_core_status_.shape()), l0_status_(core_status.l0_status()) {}

bool L1Calculator::Init(const CubeTilingParam &params) {
  (void)Calculator::Init(params);
  l1_status_.Init();
  extend_shape_k_ = 0;
  load_size_ = INT64_MAX;
  scope_ = 0;
  load2d_mode_ = false;
  best_l1_status_.Init();
  return true;
}

bool L1Calculator::Exec() {
  extend_shape_k_ = GetExtendShapeK(shape_.k, params_->load3d_special);
  // when split w, cut ho as k_dim and cut wo as k_l0, multiply them to get full k
  if (params_->split_w_flag) {
    extend_shape_k_ *= l0_status_.k;
  }

  std::array<L1Factor, kL1FactorMaxSize> l1_factors;
  load2d_mode_ = params_->load2d_flag;
  size_t l1_factors_size = GenL1Factors(l1_factors);
  for (size_t i = 0; i < l1_factors_size; i++) {
    OP_LOGE_IF(!CalcL1Status(l1_factors[i]), false, params_->op_type, "fail to calculate L1 status");
  }
  UpdateSinleCoreStatus();
  return true;
}

size_t L1Calculator::GenL1Factors(std::array<L1Factor, kL1FactorMaxSize> &l1_factors) const {
  int64_t max_m_al1 = MathUtil::CeilDivision(shape_.m, l0_status_.m);
  int64_t max_n_bl1 = MathUtil::CeilDivision(shape_.n, l0_status_.n);
  size_t idx = 0;
  bool img2colV2_posk_constraint = params_->platform_info.support_data_move_out2l1_nd2nz() &&
                                   params_->load3d_flag &&
                                   max_n_bl1 * l0_status_.n * params_->b_shape.c0 > kMaxLoad3dV2Kstart;
  bool load3d_wo_constraint = params_->load3d_special != 1 && max_n_bl1 != 1;
  // wo is big enough then have to cut wo in L0, so (k)full load is impossible when split w
  bool full_load_invalid =
      img2colV2_posk_constraint || load3d_wo_constraint || params_->split_w_flag || params_->dma_flag;
  bool bl1_full_load_invalid =
      img2colV2_posk_constraint || load3d_wo_constraint || params_->split_w_flag || params_->dma_flag;
  bool al1_full_load_invalid = params_->split_w_flag || params_->dma_flag;
  // neight AL1 nor BL1 full load
  l1_factors[idx++] = L1Factor(kNeitherFullLoad, 1, l0_status_.k, l0_status_.k, 1);
  // only al1 full load
  if (!al1_full_load_invalid) {
    l1_factors[idx++] = L1Factor(kAl1FullLoad, max_m_al1, extend_shape_k_, l0_status_.k, 1);
  }
  if (!bl1_full_load_invalid) {
    // only bl1 full load
    l1_factors[idx++] = L1Factor(kBl1FullLoad, 1, l0_status_.k, shape_.k, max_n_bl1);
  }
  if (!full_load_invalid) {
    // full load
    l1_factors[idx++] = L1Factor(kFullLoad, max_m_al1, extend_shape_k_, shape_.k, max_n_bl1);
  }
  return idx;
}

bool L1Calculator::CalcL1Status(const L1Factor &factor) {
  if (!factor.IsValid()) {
    return true;
  }

  l1_status_.m_al1 = factor.m;
  l1_status_.k_al1 = factor.k_a;
  l1_status_.k_bl1 = factor.k_b;
  l1_status_.n_bl1 = factor.n;
  l1_status_.al1_repeat_time = 1;
  l1_status_.bl1_repeat_time = 1;
  l1_status_.db_al1 = kDbOff;
  l1_status_.db_bl1 = kDbOff;

  OP_LOGD(params_->op_type, "[l1_status][%s]", l1_status_.ToString().c_str());
  int64_t load_size = CalcLoadSize(factor.load_type);
  OP_LOGE_IF(factor.load_type == kNeitherFullLoad && load_size == INT64_MAX, false, params_->op_type,
             "get invalid load size when nerther full load");
  int64_t scope = l1_status_.m_al1 + l1_status_.n_bl1;
  OP_LOGD(params_->op_type, "load_size: %ld scope: %ld", load_size, scope);

  if (NeedUpdate(load_size, scope)) {
    CubeUtil::CalcL1RemainStatus(params_, l0_status_, l1_status_);
    Update(load_size, scope);
  }

  return true;
}

int64_t L1Calculator::CalcLoadSize(KLoadType load_type) {
  int64_t load_size = INT64_MAX;
  OP_LOGD(params_->op_type, "load_type %d", load_type);
  switch (load_type) {
    case kFullLoad:
      load_size = FullLoadSize();
      break;
    case kAl1FullLoad:
      load_size = Al1FullLoadSize();
      break;
    case kBl1FullLoad:
      load_size = Bl1FullLoadSize();
      break;
    case kNeitherFullLoad:
      load_size = NeitherFullLoadSize();
      break;
    default:
      load_size = FullLoadSize();
      break;
  }
  return load_size;
}

int64_t L1Calculator::FullLoadSize() const {
  bool is_full_load =
      shape_.batch == 1 && shape_.group == 1 && IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
  if (is_full_load) {
    return shape_.m + shape_.n;
  }
  return INT64_MAX;
}

int64_t L1Calculator::Al1FullLoadSize() {
  bool is_al1_full_load =
      shape_.batch == 1 && shape_.group == 1 && IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
  if (!is_al1_full_load) {
    return INT64_MAX;
  }

  l1_status_.db_bl1 = kDbOn;
  if (!IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_))) {
    l1_status_.db_bl1 = kDbOff;
  }

  CubeUtil::CalcKbl1(params_, extend_shape_k_, l0_status_, l1_status_);
  CubeUtil::CalcNbl1(params_, shape_, l0_status_, shape_.n, l1_status_);
  return shape_.m + shape_.n;
}

int64_t L1Calculator::Bl1FullLoadSize() {
  bool is_bl1_full_load =
      shape_.batch == 1 && shape_.group == 1 && IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
  if (!is_bl1_full_load) {
    return INT64_MAX;
  }

  l1_status_.db_al1 = kDbOn;
  if (!IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_))) {
    l1_status_.db_al1 = kDbOff;
  }

  CubeUtil::CalcKal1(params_, shape_, extend_shape_k_, l0_status_, l1_status_);
  CubeUtil::CalcMal1(params_, shape_, extend_shape_k_, l0_status_, l1_status_);

  return shape_.m + shape_.n;
}

bool L1Calculator::NeitherFullLoadHelper() {
  CubeUtil::CalcKbl1(params_, extend_shape_k_, l0_status_, l1_status_);
  CubeUtil::CalcKal1(params_, shape_, extend_shape_k_, l0_status_, l1_status_);
  OP_LOGE_IF(l1_status_.k_al1 <= 0, false, params_->op_type, "invalid kal1: %d", l1_status_.k_al1);
  OP_LOGE_IF(l1_status_.k_bl1 <= 0, false, params_->op_type, "invalid kbl1: %d", l1_status_.k_bl1);

  // k_al1 and k_bl1 must be a factor of each other
  auto invalid_kal1_kbl1_relation = [this]() -> bool {
    if (l1_status_.k_al1 > l1_status_.k_bl1) {
      return l1_status_.k_al1 % l1_status_.k_bl1 != 0 ||
             !IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
    } else if (l1_status_.k_al1 == l1_status_.k_bl1) {
      return !IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
    } else {
      return l1_status_.k_bl1 % l1_status_.k_al1 != 0 ||
             !IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_));
    }
  };

  if (l1_status_.k_al1 > l1_status_.k_bl1) {
    while (l1_status_.k_al1 > 0 && (extend_shape_k_ % l1_status_.k_al1 != 0 || invalid_kal1_kbl1_relation())) {
      l1_status_.k_al1 -= l0_status_.k;
    }
  } else if (l1_status_.k_al1 < l1_status_.k_bl1) {
    int32_t real_kb_l0 = l0_status_.k / params_->load3d_special;
    while (l1_status_.k_bl1 > 0 && (extend_shape_k_ % l1_status_.k_bl1 != 0 || invalid_kal1_kbl1_relation())) {
      l1_status_.k_bl1 -= real_kb_l0;
    }
  }

  l1_status_.db_bl1 = kDbOn;
  if (!IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_))) {
    l1_status_.db_bl1 = kDbOff;
  }
  l1_status_.db_al1 = kDbOn;
  if (!IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_))) {
    l1_status_.db_al1 = kDbOff;
  }

  CubeUtil::CalcNbl1(params_, shape_, l0_status_, shape_.n, l1_status_);
  CubeUtil::CalcMal1(params_, shape_, extend_shape_k_, l0_status_, l1_status_);

  OP_LOGD(params_->op_type, "l1_status_[%s]", l1_status_.ToString().c_str());
  return true;
}

int64_t L1Calculator::NeitherFullLoadSize() {
  if (!NeitherFullLoadHelper() || l1_status_.m_al1 < 1 || l1_status_.n_bl1 < 1 || l1_status_.k_bl1 < 1 ||
      l1_status_.k_al1 < 1 || !IsL1SizeValid(CubeUtil::CalcL1Size(params_, l1_status_, l0_status_))) {
    return INT64_MAX;
  }
  int64_t max_m_al1 = MathUtil::CeilDivision(shape_.m, l0_status_.m);
  int64_t max_n_bl1 = MathUtil::CeilDivision(shape_.n, l0_status_.n);
  bool is_al1_k_full_load = shape_.batch == 1 && l1_status_.k_al1 == extend_shape_k_;
  bool is_al1_full_load = is_al1_k_full_load && l1_status_.m_al1 == max_m_al1;
  bool is_bl1_k_full_load = shape_.batch == 1 && l1_status_.k_bl1 == shape_.k;
  bool is_bl1_full_load = is_bl1_k_full_load && l1_status_.n_bl1 == max_n_bl1;
  int64_t al1_repeat_time = 1;
  int64_t bl1_repeat_time = 1;
  if (is_al1_full_load || is_bl1_full_load) {
    al1_repeat_time = 1;
    bl1_repeat_time = 1;
  } else if (is_al1_k_full_load && is_bl1_k_full_load) {
    al1_repeat_time = max_n_bl1 / l1_status_.n_bl1;
    bl1_repeat_time = max_m_al1 / l1_status_.m_al1;
    if (shape_.m + bl1_repeat_time * shape_.n < al1_repeat_time * shape_.m + shape_.n) {
      al1_repeat_time = 1;
    } else {
      bl1_repeat_time = 1;
    }
  } else if (is_al1_k_full_load) {
    al1_repeat_time = 1;
    bl1_repeat_time = max_m_al1 / l1_status_.m_al1;
  } else if (is_bl1_k_full_load) {
    al1_repeat_time = max_n_bl1 / l1_status_.n_bl1;
    bl1_repeat_time = 1;
  } else {
    al1_repeat_time = max_n_bl1 / l1_status_.n_bl1;
    bl1_repeat_time = max_m_al1 / l1_status_.m_al1;
  }
  l1_status_.al1_repeat_time = al1_repeat_time;
  l1_status_.bl1_repeat_time = bl1_repeat_time;

  OP_LOGD(params_->op_type, "is_al1_full_load: %d is_bl1_full_load: %d is_al1_k_full_load: %d is_bl1_k_full_load: %d",
          is_al1_full_load, is_bl1_full_load, is_al1_k_full_load, is_bl1_k_full_load);
  OP_LOGD(params_->op_type, "al1_repeat_time: %ld bl1_repeat_time: %ld", al1_repeat_time, bl1_repeat_time);
  return al1_repeat_time * shape_.m + bl1_repeat_time * shape_.n;
}

void L1Calculator::UpdateSinleCoreStatus() {
  single_core_status_.UpdateL1Status(best_l1_status_);
  OP_LOGD(params_->op_type, "Update single core status [l1_status][%s]", best_l1_status_.ToString().c_str());
}

bool L1Calculator::NeedUpdate(int64_t load_size, int64_t scope) const {
  if (load_size == INT64_MAX) {
    return false;
  }

  if (load_size > load_size_) {
    return false;
  }

  // the smaller load_size, a better strategy
  if (load_size < load_size_) {
    return true;
  }

  // if load_size is the same, a larger value of scope indicates a better strategy
  if (scope > scope_) {
    return true;
  }

  return false;
}

void L1Calculator::Update(int64_t load_size, int64_t scope) {
  best_l1_status_ = l1_status_;
  load_size_ = load_size;
  scope_ = scope;
  OP_LOGD(params_->op_type, "Find better l1_status [best_l1_status_][%s] [load_size_][%ld] [scope_][%ld]",
          best_l1_status_.ToString().c_str(), load_size_, scope_);
}
}  // namespace cachetiling
}  // namespace optiling
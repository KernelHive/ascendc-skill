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
 * \file ub_calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/ub_calculator.h"

#include <cmath>

namespace optiling {
namespace cachetiling {
UbCalculator::UbCalculator(SingleCoreStatus &core_status) : Calculator(core_status) {}

bool UbCalculator::Init(const CubeTilingParam &params) {
  (void)Calculator::Init(params);
  ub_status_.Init();
  wi_bub_ = 0;
  return true;
}

bool UbCalculator::Exec() {
  if (params_->platform_info.support_ub()) {
    OP_LOGD(params_->op_type, "The platform supports UB");
    CalcUbStatus();
    UpdateSinleCoreStatus();
  }
  return true;
}

int32_t UbCalculator::CalcWiBub(int32_t limit_hn, int32_t k_bub) const {
  // It can still be put down in L1 when wi = 2016
  int32_t wi = std::min(static_cast<int64_t>(limit_hn) / k_bub, params_->b_shape.w);
  int32_t actual_hn = k_bub * wi;
  int32_t algined_hn = MathUtil::Align(actual_hn, kBlockSize);
  if (algined_hn == actual_hn) {
    return wi;
  }
  // eg: algined = 1200, limit_hn = 1199,  k_bub = 3, wi = 399, ceil((1197 - (1200 - 16)) / 3) = 5
  int32_t dimi = static_cast<int32_t>(std::ceil((actual_hn - (algined_hn -
    MathUtil::Align(algined_hn - actual_hn, kBlockSize))) / k_bub));
  return wi - dimi;
}

void UbCalculator::CalcKBub(int32_t limit_hn, int32_t bl1_hi) {
  ub_status_.k_bub = MathUtil::NearestFactor(bl1_hi, limit_hn);
  // k_bub is the factor of bl1_hi
  vector<int32_t> bl1_hi_factors;
  MathUtil::GetFactors(bl1_hi_factors, bl1_hi, 1, ub_status_.k_bub);
  for (size_t i = 0; i < bl1_hi_factors.size(); ++i) {
    ub_status_.k_bub = bl1_hi_factors[i];
    if (MathUtil::Align(ub_status_.k_bub * params_->b_shape.w, kBlockSize) <= limit_hn) {
      return;
    }
  }
}

void UbCalculator::CalcUbStatus() {
  const L0Status &l0_status = single_core_status_.l0_status();
  const L1Status &l1_status = single_core_status_.l1_status();

  int32_t ub_size = params_->platform_info.ub_size();
  int32_t cub_min_size = (params_->cub_fused_num + 1) * l0_status.m * kCubeTileNumSize * ub_status_.db_cub;
  // if binarymode is NC1HWC0, aub and bub size is 0.
  int32_t bub_size = 0;
  int32_t aub_size = 0;
  wi_bub_ = params_->b_shape.w;
  if (params_->binary_mode == kBinaryModeNCHW || params_->binary_mode == kBinaryModeNHWC) {
    // aub, bub min size
    int32_t aub_min_size = (params_->aub_fused_num + 1) * kCubeTileNumSize * ub_status_.db_aub;
    int32_t algined_wi = IsLargeWi(params_->b_shape.w) || params_->conv1d_flag
                             ? kBlockSize
                             : MathUtil::Align(params_->b_shape.w, kBlockSize);
    int32_t bub_min_size = (params_->bub_fused_num + 1) * kBlockSize * algined_wi * ub_status_.db_bub;
    if (params_->binary_mode == kBinaryModeNHWC) {
      // need extra space for vector_or instruction
      ub_size -= kBlockSize * kFp16Bytes;
    }
    // aub add bub max size
    int32_t ub_rest_size = (ub_size - cub_min_size * params_->c_dtype_bytes) / params_->a_dtype_bytes;
    // aub, bub total size
    int32_t al1_size = CubeUtil::CalcAL1Size(params_, l1_status, l0_status) / params_->a_dtype_bytes;
    int32_t bl1_size = CubeUtil::CalcBL1Size(params_, l1_status, l0_status) / params_->b_dtype_bytes;
    int64_t al1_load_size = al1_size * l1_status.al1_repeat_time * l1_status.db_al1;
    int64_t bl1_load_size = bl1_size * l1_status.bl1_repeat_time * l1_status.db_bl1;
    int64_t aub_move_size = (params_->aub_fused_num + 1) * al1_load_size * ub_status_.db_aub;
    int64_t bub_move_size = (params_->bub_fused_num + 1) * bl1_load_size * ub_status_.db_bub;
    // maybe used_size > real_size
    int32_t aub_rate_size =
        static_cast<int32_t>(ub_rest_size * (static_cast<float>(aub_move_size) / (aub_move_size + bub_move_size)));
    int32_t used_aub_size = std::min(std::min(aub_rate_size, al1_size), (ub_rest_size - bub_min_size));
    // AUB_size = mAUB * kAUB * 16 * 16
    // BUB_size = nBUB * 16 * align(kBUB * wi, 16)
    int32_t limit_mk = std::max(used_aub_size / aub_min_size, 1);
    ub_status_.m_aub = MathUtil::NearestFactor(l1_status.m_al1 * l0_status.m, limit_mk);
    ub_status_.k_aub = limit_mk / ub_status_.m_aub;
    ub_status_.k_aub = MathUtil::NearestFactor(l1_status.k_al1, ub_status_.k_aub);
    aub_size = ub_status_.m_aub * ub_status_.k_aub * aub_min_size;
    // align(kBUB * wi, 16) * nBUB < limit_hn
    int32_t limit_hn = (ub_rest_size - aub_size) /
                       ((static_cast<int32_t>(params_->bub_fused_num) + 1) * kBlockSize * ub_status_.db_bub);
    if (IsLargeWi(params_->b_shape.w) && !params_->conv1d_flag) {
      ub_status_.k_bub = 1; // hi = 1, ub tiling w
      wi_bub_ = CalcWiBub(limit_hn, ub_status_.k_bub);
    } else {
      // get min value in CalcHi/CalcWi with fmap, schedule set buffer tile with calculated Hi and Wi, not use min value
      constexpr int32_t max_dim = std::numeric_limits<int32_t>::max();
      int32_t bl1_ho = CubeUtil::CalcHo(l1_status.k_bl1 * kBlockSize, params_->a_shape.w, params_->op_type);
      int32_t bl1_hi = CubeUtil::CalcHi(bl1_ho, params_->stride_h, params_->kernel_h_dilation, max_dim);
      if (params_->conv1d_flag) {
        bl1_hi = 1;
        wi_bub_ =
            CubeUtil::CalcWi(l1_status.k_bl1 * kBlockSize, params_->stride_w, params_->kernel_w_dilation, max_dim);
        wi_bub_ = (limit_hn <= MathUtil::Align(wi_bub_, kBlockSize))
                      ? MathUtil::Align(limit_hn, kBlockSize) - kBlockSize
                      : wi_bub_;
      }
      CalcKBub(limit_hn, bl1_hi);
    }

    ub_status_.n_bub = limit_hn / MathUtil::Align(ub_status_.k_bub * wi_bub_, kBlockSize);
    int32_t ci1_factor = MathUtil::CeilDivision(l1_status.n_bl1 * l0_status.n, params_->kernel_h * params_->kernel_w);
    ub_status_.n_bub = MathUtil::NearestFactor(ci1_factor, ub_status_.n_bub);
    bub_size = (params_->bub_fused_num + 1) * ub_status_.n_bub * kBlockSize *
               MathUtil::Align(ub_status_.k_bub * wi_bub_, kBlockSize) * ub_status_.db_bub;
  } else if (params_->dma_flag && !params_->platform_info.support_l0c2out()) {
    bub_size = KDmaFmapUbSize;
  }
  ub_status_.n_cub = (ub_size - (aub_size * params_->a_dtype_bytes + bub_size * params_->b_dtype_bytes)) /
                     params_->c_dtype_bytes / cub_min_size;
  ub_status_.n_cub = MathUtil::NearestFactor(l0_status.n, ub_status_.n_cub);
  // prevent dst_stride exceed 65536 limit in ub2out cce instruct
  if (params_->a_shape.c1 * params_->c_dtype_bytes * kBlockSize > std::numeric_limits<uint16_t>::max()) {
    ub_status_.n_cub = 1;
  }
}

int32_t UbCalculator::GetTilingWiBub() const {
  return wi_bub_;
}

void UbCalculator::UpdateSinleCoreStatus() {
  single_core_status_.UpdateUbStatus(ub_status_);
  OP_LOGD(params_->op_type, "Update single core status [ub_status][%s]", ub_status_.ToString().c_str());
}
}  // namespace cachetiling
}  // namespace optiling

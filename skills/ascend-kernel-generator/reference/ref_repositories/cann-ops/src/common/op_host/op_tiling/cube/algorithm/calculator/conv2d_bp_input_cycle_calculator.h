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
 * \file conv2d_bp_input_cycle_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_CYCLE_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_CYCLE_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
// (abkl1_attach_flag, al1_attach_flag, bl1_attach_flag)
typedef std::tuple<int32_t, int32_t, int32_t> AttachFlags;
typedef std::tuple<bool, AttachFlags> AttachCondition;

struct MadTilingInfo {
  int64_t m0_single_core;
  int64_t n0_single_core;
  int64_t k0_single_core;
  int64_t k_div_max_kl1;
  int64_t max_kl1_div_min_kl1;
  int64_t min_kl1_div_kl0;
};

struct CycleStatus {
  int64_t mte1_al0_cycle = 0;
  int64_t mte1_bl0_cycle = 0;
  int64_t mte2_a_cycle = 0;
  int64_t mte2_b_cycle = 0;
  int64_t fixp_cycle = 0;
  int64_t mad_cycle = 0;
};

class Conv2DBpInputCycleCalculator : public Calculator {
 public:
  Conv2DBpInputCycleCalculator(SingleCoreStatus &core_status);
  ~Conv2DBpInputCycleCalculator() override = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  void Clear();

 private:
  bool IsBadDim(int32_t core_use) const { return core_use < min_use_core_; }
  bool CheckExpand(int32_t shape, int32_t dim);
  bool CheckL1Size() const;
  bool CheckAl1FullLoad();
  bool CheckBl1FullLoad();
  void UpdateSingleCoreStatus();
  void GetNonFactorN0();
  void FastFindAl1FullLoad();
  void FastFindBl1FullLoad();
  void FastFindNotFullLoad();
  void FastFindPatternTiling(int32_t batch_dim, int32_t m_dim, int32_t n_dim);
  void GetAttachFlag();
  void SetDoubleBuffer();
  int64_t GetCycleByModel();
  int64_t GetPkgNumByCacheline(int64_t burst_len) const;
  int64_t GetMte1ACycle() const;
  int64_t GetMte1BCycle() const;
  int64_t GetMte2ACycle() const;
  int64_t GetMte2BCycle() const;
  int64_t GetMadCycle() const;
  int64_t GetFixpCycle() const;
  void CalCycleBothFullLoad(int64_t &cycle);
  void CalCycleBL1FullLoad(int64_t &cycle);
  void CalCycleAL1FullLoad(int64_t &cycle);
  void CalCycleNeitherFullLoad(int64_t &cycle);

  const TilingShape &orig_shape_;
  const Conv2DBpInputTilingParam *dx_param_ = nullptr;
  const Shape *dedy_ = nullptr;
  const Shape *kernel_ = nullptr;
  const Shape *dedx_ = nullptr;
  bool tiling_pattern_flag_ = false;
  bool is_al1_full_load_pattern_ = false;
  bool is_bl1_full_load_pattern_ = false;
  bool al1_full_load_ = false;
  bool bl1_full_load_ = false;
  TilingShape shape_;
  DimFactor block_dims_;
  L1Status l1_status_;
  L0Status l0_status_;
  CycleStatus cycle_status_;
  AttachFlags attach_flags_;
  MadTilingInfo mad_tiling_info_;
  int32_t kl0_max_size_ = 0;
  int32_t kl0_min_size_ = 0;
  int32_t kl0_offset_ = 0;
  int32_t core_num_ = 0;
  int32_t min_use_core_ = 0;
  int64_t final_cycle_ = 0;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_CYCLE_CALCULATOR_H_
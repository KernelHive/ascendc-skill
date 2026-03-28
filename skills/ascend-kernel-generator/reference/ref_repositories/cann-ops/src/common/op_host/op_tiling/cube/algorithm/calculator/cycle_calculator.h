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
 * \file cycle_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CYCLE_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CYCLE_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
// (abkl1_attach_flag, al1_attach_flag, bl1_attach_flag)
typedef std::tuple<int32_t, int32_t, int32_t> AttachFlags;
typedef std::tuple<bool, AttachFlags> AttachCondition;

struct MadTilingInfo {
  int64_t m_single_core = 0;
  int64_t n_single_core = 0;
  int64_t min_kl1_div_kl0 = 0;
  int64_t k_al0_factor = 0;
  int64_t k_bl0_factor = 0;
  int64_t k_al1_factor = 0;
  int64_t k_bl1_factor = 0;
  int64_t kl1_times = 0;
};

struct CycleStatus {
  int64_t mte1_a = 0;
  int64_t mte1_b = 0;
  int64_t mte2_a = 0;
  int64_t mte2_b = 0;
  int64_t fixp = 0;
  int64_t mad = 0;
};

class CycleCalculator : public Calculator {
 public:
  CycleCalculator(SingleCoreStatus &core_status);
  ~CycleCalculator() override = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  void Clear();

 protected:
  virtual void GenBlockDimsMapFactors();
  virtual bool IsValidBatchDim(int32_t batch_dim) const { return true; }
  virtual void UpdateTilingShape();
  virtual bool PruneBlockDim(int32_t used_core) const;
  virtual bool LoopBlockDims(bool prune, int32_t used_core = 1);
  bool IsInvalidFactor(int32_t factor) const { return factor > core_num_; }

 private:
  bool CheckL1Size() const;
  bool CheckAl1FullLoad();
  bool CheckBl1FullLoad();
  void UpdateSingleCoreStatus();
  bool FastFindAl1FullLoad();
  bool FastFindBl1FullLoad();
  bool FastFindNotFullLoad();
  bool FastFindPatternTiling(const DimFactor &block_dims);
  void GetAttachFlag(const TilingShape &status);
  void SetDoubleBuffer();
  int64_t GetCycleByModel();
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
  bool GetMN0();
  int32_t GetNl0(int32_t max_nl0);
  int32_t GetMl0(int32_t max_ml0);

 protected:
  const TilingShape &orig_shape_;
  TilingShape shape_;
  TilingShape mapped_shape_;
  std::vector<int32_t> batch_dims_factors_;
  std::vector<int32_t> m_dims_factors_;
  std::vector<int32_t> n_dims_factors_;
  std::vector<int32_t> g_dims_factors_;
  std::vector<int64_t> shape_m_vec_;
  std::vector<int64_t> shape_n_vec_;
  DimFactor block_dims_;
  int64_t total_n_l0_ = 0;
  int64_t total_n_l1_ = 0;
  int32_t core_num_ = 0;
  int32_t min_use_core_ = 0;

 private:
  L1Status l1_status_;
  L0Status l0_status_;
  CycleStatus cycle_;
  AttachFlags attach_flags_;
  MadTilingInfo tiling_ { 0 };
  int64_t final_cycle_ = 0;
  int32_t kl0_min_size_ = 0;
  int32_t kl0_offset_ = 0;
  int32_t min_kl1_cmp_kl0_ = 0;
  bool load2d_mode_ = false;
  bool load3d_mode_ = false;
  bool tiling_pattern_flag_ = false;
  bool is_al1_full_load_pattern_ = false;
  bool is_bl1_full_load_pattern_ = false;
  bool al1_full_load_ = false;
  bool bl1_full_load_ = false;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CYCLE_CALCULATOR_H_
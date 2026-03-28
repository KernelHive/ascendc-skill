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
 * \file conv3d_tiling_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_TILING_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_TILING_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"
#include "cube/algorithm/calculator/conv3d_cycle_calculator.h"

namespace optiling {
namespace cachetiling {
class Conv3DTilingCalculator : public Calculator {
 public:
  Conv3DTilingCalculator(SingleCoreStatus &core_status);
  ~Conv3DTilingCalculator() = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  void Clear();

  int64_t CalcAL0Size() const;
  int64_t CalcBL0Size() const;
  int64_t CalcCL0Size() const;
  int64_t CalcAL1Size() const;
  int64_t CalcBL1Size() const;
  int64_t CalcUbSize() const;
  int64_t CalcBiasL1Size() const;
  int32_t GetFinalTilingId() const {
    return best_tiling_id_.tiling_id();
  }

 private:
  void CalcGlobalParams(const Conv3DTilingParam *conv3d_params);
  bool IsInvalidFactor(int32_t factor) const { return factor > core_num_; }
  bool IsOverL1Size() const;
  bool IsOverBiasTableSize() const;
  int64_t CalcBiasTableBound() const;
  bool CalcKal1(int64_t remain_al1_size);
  bool CalcMal1(int64_t remain_al1_size);
  bool CalcKbl1(int64_t remain_bl1_size);
  bool CalcNbl1(int64_t remain_bl1_size);
  bool CheckAl1FullLoad();
  bool UpdateSingleCoreStatus();
  bool CalcUbStatus();
  bool FastFindAl1FullLoad();
  bool FastFindBl1FullLoad();
  bool FastFindBl1NotLoad();
  bool FastFindNotFullLoad();
  bool FastFindPatternTiling(const DimFactor &block_dims);
  void SetDoubleBuffer(const Conv3DTilingIdParam &tiling_id);
  void GenBlockDimsFactors();
  bool GetMN0();
  int32_t GetNl0(int32_t max_nl0);
  int32_t GetMl0(int32_t max_ml0);

  bool NeedCalcNextKl0() const;
  bool CalcNextKl0() {
    l0_status_.k = MathUtil::NearestFactor(total_k_l0_, l0_status_.k - 1);
    return true;
  }

  bool IsBL0FullLoad() const {
    return l0_status_.k == shape_.k && l0_status_.n == shape_.n;
  }

 private:
  const TilingShape &orig_shape_;
  TilingShape shape_;

  std::vector<int32_t> batch_dims_factors_;
  std::vector<int32_t> m_dims_factors_;
  std::vector<int32_t> n_dims_factors_;
  std::vector<int32_t> g_dims_factors_;
  std::vector<int32_t> d_dims_factors_;

  DimFactor block_dims_;
  L1Status l1_status_;
  L0Status l0_status_;
  UbStatus ub_status_;

  bool find_valid_tiling_ = true;
  bool is_al1_full_load_pattern_ = false;
  bool is_filter_can_load_to_l1_ = true;
  bool al1_full_load_ = false;
  bool bl1_full_load_ = false;
  bool pad_greater_than_filter_ = false;

  int32_t core_num_ = 0;
  int32_t min_n_dim_ = 1;
  int32_t bias_dtype_bytes_ = 2;
  int64_t kernel_d_ = 1;
  int64_t weight_size_ = 0;
  int64_t total_k_l0_ = 0;
  int64_t k_l1_without_kh_kw_ = 0;

  Conv3DTilingIdParam tiling_id_;
  Conv3DTilingIdParam best_tiling_id_;
  std::unique_ptr<Conv3DCycleCalculator> cycle_calculator_;
  int64_t final_cycle_ = 0;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_CALCULATOR_CYCLE_CALCULATOR_H_
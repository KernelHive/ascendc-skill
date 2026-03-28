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
 * \file conv2d_bp_input_block_dims_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_BLOCK_DIMS_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_BLOCK_DIMS_CALCULATOR_H_

#include <vector>

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
class Conv2DBpInputBlockDimsCalculator : public Calculator {
 public:
  Conv2DBpInputBlockDimsCalculator(SingleCoreStatus &core_status);
  ~Conv2DBpInputBlockDimsCalculator() override = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  void Clear();

 private:
  void GenBlockDimsFactors(std::vector<int32_t> &n_dims_factors, size_t n_dim_max_factor);
  void GenBlockDimsMapFactors();
  bool IsInvalidFactor(int32_t factor) const {
    return factor > core_num_;
  }
  void GenGroupDimsFactors(std::vector<int32_t> &group_dims_factors, size_t group_dim_max_factor);
  void CalcBlockDims(const DimFactor &block_dims);
  void CalcL1MinLoadSize();
  bool NeedUpdate(const DimFactor &block_dims, int64_t load_size, int32_t core_used, int32_t loop_num);
  void UpdateSingleCoreStatus();
  bool IsSatisfyN(int32_t n_dim);
  bool IsSatisfyM(int32_t m_dim);
  bool MdimTune(int32_t m1, int32_t n_dim_factor, int32_t m_dim_factor);
  void UpdateSingleCoreStatusMapFactor();
  void GetBestBlockFactors();
  void GetNonFactorialBlockFactors();

  size_t idx_n_ = 0;
  size_t idx_group_ = 0;
  const TilingShape &orig_shape_;
  TilingShape shape_;
  DimFactor block_dims_;
  int32_t core_used_ = 1;
  int32_t core_num_ = 0;
  int32_t loop_num_ = 0;
  int64_t a_size_ = 0;
  int64_t b_size_ = 0;
  int64_t min_l1_load_size_ = 0;
  const Conv2DBpInputTilingParam *dx_param_ = nullptr;
  const Shape *dedy_ = nullptr;
  const Shape *kernel_ = nullptr;
  const Shape *dedx_ = nullptr;
  TilingShape mapped_shape_;
  std::vector<int32_t> batch_dims_factors_;
  std::vector<int32_t> m_dims_factors_;
  std::vector<int32_t> n_dims_factors_;
  std::vector<int32_t> g_dims_factors_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_BLOCK_DIMS_CALCULATOR_H_
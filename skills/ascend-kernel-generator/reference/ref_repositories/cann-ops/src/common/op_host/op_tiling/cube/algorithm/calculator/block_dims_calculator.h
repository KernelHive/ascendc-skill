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
 * \file block_dims_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_BLOCK_DIMS_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_BLOCK_DIMS_CALCULATOR_H_

#include <vector>

#include "cube/algorithm/calculator/calculator.h"
#include "cube/algorithm/entity/shape.h"

namespace optiling {
namespace cachetiling {
class BlockDimsCalculator : public Calculator {
 public:
  BlockDimsCalculator(SingleCoreStatus &core_status);
  virtual ~BlockDimsCalculator() = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;

 private:
  void GenBlockDimsFactors();
  void GenBlockDimsMapFactors();
  bool IsInvalidFactor(int32_t factor) const {
    return static_cast<uint32_t>(factor) > params_->platform_info.core_num();
  }
  void CalcBlockDims(const DimFactor &block_dims);
  void CalcL1MinLoadSize();
  int64_t CalcLoadSize(const DimFactor &block_dims) const;
  int64_t FullLoadSize() const;
  int64_t KFullLoadSize() const;
  int64_t SingleKFullLoadSize(KLoadType load_type, int64_t k_full_load_size) const;
  int64_t NeitherFullLoadSize() const;
  bool NeedUpdate(const DimFactor &block_dims, int64_t load_size, int32_t core_used, int64_t loop_num) const;
  void Update(const DimFactor &block_dims, int64_t load_size, int32_t core_used, int64_t loop_num);
  void UpdateSingleCoreStatus();
  void UpdateTilingShape(const DimFactor &block_dims);
  void UpdateTilingShape();

  // NOTE original shape of mmad
  TilingShape orig_shape_;
  TilingShape shape_;
  DimFactor block_dims_;
  TilingShape mapped_shape_;
  std::vector<int32_t> batch_dims_factors_;
  std::vector<int32_t> m_dims_factors_;
  std::vector<int32_t> n_dims_factors_;
  std::vector<int32_t> g_dims_factors_;
  std::vector<int64_t> shape_m_vec_;
  std::vector<int64_t> shape_n_vec_;
  int32_t al1_min_load_size_ = 0;
  int32_t bl1_min_load_size_ = 0;
  int64_t extend_shape_k_ = 0;
  int64_t load_size_ = INT64_MAX;
  int32_t core_used_ = 0;
  int32_t min_single_core_k_ = 0;
  int64_t loop_num_ = 0;
  int32_t max_mk_ = 0;
  int32_t max_nk_ = 0;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_BLOCK_DIMS_CALCULATOR_H_

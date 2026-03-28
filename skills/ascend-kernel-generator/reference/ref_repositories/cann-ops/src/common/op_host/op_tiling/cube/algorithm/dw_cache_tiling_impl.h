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
 * \file dw_cache_tiling_impl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DW_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DW_CACHE_TILING_IMPL_H_

#include <memory>
#include "cube/algorithm/cache_tiling_impl.h"
#include "cube/algorithm/calculator/block_dims_calculator.h"
#include "cube/algorithm/calculator/l0_calculator.h"
#include "cube/algorithm/calculator/l1_calculator.h"
#include "cube/algorithm/calculator/ub_calculator.h"
#include "cube/algorithm/calculator/cycle_calculator.h"
#include "op_tiling/aoe/op_tuning_tiling/conv2d_dw_tuning_tiling.h"
// #include "op_tiling/aoe/runtime_kb/runtime_bank_manager.h"

namespace optiling {
namespace cachetiling {
class TilingIdParam {
 public:
  void Calc(const CubeTilingParam *params, const SingleCoreStatus &status, const CubeTiling &tiling);
  int32_t al1_attach_flag() const { return al1_attach_flag_; }
  int32_t bl1_attach_flag() const { return bl1_attach_flag_; }
  int32_t abkl1_attach_flag() const { return abkl1_attach_flag_; }
  int32_t load_mode() const { return load_mode_; }
  int32_t min_kl1_cmp_kl0() const { return min_kl1_cmp_kl0_; }

 private:
  int32_t al1_attach_flag_ = kAttachFullLoad;
  int32_t bl1_attach_flag_ = kAttachFullLoad;
  int32_t abkl1_attach_flag_ = kAttachFullLoad;
  int32_t load_mode_ = 0;
  int32_t min_kl1_cmp_kl0_ = 0;
};

class DwCacheTilingImpl : public CacheTilingImpl {
 public:
  explicit DwCacheTilingImpl(const CubeTilingParam &params);
  virtual ~DwCacheTilingImpl() = default;
  bool GenTiling(CubeTiling &tiling) override;
  bool Init(const CubeTilingParam &params) override;
  void Clear() override;

 protected:
  virtual void SetOrigShape();
  virtual void SetTiling(CubeTiling &tiling) const;
  virtual bool CheckCycleModelUnsupport() const;

 private:
  void ShowResourceStatistics() const;
  void CalcOrigShape(TilingShape &shape) const;
  void CheckSpecialTemplate();
  void FixTilingParam(CubeTiling &tiling);
  virtual int32_t CalcTilingId(const CubeTiling &tiling, const TilingIdParam &tiling_id_param) const;
  std::string InputArgsToString(const tuningtiling::Conv2DDwInputArgs &input_args) const;
  void BuildRepoQueryParams(tuningtiling::Conv2DDwInputArgs &input_args) const;
  bool TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuning_tiling);
  bool GetTilingFromRepo();

 protected:
  std::unique_ptr<CycleCalculator> cycle_calculator_;

 private:
  BlockDimsCalculator block_dims_calculator_;
  L0Calculator l0_calculator_;
  L1Calculator l1_calculator_;
  UbCalculator ub_calculator_;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DW_CACHE_TILING_IMPL_H_
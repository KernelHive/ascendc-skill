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
 * \file conv3d_bp_input_tiling_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_BP_INPUT_TILING_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_BP_INPUT_TILING_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"
#include "cube/algorithm/calculator/conv3d_bp_input_cycle_calculator.h"

namespace optiling {
namespace cachetiling {
class Conv3DBpInputTilingCalculator : public Calculator {
 public:
  Conv3DBpInputTilingCalculator(SingleCoreStatus &core_status);
  ~Conv3DBpInputTilingCalculator() override = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  void Clear();

  int32_t CalcAL0Size() const;
  int32_t CalcBL0Size() const;
  int32_t CalcCL0Size() const;
  int32_t CalcAL1Size() const;
  int32_t CalcBL1Size() const;

  int32_t GetFinalTilingId() const {
    OP_LOGD(conv3ddx_params_->op_type, "[conv3ddx_best_tiling flag is [%s]]",
            conv3ddx_best_tiling_id_param_.ToString().c_str());
    return conv3ddx_best_tiling_id_param_.tiling_id();
  }

  int64_t GetFinalCycle() const {
    return final_cycle_;
  }

 private:
  bool IsBadDim(int32_t core_use) const { return core_use < min_use_core_; }
  bool UpdateSingleCoreStatus();
  bool FastFindAl1FullLoad();
  bool FastFindBl1FullLoad();
  bool FastFindNotFullLoad();
  bool CheckExpand(int32_t shape, int32_t dim);
  bool Al1FullLoadStatusInit();
  bool Bl1FullLoadStatusInit();
  bool BothNotFullLoadStatusInit();
  bool CheckAL0Size() const;
  bool CheckBL0Size() const;
  bool CheckCL0Size() const;
  bool CheckL1Size() const;
  bool CheckAl1FullLoad();
  bool CheckBl1FullLoad();

  bool FastFindPatternTiling(int32_t group_dim, int32_t batch_dim, int32_t d_dim, int32_t n_dim, int32_t m_dim);
  bool SpecialStatusFiltration();
  void SetDoubleBuffer(const Conv3DDxTilingIdParam &conv3ddx_tiling_id_params);
  void GetNonFactorN0();

 private:
  const TilingShape &orig_shape_;
  TilingShape singlecore_shape_;

  DimFactor block_dims_;
  L1Status l1_status_;
  L0Status l0_status_;
  UbStatus ub_status_;

  bool find_valid_tiling_ = true;
  bool is_al1_full_load_pattern_ = false;
  bool is_bl1_full_load_pattern_ = false;
  bool al1_full_load_ = false;
  bool bl1_full_load_ = false;

  int64_t kernel_d_ = 1;
  int64_t total_k_l0_ = 0;

  int32_t core_num_ = 0;
  int32_t min_use_core_ = 0;
  int32_t kl0_max_size_ = 0;

  int32_t kl0_max_size_FP32_ = 2;
  int32_t kl0_min_size_FP32_ = 2;
  int32_t kl0_max_size_FP16_ = 4;
  int32_t kl0_min_size_FP16_ = 1;
  int32_t l0_status_n_max_withBfullload = 32;
  int32_t l0_status_n_max = 16;
  int32_t l0_status_n_perfer8_withlargem = 8;
  int32_t l0_status_m_max = 16;
  int32_t l0_status_din_max = 8;

  int32_t kl0_min_size_ = 0;
  int32_t kl0_offset_ = 0;

  const Conv3DBpInputTilingParam *conv3ddx_params_ = nullptr;
  Conv3DDxTilingIdParam conv3ddx_tiling_id_params_;
  Conv3DDxTilingIdParam conv3ddx_best_tiling_id_param_;
  std::unique_ptr<Conv3DDxCycleCalculator> conv3ddx_cycle_calculator_;
  int64_t final_cycle_ = 0;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_BP_INPUT_CALCULATOR_CYCLE_CALCULATOR_H_